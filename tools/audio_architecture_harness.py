from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kajovochat.audio.runtime_resources import AudioRuntimeResources
from kajovochat.audio.session_manager import AudioSessionManager
from kajovochat.audio.session_state import SessionState
from kajovochat.settings import AppSettings


class FakeMic:
    using_resampler = False
    input_samplerate = 24000

    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.started = False
        self.stopped = False
        self.pending_chunk_count = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class FakePlayer:
    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.buffered_bytes = 0
        self.stopped = False

    def enqueue_pcm16(self, pcm: bytes) -> None:
        self.buffered_bytes += len(pcm)

    def stop(self) -> None:
        self.stopped = True
        self.buffered_bytes = 0


class FakeDuplex:
    def __init__(self, *, samplerate: int, input_device: int | None, output_device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.input_device = input_device
        self.output_device = output_device
        self.blocksize = blocksize
        self.player = FakePlayer(samplerate=samplerate, device=output_device, blocksize=blocksize)
        self.mic = FakeMic(samplerate=samplerate, device=input_device, blocksize=blocksize)
        self.started = False
        self.stopped = False

    def start_mic(self) -> None:
        self.started = True
        self.mic.start()

    def get_runtime_state(self) -> dict[str, int]:
        return {
            "pending_chunk_count": int(self.mic.pending_chunk_count),
            "buffered_bytes": int(self.player.buffered_bytes),
        }

    def stop(self) -> None:
        self.stopped = True
        self.mic.stop()
        self.player.stop()


class FakeRealtime:
    def __init__(self) -> None:
        self.is_connected = True
        self.pending_event_count = 0
        self.cleared = False
        self.committed = False
        self.requested = False

    def clear_input_audio(self) -> None:
        self.cleared = True

    def commit_input_audio(self) -> None:
        self.committed = True

    def request_response(self) -> None:
        self.requested = True

    def close(self) -> None:
        self.is_connected = False


class FakeTransport:
    def __init__(self) -> None:
        self.realtime = FakeRealtime()
        self.turn_mode = "server_vad"
        self.calls: list[tuple[str, Any]] = []

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0) -> FakeRealtime:
        self.turn_mode = turn_mode
        self.calls.append(("ensure_connected", {"turn_mode": turn_mode, "reconnect_attempts": reconnect_attempts}))
        if self.realtime is None:
            self.realtime = FakeRealtime()
        self.realtime.is_connected = True
        return self.realtime

    def close(self) -> None:
        self.calls.append(("close", {}))
        if self.realtime is not None:
            self.realtime.close()
        self.realtime = None

    def connection_health_snapshot(self) -> dict[str, object]:
        return {
            "turn_mode": self.turn_mode,
            "is_connected": bool(self.realtime is not None and self.realtime.is_connected),
            "has_realtime": self.realtime is not None,
        }


@dataclass
class ScenarioResult:
    scenario: str
    kind: str
    backend_chain: list[str]
    final_state: str
    verdict: str
    telemetry_snapshot: dict[str, object]
    session_log: list[dict[str, object]]
    captions: list[str]


class AudioArchitectureHarness:
    def __init__(
        self,
        *,
        aec_mode: str = "windows_system_aec",
        audio_mode: str = "notebook_builtin",
        windows_available: bool = True,
        webrtc_available: bool = True,
    ) -> None:
        self.state: dict[str, Any] = {
            "mode": "idle",
            "input": 1,
            "output": 2,
            "ui_states": [],
            "captions": [],
            "errors": [],
            "logs": [],
            "aec_mode": aec_mode,
            "audio_mode": audio_mode,
        }
        self.runtime_resources = AudioRuntimeResources()
        self.transport = FakeTransport()
        self.manager = AudioSessionManager(
            settings=AppSettings(audio_aec_mode=aec_mode),
            mode_supplier=lambda: str(self.state["mode"]),
            mode_setter=lambda value: self.state.__setitem__("mode", value),
            state_sink=lambda value: self.state["ui_states"].append(value),
            caption_sink=lambda value: self.state["captions"].append(value),
            error_sink=lambda value: self.state["errors"].append(value),
            resolve_devices=lambda: None,
            ensure_player=lambda: None,
            start_session_if_needed=lambda: None,
            start_rt_loop=lambda: None,
            stop_rt_loop=lambda: None,
            preferred_frame_size=lambda: 960,
            runtime_resources=self.runtime_resources,
            input_device_getter=lambda: self.state["input"],
            output_device_getter=lambda: self.state["output"],
            guard_profile_supplier=lambda: {"server_vad_threshold": 0.72},
            status_sink=lambda value: self.state["captions"].append(value),
            user_transcript_sink=lambda value: None,
            assistant_preview_sink=lambda value: None,
            assistant_done_sink=lambda value: None,
            assistant_audio_sink=lambda value: None,
            speech_started_sink=lambda: None,
            speech_stopped_sink=lambda: None,
            response_created_sink=lambda response_id: None,
            response_done_sink=lambda: None,
            log_sink=self._log_event,
            aec_mode_setter=lambda value: self.state.__setitem__("aec_mode", value),
            device_fingerprint_supplier=lambda: f"fp-{audio_mode}",
            audio_mode_supplier=lambda: str(self.state["audio_mode"]),
            model="gpt-realtime",
            voice="alloy",
            noise_reduction="far_field",
            tts_speed=1.0,
            server_vad_silence_ms=900,
            server_vad_prefix_ms=300,
            server_vad_threshold=0.72,
        )
        self.manager.transport = self.transport  # type: ignore[assignment]
        self.manager.recovery.transport = self.transport  # type: ignore[assignment]
        self.manager._probe_windows_system_aec = lambda: (
            bool(windows_available),
            "Windows System AEC backend je dostupný." if windows_available else "Windows System AEC backend není připraven.",
        )
        self.manager._probe_webrtc_apm = lambda: (
            bool(webrtc_available),
            "WebRTC APM backend je dostupný." if webrtc_available else "WebRTC APM backend není dostupný.",
        )
        import kajovochat.audio.session_manager as session_manager_module

        self._original_duplex = session_manager_module.DuplexAudioSession
        session_manager_module.DuplexAudioSession = FakeDuplex
        self._session_manager_module = session_manager_module

    def close(self) -> None:
        self._session_manager_module.DuplexAudioSession = self._original_duplex

    def _log_event(self, record_type: str, payload: object) -> None:
        self.state["logs"].append({"type": record_type, "payload": payload})

    def snapshot(self) -> dict[str, object]:
        return self.manager.telemetry.serializable_snapshot(session_state=self.manager.session_state.value).to_log_payload()

    def result(self, *, scenario: str, kind: str) -> ScenarioResult:
        snap = self.snapshot()
        return ScenarioResult(
            scenario=scenario,
            kind=kind,
            backend_chain=list(snap.get("recovery_story", [])[-1].get("target_backend", "") for _ in []),
            final_state=self.manager.session_state.value,
            verdict="PASS" if self.state["errors"] == [] else "FAIL",
            telemetry_snapshot=snap,
            session_log=list(self.state["logs"]),
            captions=list(self.state["captions"]),
        )


def _extract_backend_chain(logs: list[dict[str, object]]) -> list[str]:
    chain: list[str] = []
    requested_backend = ""
    for item in logs:
        if not isinstance(item, dict):
            continue
        record_type = item.get("type")
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        if not requested_backend:
            candidate = payload.get("requested_backend_effective") or payload.get("requested_backend")
            if isinstance(candidate, str) and candidate:
                requested_backend = candidate
        backend = None
        if record_type == "audio_backend_selected":
            backend = payload.get("selected_backend")
        elif record_type == "audio_backend_fallback":
            backend = payload.get("to_backend")
        elif record_type == "reconnect_ok" and chain:
            backend = chain[-1]
        if isinstance(backend, str) and backend and (not chain or chain[-1] != backend):
            chain.append(backend)
    should_prepend_requested = bool(
        requested_backend
        and chain
        and chain[0] != requested_backend
        and chain[0] in {"webrtc_apm", "degraded_no_aec"}
    )
    if should_prepend_requested:
        chain.insert(0, requested_backend)
    if not chain:
        snapshot_backend = next((item.get("payload", {}).get("selected_backend") for item in logs if isinstance(item, dict) and isinstance(item.get("payload"), dict) and item.get("payload", {}).get("selected_backend")), "")
        if isinstance(snapshot_backend, str) and snapshot_backend:
            chain.append(snapshot_backend)
    return chain


def _scenario_result(h: AudioArchitectureHarness, *, scenario: str, kind: str) -> ScenarioResult:
    snap = h.snapshot()
    return ScenarioResult(
        scenario=scenario,
        kind=kind,
        backend_chain=_extract_backend_chain(h.state["logs"]),
        final_state=h.manager.session_state.value,
        verdict="PASS" if not h.state["errors"] else "FAIL",
        telemetry_snapshot=snap,
        session_log=list(h.state["logs"]),
        captions=list(h.state["captions"]),
    )


def start_handsfree_session(*, windows_available: bool = True, webrtc_available: bool = True, audio_mode: str = "notebook_builtin") -> ScenarioResult:
    h = AudioArchitectureHarness(
        aec_mode="windows_system_aec",
        audio_mode=audio_mode,
        windows_available=windows_available,
        webrtc_available=webrtc_available,
    )
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="start_handsfree_session", kind="integration")
    finally:
        h.close()


def start_ptt_session() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.ptt_pressed()
        h.manager.ptt_released()
        return _scenario_result(h, scenario="start_ptt_session", kind="integration")
    finally:
        h.close()


def fallback_windows_to_webrtc() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", windows_available=False, webrtc_available=True)
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="windows_system_aec_to_webrtc_apm", kind="integration")
    finally:
        h.close()


def fallback_webrtc_to_degraded() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", windows_available=False, webrtc_available=True)
    try:
        h.manager.start_handsfree()
        for _ in range(12):
            h.manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)
        return _scenario_result(h, scenario="webrtc_apm_to_degraded_no_aec", kind="integration")
    finally:
        h.close()


def headset_clean_path() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="wired_headset")
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="headset_clean_path", kind="integration")
    finally:
        h.close()


def transport_reconnect_without_backend_change() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        initial_backend = h.manager.telemetry.selected_backend
        h.manager.handle_transport_error("connection reset by peer")
        h.manager.telemetry.scheduled_reconnect_at = 0.0
        h.manager.recovery.tick()
        assert h.manager.telemetry.selected_backend == initial_backend
        return _scenario_result(h, scenario="transport_reconnect_without_backend_change", kind="integration")
    finally:
        h.close()


def device_reset_and_xrun_escalation() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        h.manager.note_xrun(source="fault_injection")
        h.manager.note_xrun(source="fault_injection")
        h.manager.note_xrun(source="fault_injection")
        h.manager.note_device_reset(source="fault_injection")
        h.manager.note_device_reset(source="fault_injection")
        return _scenario_result(h, scenario="device_reset_and_xrun_escalation", kind="integration")
    finally:
        h.close()


def acceptance_builtin_windows_available() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="notebook_builtin", windows_available=True)
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="acceptance_builtin_windows_available", kind="acceptance")
    finally:
        h.close()


def acceptance_builtin_windows_fallback_webrtc() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="notebook_builtin", windows_available=False, webrtc_available=True)
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="acceptance_builtin_windows_fallback_webrtc", kind="acceptance")
    finally:
        h.close()


def acceptance_builtin_reference_pipeline_unhealthy() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="notebook_builtin", windows_available=False, webrtc_available=True)
    try:
        h.manager.start_handsfree()
        for _ in range(12):
            h.manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=240)
        return _scenario_result(h, scenario="acceptance_builtin_reference_pipeline_unhealthy", kind="acceptance")
    finally:
        h.close()


def acceptance_wired_headset_clean() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="wired_headset")
    try:
        h.manager.start_handsfree()
        return _scenario_result(h, scenario="acceptance_wired_headset_clean", kind="acceptance")
    finally:
        h.close()


def acceptance_reconnect_during_active_handsfree() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec", audio_mode="notebook_builtin")
    try:
        h.manager.start_handsfree()
        h.manager.handle_transport_error("socket closed")
        assert h.manager.session_state == SessionState.RECOVERING
        h.manager.telemetry.scheduled_reconnect_at = 0.0
        h.manager.recovery.tick()
        assert h.manager.session_state == SessionState.ACTIVE
        return _scenario_result(h, scenario="acceptance_reconnect_during_active_handsfree", kind="acceptance")
    finally:
        h.close()


def soak_long_running_handsfree() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        for i in range(40):
            ready = (i % 5) != 0
            h.manager.note_reference_health(ready=ready, available_samples=480 if ready else 0, callback_age_ms=24 if ready else 90)
            if ready:
                h.manager.handle_user_transcript(f"turn-{i}")
                h.manager.note_assistant_output_started()
                h.manager.handle_assistant_audio(b"\x00\x00" * 120)
                h.manager.handle_response_done()
        return _scenario_result(h, scenario="soak_long_running_handsfree", kind="soak")
    finally:
        h.close()


def soak_repeated_reconnects() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        for _ in range(3):
            h.manager.handle_transport_error("connection reset by peer")
            h.manager.telemetry.scheduled_reconnect_at = 0.0
            h.manager.recovery._last_reconnect_at = 0.0
            h.manager.recovery.tick()
        return _scenario_result(h, scenario="soak_repeated_reconnects", kind="soak")
    finally:
        h.close()


def soak_repeated_tts_and_barge_in() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        for idx in range(6):
            h.manager.telemetry.note_turn_committed()
            h.manager.handle_user_transcript(f"barge-{idx}")
            h.manager.note_assistant_output_started()
            h.manager.handle_assistant_audio(b"\x00\x00" * 80)
            decision = h.manager.evaluate_capture_gate(
                mode="handsfree",
                guard_active=True,
                playback_active=True,
                similarity=0.08,
                input_level=0.12,
                output_level=0.05,
                default_profile={
                    "echo_similarity_drop": 0.8,
                    "echo_similarity_soft": 0.6,
                    "barge_in_min_input_level": 0.06,
                    "barge_in_output_ratio": 1.35,
                },
                voice_likelihood=0.7,
                aec_quality=0.25,
            )
            h.manager.note_barge_in_result(success=bool(decision.barge_in_confirmed), reason=decision.drop_reason)
            h.manager.handle_response_done()
        return _scenario_result(h, scenario="soak_repeated_tts_and_barge_in", kind="soak")
    finally:
        h.close()


def soak_backlog_playback_stagnation_detection() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        h.manager.telemetry.last_player_progress_at = 0.0
        h.runtime_resources.rt.pending_event_count = 7
        h.runtime_resources.mic.pending_chunk_count = 3
        h.runtime_resources.player.buffered_bytes = 4096
        h.manager.telemetry.last_player_progress_at = 0.0
        h.manager.telemetry.last_player_buffer_bytes = 4096
        h.manager.check_runtime_health()
        return _scenario_result(h, scenario="soak_backlog_playback_stagnation_detection", kind="soak")
    finally:
        h.close()


def soak_device_reset_xrun_fault_injection() -> ScenarioResult:
    h = AudioArchitectureHarness(aec_mode="windows_system_aec")
    try:
        h.manager.start_handsfree()
        h.manager.note_device_reset(source="fault_injection")
        h.manager.note_xrun(source="fault_injection")
        h.manager.note_xrun(source="fault_injection")
        h.manager.note_xrun(source="fault_injection")
        return _scenario_result(h, scenario="soak_device_reset_xrun_fault_injection", kind="soak")
    finally:
        h.close()


ALL_SCENARIOS = [
    start_handsfree_session,
    start_ptt_session,
    fallback_windows_to_webrtc,
    fallback_webrtc_to_degraded,
    headset_clean_path,
    transport_reconnect_without_backend_change,
    device_reset_and_xrun_escalation,
    acceptance_builtin_windows_available,
    acceptance_builtin_windows_fallback_webrtc,
    acceptance_builtin_reference_pipeline_unhealthy,
    acceptance_wired_headset_clean,
    acceptance_reconnect_during_active_handsfree,
    soak_long_running_handsfree,
    soak_repeated_reconnects,
    soak_repeated_tts_and_barge_in,
    soak_backlog_playback_stagnation_detection,
    soak_device_reset_xrun_fault_injection,
]


def run_all_scenarios() -> list[ScenarioResult]:
    return [scenario() for scenario in ALL_SCENARIOS]


def write_evidence(output_dir: str | Path) -> list[ScenarioResult]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = run_all_scenarios()
    for result in results:
        scenario_dir = out / result.kind
        scenario_dir.mkdir(parents=True, exist_ok=True)
        stem = result.scenario
        (scenario_dir / f"{stem}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in result.session_log) + "\n",
            encoding="utf-8",
        )
        (scenario_dir / f"{stem}_snapshot.json").write_text(
            json.dumps(result.telemetry_snapshot, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (scenario_dir / f"{stem}_verdict.json").write_text(
            json.dumps(
                {
                    "scenario": result.scenario,
                    "kind": result.kind,
                    "backend_chain": result.backend_chain,
                    "final_state": result.final_state,
                    "verdict": result.verdict,
                    "captions": result.captions,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministický acceptance/integration/soak harness pro audio architekturu KajovoChat.")
    parser.add_argument("--out", required=True, help="Cílový adresář pro evidence soubory.")
    args = parser.parse_args()
    results = write_evidence(args.out)
    summary = [
        {
            "scenario": item.scenario,
            "kind": item.kind,
            "backend_chain": item.backend_chain,
            "final_state": item.final_state,
            "verdict": item.verdict,
        }
        for item in results
    ]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
