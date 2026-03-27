from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from kajovochat.audio.runtime_resources import AudioRuntimeResources
from kajovochat.audio.session_manager import AudioSessionManager
from kajovochat.audio.session_state import SessionPresentationState, SessionState
from kajovochat.settings import AppSettings


class DummyMic:
    using_resampler = False
    input_samplerate = 24000

    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.pending_chunk_count = 0
        self.started = False
        self.stopped = False
        self._captured_samples = 0
        self._last_capture_mono_ns = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class DummyPlayer:
    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.buffered_bytes = 0
        self.stopped = False
        self.enqueued: list[bytes] = []

    def stop(self) -> None:
        self.stopped = True
        self.buffered_bytes = 0

    def enqueue_pcm16(self, pcm: bytes) -> None:
        self.enqueued.append(bytes(pcm))
        self.buffered_bytes += len(pcm)


class DummyDuplex:
    def __init__(self, *, samplerate: int, input_device: int | None, output_device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.input_device = input_device
        self.output_device = output_device
        self.blocksize = blocksize
        self.player = DummyPlayer(samplerate=samplerate, device=output_device, blocksize=blocksize)
        self.mic = DummyMic(samplerate=samplerate, device=input_device, blocksize=blocksize)
        self.mic_started = False
        self.stopped = False
        self.runtime_state = {
            "pending_chunk_count": 0,
            "buffered_bytes": 0,
            "captured_samples": 0,
            "capture_age_ms": -1,
        }

    def start_mic(self) -> None:
        self.mic_started = True
        self.mic.start()
        self.mic.pending_chunk_count = 1
        self.mic._captured_samples = max(int(self.blocksize), 1)
        self.mic._last_capture_mono_ns = time.monotonic_ns()
        self.runtime_state.update(
            pending_chunk_count=1,
            captured_samples=int(self.mic._captured_samples),
            capture_age_ms=0,
        )

    def stop(self) -> None:
        self.stopped = True
        self.mic.stop()
        self.player.stop()

    def get_runtime_state(self) -> dict[str, int]:
        snapshot = dict(self.runtime_state)
        last_capture_ns = int(getattr(self.mic, "_last_capture_mono_ns", 0) or 0)
        if last_capture_ns > 0:
            snapshot["capture_age_ms"] = int((time.monotonic_ns() - last_capture_ns) / 1_000_000)
        snapshot["captured_samples"] = int(getattr(self.mic, "_captured_samples", 0) or 0)
        snapshot["pending_chunk_count"] = int(self.mic.pending_chunk_count)
        return snapshot


class DummyRT:
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


class DummyTransport:
    def __init__(self) -> None:
        self.turn_mode = "semantic_vad"
        self.calls: list[tuple[str, int]] = []
        self.close_calls = 0
        self.realtime: DummyRT | None = DummyRT()

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0):
        self.turn_mode = turn_mode
        self.calls.append((turn_mode, reconnect_attempts))
        self.realtime = DummyRT()
        return self.realtime

    def close(self) -> None:
        self.close_calls += 1
        if self.realtime is not None:
            self.realtime.close()
        self.realtime = None

    @property
    def is_connected(self) -> bool:
        return bool(self.realtime is not None and self.realtime.is_connected)

    def connection_health_snapshot(self) -> dict[str, object]:
        return {
            "turn_mode": self.turn_mode,
            "is_connected": self.is_connected,
            "has_realtime": self.realtime is not None,
        }


@dataclass
class ScenarioResult:
    scenario: str
    verification: str
    expected_final_state: str
    backend_chain: list[str]
    pass_fail: str
    requested_backend: str
    selected_backend: str
    session_state: str
    telemetry_snapshot: dict[str, object]
    session_state_story: list[str]
    recovery_actions: list[str]
    captions: list[str]
    log_excerpt: list[dict[str, object]]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


class ScenarioHarness:
    def __init__(
        self,
        *,
        aec_mode: str = "windows_system_aec",
        audio_mode: str = "notebook_builtin",
        windows_available: bool = True,
        windows_reason: str = "Windows System AEC backend je připraven.",
        webrtc_available: bool = True,
    ) -> None:
        self.state: dict[str, Any] = {
            "mode": "idle",
            "input": 1,
            "output": 2,
            "states": [],
            "captions": [],
            "errors": [],
            "aec_mode": aec_mode,
            "logs": [],
        }
        self.runtime_resources = AudioRuntimeResources()
        self.transport = DummyTransport()
        self._stack = ExitStack()
        self._stack.enter_context(patch("kajovochat.audio.session_manager.DuplexAudioSession", DummyDuplex))
        self._stack.enter_context(
            patch(
                "kajovochat.audio.session_manager.windows_system_aec_healthcheck",
                lambda: (bool(windows_available), str(windows_reason)),
            )
        )
        self._stack.enter_context(
            patch(
                "kajovochat.audio.session_manager._WebRTCAudioProcessor",
                object() if webrtc_available else None,
            )
        )
        settings = AppSettings(audio_aec_mode=aec_mode)
        settings.openai_api_key = "test-key"
        self.manager = AudioSessionManager(
            settings=settings,
            mode_supplier=lambda: self.state["mode"],
            mode_setter=lambda value: self.state.__setitem__("mode", value),
            state_sink=lambda value: self.state["states"].append(value),
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
            response_done_sink=lambda: None,
            log_sink=lambda record_type, payload: self.state["logs"].append((record_type, payload)),
            aec_mode_setter=lambda value: self.state.__setitem__("aec_mode", value),
            device_fingerprint_supplier=lambda: f"fp-{audio_mode}-1",
            audio_mode_supplier=lambda: audio_mode,
            model="gpt-realtime",
            voice="alloy",
            noise_reduction="far_field",
            semantic_vad_eagerness="low",
            tts_speed=1.0,
            server_vad_silence_ms=900,
            server_vad_prefix_ms=300,
            server_vad_threshold=0.72,
        )
        self.manager.transport = self.transport  # type: ignore[assignment]
        self.manager.recovery.transport = self.transport  # type: ignore[assignment]
        self.runtime_resources.rt = self.transport.realtime

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> "ScenarioHarness":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def session_state_story(self) -> list[str]:
        return [payload["session_state"] for record_type, payload in self.state["logs"] if record_type == "audio_session_state"]

    def recovery_actions(self) -> list[str]:
        return [event["action"] for event in self.manager.telemetry.serializable_snapshot(session_state=self.manager.session_state.value).recovery_story]

    def log_excerpt(self) -> list[dict[str, object]]:
        excerpt: list[dict[str, object]] = []
        for record_type, payload in self.state["logs"][-10:]:
            excerpt.append({"record_type": record_type, "payload": payload})
        return excerpt

    def serializable_snapshot(self) -> dict[str, object]:
        return self.manager.telemetry.serializable_snapshot(session_state=self.manager.session_state.value).to_log_payload()

    def result(self, *, scenario: str, verification: str, expected_final_state: str, passed: bool) -> ScenarioResult:
        snapshot = self.serializable_snapshot()
        return ScenarioResult(
            scenario=scenario,
            verification=verification,
            expected_final_state=expected_final_state,
            backend_chain=list(self.manager.aec_engine.backend_chain_for(self.manager.telemetry.requested_backend)),
            pass_fail="PASS" if passed else "FAIL",
            requested_backend=self.manager.telemetry.requested_backend,
            selected_backend=self.manager.telemetry.selected_backend,
            session_state=self.manager.session_state.value,
            telemetry_snapshot=snapshot,
            session_state_story=self.session_state_story(),
            recovery_actions=self.recovery_actions(),
            captions=list(self.state["captions"]),
            log_excerpt=self.log_excerpt(),
        )


def scenario_acceptance_windows_system_aec() -> ScenarioResult:
    with ScenarioHarness() as harness:
        harness.manager.start_handsfree()
        passed = harness.manager.session_state == SessionState.ACTIVE and harness.manager.telemetry.selected_backend == "windows_system_aec"
        return harness.result(
            scenario="interní notebook mic + speakers, systémové AEC dostupné",
            verification="scripted run přes ScenarioHarness.start_handsfree()",
            expected_final_state="active",
            passed=passed,
        )


def scenario_acceptance_webrtc_fallback() -> ScenarioResult:
    with ScenarioHarness(windows_available=False, windows_reason="Windows System AEC backend není připraven.") as harness:
        harness.manager.start_handsfree()
        passed = harness.manager.session_state == SessionState.ACTIVE and harness.manager.telemetry.selected_backend == "webrtc_apm"
        return harness.result(
            scenario="interní notebook mic + speakers, systémové AEC nedostupné, fallback na webrtc_apm",
            verification="scripted run s Windows healthcheck=False",
            expected_final_state="active",
            passed=passed,
        )


def scenario_acceptance_degraded_reference_unhealthy() -> ScenarioResult:
    with ScenarioHarness(windows_available=False, windows_reason="Windows System AEC backend není připraven.") as harness:
        harness.manager.start_handsfree()
        for _ in range(12):
            harness.manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)
        passed = harness.manager.session_state == SessionState.DEGRADED and harness.manager.telemetry.selected_backend == "degraded_no_aec"
        return harness.result(
            scenario="interní notebook mic + speakers, reference pipeline unhealthy, fallback na degraded_no_aec",
            verification="scripted run s reference miss fault injection",
            expected_final_state="degraded",
            passed=passed,
        )


def scenario_acceptance_headset_clean() -> ScenarioResult:
    with ScenarioHarness(audio_mode="wired_headset") as harness:
        harness.manager.start_handsfree()
        passed = harness.manager.session_state == SessionState.ACTIVE and harness.manager.telemetry.selected_backend == "headset_clean"
        return harness.result(
            scenario="wired headset, přímý headset_clean",
            verification="scripted run s audio_mode=wired_headset",
            expected_final_state="active",
            passed=passed,
        )


def scenario_acceptance_transport_reconnect() -> ScenarioResult:
    with ScenarioHarness() as harness:
        harness.manager.start_handsfree()
        backend_before = harness.manager.telemetry.selected_backend
        harness.manager.handle_transport_error("socket timed out")
        harness.manager.telemetry.scheduled_reconnect_at = 0.0
        harness.manager.tick()
        session_story = harness.session_state_story()
        passed = (
            harness.manager.session_state == SessionState.ACTIVE
            and harness.manager.telemetry.selected_backend == backend_before
            and "recovering" in session_story
            and session_story.count("failed") == 0
        )
        return harness.result(
            scenario="reconnect při aktivní hands-free session bez session-state chaosu",
            verification="scripted run s transport timeout + manager.tick() reconnect",
            expected_final_state="active",
            passed=passed,
        )


def scenario_integration_ptt_session() -> ScenarioResult:
    with ScenarioHarness(aec_mode="degraded_no_aec") as harness:
        harness.manager.ptt_pressed()
        harness.manager.ptt_released()
        realtime = harness.transport.realtime
        passed = (
            harness.state["mode"] == "ptt"
            and realtime is not None
            and realtime.committed is True
            and realtime.requested is True
            and harness.manager.presentation_state == SessionPresentationState.TRANSCRIBING
        )
        return harness.result(
            scenario="integration: start PTT session",
            verification="public PTT API ptt_pressed()/ptt_released()",
            expected_final_state="active",
            passed=passed,
        )


def scenario_integration_transport_reconnect_without_backend_change() -> ScenarioResult:
    with ScenarioHarness(aec_mode="degraded_no_aec") as harness:
        harness.manager.start_handsfree()
        selected = harness.manager.telemetry.selected_backend
        harness.manager.handle_transport_error("socket timed out")
        harness.manager.telemetry.scheduled_reconnect_at = 0.0
        harness.manager.tick()
        passed = harness.manager.telemetry.selected_backend == selected and harness.manager.telemetry.recovery_successes_total >= 1
        return harness.result(
            scenario="integration: transport reconnect bez backend změny",
            verification="timeout + reconnect přes RecoverySupervisor bez backend switch",
            expected_final_state="degraded",
            passed=passed,
        )


def scenario_integration_device_reset_and_xrun_escalation() -> ScenarioResult:
    with ScenarioHarness(windows_available=False, windows_reason="Windows System AEC backend není připraven.") as harness:
        harness.manager.start_handsfree()
        harness.manager.note_xrun(source="capture")
        harness.manager.note_xrun(source="capture")
        harness.manager.note_xrun(source="capture")
        harness.manager.note_device_reset(source="render")
        harness.manager.note_device_reset(source="render")
        passed = harness.manager.telemetry.selected_backend == "degraded_no_aec" and harness.manager.session_state == SessionState.DEGRADED
        return harness.result(
            scenario="integration: device reset a xrun escalation",
            verification="fault injection přes note_xrun()/note_device_reset()",
            expected_final_state="degraded",
            passed=passed,
        )


def scenario_soak_long_running_handsfree() -> ScenarioResult:
    with ScenarioHarness() as harness:
        harness.manager.start_handsfree()
        for i in range(24):
            harness.manager.note_reference_health(ready=True, available_samples=2400 + i, callback_age_ms=8)
            harness.manager.handle_speech_stopped()
            harness.manager.telemetry.note_response_started()
            harness.manager.handle_assistant_audio(b"\x00\x00" * 240)
            harness.manager.handle_response_done()
        snapshot = harness.manager.telemetry.serializable_snapshot(session_state=harness.manager.session_state.value).to_log_payload()
        passed = harness.manager.session_state == SessionState.ACTIVE and int(snapshot["turn_latency"]["responses_completed_total"]) >= 24
        return harness.result(
            scenario="soak: dlouhý běh hands-free relace",
            verification="24 syntetických turnů v jedné relaci",
            expected_final_state="active",
            passed=passed,
        )


def scenario_soak_repeated_tts_barge_in() -> ScenarioResult:
    with ScenarioHarness(aec_mode="windows_system_aec") as harness:
        harness.manager.start_handsfree()
        profile = {
            "echo_similarity_drop": 0.8,
            "echo_similarity_soft": 0.6,
            "barge_in_min_input_level": 0.06,
            "barge_in_output_ratio": 1.35,
        }
        for _ in range(12):
            harness.manager.note_assistant_output_started()
            harness.manager.evaluate_capture_gate(
                mode="handsfree",
                guard_active=True,
                playback_active=True,
                similarity=0.12,
                input_level=0.12,
                output_level=0.05,
                default_profile=profile,
                voice_likelihood=0.65,
                aec_quality=0.25,
            )
            harness.manager.evaluate_capture_gate(
                mode="handsfree",
                guard_active=True,
                playback_active=True,
                similarity=0.12,
                input_level=0.12,
                output_level=0.05,
                default_profile=profile,
                voice_likelihood=0.65,
                aec_quality=0.25,
            )
            third = harness.manager.evaluate_capture_gate(
                mode="handsfree",
                guard_active=True,
                playback_active=True,
                similarity=0.12,
                input_level=0.12,
                output_level=0.05,
                default_profile=profile,
                voice_likelihood=0.65,
                aec_quality=0.25,
            )
            harness.manager.note_barge_in_result(success=bool(third.barge_in_confirmed), reason=third.drop_reason)
            harness.manager.note_response_done()
        snapshot = harness.manager.telemetry.snapshot(session_state=harness.manager.session_state.value)
        passed = snapshot.barge_in_attempts_total == 12 and snapshot.barge_in_successes_total >= 10
        return harness.result(
            scenario="soak: opakované TTS/render okno + barge-in",
            verification="12 opakování capture gate během assistant output",
            expected_final_state="active",
            passed=passed,
        )


def scenario_soak_backlog_playback_stagnation() -> ScenarioResult:
    with ScenarioHarness(aec_mode="degraded_no_aec") as harness:
        harness.manager.start_handsfree()
        harness.manager.set_presentation_state(SessionPresentationState.TRANSCRIBING, reason="stagnation_test")
        harness.manager.telemetry.last_player_progress_at = time.monotonic() - 10.0
        harness.manager.telemetry.last_player_buffer_bytes = 4096
        if harness.manager._runtime_resources.duplex is not None:
            harness.manager._runtime_resources.duplex.runtime_state = {"pending_chunk_count": 0, "buffered_bytes": 4096}
            harness.manager._runtime_resources.duplex.player.buffered_bytes = 4096
        harness.manager.check_runtime_health()
        passed = any(record_type == "audio_session_error" and payload.get("failure_reason") == "playback_stagnation" for record_type, payload in harness.state["logs"])
        return harness.result(
            scenario="soak: backlog/playback stagnation detekce",
            verification="fault injection přes duplex runtime backlog state",
            expected_final_state="degraded",
            passed=passed,
        )


def scenario_soak_repeated_reconnects() -> ScenarioResult:
    with ScenarioHarness() as harness:
        harness.manager.start_handsfree()
        backend = harness.manager.telemetry.selected_backend
        for _ in range(4):
            harness.manager.recovery._last_reconnect_at = time.monotonic() - 2.0
            harness.manager.recovery._last_reconnect_reason = ""
            harness.manager.handle_transport_error("socket timed out")
            harness.manager.telemetry.scheduled_reconnect_at = 0.0
            harness.manager.tick()
        passed = harness.manager.session_state == SessionState.ACTIVE and harness.manager.telemetry.selected_backend == backend and harness.manager.telemetry.recovery_successes_total >= 4
        return harness.result(
            scenario="soak: opakované reconnecty",
            verification="4x timeout + reconnect loop s dodrženým anti-oscillation guard oknem",
            expected_final_state="active",
            passed=passed,
        )


def scenario_soak_device_reset_xrun_faults() -> ScenarioResult:
    with ScenarioHarness(windows_available=False, windows_reason="Windows System AEC backend není připraven.") as harness:
        harness.manager.start_handsfree()
        for _ in range(3):
            harness.manager.note_xrun(source="duplex")
        for _ in range(2):
            harness.manager.note_device_reset(source="render")
        snapshot = harness.manager.telemetry.snapshot(session_state=harness.manager.session_state.value)
        passed = snapshot.xrun_events_total == 3 and snapshot.device_resets_total == 2 and harness.manager.telemetry.selected_backend == "degraded_no_aec"
        return harness.result(
            scenario="soak: device reset / xrun fault injection",
            verification="kombinovaný xrun + device reset fault injection",
            expected_final_state="degraded",
            passed=passed,
        )


ACCEPTANCE_SCENARIOS: list[Callable[[], ScenarioResult]] = [
    scenario_acceptance_windows_system_aec,
    scenario_acceptance_webrtc_fallback,
    scenario_acceptance_degraded_reference_unhealthy,
    scenario_acceptance_headset_clean,
    scenario_acceptance_transport_reconnect,
]

INTEGRATION_SCENARIOS: list[Callable[[], ScenarioResult]] = [
    scenario_acceptance_windows_system_aec,
    scenario_integration_ptt_session,
    scenario_acceptance_webrtc_fallback,
    scenario_acceptance_degraded_reference_unhealthy,
    scenario_acceptance_headset_clean,
    scenario_integration_transport_reconnect_without_backend_change,
    scenario_integration_device_reset_and_xrun_escalation,
]

SOAK_SCENARIOS: list[Callable[[], ScenarioResult]] = [
    scenario_soak_long_running_handsfree,
    scenario_soak_repeated_reconnects,
    scenario_soak_repeated_tts_barge_in,
    scenario_soak_backlog_playback_stagnation,
    scenario_soak_device_reset_xrun_faults,
]


def run_scenarios(scenarios: list[Callable[[], ScenarioResult]]) -> list[ScenarioResult]:
    return [scenario() for scenario in scenarios]


def write_results(results: list[ScenarioResult], out_dir: str | Path) -> list[Path]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for result in results:
        filename = (
            result.scenario.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("+", "plus")
            .replace(",", "")
            .replace(":", "")
            .replace("-", "_")
        )
        target = path / f"{filename}.json"
        target.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(target)
    return written
