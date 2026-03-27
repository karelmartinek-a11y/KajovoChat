from __future__ import annotations
import time

from kajovochat.audio.contracts import SessionHealth
from kajovochat.audio.runtime_resources import AudioRuntimeResources
from kajovochat.audio.session_manager import AudioSessionManager
from kajovochat.audio.session_state import SessionPresentationState, SessionState, validate_session_transition
from kajovochat.audio.aec_engine import AecEngine
from kajovochat.settings import AppSettings


class _DummyMic:
    using_resampler = False
    input_samplerate = 24000

    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _DummyPlayer:
    def __init__(self, *, samplerate: int, device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.device = device
        self.blocksize = blocksize
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _DummyDuplex:
    def __init__(self, *, samplerate: int, input_device: int | None, output_device: int | None, blocksize: int) -> None:
        self.samplerate = samplerate
        self.input_device = input_device
        self.output_device = output_device
        self.blocksize = blocksize
        self.player = _DummyPlayer(samplerate=samplerate, device=output_device, blocksize=blocksize)
        self.mic = _DummyMic(samplerate=samplerate, device=input_device, blocksize=blocksize)
        self.mic_started = False
        self.stopped = False

    def start_mic(self) -> None:
        self.mic_started = True
        self.mic.start()

    def stop(self) -> None:
        self.stopped = True
        self.mic.stop()
        self.player.stop()


class _DummyRT:
    is_connected = True

    def __init__(self) -> None:
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


class _DummyTransport:
    def __init__(self, realtime: _DummyRT) -> None:
        self.realtime = realtime
        self.turn_mode = "server_vad"
        self.calls: list[str] = []

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0):
        del reconnect_attempts
        self.turn_mode = turn_mode
        self.calls.append(turn_mode)
        self.realtime.is_connected = True
        return self.realtime

    def close(self) -> None:
        self.calls.append("close")
        if self.realtime is not None:
            self.realtime.close()
        self.realtime = None


def _build_manager(monkeypatch, *, aec_mode: str = "windows_system_aec", audio_mode: str = "notebook_builtin"):
    state: dict[str, object] = {
        "mode": "idle",
        "input": 1,
        "output": 2,
        "states": [],
        "captions": [],
        "errors": [],
        "aec_mode": aec_mode,
        "logs": [],
    }
    runtime_resources = AudioRuntimeResources()
    manager = AudioSessionManager(
        settings=AppSettings(audio_aec_mode=aec_mode),
        mode_supplier=lambda: state["mode"],
        mode_setter=lambda value: state.__setitem__("mode", value),
        state_sink=lambda value: state["states"].append(value),
        caption_sink=lambda value: state["captions"].append(value),
        error_sink=lambda value: state["errors"].append(value),
        resolve_devices=lambda: None,
        ensure_player=lambda: None,
        start_session_if_needed=lambda: None,
        start_rt_loop=lambda: None,
        stop_rt_loop=lambda: None,
        preferred_frame_size=lambda: 960,
        runtime_resources=runtime_resources,
        input_device_getter=lambda: state["input"],
        output_device_getter=lambda: state["output"],
        guard_profile_supplier=lambda: {"server_vad_threshold": 0.72},
        status_sink=lambda value: state["captions"].append(value),
        user_transcript_sink=lambda value: None,
        assistant_preview_sink=lambda value: None,
        assistant_done_sink=lambda value: None,
        assistant_audio_sink=lambda value: None,
        speech_started_sink=lambda: None,
        speech_stopped_sink=lambda: None,
        response_created_sink=lambda response_id: None,
        response_done_sink=lambda: None,
        log_sink=lambda record_type, payload: state["logs"].append((record_type, payload)),
        aec_mode_setter=lambda value: state.__setitem__("aec_mode", value),
        device_fingerprint_supplier=lambda: "fp-notebook-1",
        audio_mode_supplier=lambda: audio_mode,
        model="gpt-realtime",
        voice="alloy",
        noise_reduction="far_field",
        tts_speed=1.0,
        server_vad_silence_ms=900,
        server_vad_prefix_ms=300,
        server_vad_threshold=0.72,
    )
    realtime = _DummyRT()
    transport = _DummyTransport(realtime)
    runtime_resources.rt = realtime
    manager.transport = transport  # type: ignore[assignment]
    manager.recovery.transport = transport  # type: ignore[assignment]
    monkeypatch.setattr("kajovochat.audio.session_manager.DuplexAudioSession", _DummyDuplex)
    state["runtime_resources"] = runtime_resources
    return manager, state, realtime


def test_aec_engine_exposes_production_backend_chain() -> None:
    assert AecEngine("windows_system_aec").backend_chain == (
        "windows_system_aec",
        "webrtc_apm",
        "degraded_no_aec",
    )
    assert AecEngine("headset_clean").backend_chain == ("headset_clean",)
    assert AecEngine("custom_lab").backend_chain == ("custom_lab",)
    assert AecEngine("windows_system_aec").next_backend_after("windows_system_aec") == "webrtc_apm"


def test_aec_engine_routes_headset_topology_to_headset_clean() -> None:
    engine = AecEngine("windows_system_aec")

    assert engine.requested_backend_for_audio_mode("wired_headset") == "headset_clean"
    assert engine.backend_chain_for("headset_clean") == ("headset_clean",)


def test_session_manager_selects_webrtc_when_windows_backend_is_unavailable(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch)

    class _Probe:
        available = False
        reason = "helper missing"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()

    assert state["mode"] == "handsfree"
    assert state["aec_mode"] == "webrtc_apm"
    assert manager.session_state == SessionState.ACTIVE
    backend_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_selected"]
    assert backend_logs
    assert backend_logs[-1]["selected_backend"] == "webrtc_apm"
    assert backend_logs[-1]["requested_backend"] == "windows_system_aec"


def test_session_manager_uses_headset_clean_for_headset_topology(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="windows_system_aec", audio_mode="wired_headset")

    class _Probe:
        available = True
        reason = "native ready"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()

    assert state["aec_mode"] == "headset_clean"
    assert manager.session_state == SessionState.ACTIVE
    backend_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_selected"]
    assert backend_logs
    assert backend_logs[-1]["selected_backend"] == "headset_clean"
    assert backend_logs[-1]["requested_backend_effective"] == "headset_clean"
    assert backend_logs[-1]["backend_chain"] == ["headset_clean"]
    runtime_resources = state["runtime_resources"]
    assert isinstance(runtime_resources.duplex, _DummyDuplex)
    assert runtime_resources.duplex.mic_started is True
    assert manager.device_graph.duplex is runtime_resources.duplex
    assert manager.device_graph.player is runtime_resources.duplex.player
    assert manager.device_graph.mic is runtime_resources.duplex.mic


def test_session_manager_reference_health_triggers_controlled_fallback(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch)

    class _Probe:
        available = False
        reason = "helper missing"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    assert state["aec_mode"] == "webrtc_apm"

    for _ in range(12):
        manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)

    assert state["aec_mode"] == "degraded_no_aec"
    assert manager.session_state == SessionState.DEGRADED
    fallback_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_fallback"]
    assert fallback_logs
    assert fallback_logs[-1]["from_backend"] == "webrtc_apm"
    assert fallback_logs[-1]["to_backend"] == "degraded_no_aec"
    session_states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert "recovering" in session_states


def test_session_manager_windows_backend_falls_back_when_quality_stays_poor(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="windows_system_aec")

    class _Probe:
        available = True
        reason = "native ready"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    assert state["aec_mode"] == "windows_system_aec"

    for _ in range(6):
        manager.note_aec_observation(
            backend="windows_system_aec",
            reference_miss=False,
            aec_quality=0.01,
            improvement_ratio=0.05,
            delay_samples=360,
            calibration_latency=722,
            similarity=0.21,
            webrtc_success=False,
        )

    assert state["aec_mode"] == "webrtc_apm"
    fallback_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_fallback"]
    assert fallback_logs
    assert fallback_logs[-1]["from_backend"] == "windows_system_aec"
    assert fallback_logs[-1]["to_backend"] == "webrtc_apm"
    assert fallback_logs[-1]["reason"] == "windows_system_aec_unhealthy"


def test_session_manager_treats_windows_system_aec_capture_contract_as_healthy(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="windows_system_aec")

    class _Probe:
        available = True
        reason = "native ready"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    assert state["aec_mode"] == "windows_system_aec"

    for _ in range(8):
        manager.note_aec_observation(
            backend="windows_system_aec",
            reference_miss=False,
            aec_quality=0.0,
            improvement_ratio=0.0,
            delay_samples=0,
            calibration_latency=722,
            similarity=0.0,
            webrtc_success=False,
        )

    assert state["aec_mode"] == "windows_system_aec"
    fallback_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_fallback"]
    assert fallback_logs == []


def test_session_manager_ptt_lifecycle(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.ptt_pressed()
    assert state["mode"] == "ptt"
    assert realtime.cleared is True
    assert state["states"][-1] == "listening"

    manager.ptt_released()
    assert realtime.committed is True
    assert realtime.requested is True
    assert state["states"][-1] == "transcribing"


def test_session_manager_logs_internal_state_transitions(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.request_stop()

    states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert "starting" in states
    assert "probing" in states
    assert "active" in states or "degraded" in states
    assert "stopping" in states
    assert states[-1] == "idle"


def test_session_manager_keeps_runtime_render_and_barge_in_as_ui_only(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_assistant_output_started()
    manager.note_speech_started(during_assistant_output=True)
    manager.note_user_turn_committed()
    manager.note_response_done()

    session_states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    ui_states = list(state["states"])

    assert "speaking" in ui_states
    assert "transcribing" in ui_states
    assert session_states[-1] == "degraded"
    assert all(value not in session_states for value in {"during_assistant_output", "double_talk", "barge_in_transition"})


def test_audio_session_manager_survives_ten_consecutive_handsfree_sessions(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    for _ in range(10):
        realtime = _DummyRT()
        transport = _DummyTransport(realtime)
        manager.transport = transport  # type: ignore[assignment]
        manager.recovery.transport = transport  # type: ignore[assignment]
        manager.start_handsfree()
        assert manager.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}
        manager.request_stop()
        assert manager.session_state == SessionState.IDLE

    states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert states.count("idle") >= 10


def test_duplicate_handsfree_start_is_ignored_without_invalid_transition(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.start_handsfree()

    assert manager.session_state == SessionState.DEGRADED
    ignored_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_start_ignored"]
    assert ignored_logs
    assert ignored_logs[-1]["reason"] == "duplicate_start"
    session_states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert session_states.count("starting") == 1


def test_audio_session_manager_recovers_to_fallback_without_failing_session(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="windows_system_aec")

    class _Probe:
        available = True
        reason = "native ready"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    assert state["aec_mode"] == "windows_system_aec"

    for _ in range(6):
        manager.note_aec_observation(
            backend="windows_system_aec",
            reference_miss=False,
            aec_quality=0.01,
            improvement_ratio=0.04,
            delay_samples=380,
            calibration_latency=722,
            similarity=0.2,
            webrtc_success=False,
        )

    assert state["aec_mode"] == "webrtc_apm"
    assert manager.session_state in {SessionState.RECOVERING, SessionState.ACTIVE}
    manager.note_response_done()
    assert manager.session_state == SessionState.ACTIVE


def test_handsfree_local_fallback_starts_without_playback_even_in_server_vad(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()
    forced = False
    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.09,
            voice_likelihood=0.42,
            now_monotonic=now + 0.4 + (idx * 0.04),
        )

    assert forced is False
    assert realtime.committed is False
    assert realtime.requested is False
    assert manager.awaiting_transcript is False
    assert manager.voice_gate_runtime.local_turn_active is True
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert started_logs
    blocked_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback_blocked"]
    assert blocked_logs == []


def test_handsfree_local_turn_detection_starts_without_server_speech_events(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    now = time.monotonic()
    manager.note_input_audio_appended(4800)
    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.09,
            voice_likelihood=0.42,
            now_monotonic=now + (idx * 0.04),
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is True
    for idx in range(18):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.01,
            voice_likelihood=0.05,
            now_monotonic=now + 0.2 + (idx * 0.04),
        )

    assert forced is True
    assert realtime.committed is True
    assert realtime.requested is True
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert started_logs
    blocked_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback_blocked"]
    assert blocked_logs == []


def test_handsfree_local_fallback_does_not_start_on_dropped_echo(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    now = time.monotonic()
    for idx in range(10):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.18,
            voice_likelihood=0.62,
            now_monotonic=now + (idx * 0.04),
            drop_chunk=True,
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is False
    assert manager.voice_gate_runtime.local_voice_streak == 0
    assert not [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert realtime.committed is False


def test_handsfree_local_fallback_does_not_start_without_buffered_input_audio(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    now = time.monotonic()
    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.09,
            voice_likelihood=0.42,
            now_monotonic=now + (idx * 0.04),
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is False
    assert manager.voice_gate_runtime.local_voice_streak == 0
    assert realtime.committed is False
    blocked_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback_blocked"]
    assert blocked_logs
    assert blocked_logs[-1]["reason"] == "missing_input_audio"


def test_handsfree_local_fallback_waits_after_response_done(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    manager.note_input_audio_appended(4800)
    hold_until = manager.voice_gate_runtime.post_response_hold_until
    now = time.monotonic()

    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.09,
            voice_likelihood=0.46,
            now_monotonic=now + 0.05 + (idx * 0.05),
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is False
    assert manager.voice_gate_runtime.local_voice_streak == 0
    assert hold_until > now
    assert hold_until - now <= 0.2
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert started_logs == []
    assert realtime.committed is False

    forced = False
    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.09,
            voice_likelihood=0.46,
            now_monotonic=max(now + 0.5, hold_until + 0.01) + (idx * 0.04),
        )

    assert forced is False
    assert manager.voice_gate_runtime.local_turn_active is True
    blocked_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback_blocked"]
    assert blocked_logs == []


def test_handsfree_local_fallback_does_not_start_while_playback_is_active(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()

    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.12,
            voice_likelihood=0.52,
            now_monotonic=now + 0.5 + (idx * 0.04),
            playback_active=True,
            barge_in_confirmed=False,
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is False
    assert manager.voice_gate_runtime.local_voice_streak == 0
    blocked_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback_blocked"]
    assert blocked_logs
    assert blocked_logs[-1]["reason"] == "server_vad_authority"
    assert realtime.committed is False

    forced = False
    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.12,
            voice_likelihood=0.52,
            now_monotonic=now + 0.7 + (idx * 0.04),
            playback_active=False,
            barge_in_confirmed=True,
        )

    assert forced is False
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert started_logs


def test_handsfree_local_fallback_does_not_start_when_barge_in_is_confirmed(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()

    for idx in range(3):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.12,
            voice_likelihood=0.52,
            now_monotonic=now + 0.5 + (idx * 0.04),
            playback_active=False,
            barge_in_confirmed=True,
        )
        assert forced is False

    assert manager.voice_gate_runtime.local_turn_active is True
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_started_local_fallback"]
    assert started_logs
    assert realtime.committed is False


def test_server_speech_stopped_commits_normally_without_local_fallback(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()
    for idx in range(18):
        manager.maybe_force_server_vad_turn_commit(
            input_level=0.01,
            voice_likelihood=0.05,
            now_monotonic=now + 0.4 + (idx * 0.04),
        )
    turns_before = manager.telemetry.turns_total

    manager.handle_speech_stopped()

    assert manager.telemetry.turns_total == turns_before + 1
    assert not [payload for record_type, payload in state["logs"] if record_type == "speech_stopped_ignored"]


def test_server_speech_stopped_commits_even_after_transcript_arrives_without_local_fallback(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()
    for idx in range(18):
        manager.maybe_force_server_vad_turn_commit(
            input_level=0.01,
            voice_likelihood=0.05,
            now_monotonic=now + 0.4 + (idx * 0.04),
        )

    manager.handle_user_transcript("ahoj")
    turns_before = manager.telemetry.turns_total
    manager.handle_speech_stopped()

    assert manager.telemetry.turns_total == turns_before + 1
    assert not [payload for record_type, payload in state["logs"] if record_type == "speech_stopped_ignored"]


def test_handsfree_watchdog_commit_requests_response_for_server_turn(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    manager.telemetry.last_server_activity_at = time.monotonic() - 5.0
    manager.voice_gate_runtime.last_voice_activity_at = time.monotonic()

    forced = manager.maybe_force_server_vad_turn_commit(
        input_level=0.22,
        voice_likelihood=0.61,
        now_monotonic=time.monotonic() + 7.0,
    )

    assert forced is True
    assert realtime.committed is True
    assert realtime.requested is True
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_stopped_local_fallback"]
    assert started_logs
    assert started_logs[-1]["fallback_reason"] == "server_vad_watchdog"


def test_handsfree_idle_watchdog_requests_response_for_server_turn(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()
    manager.telemetry.last_server_activity_at = now - 5.0
    manager.voice_gate_runtime.speech_started_at = now - 7.0
    manager.voice_gate_runtime.last_voice_activity_at = now - 4.5
    manager.voice_gate_runtime.server_turn_active = True

    forced = manager.maybe_force_server_vad_idle_watchdog(now_monotonic=now)

    assert forced is True
    assert realtime.committed is True
    assert realtime.requested is True
    started_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_stopped_local_fallback"]
    assert started_logs
    assert started_logs[-1]["fallback_reason"] == "server_vad_watchdog_idle"


def test_handsfree_idle_watchdog_skips_commit_when_buffer_is_too_small(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(960)
    now = time.monotonic()
    manager.telemetry.last_server_activity_at = now - 5.0
    manager.voice_gate_runtime.speech_started_at = now - 7.0
    manager.voice_gate_runtime.last_voice_activity_at = now - 4.5
    manager.voice_gate_runtime.server_turn_active = False
    manager.voice_gate_runtime.local_turn_active = True

    forced = manager.maybe_force_server_vad_idle_watchdog(now_monotonic=now)

    assert forced is False
    assert realtime.committed is False
    assert realtime.requested is False
    skipped_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_commit_skipped_small_buffer"]
    assert skipped_logs
    assert skipped_logs[-1]["buffered_audio_ms"] < skipped_logs[-1]["min_commit_audio_ms"]
    assert manager.voice_gate_runtime.local_turn_active is False


def test_handsfree_server_speech_started_commits_buffered_audio_and_requests_response(monkeypatch) -> None:
    manager, _state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_input_audio_appended(4800)
    manager.handle_speech_started()
    now = time.monotonic()
    forced = False
    for idx in range(18):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.01,
            voice_likelihood=0.05,
            now_monotonic=now + 0.4 + (idx * 0.04),
        )

    assert forced is False
    assert realtime.committed is False
    assert realtime.requested is False
    assert manager.telemetry.current_turn_input_audio_ms() > 0.0


def test_handsfree_local_fallback_does_not_commit_or_request_response(monkeypatch) -> None:
    manager, _state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    call_order: list[tuple[str, float]] = []

    original_commit_input_audio = realtime.commit_input_audio
    original_request_response = realtime.request_response
    original_note_turn_committed = manager.telemetry.note_turn_committed

    def commit_spy() -> None:
        call_order.append(("commit", manager.telemetry.current_turn_input_audio_ms()))
        original_commit_input_audio()

    def request_spy() -> None:
        call_order.append(("request", manager.telemetry.current_turn_input_audio_ms()))
        original_request_response()

    def note_turn_spy() -> None:
        call_order.append(("note_turn", manager.telemetry.current_turn_input_audio_ms()))
        original_note_turn_committed()

    monkeypatch.setattr(realtime, "commit_input_audio", commit_spy)
    monkeypatch.setattr(realtime, "request_response", request_spy)
    monkeypatch.setattr(manager.telemetry, "note_turn_committed", note_turn_spy)

    manager.start_handsfree()
    manager.handle_speech_started()
    manager.note_input_audio_appended(4800)
    now = time.monotonic()

    forced = False
    for idx in range(2):
        forced = manager.maybe_force_server_vad_turn_commit(
            input_level=0.01,
            voice_likelihood=0.05,
            now_monotonic=now + 0.4 + (idx * 0.04),
        )

    assert forced is False
    assert realtime.committed is False
    assert realtime.requested is False
    assert call_order == []


def test_handsfree_idle_watchdog_clears_stale_server_turn_without_audio(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_speech_started()
    now = time.monotonic()
    manager.telemetry.last_server_activity_at = now - 5.0
    manager.voice_gate_runtime.speech_started_at = now - 7.0
    manager.voice_gate_runtime.last_voice_activity_at = now - 4.5
    manager.voice_gate_runtime.server_turn_active = True

    forced = manager.maybe_force_server_vad_idle_watchdog(now_monotonic=now)

    assert forced is False
    assert realtime.committed is False
    assert manager.voice_gate_runtime.server_turn_active is False
    skipped_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_commit_skipped_small_buffer"]
    assert skipped_logs
    assert skipped_logs[-1]["stale_server_turn_cleared"] is True


def test_late_user_transcript_requests_followup_response_when_previous_response_already_finished(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    realtime.requested = False

    manager.handle_user_transcript("ahoj")

    assert realtime.requested is True
    late_logs = [payload for record_type, payload in state["logs"] if record_type == "late_transcript_response_requested"]
    assert late_logs


def test_late_user_transcript_does_not_request_followup_while_server_response_is_still_active(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    manager._server_response_active = True
    realtime.requested = False

    manager.handle_user_transcript("ahoj")

    assert realtime.requested is False
    late_logs = [payload for record_type, payload in state["logs"] if record_type == "late_transcript_response_requested"]
    assert late_logs == []


def test_late_user_transcript_does_not_request_followup_after_server_response_created_event(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_response_done()
    manager.handle_response_created("resp_forensic_123")
    realtime.requested = False

    manager.handle_user_transcript("ahoj")

    assert realtime.requested is False
    response_created_logs = [payload for record_type, payload in state["logs"] if record_type == "response_created"]
    assert response_created_logs
    assert response_created_logs[-1]["response_id"] == "resp_forensic_123"
    late_logs = [payload for record_type, payload in state["logs"] if record_type == "late_transcript_response_requested"]
    assert late_logs == []


def test_user_transcript_clears_stale_turn_input_audio_before_next_fallback(monkeypatch) -> None:
    manager, state, realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_input_audio_appended(4800)
    manager.handle_response_done()
    manager.handle_user_transcript("ahoj")

    assert manager.telemetry.current_turn_input_audio_ms() == 0.0

    now = time.monotonic()
    manager.telemetry.last_server_activity_at = now - 5.0
    manager.voice_gate_runtime.local_turn_active = True
    manager.voice_gate_runtime.speech_started_at = now - 7.0
    manager.voice_gate_runtime.last_voice_activity_at = now - 4.5
    forced = manager.maybe_force_server_vad_idle_watchdog(now_monotonic=now)

    assert forced is False
    assert realtime.committed is False
    skipped_logs = [payload for record_type, payload in state["logs"] if record_type == "speech_commit_skipped_small_buffer"]
    assert skipped_logs


def test_handsfree_restart_after_failed_state_resets_to_idle_first(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.fail("test failure", reason="unknown")
    manager.start_handsfree()

    session_states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert "failed" in session_states
    failed_index = session_states.index("failed")
    assert session_states[failed_index + 1] == "idle"
    assert session_states[failed_index + 2] == "starting"


def test_audio_telemetry_snapshot_returns_session_health_contract(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()

    snapshot = manager.telemetry.snapshot(session_state=manager.session_state)

    assert isinstance(snapshot, SessionHealth)
    assert snapshot.selected_backend == "degraded_no_aec"
    assert snapshot.backend_health.backend == "degraded_no_aec"
    state_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert state_logs
    assert "backend_health" in state_logs[-1]


def test_audio_telemetry_counts_backend_switches_and_degraded_transition(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch)

    class _Probe:
        available = False
        reason = "helper missing"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    for _ in range(12):
        manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)

    snapshot = manager.telemetry.snapshot(session_state=manager.session_state)

    assert snapshot.selected_backend == "degraded_no_aec"
    assert snapshot.backend_switches_total >= 2
    assert snapshot.degraded_transitions_total >= 1
    assert snapshot.health_score < 1.0


def test_audio_session_manager_acceptance_like_barge_in_flow(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_assistant_output_started()
    manager.note_speech_started(during_assistant_output=True)
    manager.note_user_turn_committed()
    manager.note_response_done()

    transitions = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    ui_states = list(state["states"])
    assert all(value not in transitions for value in {"during_assistant_output", "double_talk", "barge_in_transition"})
    assert "speaking" in ui_states
    assert "transcribing" in ui_states
    assert transitions[-1] == "degraded"


def test_audio_session_manager_tracks_barge_in_success_in_health(monkeypatch) -> None:
    manager, _state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_barge_in_result(success=True, reason="user_took_turn")
    manager.note_barge_in_result(success=False, reason="echo_reject")

    snapshot = manager.telemetry.snapshot(session_state=manager.session_state)
    assert snapshot.barge_in_attempts_total == 2
    assert snapshot.barge_in_successes_total == 1
    assert snapshot.backend_health.barge_in_success_ratio == 0.5


def test_audio_session_manager_tracks_xruns_and_device_resets(monkeypatch) -> None:
    manager, _state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_xrun(source="duplex")
    manager.note_xrun(source="duplex")
    manager.note_device_reset(source="render")

    snapshot = manager.telemetry.snapshot(session_state=manager.session_state)
    assert snapshot.xrun_events_total == 2
    assert snapshot.device_resets_total == 1
    assert snapshot.backend_health.xruns == 2
    assert snapshot.backend_health.device_resets == 1
    assert snapshot.health_score < 1.0


def test_audio_session_manager_device_instability_triggers_fallback(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch)

    class _Probe:
        available = False
        reason = "helper missing"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    assert state["aec_mode"] == "webrtc_apm"

    manager.note_xrun(source="capture")
    manager.note_xrun(source="capture")
    manager.note_xrun(source="capture")

    assert state["aec_mode"] == "degraded_no_aec"
    fallback_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_backend_fallback"]
    assert fallback_logs
    assert fallback_logs[-1]["reason"] == "device_unavailable"
    session_states = [payload["session_state"] for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert "recovering" in session_states


def test_session_state_transition_validator_rejects_skipped_path() -> None:
    validate_session_transition(SessionState.IDLE, SessionState.STARTING)
    validate_session_transition(SessionState.STARTING, SessionState.PROBING)
    validate_session_transition(SessionState.PROBING, SessionState.ACTIVE)

    try:
        validate_session_transition(SessionState.IDLE, SessionState.ACTIVE)
    except Exception:
        pass
    else:
        raise AssertionError("Přechod idle -> active má být neplatný bez startingu a probingu.")


def test_aec_engine_select_backend_degrades_when_all_production_backends_are_unavailable() -> None:
    decision = AecEngine("windows_system_aec").select_backend(
        audio_mode="notebook_builtin",
        windows_healthcheck=lambda: (False, "native missing"),
        webrtc_healthcheck=lambda: (False, "webrtc missing"),
    )

    assert decision.requested_backend == "windows_system_aec"
    assert decision.selected_backend == "degraded_no_aec"
    assert decision.degraded is True
    assert decision.fallback_reason == "webrtc_apm_unavailable"
    assert decision.degradation_cause == "webrtc_apm_unavailable"
    assert decision.probe_details["windows_system_aec"] == "native missing"
    assert decision.probe_details["webrtc_apm"] == "webrtc missing"


def test_session_manager_log_payload_exposes_backend_chain(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()

    state_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_session_state"]
    assert state_logs
    assert state_logs[-1]["backend_chain"] == ["degraded_no_aec"]
    assert state_logs[-1]["turn_mode"] == "server_vad"


def test_transport_error_marks_recoverable_failures_in_log(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager._handle_transport_error("socket timed out")

    error_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_session_error"]
    assert error_logs
    assert error_logs[-1]["failure_reason"] == "transport_timeout"
    assert error_logs[-1]["recoverable"] is True


def test_recovery_exhaustion_is_logged(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    class _BrokenTransport(_DummyTransport):
        def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0):
            self.turn_mode = turn_mode
            raise RuntimeError("socket closed")

    broken_transport = _BrokenTransport(_DummyRT())
    broken_transport.realtime = None
    manager.transport = broken_transport  # type: ignore[assignment]
    manager.recovery.transport = broken_transport  # type: ignore[assignment]
    state["mode"] = "handsfree"
    manager.telemetry.reconnect_attempts = 5
    manager.telemetry.scheduled_reconnect_at = 0.0

    manager.tick()

    exhausted_logs = [payload for record_type, payload in state["logs"] if record_type == "recovery_exhausted"]
    assert exhausted_logs
    assert exhausted_logs[-1]["failure_reason"] == "recovery_exhausted"
    assert state["errors"]


def test_audio_telemetry_serializable_snapshot_exposes_recovery_story(monkeypatch) -> None:
    manager, _state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)
    manager.note_barge_in_result(success=True, reason="user_took_turn")

    snapshot = manager.telemetry.serializable_snapshot(session_state=manager.session_state.value).to_log_payload()

    assert snapshot["selected_backend"] == "degraded_no_aec"
    assert snapshot["fallback_chain_step"] == 0
    assert snapshot["reference_health_timeline"]
    assert snapshot["recovery_story"]
    assert "timings" in snapshot
    assert "turn_latency" in snapshot



def test_runtime_watchdog_reconnect_is_centralized_in_recovery_supervisor(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.set_presentation_state(SessionPresentationState.TRANSCRIBING, reason="test_watchdog")
    manager.telemetry.last_server_activity_at = time.monotonic() - 30.0
    manager._runtime_resources.duplex = None
    manager._runtime_resources.player = None
    manager.runtime_pending_snapshot = lambda: {"pending_events": 0, "pending_mic": 0, "pending_player_bytes": 0}  # type: ignore[method-assign]

    manager.check_runtime_health()

    error_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_session_error"]
    reconnect_logs = [payload for record_type, payload in state["logs"] if record_type == "reconnect_scheduled"]
    assert error_logs
    assert error_logs[-1]["failure_reason"] == "transport_timeout"
    assert error_logs[-1]["recovery_action"] == "transport_reconnect"
    assert reconnect_logs


def test_aec_engine_product_mode_contracts_make_product_modes_explicit() -> None:
    engine = AecEngine("windows_system_aec")

    degraded = engine.product_mode_contract_for(
        selected_backend="degraded_no_aec",
        requested_backend="windows_system_aec",
        audio_mode="notebook_builtin",
        degradation_cause="windows_system_aec_unavailable",
    )
    headset = engine.product_mode_contract_for(
        selected_backend="headset_clean",
        requested_backend="headset_clean",
        audio_mode="wired_headset",
    )

    assert degraded.key == "notebook_builtin_degraded_no_aec"
    assert degraded.capture_gate_policy == "degraded_no_aec"
    assert degraded.recovery_policy == "probe_richer_backend_again"
    assert degraded.recovery_retry_budget == 2
    assert headset.key == "headset_clean"
    assert headset.requires_reference is False
    assert headset.recovery_retry_budget == 0


def test_session_manager_telemetry_exposes_product_mode_for_headset(monkeypatch) -> None:
    manager, _state, _realtime = _build_manager(monkeypatch, aec_mode="windows_system_aec", audio_mode="wired_headset")

    class _Probe:
        available = True
        reason = "native ready"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    snapshot = manager.telemetry.serializable_snapshot(session_state=manager.session_state.value).to_log_payload()

    assert snapshot["product_mode_key"] == "headset_clean"
    assert snapshot["capture_gate_policy"] == "headset_clean"
    assert snapshot["recovery_policy"] == "topology_locked"


def test_session_manager_logs_degraded_reason_and_product_mode(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch)

    class _Probe:
        available = False
        reason = "helper missing"

    monkeypatch.setattr("kajovochat.audio.session_manager.windows_system_aec_healthcheck", lambda: _Probe().as_tuple() if hasattr(_Probe(), "as_tuple") else (bool(_Probe().available), _Probe().reason))
    monkeypatch.setattr("kajovochat.audio.session_manager._WebRTCAudioProcessor", object())

    manager.start_handsfree()
    for _ in range(12):
        manager.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)

    snapshot = manager.telemetry.serializable_snapshot(session_state=manager.session_state.value).to_log_payload()
    captions = "\n".join(state["captions"])

    assert snapshot["product_mode_key"] == "notebook_builtin_degraded_no_aec"
    assert snapshot["fallback_reason"] == "reference_pipeline_unhealthy"
    assert snapshot["degradation_cause"] == "reference_pipeline_unhealthy"
    assert "Důvod: reference_pipeline_unhealthy" in captions


def test_session_manager_fails_when_capture_callback_never_becomes_ready(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")
    manager._capture_ready_timeout_s = 0.01
    manager._capture_ready_poll_s = 0.001

    class _NoCaptureDuplex(_DummyDuplex):
        def start_mic(self) -> None:
            self.mic_started = True

        def get_runtime_state(self) -> dict[str, int]:
            return {"captured_samples": 0, "pending_chunk_count": 0, "capture_age_ms": -1}

    monkeypatch.setattr("kajovochat.audio.session_manager.DuplexAudioSession", _NoCaptureDuplex)

    manager.start_handsfree()

    assert manager.session_state == SessionState.FAILED
    assert state["mode"] == "idle"
    assert state["errors"]
    assert state["runtime_resources"].duplex is None
    failure_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_session_start_failed"]
    assert failure_logs
    assert failure_logs[-1]["error_type"] == "RuntimeError"


def test_runtime_fatal_failure_is_logged_and_stops_session(monkeypatch) -> None:
    manager, state, _realtime = _build_manager(monkeypatch, aec_mode="degraded_no_aec")

    manager.start_handsfree()
    manager.handle_runtime_fatal(RuntimeError("socket closed"), stage="runtime_loop")

    assert manager.session_state == SessionState.FAILED
    assert state["mode"] == "idle"
    runtime_logs = [payload for record_type, payload in state["logs"] if record_type == "audio_runtime_fatal"]
    assert runtime_logs
    assert runtime_logs[-1]["stage"] == "runtime_loop"
    assert runtime_logs[-1]["failure_reason"] == "transport_disconnect"
