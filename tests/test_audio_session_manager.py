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
