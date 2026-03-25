from __future__ import annotations

from kajovochat.audio.aec_engine import AecEngine
from kajovochat.audio.recovery import FailureReason, RecoverySupervisor
from kajovochat.audio.session_state import SessionPresentationState, SessionState, session_state_to_ui_state
from kajovochat.audio.telemetry import AudioTelemetry


class _DummyTransport:
    def __init__(self) -> None:
        self.turn_mode = "server_vad"
        self.realtime = None
        self.ensure_connected_calls = 0
        self.closed = False

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0) -> None:
        self.turn_mode = turn_mode
        self.ensure_connected_calls += 1

    def close(self) -> None:
        self.closed = True
        self.realtime = None


def test_aec_engine_exposes_first_class_product_modes() -> None:
    engine = AecEngine("windows_system_aec")

    degraded = engine.product_mode_contract_for(
        selected_backend="degraded_no_aec",
        requested_backend="windows_system_aec",
        audio_mode="notebook_builtin",
        degradation_cause="reference_pipeline_unhealthy",
    )
    headset = engine.product_mode_contract_for(
        selected_backend="headset_clean",
        requested_backend="headset_clean",
        audio_mode="wired_headset",
    )

    assert degraded.capture_gate_policy == "degraded_no_aec"
    assert degraded.recovery_policy == "probe_richer_backend_again"
    assert degraded.recovery_retry_budget == 2
    assert headset.requires_reference is False
    assert headset.recovery_policy == "topology_locked"


def test_audio_telemetry_serializable_snapshot_contains_recovery_story() -> None:
    telemetry = AudioTelemetry()
    telemetry.mark_session_started(
        requested_backend="windows_system_aec",
        device_fingerprint="fp1",
        audio_mode="notebook_builtin",
    )
    telemetry.mark_probe_started()
    telemetry.mark_probe_completed()
    telemetry.mark_session_activated()
    telemetry.note_backend_selected(selected_backend="webrtc_apm", fallback_reason="windows_system_aec_unavailable")
    telemetry.note_reference_health(ready=False, available_samples=0, callback_age_ms=190)
    telemetry.note_xrun()
    telemetry.record_recovery_story(
        category="fallback",
        reason="windows_system_aec_unavailable",
        action="backend_fallback",
        session_state="recovering",
        selected_backend="windows_system_aec",
        target_backend="webrtc_apm",
        cooldown_s=8.0,
    )
    snapshot = telemetry.serializable_snapshot(session_state="active").to_log_payload()

    assert snapshot["selected_backend"] == "webrtc_apm"
    assert snapshot["fallback_reason"] == "windows_system_aec_unavailable"
    assert snapshot["xrun_events_total"] == 1
    assert snapshot["recovery_story"]
    assert snapshot["timings"]["session_start_mono"] > 0.0


def test_recovery_supervisor_applies_transport_reconnect_without_parallel_policy() -> None:
    telemetry = AudioTelemetry()
    telemetry.mark_session_started(
        requested_backend="windows_system_aec",
        device_fingerprint="fp1",
        audio_mode="notebook_builtin",
    )
    transport = _DummyTransport()
    logs: list[tuple[str, object]] = []
    states: list[str] = []
    captions: list[str] = []

    supervisor = RecoverySupervisor(
        telemetry=telemetry,
        transport=transport,
        mode_supplier=lambda: "handsfree",
        state_sink=lambda value: states.append(value),
        caption_sink=lambda value: captions.append(value),
        log_sink=lambda record_type, payload: logs.append((record_type, payload)),
        enter_recovering=lambda reason: states.append(f"recovering:{reason}"),
        stop_session=lambda: None,
        fail_session=lambda message, reason: states.append(f"failed:{reason}:{message}"),
        error_sink=lambda message: captions.append(message),
        selected_backend_supplier=lambda: telemetry.selected_backend,
        fallback_handler=lambda reason: False,
        restore_session_state=lambda reason: states.append(f"restored:{reason}"),
        stop_playback=lambda: None,
    )

    supervisor.handle_transport_error("socket closed", session_state="active")

    assert telemetry.reconnect_attempts == 1
    assert telemetry.last_failure_reason == FailureReason.TRANSPORT_DISCONNECT.value
    assert transport.closed is True
    assert any(record_type == "reconnect_scheduled" for record_type, _ in logs)
    assert any(item.startswith("recovering:") for item in states)


def test_session_state_ui_mapping_uses_single_official_translation() -> None:
    assert session_state_to_ui_state(SessionState.STARTING) == "connecting"
    assert session_state_to_ui_state(SessionState.RECOVERING) == "reconnecting"
    assert session_state_to_ui_state(SessionState.ACTIVE, SessionPresentationState.SPEAKING) == "speaking"
    assert session_state_to_ui_state(SessionState.DEGRADED, SessionPresentationState.TRANSCRIBING) == "transcribing"
