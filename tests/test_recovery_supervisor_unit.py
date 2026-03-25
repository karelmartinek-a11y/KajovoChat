from __future__ import annotations

from kajovochat.audio.recovery import FailureReason, RecoverySupervisor
from kajovochat.audio.telemetry import AudioTelemetry


class _Transport:
    def __init__(self) -> None:
        self.turn_mode = "server_vad"
        self.realtime = None
        self.closed = 0

    def close(self) -> None:
        self.closed += 1

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0):
        del reconnect_attempts
        self.turn_mode = turn_mode
        self.realtime = object()
        return self.realtime



def _build_supervisor(logs: list[tuple[str, object]]):
    telemetry = AudioTelemetry()
    transport = _Transport()
    flags = {"recovering": 0, "failed": 0, "fallbacks": 0, "restores": 0}
    supervisor = RecoverySupervisor(
        telemetry=telemetry,
        transport=transport,
        mode_supplier=lambda: "handsfree",
        state_sink=lambda value: None,
        caption_sink=lambda value: logs.append(("caption", value)),
        log_sink=lambda record_type, payload: logs.append((record_type, payload)),
        enter_recovering=lambda reason: flags.__setitem__("recovering", flags["recovering"] + 1),
        stop_session=lambda: flags.__setitem__("failed", flags["failed"] + 1),
        fail_session=lambda message, reason: flags.__setitem__("failed", flags["failed"] + 1),
        error_sink=lambda message: logs.append(("error", message)),
        selected_backend_supplier=lambda: telemetry.selected_backend,
        fallback_handler=lambda reason: flags.__setitem__("fallbacks", flags["fallbacks"] + 1) or True,
        restore_session_state=lambda reason: flags.__setitem__("restores", flags["restores"] + 1),
        stop_playback=lambda: logs.append(("stop_playback", {})),
    )
    return supervisor, telemetry, transport, flags


def test_recovery_supervisor_suppresses_oscillating_fallbacks() -> None:
    logs: list[tuple[str, object]] = []
    supervisor, telemetry, _transport, flags = _build_supervisor(logs)
    telemetry.selected_backend = "webrtc_apm"

    supervisor.handle_failure(
        FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value,
        session_state="active",
        message="reference stale",
    )
    supervisor.handle_failure(
        FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value,
        session_state="active",
        message="reference stale again",
    )

    assert flags["fallbacks"] == 1
    assert any(record_type == "audio_backend_fallback_suppressed" for record_type, _payload in logs)


def test_recovery_supervisor_reconnect_policy_stays_transport_only() -> None:
    logs: list[tuple[str, object]] = []
    supervisor, telemetry, transport, flags = _build_supervisor(logs)
    telemetry.selected_backend = "windows_system_aec"

    supervisor.handle_failure(
        FailureReason.TRANSPORT_TIMEOUT.value,
        session_state="active",
        message="socket timed out",
    )
    telemetry.scheduled_reconnect_at = 0.0
    supervisor.tick()

    assert flags["fallbacks"] == 0
    assert flags["restores"] == 1
    assert transport.turn_mode == "server_vad"
    assert any(record_type == "reconnect_scheduled" for record_type, _payload in logs)
    assert any(record_type == "reconnect_ok" for record_type, _payload in logs)
