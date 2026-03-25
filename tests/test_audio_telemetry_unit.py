from __future__ import annotations

from kajovochat.audio.aec_engine import AecEngine
from kajovochat.audio.telemetry import AudioTelemetry


def test_audio_telemetry_serializable_snapshot_contains_single_story() -> None:
    telemetry = AudioTelemetry()
    telemetry.mark_session_started(requested_backend="windows_system_aec", device_fingerprint="fp-1", audio_mode="notebook_builtin")
    telemetry.mark_probe_started()
    telemetry.mark_probe_completed()
    telemetry.note_backend_selected(
        selected_backend="webrtc_apm",
        fallback_reason="windows_system_aec_unavailable",
        mode_contract=AecEngine("windows_system_aec").product_mode_contract_for(
            selected_backend="webrtc_apm",
            requested_backend="windows_system_aec",
            audio_mode="notebook_builtin",
        ),
    )
    telemetry.schedule_reconnect(delay_s=1.5, failure_reason="transport_timeout")
    telemetry.note_recovery_success()
    telemetry.note_reference_health(ready=False, available_samples=0, callback_age_ms=180)
    telemetry.note_xrun()
    telemetry.note_device_reset()

    snapshot = telemetry.serializable_snapshot(session_state="recovering").to_log_payload()

    assert snapshot["requested_backend"] == "windows_system_aec"
    assert snapshot["selected_backend"] == "webrtc_apm"
    assert snapshot["fallback_reason"] == "windows_system_aec_unavailable"
    assert snapshot["reconnect_attempts"] == 1
    assert snapshot["xrun_events_total"] == 1
    assert snapshot["device_resets_total"] == 1
    assert snapshot["reference_health_timeline"]
    assert snapshot["recovery_story"]
    assert snapshot["backend_health_score"] < 1.0


def test_audio_telemetry_caps_story_and_timeline_lengths() -> None:
    telemetry = AudioTelemetry()
    telemetry.mark_session_started(requested_backend="degraded_no_aec", device_fingerprint="fp-2", audio_mode="notebook_builtin")

    for index in range(80):
        telemetry.record_recovery_story(
            category="test",
            reason=f"reason-{index}",
            action="observe",
            session_state="active",
        )
    for index in range(60):
        telemetry.note_reference_health(ready=bool(index % 2), available_samples=index, callback_age_ms=index)

    snapshot = telemetry.serializable_snapshot(session_state="active")
    assert len(snapshot.recovery_story) == 64
    assert len(snapshot.reference_health_timeline) == 48
