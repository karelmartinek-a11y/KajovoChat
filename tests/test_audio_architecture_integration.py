from __future__ import annotations

from tools.audio_architecture_harness import (
    device_reset_and_xrun_escalation,
    fallback_webrtc_to_degraded,
    fallback_windows_to_webrtc,
    headset_clean_path,
    start_handsfree_session,
    start_ptt_session,
    transport_reconnect_without_backend_change,
)


def test_integration_start_handsfree_session() -> None:
    result = start_handsfree_session()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec"]
    assert result.final_state == "active"


def test_integration_start_ptt_session() -> None:
    result = start_ptt_session()
    assert result.verdict == "PASS"
    assert result.final_state == "active"
    assert any(item["type"] == "audio_push_to_realtime" for item in result.session_log)


def test_integration_windows_to_webrtc_fallback() -> None:
    result = fallback_windows_to_webrtc()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec", "webrtc_apm"]
    assert result.telemetry_snapshot["fallback_reason"] == "windows_system_aec_unavailable"


def test_integration_webrtc_to_degraded_fallback() -> None:
    result = fallback_webrtc_to_degraded()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec", "webrtc_apm", "degraded_no_aec"]
    assert result.final_state == "degraded"
    assert result.telemetry_snapshot["degradation_cause"] == "reference_pipeline_unhealthy"


def test_integration_headset_clean_path() -> None:
    result = headset_clean_path()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["headset_clean"]
    assert result.telemetry_snapshot["product_mode_key"] == "headset_clean"


def test_integration_transport_reconnect_without_backend_change() -> None:
    result = transport_reconnect_without_backend_change()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec"]
    assert result.telemetry_snapshot["recovery_successes_total"] == 1


def test_integration_device_reset_and_xrun_escalation() -> None:
    result = device_reset_and_xrun_escalation()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec", "webrtc_apm"]
    assert result.telemetry_snapshot["xrun_events_total"] == 3
    assert result.telemetry_snapshot["device_resets_total"] == 2
