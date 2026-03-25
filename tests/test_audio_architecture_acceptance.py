from __future__ import annotations

from tools.audio_architecture_harness import (
    acceptance_builtin_reference_pipeline_unhealthy,
    acceptance_builtin_windows_available,
    acceptance_builtin_windows_fallback_webrtc,
    acceptance_reconnect_during_active_handsfree,
    acceptance_wired_headset_clean,
)


def test_acceptance_builtin_windows_available() -> None:
    result = acceptance_builtin_windows_available()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec"]
    assert result.final_state == "active"


def test_acceptance_builtin_windows_fallback_webrtc() -> None:
    result = acceptance_builtin_windows_fallback_webrtc()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec", "webrtc_apm"]
    assert result.telemetry_snapshot["product_mode_key"] == "notebook_builtin_webrtc_apm"


def test_acceptance_builtin_reference_pipeline_unhealthy() -> None:
    result = acceptance_builtin_reference_pipeline_unhealthy()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["windows_system_aec", "webrtc_apm", "degraded_no_aec"]
    assert result.final_state == "degraded"


def test_acceptance_wired_headset_clean() -> None:
    result = acceptance_wired_headset_clean()
    assert result.verdict == "PASS"
    assert result.backend_chain == ["headset_clean"]
    assert result.telemetry_snapshot["capture_gate_policy"] == "headset_clean"


def test_acceptance_reconnect_during_active_handsfree() -> None:
    result = acceptance_reconnect_during_active_handsfree()
    assert result.verdict == "PASS"
    states = [item["payload"].get("session_state") for item in result.session_log if item["type"] == "audio_session_state"]
    assert "recovering" in states
    assert states[-1] == "active"
