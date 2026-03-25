from __future__ import annotations

from tools.audio_architecture_harness import (
    soak_backlog_playback_stagnation_detection,
    soak_device_reset_xrun_fault_injection,
    soak_long_running_handsfree,
    soak_repeated_reconnects,
    soak_repeated_tts_and_barge_in,
)


def test_soak_long_running_handsfree_session() -> None:
    result = soak_long_running_handsfree()
    assert result.verdict == "PASS"
    assert result.telemetry_snapshot["turn_latency"]["responses_completed_total"] >= 1


def test_soak_repeated_reconnects() -> None:
    result = soak_repeated_reconnects()
    assert result.verdict == "PASS"
    assert result.telemetry_snapshot["recovery_successes_total"] == 3


def test_soak_repeated_tts_and_barge_in() -> None:
    result = soak_repeated_tts_and_barge_in()
    assert result.verdict == "PASS"
    assert result.telemetry_snapshot["health"]["pending_player_bytes"] == 0
    assert result.telemetry_snapshot["turn_latency"]["responses_completed_total"] >= 1


def test_soak_backlog_and_playback_stagnation_detection() -> None:
    result = soak_backlog_playback_stagnation_detection()
    assert result.verdict == "PASS"
    assert any(item["type"] == "watchdog" for item in result.session_log)
    assert result.telemetry_snapshot["reconnect_attempts"] == 1


def test_soak_device_reset_xrun_fault_injection() -> None:
    result = soak_device_reset_xrun_fault_injection()
    assert result.verdict == "PASS"
    assert result.telemetry_snapshot["xrun_events_total"] >= 3
