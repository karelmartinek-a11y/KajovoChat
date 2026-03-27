from __future__ import annotations

import time

from kajovochat.audio.voice_gate import (
    VoiceGateRuntimeState,
    can_use_cached_reference,
    evaluate_capture_gate,
    is_guard_active,
    note_tts_window,
    note_diag_logged,
    note_reference_cache,
    record_gate_outcome,
    resolve_reference_gate,
    should_log_problem_diag,
    should_log_success_diag,
    update_playback_reference_state,
)


def test_playback_reference_state_arms_and_suppresses_mic() -> None:
    runtime = VoiceGateRuntimeState()
    now = time.monotonic()

    update_playback_reference_state(
        runtime,
        is_playing_out=True,
        now_monotonic=now,
        trailing_hold_s=0.2,
    )

    assert runtime.playback_reference_armed is True
    assert runtime.reference_warmup_until >= now + 0.02
    assert runtime.mic_suppressed_until >= now + 0.19
    assert is_guard_active(runtime, mode="handsfree", is_playing_out=False) is True


def test_reference_gate_respects_warmup_and_live_samples() -> None:
    runtime = VoiceGateRuntimeState(playback_reference_armed=True, reference_warmup_until=time.monotonic() + 0.5)

    early = resolve_reference_gate(
        runtime,
        aec_requires_reference=True,
        now_monotonic=time.monotonic(),
        reference_needed=960,
        available_samples=1100,
        played_samples=100,
        callback_age_ms=10,
    )
    late = resolve_reference_gate(
        runtime,
        aec_requires_reference=True,
        now_monotonic=time.monotonic(),
        reference_needed=960,
        available_samples=1400,
        played_samples=320,
        callback_age_ms=10,
    )

    assert early.ready is False
    assert early.miss is True
    assert late.ready is True
    assert late.source == "live"


def test_cached_reference_uses_runtime_state() -> None:
    runtime = VoiceGateRuntimeState()
    note_reference_cache(runtime, b"\x01\x00" * 800, now_monotonic=time.monotonic())

    assert can_use_cached_reference(runtime, now_monotonic=time.monotonic(), reference_needed=960) is True


def test_gate_outcome_updates_session_owned_counters() -> None:
    runtime = VoiceGateRuntimeState()

    first = record_gate_outcome(runtime, drop_chunk=True, barge_in_candidate=False)
    second = record_gate_outcome(runtime, drop_chunk=False, barge_in_candidate=True)

    assert first.echo_drop_count == 1
    assert first.should_log_echo_drop is True
    assert second.barge_in_chunk_count == 1
    assert runtime.echo_drop_count == 1
    assert runtime.barge_in_chunk_count == 1


def test_diag_throttle_uses_runtime_state() -> None:
    runtime = VoiceGateRuntimeState()
    now = time.monotonic()

    assert should_log_problem_diag(runtime, now_monotonic=now, min_interval_s=0.5) is True
    note_diag_logged(runtime, success=False, now_monotonic=now)
    assert should_log_problem_diag(runtime, now_monotonic=now + 0.1, min_interval_s=0.5) is False

    assert should_log_success_diag(runtime, now_monotonic=now, min_interval_s=0.5) is True
    note_diag_logged(runtime, success=True, now_monotonic=now)
    assert should_log_success_diag(runtime, now_monotonic=now + 0.1, min_interval_s=0.5) is False


def test_voice_gate_confirms_barge_in_after_streak() -> None:
    runtime = VoiceGateRuntimeState()
    profile = {
        "echo_similarity_drop": 0.8,
        "echo_similarity_soft": 0.6,
        "barge_in_min_input_level": 0.06,
        "barge_in_output_ratio": 1.35,
    }

    first = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.11,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime,
        voice_likelihood=0.6,
        aec_quality=0.2,
    )
    second = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.11,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime,
        voice_likelihood=0.6,
        aec_quality=0.2,
    )

    assert first.barge_in_candidate is True
    assert first.barge_in_confirmed is False
    assert second.barge_in_confirmed is True


def test_voice_gate_requires_longer_barge_in_streak_during_tts_hold() -> None:
    runtime = VoiceGateRuntimeState()
    note_tts_window(runtime, rendering_active=True, now_monotonic=time.monotonic(), start_hold_s=0.5, tail_hold_s=0.0)
    profile = {
        "echo_similarity_drop": 0.8,
        "echo_similarity_soft": 0.6,
        "barge_in_min_input_level": 0.06,
        "barge_in_output_ratio": 1.35,
    }

    first = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.12,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime,
        voice_likelihood=0.62,
        aec_quality=0.2,
    )
    second = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.12,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime,
        voice_likelihood=0.62,
        aec_quality=0.2,
    )
    third = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.12,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime,
        voice_likelihood=0.62,
        aec_quality=0.2,
    )

    assert first.barge_in_candidate is True
    assert first.barge_in_confirmed is False
    assert second.barge_in_confirmed is False
    assert third.barge_in_confirmed is True


def test_degraded_no_aec_policy_is_more_conservative_than_standard() -> None:
    runtime_standard = VoiceGateRuntimeState()
    runtime_degraded = VoiceGateRuntimeState()
    profile = {
        "echo_similarity_drop": 0.8,
        "echo_similarity_soft": 0.6,
        "barge_in_min_input_level": 0.06,
        "barge_in_output_ratio": 1.35,
    }

    standard = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.11,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime_standard,
        voice_likelihood=0.6,
        aec_quality=0.2,
    )
    degraded = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.12,
        input_level=0.11,
        output_level=0.05,
        default_profile=profile,
        runtime=runtime_degraded,
        voice_likelihood=0.6,
        aec_quality=0.2,
        capture_gate_policy="degraded_no_aec",
    )

    assert standard.barge_in_candidate is True
    assert degraded.barge_in_candidate is False
    assert degraded.capture_gate_policy == "degraded_no_aec"


def test_headset_clean_policy_disables_echo_drop() -> None:
    profile = {
        "echo_similarity_drop": 0.8,
        "echo_similarity_soft": 0.6,
        "barge_in_min_input_level": 0.06,
        "barge_in_output_ratio": 1.35,
    }

    standard = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.92,
        input_level=0.015,
        output_level=0.08,
        default_profile=profile,
        runtime=VoiceGateRuntimeState(),
        voice_likelihood=0.05,
        aec_quality=0.01,
    )
    headset = evaluate_capture_gate(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.92,
        input_level=0.015,
        output_level=0.08,
        default_profile=profile,
        runtime=VoiceGateRuntimeState(),
        voice_likelihood=0.05,
        aec_quality=0.01,
        capture_gate_policy="headset_clean",
    )

    assert standard.drop_chunk is True
    assert headset.drop_chunk is False
    assert headset.capture_gate_policy == "headset_clean"
