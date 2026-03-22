from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from kajovochat.services.guard_adaptation import GuardAdaptor
from kajovochat.services.guard_replay import append_guard_replay_metrics
from kajovochat.services.guard_telemetry import GuardTelemetry
from kajovochat.services.voice_features import estimate_voice_likelihood_from_pcm16
from kajovochat.services.audio_service import suppress_echo_from_pcm16
from kajovochat.settings import DEFAULT_AUDIO_GUARD_PROFILE


def test_voice_likelihood_prefers_speech_like_signal() -> None:
    samplerate = 24000
    t = np.linspace(0.0, 0.04, int(0.04 * samplerate), endpoint=False, dtype=np.float32)
    speech_like = (
        0.5 * np.sin(2.0 * math.pi * 220.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 440.0 * t)
        + 0.12 * np.sin(2.0 * math.pi * 880.0 * t)
    )
    raw = np.clip(speech_like, -1.0, 1.0)
    pcm = (raw * 32767.0).astype(np.int16).tobytes()

    value = estimate_voice_likelihood_from_pcm16(pcm)

    assert value > 0.35


def test_guard_telemetry_aggregates_recent_samples() -> None:
    telemetry = GuardTelemetry()
    for index in range(8):
        telemetry.add_sample(
            input_level=0.1 + index * 0.01,
            output_level=0.2,
            similarity=0.15,
            voice_likelihood=0.5,
            dropped=index % 2 == 0,
            playback_active=True,
            reason="echo_similarity" if index % 2 == 0 else "",
            barge_in_candidate=index % 3 == 0,
        )

    snapshot = telemetry.snapshot(window_s=60.0)

    assert snapshot["samples"] == 8
    assert snapshot["drop_rate"] > 0.4
    assert snapshot["top_reason"] == "echo_similarity"


def test_guard_adaptor_tightens_echo_thresholds_under_heavy_echo() -> None:
    adaptor = GuardAdaptor()
    result = adaptor.adapt(
        dict(DEFAULT_AUDIO_GUARD_PROFILE),
        {
            "drop_rate": 0.22,
            "avg_similarity": 0.39,
            "avg_voice_likelihood": 0.21,
            "playback_ratio": 0.62,
            "barge_in_ratio": 0.01,
        },
    )

    assert result.state == "echo_heavy"
    assert result.profile["echo_similarity_drop"] > DEFAULT_AUDIO_GUARD_PROFILE["echo_similarity_drop"]


def test_guard_adaptor_relaxes_barge_in_when_voice_is_present() -> None:
    adaptor = GuardAdaptor()
    result = adaptor.adapt(
        dict(DEFAULT_AUDIO_GUARD_PROFILE),
        {
            "drop_rate": 0.04,
            "avg_similarity": 0.11,
            "avg_voice_likelihood": 0.58,
            "playback_ratio": 0.28,
            "barge_in_ratio": 0.16,
        },
    )

    assert result.state == "barge_ready"
    assert result.profile["barge_in_min_input_level"] < DEFAULT_AUDIO_GUARD_PROFILE["barge_in_min_input_level"]


def test_guard_adaptor_supports_aec_aware_mode() -> None:
    adaptor = GuardAdaptor()
    result = adaptor.adapt(
        dict(DEFAULT_AUDIO_GUARD_PROFILE),
        {
            "drop_rate": 0.03,
            "avg_similarity": 0.03,
            "avg_voice_likelihood": 0.19,
            "playback_ratio": 0.5,
            "avg_output": 0.05,
            "barge_in_ratio": 0.01,
        },
        aec_aware=True,
    )

    assert result.state == "aec_aware"
    assert result.profile["barge_in_output_ratio"] > DEFAULT_AUDIO_GUARD_PROFILE["barge_in_output_ratio"]


def test_guard_replay_metrics_append_jsonl() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = append_guard_replay_metrics(Path(temp_dir), {"session": "s1", "value": 1})
        assert path.exists()
        assert '"session": "s1"' in path.read_text(encoding="utf-8")


def test_echo_suppression_returns_similarity_and_cleaned_chunk() -> None:
    ref = (np.sin(np.linspace(0.0, math.tau * 4.0, 480, endpoint=False)) * 12000.0).astype(np.int16)
    mic = np.clip(ref.astype(np.int32) + 400, -32768, 32767).astype(np.int16)

    cleaned, similarity = suppress_echo_from_pcm16(mic.tobytes(), ref, max_shift_samples=0)

    assert similarity > 0.8
    assert cleaned != mic.tobytes()
