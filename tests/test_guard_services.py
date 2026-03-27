from __future__ import annotations

import math
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from kajovochat.services.guard_adaptation import GuardAdaptor
from kajovochat.services.guard_replay import append_guard_replay_metrics
from kajovochat.services.guard_telemetry import GuardTelemetry
from kajovochat.services.voice_features import estimate_voice_likelihood_from_pcm16
from kajovochat.audio.dsp_helpers import AdaptiveEchoCanceller, suppress_echo_from_pcm16, _find_best_alignment
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


def test_echo_suppression_handles_shifted_reference_tail() -> None:
    ref = (np.sin(np.linspace(0.0, math.tau * 6.0, 960, endpoint=False)) * 9000.0).astype(np.int16)
    mic = np.concatenate([np.zeros((72,), dtype=np.int16), ref[:720]]).astype(np.int16)

    cleaned, similarity = suppress_echo_from_pcm16(mic.tobytes(), ref, max_shift_samples=120)

    cleaned_pcm = np.frombuffer(cleaned, dtype=np.int16)
    assert similarity > 0.45
    assert np.sqrt(np.mean(cleaned_pcm.astype(np.float32) ** 2)) < np.sqrt(np.mean(mic.astype(np.float32) ** 2))


def test_guard_telemetry_exposes_aec_metrics() -> None:
    telemetry = GuardTelemetry()
    telemetry.add_sample(
        input_level=0.08,
        output_level=0.12,
        similarity=0.7,
        voice_likelihood=0.2,
        dropped=True,
        playback_active=True,
        reason="echo_similarity",
        barge_in_candidate=False,
        residual_level=0.02,
        aec_quality=0.44,
        double_talk=False,
    )

    snapshot = telemetry.snapshot(window_s=60.0)

    assert snapshot["avg_aec_quality"] == 0.44
    assert snapshot["avg_residual"] == 0.02


def test_adaptive_echo_canceller_finds_realistic_latency_without_prior_hint() -> None:
    samplerate = 24000
    rng = np.random.default_rng(7)
    total = 12000
    reference = (
        rng.normal(0.0, 0.22, size=total)
        + 0.28 * np.sin(np.arange(total, dtype=np.float32) * 0.17)
        + 0.12 * np.sin(np.arange(total, dtype=np.float32) * 0.043 + 0.6)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)

    chunk_size = 960
    true_shift = 620
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    mic = 0.72 * direct + 0.18 * np.roll(direct, 70) + 0.06 * np.roll(direct, 140)
    mic += rng.normal(0.0, 0.01, size=chunk_size)
    mic = np.clip(mic, -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    result = canceller.process((mic * 32767.0).astype(np.int16).tobytes(), (reference * 20000.0).astype(np.int16))
    cleaned = np.frombuffer(result["pcm"], dtype=np.int16).astype(np.float32) / 32767.0

    assert result["similarity"] > 0.9
    assert abs(int(result["delay_samples"]) - true_shift) <= 8
    assert np.sqrt(np.mean(cleaned * cleaned)) < np.sqrt(np.mean(mic * mic)) * 0.35


def test_adaptive_echo_canceller_flags_double_talk_when_user_voice_is_present() -> None:
    samplerate = 24000
    rng = np.random.default_rng(11)
    total = 16000
    reference = (
        rng.normal(0.0, 0.18, size=total)
        + 0.22 * np.sin(np.arange(total, dtype=np.float32) * 0.13)
        + 0.14 * np.sin(np.arange(total, dtype=np.float32) * 0.051 + 0.5)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)

    chunk_size = 960
    true_shift = 540
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    echo = 0.66 * direct + 0.2 * np.roll(direct, 50) - 0.08 * np.roll(direct, 110)
    user = 0.18 * np.sin(2.0 * math.pi * 240.0 * np.arange(chunk_size, dtype=np.float32) / samplerate)
    user += 0.1 * np.sin(2.0 * math.pi * 480.0 * np.arange(chunk_size, dtype=np.float32) / samplerate + 0.3)
    mic = np.clip(echo + user + rng.normal(0.0, 0.004, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    baseline = canceller.process((echo * 32767.0).astype(np.int16).tobytes(), (reference * 22000.0).astype(np.int16))
    result = canceller.process((mic * 32767.0).astype(np.int16).tobytes(), (reference * 22000.0).astype(np.int16))
    cleaned = np.frombuffer(result["pcm"], dtype=np.int16).astype(np.float32) / 32767.0
    baseline_cleaned = np.frombuffer(baseline["pcm"], dtype=np.int16).astype(np.float32) / 32767.0

    assert result["double_talk"] is True
    assert result["similarity"] > 0.45
    assert np.sqrt(np.mean(cleaned * cleaned)) > np.sqrt(np.mean(baseline_cleaned * baseline_cleaned)) * 2.0


def test_adaptive_echo_canceller_recovers_when_expected_shift_is_wrong() -> None:
    samplerate = 24000
    rng = np.random.default_rng(5)
    chunk_size = 960
    true_shift = 700
    total = 16000
    reference = (
        rng.normal(0.0, 0.18, size=total)
        + 0.24 * np.sin(np.arange(total, dtype=np.float32) * 0.11)
        + 0.17 * np.sin(np.arange(total, dtype=np.float32) * 0.037 + 0.7)
        + rng.normal(0.0, 0.02, size=total)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)

    ir = np.zeros((1300,), dtype=np.float32)
    ir[true_shift] = 0.6
    ir[true_shift + 90] = 0.2
    ir[true_shift + 170] = -0.08
    bleed = np.convolve(reference, ir, mode="full")[:total]
    bleed = np.clip(bleed, -1.0, 1.0)

    mic = bleed[-chunk_size:]
    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1600)
    result = canceller.process(
        (mic * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1600,
        expected_shift=300,
    )
    cleaned = np.frombuffer(result["pcm"], dtype=np.int16).astype(np.float32) / 32767.0

    assert abs(int(result["delay_samples"]) - true_shift) <= 16
    assert result["similarity"] > 0.9
    assert np.sqrt(np.mean(cleaned * cleaned)) < np.sqrt(np.mean(mic * mic)) * 0.5


def test_alignment_prefers_expected_shift_for_repetitive_signal() -> None:
    cycle = (
        0.5 * np.sin(np.linspace(0.0, math.tau * 3.0, 240, endpoint=False, dtype=np.float32))
        + 0.22 * np.sin(np.linspace(0.0, math.tau * 7.0, 240, endpoint=False, dtype=np.float32) + 0.4)
    ).astype(np.float32)
    reference = np.tile(cycle, 14)
    true_shift = 720
    chunk_size = 960
    mic = reference[-(chunk_size + true_shift) : -true_shift].copy()

    _segment, similarity, delay = _find_best_alignment(
        mic,
        reference,
        max_shift_samples=960,
        expected_shift=true_shift,
    )

    assert similarity > 0.95
    assert abs(int(delay) - true_shift) <= 8


def test_adaptive_echo_canceller_uses_webrtc_with_stable_delay_lock() -> None:
    samplerate = 24000
    rng = np.random.default_rng(21)
    total = 18000
    chunk_size = 960
    true_shift = 408
    reference = (
        rng.normal(0.0, 0.16, size=total)
        + 0.2 * np.sin(np.arange(total, dtype=np.float32) * 0.09)
        + 0.12 * np.sin(np.arange(total, dtype=np.float32) * 0.031 + 0.4)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)

    direct = reference[-(chunk_size + true_shift) : -true_shift]
    echo = np.clip(0.58 * direct + 0.16 * np.roll(direct, 36), -1.0, 1.0)
    weak_user = 0.015 * np.sin(2.0 * math.pi * 180.0 * np.arange(chunk_size, dtype=np.float32) / samplerate)
    mic = np.clip(echo + weak_user + rng.normal(0.0, 0.002, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    for _ in range(4):
        canceller.process((echo * 32767.0).astype(np.int16).tobytes(), (reference * 22000.0).astype(np.int16))

    result = canceller.process((mic * 32767.0).astype(np.int16).tobytes(), (reference * 22000.0).astype(np.int16))

    assert int(result["delay_samples"]) > 0
    assert result["backend"] in {"custom", "webrtc"}
    if result["similarity"] < 0.22:
        assert result["backend"] == "webrtc"


def test_custom_branch_with_no_improvement_does_not_report_fake_prediction() -> None:
    samplerate = 24000
    rng = np.random.default_rng(41)
    total = 12000
    chunk_size = 960
    true_shift = 520
    reference = rng.normal(0.0, 0.15, size=total).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    mic = np.clip(0.12 * direct + rng.normal(0.0, 0.03, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    result = canceller.process(
        (mic * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1200,
        expected_shift=true_shift,
    )

    if result["backend"] == "custom" and float(result["similarity"]) >= 0.28 and float(result["improvement_ratio"]) < 0.01:
        assert float(result["predicted_level"]) == 0.0


def test_custom_branch_with_no_improvement_downranks_similarity() -> None:
    samplerate = 24000
    rng = np.random.default_rng(42)
    total = 12000
    chunk_size = 960
    true_shift = 520
    reference = rng.normal(0.0, 0.14, size=total).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    mic = np.clip(0.1 * direct + rng.normal(0.0, 0.035, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    result = canceller.process(
        (mic * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1200,
        expected_shift=true_shift,
    )

    if result["backend"] == "custom" and float(result["improvement_ratio"]) < 0.01:
        assert float(result["similarity"]) <= 0.08
        assert int(result["delay_samples"]) == 0


def test_windows_native_preferred_keeps_native_when_webrtc_is_only_marginally_better() -> None:
    samplerate = 24000
    rng = np.random.default_rng(17)
    total = 12000
    chunk_size = 960
    true_shift = 640
    reference = (
        rng.normal(0.0, 0.16, size=total)
        + 0.24 * np.sin(np.arange(total, dtype=np.float32) * 0.13)
        + 0.11 * np.sin(np.arange(total, dtype=np.float32) * 0.051 + 0.2)
    ).astype(np.float32)
    reference = np.clip(reference, -1.0, 1.0)
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    mic = np.clip(0.72 * direct + rng.normal(0.0, 0.004, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1400)
    canceller._windows_native_probe = SimpleNamespace(available=True, installed_driver=True)
    canceller._ridge_candidate = lambda design, target, ridge=None: (
        np.zeros((design.shape[1],), dtype=np.float32),
        np.zeros((target.shape[0],), dtype=np.float32),
    )
    canceller._nlms_candidate = lambda design, target, iterations, initial_weights=None: (
        np.zeros((design.shape[1],), dtype=np.float32),
        np.zeros((target.shape[0],), dtype=np.float32),
    )

    class DummyNativeBackend:
        def process(self, *, mic_pcm: bytes, reference_pcm: np.ndarray, delay_ms: int) -> bytes:
            del reference_pcm, delay_ms
            mic_local = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
            cleaned = np.clip(mic_local * 0.08, -32768.0, 32767.0).astype(np.int16)
            return cleaned.tobytes()

    class DummyWebRTCBackend:
        def process(self, *, mic_pcm: bytes, reference_pcm: np.ndarray, delay_ms: int) -> bytes:
            del reference_pcm, delay_ms
            mic_local = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
            cleaned = np.clip(mic_local * 0.2, -32768.0, 32767.0).astype(np.int16)
            return cleaned.tobytes()

    canceller._windows_native_backend = DummyNativeBackend()
    canceller._windows_native_backend_attempted = True
    canceller._external_backend = DummyWebRTCBackend()

    result = canceller.process(
        (mic * 32767.0).astype(np.int16).tobytes(),
        (reference * 22000.0).astype(np.int16),
        max_shift_samples=1400,
        expected_shift=true_shift,
        aec_mode="windows_native_preferred",
    )

    assert result["native_attempted"] is True
    assert result["native_selected"] is True
    assert result["backend"] == "windows_system_aec"


def test_production_webrtc_mode_falls_back_to_degraded_instead_of_custom_output() -> None:
    samplerate = 24000
    rng = np.random.default_rng(7)
    total = 8000
    chunk_size = 960
    true_shift = 220
    reference = rng.normal(0.0, 0.22, size=total).astype(np.float32)
    direct = reference[-(chunk_size + true_shift) : -true_shift]
    mic = np.clip(0.62 * direct + rng.normal(0.0, 0.01, size=chunk_size), -1.0, 1.0)

    canceller = AdaptiveEchoCanceller(samplerate=samplerate, filter_length=1024, max_shift_samples=1200)
    canceller._windows_native_probe = SimpleNamespace(available=False)
    canceller._external_backend = None
    canceller._ridge_candidate = lambda design, target, ridge=None: (
        np.zeros((design.shape[1],), dtype=np.float32),
        np.zeros((target.shape[0],), dtype=np.float32),
    )
    canceller._nlms_candidate = lambda design, target, iterations, initial_weights=None: (
        np.zeros((design.shape[1],), dtype=np.float32),
        np.zeros((target.shape[0],), dtype=np.float32),
    )

    result = canceller.process(
        (mic * 32767.0).astype(np.int16).tobytes(),
        (reference * 32767.0).astype(np.int16),
        max_shift_samples=1200,
        expected_shift=true_shift,
        aec_mode="webrtc_apm",
    )

    assert result["backend"] == "webrtc"
    assert result["selection_reason"] == "webrtc_no_gain"


def test_windows_system_aec_with_installed_apo_uses_system_capture_path() -> None:
    canceller = AdaptiveEchoCanceller(samplerate=24000, filter_length=512, max_shift_samples=960)
    canceller._windows_native_probe = SimpleNamespace(available=True, installed_driver=True)

    mic = (np.random.default_rng(11).normal(0.0, 0.03, size=960).astype(np.float32) * 32767.0).astype(np.int16)
    result = canceller.process(
        mic.tobytes(),
        np.zeros((0,), dtype=np.int16),
        max_shift_samples=960,
        expected_shift=480,
        aec_mode="windows_system_aec",
    )

    assert result["backend"] == "windows_system_aec"
    assert result["native_attempted"] is True
    assert result["native_selected"] is True
    assert result["selection_reason"] == "windows_system_aec"
