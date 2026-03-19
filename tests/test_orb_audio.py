from __future__ import annotations

import numpy as np

from kajovochat.orb.audio import AudioAnalyzer, AudioFeatureFrame
from kajovochat.orb.config import create_default_config


def test_audio_analyzer_extracts_expected_bands() -> None:
    analyzer = AudioAnalyzer(create_default_config())
    t = np.linspace(0.0, 0.08, int(24000 * 0.08), endpoint=False, dtype=np.float32)
    samples = 0.55 * np.sin(2.0 * np.pi * 220.0 * t) + 0.20 * np.sin(2.0 * np.pi * 2200.0 * t)
    frame = analyzer.extract(samples, sample_rate=24000)

    assert frame.loudness > 0.1
    assert frame.low_band > frame.mid_band
    assert frame.high_band > 0.0
    assert 0.0 <= frame.spectral_centroid <= 1.0


def test_audio_analyzer_smoothing_holds_speaking_gate() -> None:
    analyzer = AudioAnalyzer(create_default_config())
    hot = AudioFeatureFrame(loudness=0.7, speaking_gate=1.0)
    cool = AudioFeatureFrame()

    analyzer.smooth(hot, 0.016)
    held = analyzer.smooth(cool, 0.05)

    assert held.speaking_gate > 0.1


def test_audio_analyzer_sanitizes_nan_inputs() -> None:
    analyzer = AudioAnalyzer(create_default_config())
    samples = np.asarray([0.0, np.nan, np.inf, -np.inf, 0.1], dtype=np.float32)
    frame = analyzer.extract(samples, sample_rate=24000)

    assert frame.loudness >= 0.0
    assert frame.peak_envelope >= 0.0
