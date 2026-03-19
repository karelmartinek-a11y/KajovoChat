from __future__ import annotations

import math

import numpy as np

from kajovochat.animation.types import VisemeFrame
from kajovochat.animation.viseme_engine import VisemeEngine


def _sine(freq: float, *, amp: float = 0.7, sr: int = 24000, duration_s: float = 0.08) -> bytes:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / float(sr)
    wave = np.sin(2.0 * math.pi * freq * t) * amp
    return np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def _mix_tones(tones: list[tuple[float, float]], *, sr: int = 24000, duration_s: float = 0.08) -> bytes:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / float(sr)
    wave = np.zeros_like(t)
    for freq, amp in tones:
        wave += np.sin(2.0 * math.pi * freq * t) * amp
    wave = np.clip(wave, -1.0, 1.0)
    return np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def test_viseme_engine_returns_silence_on_zero_pcm() -> None:
    engine = VisemeEngine()
    engine.consume_playback_pcm16(np.zeros((1920,), dtype=np.int16).tobytes(), 24000, timestamp_s=1.0)
    frame = engine.snapshot(timestamp_s=1.0)
    assert frame.cluster == "sil"
    assert frame.pose == "closed"
    assert frame.mouth_open < 0.05
    assert frame.jaw_open < 0.05


def test_viseme_engine_low_band_signal_opens_jaw_more() -> None:
    engine = VisemeEngine()
    for idx in range(4):
        engine.consume_playback_pcm16(_sine(180.0, amp=0.95), 24000, timestamp_s=1.0 + idx * 0.02)
    frame = engine.snapshot(timestamp_s=1.09)
    assert frame.jaw_open > 0.30
    assert frame.mouth_open > 0.25


def test_viseme_engine_rounded_profile_raises_round_and_funnel() -> None:
    engine = VisemeEngine()
    rounded = _mix_tones([(240.0, 0.65), (520.0, 0.38), (820.0, 0.18)])
    for idx in range(5):
        engine.consume_playback_pcm16(rounded, 24000, timestamp_s=2.0 + idx * 0.02)
    frame = engine.snapshot(timestamp_s=2.12)
    assert frame.lip_round > 0.25
    assert frame.lip_funnel > 0.22


def test_viseme_engine_normalizes_cluster_weights() -> None:
    engine = VisemeEngine()
    engine.consume_playback_pcm16(_sine(440.0, amp=0.9), 24000, timestamp_s=3.0)
    frame = engine.snapshot(timestamp_s=3.03)
    assert abs(sum(frame.weights.values()) - 1.0) < 1e-6
    assert abs(sum(frame.legacy_weights.values()) - 1.0) < 1e-6


def test_viseme_engine_suppresses_jitter_near_silence() -> None:
    engine = VisemeEngine()
    tiny = _sine(260.0, amp=0.02)
    for idx in range(3):
        engine.consume_playback_pcm16(tiny, 24000, timestamp_s=4.0 + idx * 0.02)
    frame = engine.snapshot(timestamp_s=4.08)
    assert frame.pose == "closed"
    assert frame.mouth_open < 0.10


def test_viseme_frame_legacy_conversion_roundtrip() -> None:
    legacy = {
        "pose": "oo",
        "openness": 0.44,
        "energy": 0.38,
        "weights": {"closed": 0.1, "small": 0.1, "aa": 0.1, "ee": 0.1, "oo": 0.6},
    }
    frame = VisemeFrame.from_legacy_snapshot(legacy)
    restored = frame.to_legacy_snapshot()
    assert restored["pose"] == "oo"
    assert restored["weights"]["oo"] == 0.6
