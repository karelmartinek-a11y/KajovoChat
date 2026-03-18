from __future__ import annotations

import math

import numpy as np

from kajovochat.services.lip_sync_engine import LipSyncEngine


def _tone(freq: float, amp: float = 0.7, sr: int = 24000, duration_s: float = 0.08) -> bytes:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / float(sr)
    wave = np.sin(2.0 * math.pi * freq * t) * amp
    return np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16).tobytes()


def test_lip_sync_engine_stays_closed_on_silence() -> None:
    engine = LipSyncEngine()
    snap = engine.snapshot()
    assert snap.pose == "closed"
    assert snap.weights["closed"] > 0.9


def test_lip_sync_engine_opens_on_playback_audio() -> None:
    engine = LipSyncEngine()
    for _ in range(4):
        engine.consume_playback_pcm16(_tone(440.0, amp=0.9), samplerate=24000)
    snap = engine.snapshot()
    assert snap.openness > 0.03
    open_weight = snap.weights["small"] + snap.weights["aa"] + snap.weights["ee"] + snap.weights["oo"]
    assert open_weight > 0.05
