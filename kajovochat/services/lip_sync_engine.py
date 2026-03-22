from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np


_POSES = ("closed", "small", "aa", "ee", "oo")


def _default_weights() -> dict[str, float]:
    return {pose: (1.0 if pose == "closed" else 0.0) for pose in _POSES}


@dataclass(frozen=True)
class LipSyncSnapshot:
    pose: str
    openness: float
    energy: float
    weights: dict[str, float]


class LipSyncEngine:
    """Lehký odhad mouth pose z PCM16 chunků skutečně přehrávaných callbackem."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights = _default_weights()
        self._energy = 0.0
        self._last_tick = time.perf_counter()
        self._last_audio_at = 0.0

    @staticmethod
    def _smooth_exp(current: float, target: float, dt: float, tau_up: float, tau_down: float) -> float:
        if dt <= 0.0:
            return current
        tau = tau_up if target > current else tau_down
        tau = max(0.001, float(tau))
        alpha = 1.0 - math.exp(-dt / tau)
        return current + (target - current) * alpha

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        total = float(sum(max(0.0, value) for value in scores.values()))
        if total <= 1e-9:
            return _default_weights()
        return {key: max(0.0, value) / total for key, value in scores.items()}

    def reset(self) -> None:
        with self._lock:
            now = time.perf_counter()
            self._weights = _default_weights()
            self._energy = 0.0
            self._last_tick = now
            self._last_audio_at = 0.0

    def consume_playback_pcm16(self, pcm_bytes: bytes, samplerate: int) -> None:
        if not pcm_bytes:
            return
        try:
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        except Exception:
            return
        if pcm.size == 0:
            return

        pcm /= 32768.0
        rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        energy = float(max(0.0, min(1.0, max(rms * 2.8, peak * 1.05))))

        window = np.hanning(pcm.size).astype(np.float32) if pcm.size > 8 else np.ones_like(pcm)
        spectrum = np.abs(np.fft.rfft(pcm * window))
        freqs = np.fft.rfftfreq(pcm.size, d=1.0 / max(1, int(samplerate)))
        total_spec = float(np.sum(spectrum) + 1e-9)

        def band(low: float, high: float) -> float:
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                return 0.0
            return float(np.sum(spectrum[mask]) / total_spec)

        low = band(120.0, 700.0)
        mid = band(700.0, 1800.0)
        high = band(1800.0, 4200.0)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(pcm).astype(np.int8))))) if pcm.size > 1 else 0.0
        zcr = max(0.0, min(1.0, zcr * 2.4))
        open_factor = max(0.0, min(1.0, (energy - 0.045) / 0.32))

        scores = {
            "closed": max(0.0, 1.15 - open_factor * 1.9) + max(0.0, 0.18 - energy),
            "small": (0.30 + (1.0 - open_factor) * 0.85) * (0.35 + low * 0.25 + mid * 0.20 + high * 0.20),
            "aa": open_factor * (0.42 + low * 1.35 + mid * 0.35 + (1.0 - zcr) * 0.30),
            "ee": open_factor * (0.28 + high * 1.40 + mid * 0.55 + zcr * 0.22),
            "oo": open_factor * (0.24 + low * 1.05 + (1.0 - high) * 0.40 + (1.0 - zcr) * 0.24),
        }
        targets = self._normalize(scores)

        with self._lock:
            now = time.perf_counter()
            chunk_dt = float(pcm.size) / max(1.0, float(samplerate))
            dt = max(0.0, min(0.08, max(now - self._last_tick, chunk_dt)))
            self._last_tick = now
            self._last_audio_at = now
            for pose in _POSES:
                self._weights[pose] = self._smooth_exp(self._weights[pose], targets[pose], dt, 0.028, 0.11)
            self._weights = self._normalize(self._weights)
            self._energy = self._smooth_exp(self._energy, energy, dt, 0.03, 0.14)

    def _advance_idle_locked(self, now: float) -> None:
        dt = max(0.0, min(0.05, now - self._last_tick))
        self._last_tick = now
        if now - self._last_audio_at < 0.045:
            return
        targets = _default_weights()
        for pose in _POSES:
            self._weights[pose] = self._smooth_exp(self._weights[pose], targets[pose], dt, 0.04, 0.18)
        self._weights = self._normalize(self._weights)
        self._energy = self._smooth_exp(self._energy, 0.0, dt, 0.05, 0.22)

    def snapshot(self) -> LipSyncSnapshot:
        with self._lock:
            now = time.perf_counter()
            self._advance_idle_locked(now)
            weights = dict(self._weights)
            pose = max(weights.items(), key=lambda item: item[1])[0]
            openness = (
                weights["small"] * 0.22
                + weights["aa"] * 1.00
                + weights["ee"] * 0.52
                + weights["oo"] * 0.64
            )
            openness = max(0.0, min(1.0, openness))
            return LipSyncSnapshot(pose=pose, openness=openness, energy=float(max(0.0, min(1.0, self._energy))), weights=weights)
