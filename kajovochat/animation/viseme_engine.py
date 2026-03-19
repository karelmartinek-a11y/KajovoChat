from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from .types import VisemeFrame


_CLUSTERS = ("sil", "aa", "ee", "ih", "oh", "oo", "fv", "mbp", "lntd", "szchj", "wq")
_LEGACY_POSES = ("closed", "small", "aa", "ee", "oo")

_CHANNEL_MAP: dict[str, dict[str, float]] = {
    "sil": {"jaw_open": 0.0, "mouth_open": 0.0, "lip_funnel": 0.0, "lip_round": 0.0, "lip_spread": 0.0, "lip_press": 0.20, "upper_lip_raise": 0.0, "lower_lip_drop": 0.0, "cheek_raise": 0.0},
    "aa": {"jaw_open": 0.92, "mouth_open": 0.86, "lip_funnel": 0.02, "lip_round": 0.05, "lip_spread": 0.08, "lip_press": 0.0, "upper_lip_raise": 0.05, "lower_lip_drop": 0.88, "cheek_raise": 0.08},
    "ee": {"jaw_open": 0.42, "mouth_open": 0.44, "lip_funnel": 0.0, "lip_round": 0.04, "lip_spread": 0.95, "lip_press": 0.0, "upper_lip_raise": 0.16, "lower_lip_drop": 0.34, "cheek_raise": 0.60},
    "ih": {"jaw_open": 0.34, "mouth_open": 0.38, "lip_funnel": 0.0, "lip_round": 0.02, "lip_spread": 0.70, "lip_press": 0.0, "upper_lip_raise": 0.08, "lower_lip_drop": 0.30, "cheek_raise": 0.30},
    "oh": {"jaw_open": 0.58, "mouth_open": 0.56, "lip_funnel": 0.52, "lip_round": 0.68, "lip_spread": 0.04, "lip_press": 0.0, "upper_lip_raise": 0.05, "lower_lip_drop": 0.46, "cheek_raise": 0.08},
    "oo": {"jaw_open": 0.36, "mouth_open": 0.42, "lip_funnel": 0.92, "lip_round": 0.96, "lip_spread": 0.0, "lip_press": 0.0, "upper_lip_raise": 0.0, "lower_lip_drop": 0.20, "cheek_raise": 0.02},
    "fv": {"jaw_open": 0.16, "mouth_open": 0.22, "lip_funnel": 0.0, "lip_round": 0.02, "lip_spread": 0.32, "lip_press": 0.72, "upper_lip_raise": 0.58, "lower_lip_drop": 0.28, "cheek_raise": 0.04},
    "mbp": {"jaw_open": 0.04, "mouth_open": 0.06, "lip_funnel": 0.0, "lip_round": 0.02, "lip_spread": 0.02, "lip_press": 1.0, "upper_lip_raise": 0.04, "lower_lip_drop": 0.04, "cheek_raise": 0.04},
    "lntd": {"jaw_open": 0.28, "mouth_open": 0.30, "lip_funnel": 0.0, "lip_round": 0.0, "lip_spread": 0.22, "lip_press": 0.08, "upper_lip_raise": 0.18, "lower_lip_drop": 0.22, "cheek_raise": 0.04},
    "szchj": {"jaw_open": 0.24, "mouth_open": 0.26, "lip_funnel": 0.0, "lip_round": 0.06, "lip_spread": 0.54, "lip_press": 0.12, "upper_lip_raise": 0.10, "lower_lip_drop": 0.18, "cheek_raise": 0.18},
    "wq": {"jaw_open": 0.22, "mouth_open": 0.26, "lip_funnel": 0.72, "lip_round": 0.80, "lip_spread": 0.0, "lip_press": 0.04, "upper_lip_raise": 0.0, "lower_lip_drop": 0.16, "cheek_raise": 0.02},
}


def _default_cluster_weights() -> dict[str, float]:
    return {cluster: (1.0 if cluster == "sil" else 0.0) for cluster in _CLUSTERS}


class VisemeEngine:
    """Audio-driven viseme engine nad skutečně přehrávaným PCM16 mono audiem."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights = _default_cluster_weights()
        self._channels = dict(_CHANNEL_MAP["sil"])
        self._energy = 0.0
        self._attack = 0.0
        self._voicing_confidence = 0.0
        self._legacy_weights = {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0}
        self._last_tick = time.perf_counter()
        self._last_audio_at = 0.0
        self._prev_energy = 0.0
        self._mbp_hold_until = 0.0

    @staticmethod
    def _smooth_exp(current: float, target: float, dt: float, tau_up: float, tau_down: float) -> float:
        if dt <= 0.0:
            return current
        tau = tau_up if target > current else tau_down
        tau = max(0.001, float(tau))
        alpha = 1.0 - math.exp(-dt / tau)
        return current + (target - current) * alpha

    @staticmethod
    def _normalize(scores: dict[str, float], *, fallback: str = "sil") -> dict[str, float]:
        total = float(sum(max(0.0, value) for value in scores.values()))
        if total <= 1e-9:
            return {name: (1.0 if name == fallback else 0.0) for name in scores}
        return {name: max(0.0, value) / total for name, value in scores.items()}

    @staticmethod
    def _band_ratio(freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float, total: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        return float(np.sum(spectrum[mask]) / max(1e-9, total))

    @staticmethod
    def _to_numpy_pcm(pcm_bytes: bytes) -> np.ndarray:
        if not pcm_bytes:
            return np.zeros((0,), dtype=np.float32)
        try:
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        except Exception:
            return np.zeros((0,), dtype=np.float32)
        return pcm / 32768.0 if pcm.size else pcm

    def reset(self) -> None:
        with self._lock:
            self._weights = _default_cluster_weights()
            self._channels = dict(_CHANNEL_MAP["sil"])
            self._energy = 0.0
            self._attack = 0.0
            self._voicing_confidence = 0.0
            self._legacy_weights = {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0}
            self._last_tick = time.perf_counter()
            self._last_audio_at = 0.0
            self._prev_energy = 0.0
            self._mbp_hold_until = 0.0

    def consume_playback_pcm16(self, pcm_bytes: bytes, samplerate: int, *, timestamp_s: float | None = None) -> None:
        pcm = self._to_numpy_pcm(pcm_bytes)
        if pcm.size == 0:
            return

        rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
        peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
        energy = max(0.0, min(1.0, max(rms * 2.8, peak * 1.02)))

        window = np.hanning(pcm.size).astype(np.float32) if pcm.size > 8 else np.ones_like(pcm)
        spectrum = np.abs(np.fft.rfft(pcm * window))
        freqs = np.fft.rfftfreq(pcm.size, d=1.0 / max(1, int(samplerate)))
        total_spec = float(np.sum(spectrum) + 1e-9)

        low = self._band_ratio(freqs, spectrum, 80.0, 450.0, total_spec)
        low_mid = self._band_ratio(freqs, spectrum, 450.0, 1200.0, total_spec)
        mid = self._band_ratio(freqs, spectrum, 1200.0, 2600.0, total_spec)
        high = self._band_ratio(freqs, spectrum, 2600.0, 5000.0, total_spec)
        centroid = float(np.sum(freqs * spectrum) / total_spec) / max(1.0, samplerate * 0.5)
        centroid = max(0.0, min(1.0, centroid))
        zcr = 0.0 if pcm.size < 2 else float(np.mean(np.abs(np.diff(np.signbit(pcm).astype(np.int8)))))
        zcr = max(0.0, min(1.0, zcr * 2.0))

        attack_raw = max(0.0, energy - self._prev_energy)
        transient = max(0.0, min(1.0, attack_raw * 5.0 + max(0.0, peak - rms) * 1.4))
        voicing = max(0.0, min(1.0, 0.55 * (low + low_mid) + 0.35 * (1.0 - zcr) + 0.20 * (1.0 - centroid)))
        rounded = max(0.0, min(1.0, low * 1.3 + low_mid * 0.6 - high * 0.5 + (1.0 - centroid) * 0.3))
        silence_gate = max(0.0, min(1.0, (0.065 - energy) / 0.065))

        scores = {
            "sil": 1.25 * silence_gate + max(0.0, 0.18 - energy) * 4.0,
            "aa": energy * (0.45 + low * 1.7 + low_mid * 0.8 + voicing * 0.45),
            "ee": energy * (0.25 + mid * 1.2 + high * 0.85 + centroid * 0.5 + zcr * 0.15),
            "ih": energy * (0.22 + low_mid * 1.0 + mid * 0.75 + centroid * 0.18),
            "oh": energy * (0.24 + rounded * 1.0 + low * 0.7 + low_mid * 0.4),
            "oo": energy * (0.16 + rounded * 1.55 + voicing * 0.18),
            "fv": energy * (0.05 + high * 1.65 + zcr * 0.9 + transient * 0.2),
            "mbp": energy * (0.08 + transient * 1.8 + max(0.0, 0.45 - zcr) * 0.25),
            "lntd": energy * (0.10 + low_mid * 0.75 + transient * 0.65 + centroid * 0.2),
            "szchj": energy * (0.08 + high * 1.15 + mid * 0.5 + zcr * 0.7),
            "wq": energy * (0.06 + rounded * 1.05 + low * 0.55 + (1.0 - zcr) * 0.12),
        }

        now = time.perf_counter() if timestamp_s is None else float(timestamp_s)
        if scores["mbp"] >= 0.22:
            self._mbp_hold_until = max(self._mbp_hold_until, now + 0.075)
        if now < self._mbp_hold_until:
            scores["mbp"] += 0.32
            scores["sil"] *= 0.75

        if energy < 0.08:
            scores["sil"] += max(0.0, min(1.0, (0.11 - energy) / 0.11)) * 1.5
            for cluster in ("aa", "ee", "ih", "oh", "oo", "szchj", "fv"):
                scores[cluster] *= 0.55

        targets = self._normalize(scores)
        if energy < 0.045 and self._energy < 0.06:
            targets["sil"] = max(targets["sil"], 0.82)
            remainder = 1.0 - targets["sil"]
            carry = self._normalize({name: value for name, value in targets.items() if name != "sil"}, fallback="aa")
            for name in targets:
                if name != "sil":
                    targets[name] = carry.get(name, 0.0) * remainder

        with self._lock:
            dt = max(0.012, min(0.05, now - self._last_tick))
            self._last_tick = now
            self._last_audio_at = now
            self._prev_energy = energy
            self._energy = self._smooth_exp(self._energy, energy, dt, 0.022, 0.135)
            self._attack = self._smooth_exp(self._attack, transient, dt, 0.018, 0.120)
            self._voicing_confidence = self._smooth_exp(self._voicing_confidence, voicing, dt, 0.025, 0.16)

            for cluster in _CLUSTERS:
                self._weights[cluster] = self._smooth_exp(self._weights[cluster], targets[cluster], dt, 0.025, 0.11)
            self._weights = self._normalize(self._weights)

            target_channels = self._cluster_weights_to_channels(self._weights)
            for channel, target in target_channels.items():
                if channel == "jaw_open":
                    self._channels[channel] = self._smooth_exp(self._channels[channel], target, dt, 0.018, 0.110)
                elif channel == "lip_round":
                    self._channels[channel] = self._smooth_exp(self._channels[channel], target, dt, 0.026, 0.145)
                else:
                    self._channels[channel] = self._smooth_exp(self._channels[channel], target, dt, 0.025, 0.12)
            self._channels["jaw_open"] = min(0.92, self._channels["jaw_open"], self._channels["mouth_open"] * 1.08 + 0.12)
            self._legacy_weights = self._to_legacy_weights_locked()

    def _cluster_weights_to_channels(self, weights: dict[str, float]) -> dict[str, float]:
        channels = {name: 0.0 for name in _CHANNEL_MAP["sil"]}
        for cluster, weight in weights.items():
            for channel, value in _CHANNEL_MAP[cluster].items():
                channels[channel] += weight * value
        channels["mouth_open"] = max(channels["mouth_open"], channels["jaw_open"] * 0.78)
        channels["lower_lip_drop"] = max(channels["lower_lip_drop"], channels["mouth_open"] * 0.72)
        return {name: max(0.0, min(1.0, value)) for name, value in channels.items()}

    def _to_legacy_weights_locked(self) -> dict[str, float]:
        aa = self._weights["aa"] + self._weights["oh"] * 0.30
        ee = self._weights["ee"] + self._weights["ih"] * 0.85 + self._weights["szchj"] * 0.30
        oo = self._weights["oo"] + self._weights["oh"] * 0.55 + self._weights["wq"] * 0.85
        small = self._weights["lntd"] * 0.55 + self._weights["fv"] * 0.52 + self._weights["szchj"] * 0.28 + self._weights["mbp"] * 0.35
        closed = self._weights["sil"] + self._weights["mbp"] * 0.70
        return self._normalize({"closed": closed, "small": small, "aa": aa, "ee": ee, "oo": oo}, fallback="closed")

    def _advance_idle_locked(self, now: float) -> None:
        dt = max(0.0, min(0.05, now - self._last_tick))
        self._last_tick = now
        if now - self._last_audio_at < 0.045:
            return
        targets = _default_cluster_weights()
        for cluster in _CLUSTERS:
            self._weights[cluster] = self._smooth_exp(self._weights[cluster], targets[cluster], dt, 0.04, 0.18)
        self._weights = self._normalize(self._weights)
        target_channels = self._cluster_weights_to_channels(self._weights)
        for channel, target in target_channels.items():
            self._channels[channel] = self._smooth_exp(self._channels[channel], target, dt, 0.04, 0.18)
        self._channels["jaw_open"] = min(0.92, self._channels["jaw_open"], self._channels["mouth_open"] * 1.05 + 0.10)
        self._energy = self._smooth_exp(self._energy, 0.0, dt, 0.05, 0.22)
        self._attack = self._smooth_exp(self._attack, 0.0, dt, 0.04, 0.18)
        self._voicing_confidence = self._smooth_exp(self._voicing_confidence, 0.0, dt, 0.05, 0.24)
        self._legacy_weights = self._to_legacy_weights_locked()

    def snapshot(self, *, timestamp_s: float | None = None) -> VisemeFrame:
        with self._lock:
            now = time.perf_counter() if timestamp_s is None else float(timestamp_s)
            self._advance_idle_locked(now)
            pose = max(self._legacy_weights.items(), key=lambda item: item[1])[0]
            cluster = max(self._weights.items(), key=lambda item: item[1])[0]
            openness = max(0.0, min(1.0, self._channels["mouth_open"] * 0.68 + self._channels["jaw_open"] * 0.32))
            return VisemeFrame(
                timestamp_s=now,
                cluster=cluster,
                pose=pose,
                openness=openness,
                energy=max(0.0, min(1.0, self._energy)),
                speech_energy=max(0.0, min(1.0, self._energy)),
                voicing_confidence=max(0.0, min(1.0, self._voicing_confidence)),
                attack=max(0.0, min(1.0, self._attack)),
                jaw_open=self._channels["jaw_open"],
                mouth_open=self._channels["mouth_open"],
                lip_funnel=self._channels["lip_funnel"],
                lip_round=self._channels["lip_round"],
                lip_spread=self._channels["lip_spread"],
                lip_press=self._channels["lip_press"],
                upper_lip_raise=self._channels["upper_lip_raise"],
                lower_lip_drop=self._channels["lower_lip_drop"],
                cheek_raise=self._channels["cheek_raise"],
                weights=dict(self._weights),
                legacy_weights=dict(self._legacy_weights),
            )

    def snapshot_dict(self, *, rich: bool = False, timestamp_s: float | None = None) -> dict[str, Any]:
        frame = self.snapshot(timestamp_s=timestamp_s)
        return frame.to_dict() if rich else frame.to_legacy_snapshot()
