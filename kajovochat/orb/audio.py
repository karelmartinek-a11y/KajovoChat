from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import LivingOrbConfig


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _smooth_exp(current: float, target: float, dt: float, attack: float, release: float) -> float:
    if dt <= 0.0:
        return current
    tau = attack if target > current else release
    tau = max(0.001, float(tau))
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


@dataclass
class AudioFeatureFrame:
    loudness: float = 0.0
    rms: float = 0.0
    peak_envelope: float = 0.0
    short_energy: float = 0.0
    low_band: float = 0.0
    mid_band: float = 0.0
    high_band: float = 0.0
    spectral_centroid: float = 0.0
    spectral_flux: float = 0.0
    transient_activity: float = 0.0
    speaking_gate: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "loudness": self.loudness,
            "rms": self.rms,
            "peak_envelope": self.peak_envelope,
            "short_energy": self.short_energy,
            "low_band": self.low_band,
            "mid_band": self.mid_band,
            "high_band": self.high_band,
            "spectral_centroid": self.spectral_centroid,
            "spectral_flux": self.spectral_flux,
            "transient_activity": self.transient_activity,
            "speaking_gate": self.speaking_gate,
        }


@dataclass
class AudioAnalyzer:
    config: LivingOrbConfig
    sample_rate: int = 24000
    _smoothed: AudioFeatureFrame = field(default_factory=AudioFeatureFrame)
    _prev_spectrum: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    _gate_hold_remaining: float = 0.0

    def extract(self, samples: np.ndarray, sample_rate: int | None = None) -> AudioFeatureFrame:
        effective_rate = int(sample_rate or self.sample_rate or 24000)
        self.sample_rate = effective_rate
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return AudioFeatureFrame()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(np.max(np.abs(x))) if x.size else 0.0
        if max_abs > 1.5:
            x = x / 32768.0
        x = np.clip(x, -1.0, 1.0)

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        short_energy = float(np.mean(np.abs(x) ** 1.4))

        fft_size = max(8, x.size)
        padded = np.zeros((fft_size,), dtype=np.float32)
        padded[: x.size] = x
        window = np.hanning(fft_size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(padded * window)).astype(np.float32)
        if spectrum.size == 0:
            spectrum = np.zeros((1,), dtype=np.float32)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(effective_rate))
        power = spectrum * spectrum
        power_sum = float(np.sum(power) + 1e-9)

        def _band(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return 0.0
            return float(np.sum(power[mask]) / power_sum)

        low_band = _band(20.0, 260.0)
        mid_band = _band(260.0, 1800.0)
        high_band = _band(1800.0, min(8000.0, effective_rate * 0.5))

        centroid = 0.0
        if power_sum > 1e-9:
            centroid = float(np.sum(freqs * power) / power_sum) / max(1.0, effective_rate * 0.5)

        if self._prev_spectrum.size != spectrum.size:
            prev = np.zeros_like(spectrum)
        else:
            prev = self._prev_spectrum
        flux = float(np.mean(np.maximum(0.0, spectrum - prev))) / max(1e-6, float(np.mean(spectrum) + 1e-6))
        transient = max(0.0, flux * 0.85 + max(0.0, peak - rms) * 0.55)
        speaking_gate = 0.0
        if rms > 0.018 or (peak > 0.08 and flux > 0.10):
            speaking_gate = 1.0

        self._prev_spectrum = spectrum
        loudness = max(rms * 2.1, peak * 0.92, short_energy * 1.5)
        return AudioFeatureFrame(
            loudness=_clamp01(loudness),
            rms=_clamp01(rms * 4.0),
            peak_envelope=_clamp01(peak),
            short_energy=_clamp01(short_energy * 2.0),
            low_band=_clamp01(low_band * 4.0),
            mid_band=_clamp01(mid_band * 4.0),
            high_band=_clamp01(high_band * 5.0),
            spectral_centroid=_clamp01(centroid),
            spectral_flux=_clamp01(flux * 1.8),
            transient_activity=_clamp01(transient),
            speaking_gate=_clamp01(speaking_gate),
        )

    def smooth(self, target: AudioFeatureFrame, dt: float) -> AudioFeatureFrame:
        if dt <= 0.0:
            return self._smoothed

        raw_gate = target.speaking_gate
        if raw_gate >= 0.5:
            self._gate_hold_remaining = self.config.speaking_hold_seconds
        else:
            self._gate_hold_remaining = max(0.0, self._gate_hold_remaining - dt)
        held_gate = 1.0 if self._gate_hold_remaining > 0.0 else raw_gate

        self._smoothed = AudioFeatureFrame(
            loudness=_smooth_exp(self._smoothed.loudness, target.loudness, dt, self.config.attack_seconds, self.config.release_seconds),
            rms=_smooth_exp(self._smoothed.rms, target.rms, dt, self.config.attack_seconds, self.config.release_seconds),
            peak_envelope=_smooth_exp(self._smoothed.peak_envelope, target.peak_envelope, dt, self.config.peak_attack_seconds, self.config.peak_release_seconds),
            short_energy=_smooth_exp(self._smoothed.short_energy, target.short_energy, dt, self.config.attack_seconds, self.config.release_seconds),
            low_band=_smooth_exp(self._smoothed.low_band, target.low_band, dt, self.config.attack_seconds, self.config.release_seconds),
            mid_band=_smooth_exp(self._smoothed.mid_band, target.mid_band, dt, self.config.attack_seconds, self.config.release_seconds),
            high_band=_smooth_exp(self._smoothed.high_band, target.high_band, dt, self.config.attack_seconds * 0.7, self.config.release_seconds * 0.8),
            spectral_centroid=_smooth_exp(self._smoothed.spectral_centroid, target.spectral_centroid, dt, self.config.attack_seconds, self.config.release_seconds * 0.75),
            spectral_flux=_smooth_exp(self._smoothed.spectral_flux, target.spectral_flux, dt, self.config.peak_attack_seconds, self.config.transient_decay_seconds),
            transient_activity=_smooth_exp(self._smoothed.transient_activity, target.transient_activity, dt, self.config.peak_attack_seconds, self.config.transient_decay_seconds),
            speaking_gate=_smooth_exp(self._smoothed.speaking_gate, held_gate, dt, self.config.peak_attack_seconds, self.config.release_seconds),
        )
        return self._smoothed

    def update_from_pcm(self, samples: np.ndarray, dt: float, sample_rate: int | None = None) -> AudioFeatureFrame:
        extracted = self.extract(samples, sample_rate=sample_rate)
        return self.smooth(extracted, dt)

    def update_from_features(self, features: Mapping[str, float], dt: float) -> AudioFeatureFrame:
        frame = AudioFeatureFrame(
            loudness=_clamp01(float(features.get("loudness", features.get("rms", 0.0)))),
            rms=_clamp01(float(features.get("rms", 0.0))),
            peak_envelope=_clamp01(float(features.get("peak_envelope", features.get("peak", 0.0)))),
            short_energy=_clamp01(float(features.get("short_energy", features.get("energy", 0.0)))),
            low_band=_clamp01(float(features.get("low_band", 0.0))),
            mid_band=_clamp01(float(features.get("mid_band", 0.0))),
            high_band=_clamp01(float(features.get("high_band", 0.0))),
            spectral_centroid=_clamp01(float(features.get("spectral_centroid", features.get("centroid", 0.0)))),
            spectral_flux=_clamp01(float(features.get("spectral_flux", features.get("flux", 0.0)))),
            transient_activity=_clamp01(float(features.get("transient_activity", features.get("transient", 0.0)))),
            speaking_gate=_clamp01(float(features.get("speaking_gate", features.get("vad", 0.0)))),
        )
        if frame.loudness <= 0.0:
            frame.loudness = _clamp01(max(frame.rms, frame.peak_envelope, frame.short_energy))
        return self.smooth(frame, dt)
