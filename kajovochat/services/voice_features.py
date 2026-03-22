from __future__ import annotations

import numpy as np


def estimate_voice_likelihood_from_pcm16(pcm: bytes) -> float:
    """Hrubý odhad, jestli chunk připomíná lidský hlas víc než čisté echo."""
    if not pcm:
        return 0.0

    data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if data.size < 64:
        return 0.0
    data /= 32768.0

    energy = float(np.sqrt(np.mean(data * data) + 1e-12))
    if energy <= 1e-4:
        return 0.0

    centered = data - float(np.mean(data))
    spec = np.abs(np.fft.rfft(centered))
    if spec.size < 8:
        return min(1.0, energy * 24.0)

    freqs = np.fft.rfftfreq(centered.size, d=1.0 / 24000.0)
    speech_band = spec[(freqs >= 120.0) & (freqs <= 3400.0)]
    low_band = spec[(freqs >= 20.0) & (freqs < 120.0)]
    high_band = spec[(freqs > 3400.0) & (freqs <= 7600.0)]

    speech_energy = float(np.sum(speech_band) + 1e-6)
    low_energy = float(np.sum(low_band) + 1e-6)
    high_energy = float(np.sum(high_band) + 1e-6)
    total_energy = speech_energy + low_energy + high_energy

    zero_cross = float(np.mean(np.abs(np.diff(np.signbit(centered)).astype(np.float32))))
    spectral_balance = speech_energy / total_energy
    low_penalty = min(1.0, low_energy / speech_energy)
    high_penalty = min(1.0, high_energy / speech_energy)

    score = (
        spectral_balance * 0.58
        + min(1.0, energy * 16.0) * 0.16
        + min(1.0, zero_cross * 7.0) * 0.18
        + (1.0 - min(1.0, (low_penalty * 0.75) + (high_penalty * 0.35))) * 0.08
    )
    return float(max(0.0, min(1.0, score)))
