from __future__ import annotations

import numpy as np


def estimate_voice_likelihood_from_pcm16(pcm: bytes) -> float:
    """Odhad pravděpodobnosti lidské řeči v krátkém PCM16 chunku.

    Není to ASR/VAD model. Kombinuje několik lehkých DSP feature tak, aby lépe
    odlišil řeč od čistého leakage reproduktorů a impulsního šumu.
    """
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
    window = np.hanning(centered.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(centered * window))
    if spec.size < 8:
        return min(1.0, energy * 24.0)

    freqs = np.fft.rfftfreq(centered.size, d=1.0 / 24000.0)
    speech_band = spec[(freqs >= 120.0) & (freqs <= 3400.0)]
    low_band = spec[(freqs >= 20.0) & (freqs < 120.0)]
    high_band = spec[(freqs > 3400.0) & (freqs <= 7600.0)]
    presence_band = spec[(freqs >= 1800.0) & (freqs <= 4200.0)]

    speech_energy = float(np.sum(speech_band) + 1e-6)
    low_energy = float(np.sum(low_band) + 1e-6)
    high_energy = float(np.sum(high_band) + 1e-6)
    presence_energy = float(np.sum(presence_band) + 1e-6)
    total_energy = speech_energy + low_energy + high_energy

    # ZCR je užitečný, ale příliš vysoké hodnoty indikují spíš šum.
    zero_cross = float(np.mean(np.abs(np.diff(np.signbit(centered)).astype(np.float32))))
    zcr_score = max(0.0, 1.0 - abs(zero_cross - 0.12) / 0.16)

    # Spektrální plochost: řeč bývá méně plochá než širokopásmový šum.
    speech_log = np.log(np.maximum(speech_band, 1e-7))
    spectral_flatness = float(np.exp(np.mean(speech_log)) / (np.mean(speech_band) + 1e-7))
    flatness_score = max(0.0, min(1.0, 1.0 - spectral_flatness * 1.6))

    spectral_balance = speech_energy / total_energy
    presence_ratio = presence_energy / speech_energy
    low_penalty = min(1.0, low_energy / speech_energy)
    high_penalty = min(1.0, high_energy / speech_energy)

    score = (
        spectral_balance * 0.42
        + min(1.0, energy * 15.0) * 0.14
        + zcr_score * 0.16
        + flatness_score * 0.16
        + min(1.0, presence_ratio * 3.2) * 0.08
        + (1.0 - min(1.0, (low_penalty * 0.7) + (high_penalty * 0.35))) * 0.04
    )
    return float(max(0.0, min(1.0, score)))
