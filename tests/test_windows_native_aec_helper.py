from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from kajovochat.services.windows_native_aec import WindowsNativeAECBackend, probe_windows_native_aec


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_DLL = REPO_ROOT / "native" / "windows_aec_helper" / "build" / "bin" / "Release" / "kajovochat_windows_aec.dll"


@pytest.mark.skipif(os.name != "nt", reason="Windows-native AEC helper je dostupny jen na Windows")
def test_windows_native_aec_helper_reduces_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HELPER_DLL.exists():
        pytest.skip("Windows-native helper DLL nebyla nalezena.")

    monkeypatch.setenv("KAJOVOCHAT_WINDOWS_AEC_DLL", str(HELPER_DLL))
    probe = probe_windows_native_aec()
    assert probe.available
    assert probe.helper_path

    backend = WindowsNativeAECBackend(
        input_samplerate=24000,
        filter_length=256,
        max_shift_samples=960,
    )
    try:
        rng = np.random.default_rng(42)
        ref = rng.integers(-9000, 9000, size=24000, dtype=np.int16)
        delay_samples = 288
        mic = np.zeros_like(ref)
        mic[delay_samples:] = (ref[:-delay_samples].astype(np.float32) * 0.72).astype(np.int16)
        mic_pcm = mic.tobytes()

        # Prvni pruchod helper zahreje, druhy uz musi mit viditelny efekt.
        backend.process(mic_pcm=mic_pcm, reference_pcm=ref, delay_ms=12)
        cleaned_pcm = backend.process(mic_pcm=mic_pcm, reference_pcm=ref, delay_ms=12)

        input_rms = float(np.sqrt(np.mean(mic.astype(np.float32) ** 2)))
        output_rms = float(np.sqrt(np.mean(np.frombuffer(cleaned_pcm, dtype=np.int16).astype(np.float32) ** 2)))

        assert output_rms < input_rms * 0.8
    finally:
        backend.close()
