from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from kajovochat.services.windows_native_aec import (
    WindowsNativeAECBackend,
    _probe_installed_audio_processing_object,
    probe_windows_native_aec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_DLL = REPO_ROOT / "native" / "windows_apo_helper" / "build" / "bin" / "Release" / "kajovochat_windows_apo.dll"


@pytest.mark.skipif(os.name != "nt", reason="Windows-native APO helper je dostupny jen na Windows")
def test_windows_apo_helper_reduces_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    if not HELPER_DLL.exists():
        pytest.skip("Windows-native APO helper DLL nebyla nalezena.")

    monkeypatch.setenv("KAJOVOCHAT_WINDOWS_APO_DLL", str(HELPER_DLL))
    probe = probe_windows_native_aec()
    assert probe.available
    assert probe.helper_path

    backend = WindowsNativeAECBackend(
        input_samplerate=24000,
        filter_length=256,
        max_shift_samples=960,
    )
    try:
        assert backend.using_system_capture_contract is True
        rng = np.random.default_rng(7)
        ref = rng.integers(-8500, 8500, size=24000, dtype=np.int16)
        delay_samples = 224
        mic = np.zeros_like(ref)
        mic[delay_samples:] = (ref[:-delay_samples].astype(np.float32) * 0.68).astype(np.int16)
        mic_pcm = mic.tobytes()

        cleaned_pcm = backend.process(mic_pcm=mic_pcm, reference_pcm=ref, delay_ms=9)

        input_rms = float(np.sqrt(np.mean(mic.astype(np.float32) ** 2)))
        output_rms = float(np.sqrt(np.mean(np.frombuffer(cleaned_pcm, dtype=np.int16).astype(np.float32) ** 2)))

        assert output_rms <= input_rms * 1.05
        assert len(cleaned_pcm) == len(mic_pcm)
    finally:
        backend.close()


def test_probe_installed_audio_processing_object_parses_pnputil_output(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = """
Published Name:     oem41.inf
Original Name:      kajovochat_windows_apo.inf
Provider Name:      KajovoChat
Class Name:         AudioProcessingObject
Class GUID:         {5989fce8-9cd0-467d-8a6a-5419e31529d4}
"""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=sample, stderr="")

    monkeypatch.setattr("kajovochat.services.windows_native_aec.subprocess.run", fake_run)
    monkeypatch.setattr("kajovochat.services.windows_native_aec.os.name", "nt")

    installed, published_name, original_name, provider_name = _probe_installed_audio_processing_object()

    assert installed is True
    assert published_name == "oem41.inf"
    assert original_name == "kajovochat_windows_apo.inf"
    assert provider_name == "KajovoChat"
