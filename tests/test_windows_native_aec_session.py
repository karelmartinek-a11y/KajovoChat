from __future__ import annotations

import numpy as np

from kajovochat.services.windows_native_aec import (
    WindowsNativeAECProbe,
    WindowsNativeAECSession,
    WindowsNativeAECSessionConfig,
)
from kajovochat.audio.contracts import RenderFrame


class _FakeBackend:
    def __init__(self, *, using_system_capture_contract: bool) -> None:
        self.using_system_capture_contract = using_system_capture_contract
        self.last_quality = 0.82
        self.last_voice_probability = 0.37
        self.last_improvement = 0.41
        self.last_residual = 0.013
        self.last_flags = 1 if using_system_capture_contract else 0
        self.closed = False
        self.calls: list[dict[str, object]] = []

    def process(self, *, mic_pcm: bytes, reference_pcm: np.ndarray, delay_ms: int) -> bytes:
        self.calls.append(
            {
                "mic_bytes": len(mic_pcm),
                "reference_samples": int(reference_pcm.size),
                "delay_ms": int(delay_ms),
            }
        )
        mic = np.frombuffer(mic_pcm, dtype=np.int16).copy()
        return (mic // 2).astype(np.int16, copy=False).tobytes()

    def close(self) -> None:
        self.closed = True


def _probe() -> WindowsNativeAECProbe:
    return WindowsNativeAECProbe(
        available=True,
        reason="ok",
        helper_path="C:\\fake\\kajovochat_windows_apo.dll",
        installed_driver=True,
        published_name="oem41.inf",
        original_name="kajovochat_windows_apo.inf",
        provider_name="KajovoChat",
    )


def test_windows_native_aec_session_processes_frame_with_render_reference() -> None:
    backend = _FakeBackend(using_system_capture_contract=False)
    session = WindowsNativeAECSession(
        WindowsNativeAECSessionConfig(samplerate=24000, frame_samples=240),
        probe=_probe(),
        backend=backend,
    )
    session.start()
    session.write_render_frame(RenderFrame(frame_index=3, mono_ns=123456, pcm16=(b"\x01\x00" * 240), tts_active=True))
    session.submit_capture_frame(
        raw_mic_pcm16=(b"\x10\x00" * 240),
        mono_ns=654321,
        stream_delay_ms=12,
    )

    frame = session.read_capture_frame(timeout_ms=0)
    assert frame is not None
    assert frame.frame_index == 0
    assert frame.mono_ns == 654321
    assert frame.sample_rate == 24000
    assert frame.channels == 1
    assert frame.aec_backend == "windows_native"
    assert frame.aec_quality == 0.82
    assert frame.vad_probability == 0.37
    assert frame.stream_delay_ms == 12
    assert frame.render_ref_pcm16 == (b"\x01\x00" * 240)
    assert len(frame.processed_mic_pcm16) == len(frame.raw_mic_pcm16)
    assert backend.calls[0]["reference_samples"] == 240

    health = session.get_health_snapshot()
    assert health.processed_frames == 1
    assert health.backend_snapshot.backend == "windows_native"
    assert health.backend_snapshot.reference_health_state == "render_feed"

    session.close()
    assert backend.closed is True


def test_windows_native_aec_session_uses_system_capture_contract_without_reference() -> None:
    backend = _FakeBackend(using_system_capture_contract=True)
    session = WindowsNativeAECSession(
        WindowsNativeAECSessionConfig(samplerate=24000, frame_samples=240),
        probe=_probe(),
        backend=backend,
    )
    session.start()
    session.submit_capture_frame(
        raw_mic_pcm16=(b"\x08\x00" * 240),
        mono_ns=111,
        stream_delay_ms=7,
    )

    frame = session.read_capture_frame(timeout_ms=0)
    assert frame is not None
    assert frame.aec_backend == "windows_system_capture"
    assert frame.double_talk is True
    assert backend.calls[0]["reference_samples"] == 0

    health = session.get_health_snapshot()
    assert health.backend_snapshot.backend == "windows_system_capture"
    assert health.backend_snapshot.reference_ready is True
    assert health.backend_snapshot.reference_health_state == "system_capture"
