from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from ..contracts import CaptureFrame

def _resample_pcm16_mono(pcm16: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 mono PCM from src_rate to dst_rate.

    Args:
        pcm16: 1D numpy array of dtype int16.
        src_rate: Source sample rate (Hz).
        dst_rate: Target sample rate (Hz).

    Returns:
        1D numpy array of dtype int16 at dst_rate.
    """
    if src_rate == dst_rate:
        return pcm16
    if pcm16.size == 0:
        return pcm16

    src_rate = int(src_rate)
    dst_rate = int(dst_rate)
    g = math.gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g

    x = pcm16.astype(np.float32) / 32768.0
    y = signal.resample_poly(x, up, down)
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


@dataclass
class RecordResult:
    wav_bytes: bytes
    duration_s: float
    samplerate: int
    rms_median: float

@dataclass
class CapturedAudioChunk:
    pcm_bytes: bytes
    captured_at_mono_ns: int

    def to_capture_frame(
        self,
        *,
        frame_index: int,
        sample_rate: int,
        channels: int = 1,
        processed_mic_pcm16: Optional[bytes] = None,
        render_ref_pcm16: Optional[bytes] = None,
        aec_backend: str = "unknown",
        aec_quality: float = 0.0,
        residual_level: float = 0.0,
        vad_probability: float = 0.0,
        double_talk: bool = False,
        stream_delay_ms: int = 0,
        device_clock_locked: bool = True,
    ) -> CaptureFrame:
        return CaptureFrame(
            frame_index=int(frame_index),
            mono_ns=int(self.captured_at_mono_ns),
            raw_mic_pcm16=self.pcm_bytes,
            processed_mic_pcm16=processed_mic_pcm16 if processed_mic_pcm16 is not None else self.pcm_bytes,
            render_ref_pcm16=render_ref_pcm16,
            sample_rate=int(sample_rate),
            channels=int(channels),
            aec_backend=aec_backend,
            aec_quality=float(aec_quality),
            residual_level=float(residual_level),
            vad_probability=float(vad_probability),
            double_talk=bool(double_talk),
            stream_delay_ms=int(stream_delay_ms),
            device_clock_locked=bool(device_clock_locked),
        )
