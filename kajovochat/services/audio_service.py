from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class RecordResult:
    wav_bytes: bytes
    duration_s: float
    samplerate: int


class AudioRecorder:
    """
    Records microphone audio into a WAV buffer.
    Stops on silence (simple energy-based VAD).
    """

    def __init__(
        self,
        samplerate: int = 16000,
        device: Optional[int] = None,
        rms_threshold: float = 0.012,
        silence_ms: int = 900,
        max_seconds: int = 25,
    ) -> None:
        self.samplerate = samplerate
        self.device = device
        self.rms_threshold = rms_threshold
        self.silence_ms = silence_ms
        self.max_seconds = max_seconds

    def record_once(self) -> RecordResult:
        block = 1024
        channels = 1

        frames = []
        started = time.time()
        last_loud = started

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=channels,
            dtype="float32",
            blocksize=block,
            device=self.device,
        ) as stream:
            while True:
                data, _ = stream.read(block)
                data = np.asarray(data).reshape(-1)
                frames.append(data.copy())

                rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
                now = time.time()
                if rms >= self.rms_threshold:
                    last_loud = now

                if (now - last_loud) * 1000.0 >= self.silence_ms and (now - started) > 0.6:
                    break
                if (now - started) >= self.max_seconds:
                    break

        audio = np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32)
        duration = len(audio) / float(self.samplerate)

        # write wav to bytes (16-bit PCM)
        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
        return RecordResult(wav_bytes=buf.getvalue(), duration_s=duration, samplerate=self.samplerate)


class AudioPlayer:
    """
    Plays PCM audio (int16) at a fixed samplerate.
    """

    def __init__(self, samplerate: int = 24000, device: Optional[int] = None) -> None:
        self.samplerate = samplerate
        self.device = device

    def play_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=self.samplerate, device=self.device, blocking=True)
