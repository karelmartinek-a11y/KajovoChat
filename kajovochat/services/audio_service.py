from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class RecordResult:
    wav_bytes: bytes
    duration_s: float
    samplerate: int
    rms_median: float


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


class AudioRecorder:
    """
    Records microphone audio into a WAV buffer.

    - Hands-free: simple energy-based VAD (RMS threshold + silence timeout).
    - Push-to-talk: record until external stop event.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        device: Optional[int] = None,
        rms_threshold: float = 0.012,
        silence_ms: int = 900,
        max_seconds: int = 25,
        blocksize: int = 1024,
    ) -> None:
        self.samplerate = samplerate
        self.device = device
        self.rms_threshold = rms_threshold
        self.silence_ms = silence_ms
        self.max_seconds = max_seconds
        self.blocksize = blocksize

    def calibrate_noise(self, seconds: float = 0.7) -> float:
        """Measure background RMS for a short period."""
        frames = []
        deadline = time.time() + max(0.2, seconds)
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
        ) as stream:
            while time.time() < deadline:
                data, _ = stream.read(self.blocksize)
                frames.append(_rms(data))
        if not frames:
            return 0.0
        return float(np.median(np.asarray(frames, dtype=np.float32)))

    def record_handsfree(self, cancel: Optional[threading.Event] = None, threshold: Optional[float] = None) -> RecordResult:
        block = self.blocksize
        frames = []
        rms_values = []

        started = time.time()
        last_loud = started
        thr = float(threshold if threshold is not None else self.rms_threshold)

        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=self.device,
        ) as stream:
            while True:
                if cancel and cancel.is_set():
                    break
                data, _ = stream.read(block)
                data = np.asarray(data).reshape(-1)
                frames.append(data.copy())

                r = _rms(data)
                rms_values.append(r)
                now = time.time()

                if r >= thr:
                    last_loud = now

                # Stop if enough silence after some audio.
                if (now - last_loud) * 1000.0 >= self.silence_ms and (now - started) > 0.6:
                    break
                if (now - started) >= self.max_seconds:
                    break

        audio = np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32)
        duration = len(audio) / float(self.samplerate)
        rms_med = float(np.median(np.asarray(rms_values, dtype=np.float32))) if rms_values else 0.0

        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
        return RecordResult(wav_bytes=buf.getvalue(), duration_s=duration, samplerate=self.samplerate, rms_median=rms_med)

    def record_ptt(self, stop_event: threading.Event, cancel: Optional[threading.Event] = None) -> RecordResult:
        block = self.blocksize
        frames = []
        rms_values = []

        started = time.time()
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=self.device,
        ) as stream:
            while True:
                if cancel and cancel.is_set():
                    break
                if stop_event.is_set():
                    break
                data, _ = stream.read(block)
                data = np.asarray(data).reshape(-1)
                frames.append(data.copy())
                rms_values.append(_rms(data))
                if (time.time() - started) >= self.max_seconds:
                    break

        audio = np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32)
        duration = len(audio) / float(self.samplerate)
        rms_med = float(np.median(np.asarray(rms_values, dtype=np.float32))) if rms_values else 0.0

        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV", subtype="PCM_16")
        return RecordResult(wav_bytes=buf.getvalue(), duration_s=duration, samplerate=self.samplerate, rms_median=rms_med)


class AudioPlayer:
    """Low-latency PCM16 playback with stable buffering and interruption.

    - Uses an OutputStream callback and an internal ring buffer (bytearray).
    - `stop()` clears the buffer and closes the stream immediately.
    - `play_pcm16()` blocks in the worker thread, but never blocks the GUI thread.
    """

    def __init__(self, samplerate: int = 24000, device: Optional[int] = None, blocksize: int = 1024) -> None:
        self.samplerate = int(samplerate)
        self.device = device
        self.blocksize = int(blocksize)

        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._stream: Optional[sd.OutputStream] = None
        self._closed = False

    def _ensure_stream(self) -> None:
        if self._stream:
            return

        def callback(outdata, frames, time_info, status) -> None:
            # mono float32 output
            need_bytes = frames * 2  # int16
            with self._lock:
                if self._closed:
                    outdata[:] = 0
                    return
                if len(self._buffer) >= need_bytes:
                    chunk = bytes(self._buffer[:need_bytes])
                    del self._buffer[:need_bytes]
                else:
                    chunk = bytes(self._buffer)
                    self._buffer.clear()

            if not chunk:
                outdata[:] = 0
                return

            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.shape[0] < frames:
                padded = np.zeros((frames,), dtype=np.float32)
                padded[: pcm.shape[0]] = pcm
                pcm = padded
            outdata[:, 0] = pcm

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._closed = True

        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
        self._stream = None

        with self._lock:
            self._closed = False

    def enqueue_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self._ensure_stream()
        with self._lock:
            self._buffer.extend(pcm_bytes)

    def play_pcm16(self, pcm_bytes: bytes, cancel: Optional[threading.Event] = None) -> None:
        if not pcm_bytes:
            return

        self.enqueue_pcm16(pcm_bytes)

        # Wait until the buffer drains (or cancellation requested).
        while True:
            if cancel and cancel.is_set():
                self.stop()
                return
            with self._lock:
                remaining = len(self._buffer)
            if remaining <= 0:
                # double-check after a short sleep to allow callback to run
                time.sleep(0.03)
                with self._lock:
                    if len(self._buffer) <= 0:
                        break
            time.sleep(0.01)
class VADMonitor:
    """
    Background VAD monitor for barge-in (speech start detection).
    """

    def __init__(
        self,
        samplerate: int,
        device: Optional[int],
        threshold: float,
        trigger_ms: int = 140,
        blocksize: int = 512,
    ) -> None:
        self.samplerate = samplerate
        self.device = device
        self.threshold = float(threshold)
        self.trigger_ms = int(trigger_ms)
        self.blocksize = int(blocksize)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, on_voice: Callable[[float], None]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(on_voice,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.8)

    def _run(self, on_voice: Callable[[float], None]) -> None:
        above_ms = 0.0
        dt_ms = (self.blocksize / float(self.samplerate)) * 1000.0

        try:
            with sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                blocksize=self.blocksize,
                device=self.device,
            ) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(self.blocksize)
                    r = _rms(data)
                    if r >= self.threshold:
                        above_ms += dt_ms
                        if above_ms >= self.trigger_ms:
                            on_voice(r)
                            # reset so we don't spam
                            above_ms = 0.0
                            time.sleep(0.05)
                    else:
                        above_ms = 0.0
        except Exception:
            # If mic cannot be opened concurrently, monitoring degrades gracefully.
            return

