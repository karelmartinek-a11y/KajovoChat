from __future__ import annotations

import io
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from ..contracts import RenderFrame
from ...services.lip_sync_engine import LipSyncEngine
from .common import CapturedAudioChunk, RecordResult, _resample_pcm16_mono
from ..dsp_helpers import _rms

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
        # The Realtime API returns audio at 24kHz PCM by default.
        # Some output devices/drivers do not accept 24kHz; we fall back to the
        # device default rate and resample on enqueue.
        self.target_samplerate = int(samplerate)
        self.samplerate = int(samplerate)  # actual stream samplerate (may change on fallback)
        self.device = device
        self.blocksize = int(blocksize)

        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._stream: Optional[sd.OutputStream] = None
        self._closed = False

        # Approximate current playback level (0..1). Updated in the audio
        # callback thread; read from UI/worker threads.
        self._level: float = 0.0
        self._lip_sync = LipSyncEngine()
        self._echo_reference_chunks: "deque[tuple[int, bytes]]" = deque()
        self._echo_reference_enqueued_samples = 0
        self._echo_reference_played_samples = 0
        self._echo_reference_max_samples = int(self.target_samplerate * 2.0)
        self._last_callback_mono_ns = 0
        self._echo_reference_played_end_mono_ns = 0
        self._started_mono_ns = time.monotonic_ns()

    def _ensure_stream(self) -> None:
        if self._stream:
            return

        # Try the target samplerate first; if that fails, fall back to the device
        # default samplerate and resample incoming PCM to match.
        try_rates = [self.target_samplerate]
        try:
            devinfo = sd.query_devices(self.device, "output") if self.device is not None else sd.query_devices(None, "output")
            default_rate = int(devinfo.get("default_samplerate") or 0)
            if default_rate and default_rate != self.target_samplerate:
                try_rates.append(default_rate)
        except Exception:
            pass

        def callback(outdata, frames, time_info, status) -> None:
            # mono float32 output
            need_bytes = frames * 2  # int16
            with self._lock:
                self._last_callback_mono_ns = time.monotonic_ns()
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
                self._level = 0.0
                return

            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.shape[0] < frames:
                padded = np.zeros((frames,), dtype=np.float32)
                padded[: pcm.shape[0]] = pcm
                pcm = padded
            outdata[:, 0] = pcm

            # Sleduj výstupní hlasitost pro UI animaci hlavy. Musí to být lehké.
            try:
                rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
                peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                lvl = max(rms * 1.8, peak * 1.0)
                self._level = float(max(0.0, min(1.0, lvl)))
            except Exception:
                self._level = 0.0
            try:
                played = np.clip(pcm * 32767.0, -32768.0, 32767.0).astype(np.int16, copy=False)
            except Exception:
                played = np.zeros((frames,), dtype=np.int16)

            try:
                played_target_samples = frames
                if self.samplerate != self.target_samplerate:
                    played_target_samples = int(round(frames * (self.target_samplerate / float(self.samplerate))))
                with self._lock:
                    self._echo_reference_played_samples += max(0, played_target_samples)
                    playback_horizon_ns = int(round(max(0, played_target_samples) * (1_000_000_000.0 / float(self.target_samplerate))))
                    self._echo_reference_played_end_mono_ns = self._last_callback_mono_ns + playback_horizon_ns
                    while self._echo_reference_chunks:
                        oldest_end = self._echo_reference_chunks[0][0]
                        if self._echo_reference_played_samples - oldest_end <= self._echo_reference_max_samples:
                            break
                        self._echo_reference_chunks.popleft()
            except Exception:
                pass

            try:
                self._lip_sync.consume_playback_pcm16(played.tobytes(), samplerate=self.samplerate)
            except Exception:
                pass

        last_err: Optional[Exception] = None
        for rate in try_rates:
            try:
                self.samplerate = int(rate)
                self._stream = sd.OutputStream(
                    samplerate=self.samplerate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.blocksize,
                    device=self.device,
                    callback=callback,
                )
                self._stream.start()
                last_err = None
                break
            except Exception as e:
                last_err = e
                self._stream = None

        if last_err is not None:
            raise last_err

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
            self._echo_reference_chunks.clear()
            self._echo_reference_enqueued_samples = 0
            self._echo_reference_played_samples = 0
            self._last_callback_mono_ns = 0
            self._echo_reference_played_end_mono_ns = 0
        self._level = 0.0
        self._lip_sync.reset()

    def get_level(self) -> float:
        """Return approximate current playback level in range 0..1."""
        try:
            return float(self._level)
        except Exception:
            return 0.0

    def get_lipsync_snapshot(self) -> dict[str, object]:
        try:
            snap = self._lip_sync.snapshot()
            return {
                "pose": snap.pose,
                "openness": snap.openness,
                "energy": snap.energy,
                "weights": dict(snap.weights),
            }
        except Exception:
            return {
                "pose": "closed",
                "openness": 0.0,
                "energy": 0.0,
                "weights": {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0},
            }

    def get_echo_reference(self, max_samples: int = 4096) -> np.ndarray:
        return self.get_echo_reference_for_capture(max_samples=max_samples, captured_at_mono_ns=None)

    def get_echo_reference_for_capture(self, *, max_samples: int = 4096, captured_at_mono_ns: Optional[int]) -> np.ndarray:
        try:
            need_samples = max(1, int(max_samples))
            with self._lock:
                if not self._echo_reference_chunks:
                    return np.zeros((0,), dtype=np.int16)
                played_end = self._echo_reference_played_samples
                if captured_at_mono_ns is not None and self._echo_reference_played_end_mono_ns > 0:
                    future_ns = max(0, int(self._echo_reference_played_end_mono_ns) - int(captured_at_mono_ns))
                    if future_ns > 0:
                        future_samples = int(round(future_ns * (self.target_samplerate / 1_000_000_000.0)))
                        played_end = max(0, played_end - future_samples)
                if played_end <= 0:
                    return np.zeros((0,), dtype=np.int16)
                start_sample = max(0, played_end - need_samples)
                chunks: list[bytes] = []
                cursor = played_end
                for end_sample, payload in reversed(self._echo_reference_chunks):
                    payload_samples = len(payload) // 2
                    chunk_start = end_sample - payload_samples
                    overlap_start = max(chunk_start, start_sample)
                    overlap_end = min(end_sample, played_end)
                    if overlap_end <= overlap_start:
                        continue
                    offset_start = overlap_start - chunk_start
                    offset_end = overlap_end - chunk_start
                    payload_view = memoryview(payload)[offset_start * 2 : offset_end * 2]
                    chunks.append(bytes(payload_view))
                    cursor = overlap_start
                    if cursor <= start_sample:
                        break
                tail = b"".join(reversed(chunks))
            if not tail:
                return np.zeros((0,), dtype=np.int16)
            array = np.frombuffer(tail, dtype=np.int16)
            if array.size > need_samples:
                array = array[-need_samples:]
            return array.copy()
        except Exception:
            return np.zeros((0,), dtype=np.int16)

    def get_echo_reference_stats(self) -> dict[str, int]:
        try:
            with self._lock:
                available_samples = sum(len(payload) // 2 for _, payload in self._echo_reference_chunks)
                return {
                    "available_samples": int(max(0, min(available_samples, self._echo_reference_played_samples))),
                    "total_samples": int(self._echo_reference_enqueued_samples),
                    "played_samples": int(self._echo_reference_played_samples),
                    "callback_age_ms": int((time.monotonic_ns() - self._last_callback_mono_ns) / 1_000_000) if self._last_callback_mono_ns else -1,
                    "played_end_mono_ns": int(self._echo_reference_played_end_mono_ns),
                }
        except Exception:
            return {"available_samples": 0, "total_samples": 0, "played_samples": 0, "callback_age_ms": -1, "played_end_mono_ns": 0}

    @property
    def buffered_bytes(self) -> int:
        try:
            with self._lock:
                return int(len(self._buffer))
        except Exception:
            return 0

    def enqueue_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self._ensure_stream()

        try:
            target_pcm = pcm_bytes
            target_samples = len(target_pcm) // 2
            if target_samples > 0:
                with self._lock:
                    self._echo_reference_enqueued_samples += target_samples
                    self._echo_reference_chunks.append((self._echo_reference_enqueued_samples, bytes(target_pcm)))
                    while self._echo_reference_chunks:
                        oldest_end = self._echo_reference_chunks[0][0]
                        if self._echo_reference_enqueued_samples - oldest_end <= self._echo_reference_max_samples:
                            break
                        self._echo_reference_chunks.popleft()
        except Exception:
            pass

        # If the stream is running at a different samplerate than what the model
        # produced (target_samplerate), resample before buffering.
        if self.samplerate != self.target_samplerate:
            try:
                x = np.frombuffer(pcm_bytes, dtype=np.int16)
                y = _resample_pcm16_mono(x, self.target_samplerate, self.samplerate)
                pcm_bytes = y.tobytes()
            except Exception:
                # If resampling fails for any reason, fall back to playing raw
                # bytes (will sound wrong, but avoids crashing).
                pass
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


class RealtimeMicStream:
    """Capture microphone audio as PCM16 frames suitable for Realtime API.

    The Realtime API supports PCM audio at 24kHz (mono). We expose a small queue
    of raw PCM16 bytes (little-endian) for the sender thread to Base64-encode
    and ship via `input_audio_buffer.append`.
    """

    def __init__(
        self,
        samplerate: int = 24000,
        device: Optional[int] = None,
        blocksize: int = 480,  # ~20ms @ 24kHz
    ) -> None:
        # Target samplerate expected by Realtime API for PCM.
        self.samplerate = int(samplerate)
        self.device = device
        # blocksize is specified in target-rate frames (defaults to ~20ms).
        self.blocksize = int(blocksize)

        # Actual input samplerate chosen for the device (may differ).
        self.input_samplerate = int(samplerate)
        self.using_resampler = False

        # Resampler state (small overlap to reduce chunk boundary artifacts)
        self._rs_overlap = 0
        self._rs_prev = np.zeros((0,), dtype=np.int16)

        self._stream: Optional[sd.InputStream] = None
        self._queue: "queue.Queue[CapturedAudioChunk]" = queue.Queue(maxsize=200)
        self._running = False
        self._started_mono_ns = 0
        self._last_capture_mono_ns = 0
        self._captured_samples = 0

    @property
    def queue(self) -> "queue.Queue[CapturedAudioChunk]":
        return self._queue

    @property
    def pending_chunk_count(self) -> int:
        try:
            return int(self._queue.qsize())
        except Exception:
            return 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_mono_ns = time.monotonic_ns()
        self._last_capture_mono_ns = 0
        self._captured_samples = 0

        # Try opening the mic at 24kHz; if the device/driver rejects it,
        # fall back to the device default rate and resample to 24kHz.
        try_rates = [self.samplerate]
        try:
            devinfo = sd.query_devices(self.device, "input") if self.device is not None else sd.query_devices(None, "input")
            default_rate = int(devinfo.get("default_samplerate") or 0)
            if default_rate and default_rate != self.samplerate:
                try_rates.append(default_rate)
        except Exception:
            pass

        chosen_rate: Optional[int] = None
        chosen_blocksize: Optional[int] = None

        # Convert the target blocksize (~20ms) into input-rate frames.
        for rate in try_rates:
            rate = int(rate)
            bs = int(round(rate * (self.blocksize / float(self.samplerate))))
            bs = max(128, bs)
            try:
                test = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="int16",
                    blocksize=bs,
                    device=self.device,
                )
                test.close()
                chosen_rate = rate
                chosen_blocksize = bs
                break
            except Exception:
                continue

        if chosen_rate is None or chosen_blocksize is None:
            self._running = False
            raise RuntimeError("Nepodařilo se otevřít mikrofonní stream (žádná podporovaná vzorkovací frekvence).")

        self.input_samplerate = int(chosen_rate)
        self.using_resampler = (self.input_samplerate != self.samplerate)
        if self.using_resampler:
            # Keep ~30ms overlap for smoother resampling across chunk boundaries.
            self._rs_overlap = int(round(self.input_samplerate * 0.03))
            self._rs_overlap = max(256, min(self._rs_overlap, 4096))
            self._rs_prev = np.zeros((0,), dtype=np.int16)

        def callback(indata, frames, time_info, status) -> None:
            if not self._running:
                return
            captured_at_mono_ns = time.monotonic_ns()
            self._last_capture_mono_ns = captured_at_mono_ns
            self._captured_samples += int(frames)
            try:
                # indata dtype=int16, shape=(frames, 1)
                if not self.using_resampler:
                    self._queue.put_nowait(CapturedAudioChunk(pcm_bytes=indata.tobytes(), captured_at_mono_ns=captured_at_mono_ns))
                    return

                src = np.asarray(indata).reshape(-1).astype(np.int16, copy=False)

                # Simple overlap-add style: prepend a small tail from the
                # previous chunk to reduce boundary artifacts.
                if self._rs_prev.size > 0:
                    combined = np.concatenate([self._rs_prev, src])
                else:
                    combined = src

                resampled = _resample_pcm16_mono(combined, self.input_samplerate, self.samplerate)

                # Drop the portion that corresponds to the prepended overlap to
                # avoid duplicating audio.
                if self._rs_prev.size > 0:
                    drop = int(round(self._rs_prev.size * (self.samplerate / float(self.input_samplerate))))
                    if drop > 0 and drop < resampled.size:
                        resampled = resampled[drop:]

                # Update overlap buffer (tail of combined)
                if combined.size > self._rs_overlap:
                    self._rs_prev = combined[-self._rs_overlap :].copy()
                else:
                    self._rs_prev = combined.copy()

                self._queue.put_nowait(CapturedAudioChunk(pcm_bytes=resampled.tobytes(), captured_at_mono_ns=captured_at_mono_ns))
            except Exception:
                # drop frames on backpressure
                return

        self._stream = sd.InputStream(
            samplerate=self.input_samplerate,
            channels=1,
            dtype="int16",
            blocksize=int(chosen_blocksize),
            device=self.device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
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
        self._last_capture_mono_ns = 0
        self._captured_samples = 0
        # best-effort clear
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass

class _DuplexPlayerView:
    """Kompatibilni pohled na render cast duplex session."""

    def __init__(self, session: "DuplexAudioSession") -> None:
        self._session = session

    @property
    def samplerate(self) -> int:
        return int(self._session.stream_samplerate)

    @property
    def target_samplerate(self) -> int:
        return int(self._session.samplerate)

    @property
    def device(self) -> Optional[int]:
        return self._session.output_device

    @property
    def blocksize(self) -> int:
        return int(self._session.blocksize)

    @property
    def buffered_bytes(self) -> int:
        return int(self._session.buffered_bytes)

    @property
    def _echo_reference_chunks(self) -> "deque[tuple[int, bytes]]":
        return self._session._echo_reference_chunks

    @property
    def _echo_reference_enqueued_samples(self) -> int:
        return int(self._session._echo_reference_enqueued_samples)

    @_echo_reference_enqueued_samples.setter
    def _echo_reference_enqueued_samples(self, value: int) -> None:
        self._session._echo_reference_enqueued_samples = int(value)

    @property
    def _echo_reference_played_samples(self) -> int:
        return int(self._session._echo_reference_played_samples)

    @_echo_reference_played_samples.setter
    def _echo_reference_played_samples(self, value: int) -> None:
        self._session._echo_reference_played_samples = int(value)

    @property
    def _last_callback_mono_ns(self) -> int:
        return int(self._session._last_callback_mono_ns)

    @_last_callback_mono_ns.setter
    def _last_callback_mono_ns(self, value: int) -> None:
        self._session._last_callback_mono_ns = int(value)

    @property
    def _echo_reference_played_end_mono_ns(self) -> int:
        return int(self._session._echo_reference_played_end_mono_ns)

    @_echo_reference_played_end_mono_ns.setter
    def _echo_reference_played_end_mono_ns(self, value: int) -> None:
        self._session._echo_reference_played_end_mono_ns = int(value)

    def stop(self) -> None:
        self._session.stop()

    def enqueue_pcm16(self, pcm_bytes: bytes) -> None:
        self._session.enqueue_pcm16(pcm_bytes)

    def get_level(self) -> float:
        return self._session.get_level()

    def get_lipsync_snapshot(self) -> dict[str, object]:
        return self._session.get_lipsync_snapshot()

    def get_echo_reference(self, max_samples: int = 4096) -> np.ndarray:
        return self._session.get_echo_reference(max_samples=max_samples)

    def get_echo_reference_for_capture(self, *, max_samples: int = 4096, captured_at_mono_ns: Optional[int]) -> np.ndarray:
        return self._session.get_echo_reference_for_capture(max_samples=max_samples, captured_at_mono_ns=captured_at_mono_ns)

    def get_echo_reference_stats(self) -> dict[str, int]:
        return self._session.get_echo_reference_stats()


class _DuplexMicView:
    """Kompatibilni pohled na capture cast duplex session."""

    def __init__(self, session: "DuplexAudioSession") -> None:
        self._session = session

    @property
    def samplerate(self) -> int:
        return int(self._session.samplerate)

    @property
    def device(self) -> Optional[int]:
        return self._session.input_device

    @property
    def blocksize(self) -> int:
        return int(self._session.blocksize)

    @property
    def using_resampler(self) -> bool:
        return bool(self._session.using_input_resampler)

    @property
    def input_samplerate(self) -> int:
        return int(self._session.input_samplerate)

    @property
    def queue(self) -> "queue.Queue[CapturedAudioChunk]":
        return self._session.queue

    @property
    def pending_chunk_count(self) -> int:
        return int(self._session.pending_chunk_count)

    @property
    def _last_capture_mono_ns(self) -> int:
        return int(self._session._last_capture_mono_ns)

    @_last_capture_mono_ns.setter
    def _last_capture_mono_ns(self, value: int) -> None:
        self._session._last_capture_mono_ns = int(value)

    @property
    def _captured_samples(self) -> int:
        return int(self._session._captured_samples)

    @_captured_samples.setter
    def _captured_samples(self, value: int) -> None:
        self._session._captured_samples = int(value)

    def start(self) -> None:
        self._session.start_mic()

    def stop(self) -> None:
        self._session.stop_mic()


class DuplexAudioSession:
    """Session-owned wrapper pro společné vlastnictví render a capture I/O."""

    def __init__(
        self,
        *,
        samplerate: int = 24000,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        blocksize: int = 480,
    ) -> None:
        self.started_at_mono_ns = time.monotonic_ns()
        self.samplerate = int(samplerate)
        self.input_device = input_device
        self.output_device = output_device
        self.blocksize = int(blocksize)
        self.stream_samplerate = int(samplerate)
        self.stream_blocksize = int(blocksize)
        self._lock = threading.Lock()
        self._stream: Optional[sd.Stream] = None
        self._running = False
        self._closed = False
        self._playback_buffer = bytearray()
        self._max_playback_buffer_bytes = int(self.samplerate * 2 * 2.0)
        self._capture_queue: "queue.Queue[CapturedAudioChunk]" = queue.Queue(maxsize=200)
        self._level: float = 0.0
        self._lip_sync = LipSyncEngine()
        self._echo_reference_chunks: "deque[tuple[int, bytes]]" = deque()
        self._echo_reference_enqueued_samples = 0
        self._echo_reference_played_samples = 0
        self._echo_reference_max_samples = int(self.samplerate * 2.0)
        self._last_callback_mono_ns = 0
        self._last_render_mono_ns = 0
        self._last_capture_mono_ns = 0
        self._echo_reference_played_end_mono_ns = 0
        self._captured_samples = 0
        self.input_samplerate = int(samplerate)
        self.using_input_resampler = False
        self._input_overlap = 0
        self._input_prev = np.zeros((0,), dtype=np.int16)
        self.player = _DuplexPlayerView(self)
        self.mic = _DuplexMicView(self)

    def _candidate_stream_rates(self) -> list[int]:
        candidates = [int(self.samplerate)]
        try:
            input_info = sd.query_devices(self.input_device, "input") if self.input_device is not None else sd.query_devices(None, "input")
            input_default = int(input_info.get("default_samplerate") or 0)
            if input_default > 0:
                candidates.append(input_default)
        except Exception:
            pass
        try:
            output_info = sd.query_devices(self.output_device, "output") if self.output_device is not None else sd.query_devices(None, "output")
            output_default = int(output_info.get("default_samplerate") or 0)
            if output_default > 0:
                candidates.append(output_default)
        except Exception:
            pass
        unique: list[int] = []
        for rate in candidates:
            if rate > 0 and rate not in unique:
                unique.append(rate)
        return unique or [int(self.samplerate)]

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return

        last_error: Optional[Exception] = None

        def callback(indata, outdata, frames, time_info, status) -> None:
            del time_info, status
            now_ns = time.monotonic_ns()
            need_bytes = int(frames) * 2
            with self._lock:
                self._last_callback_mono_ns = now_ns
                if self._closed:
                    outdata[:] = 0
                    return
                if len(self._playback_buffer) >= need_bytes:
                    played_bytes = bytes(self._playback_buffer[:need_bytes])
                    del self._playback_buffer[:need_bytes]
                else:
                    played_bytes = bytes(self._playback_buffer)
                    self._playback_buffer.clear()

            played = np.frombuffer(played_bytes, dtype=np.int16)
            played_frames = int(played.size)
            if played.size < frames:
                padded = np.zeros((int(frames),), dtype=np.int16)
                if played.size:
                    padded[: played.size] = played
                played = padded
            outdata[:, 0] = played

            try:
                pcm = played.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
                peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                self._level = float(max(0.0, min(1.0, max(rms * 1.8, peak))))
            except Exception:
                self._level = 0.0

            try:
                played_target_samples = max(0, int(round(played_frames * (self.samplerate / float(self.stream_samplerate)))))
                with self._lock:
                    self._echo_reference_played_samples += played_target_samples
                    playback_horizon_ns = int(round(played_target_samples * (1_000_000_000.0 / float(self.samplerate))))
                    self._echo_reference_played_end_mono_ns = now_ns + playback_horizon_ns
                    self._last_render_mono_ns = now_ns
                    while self._echo_reference_chunks:
                        oldest_end = self._echo_reference_chunks[0][0]
                        if self._echo_reference_played_samples - oldest_end <= self._echo_reference_max_samples:
                            break
                        self._echo_reference_chunks.popleft()
            except Exception:
                pass

            try:
                self._lip_sync.consume_playback_pcm16(played.tobytes(), samplerate=self.stream_samplerate)
            except Exception:
                pass

            try:
                incoming = np.asarray(indata).reshape(-1).astype(np.int16, copy=False)
                if self.using_input_resampler:
                    combined = np.concatenate([self._input_prev, incoming]) if self._input_prev.size else incoming
                    resampled = _resample_pcm16_mono(combined, self.input_samplerate, self.samplerate)
                    if self._input_prev.size > 0:
                        drop = int(round(self._input_prev.size * (self.samplerate / float(self.input_samplerate))))
                        if 0 < drop < resampled.size:
                            resampled = resampled[drop:]
                    if combined.size > self._input_overlap:
                        self._input_prev = combined[-self._input_overlap :].copy()
                    else:
                        self._input_prev = combined.copy()
                    captured = resampled
                else:
                    captured = incoming.copy()
                self._last_capture_mono_ns = now_ns
                self._captured_samples += int(captured.size)
                if captured.size > 0:
                    self._capture_queue.put_nowait(
                        CapturedAudioChunk(
                            pcm_bytes=captured.astype(np.int16, copy=False).tobytes(),
                            captured_at_mono_ns=now_ns,
                        )
                    )
            except Exception:
                return

        for rate in self._candidate_stream_rates():
            try:
                block = int(round(rate * (self.blocksize / float(self.samplerate))))
                block = max(128, block)
                stream = sd.Stream(
                    samplerate=int(rate),
                    channels=1,
                    dtype="int16",
                    blocksize=block,
                    device=(self.input_device, self.output_device),
                    callback=callback,
                )
                stream.start()
                self.stream_samplerate = int(rate)
                self.stream_blocksize = int(block)
                self.input_samplerate = int(rate)
                self.using_input_resampler = bool(self.input_samplerate != self.samplerate)
                if self.using_input_resampler:
                    self._input_overlap = max(256, min(int(round(self.input_samplerate * 0.03)), 4096))
                    self._input_prev = np.zeros((0,), dtype=np.int16)
                else:
                    self._input_overlap = 0
                    self._input_prev = np.zeros((0,), dtype=np.int16)
                self._stream = stream
                self._running = True
                self._closed = False
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                self._stream = None
        if last_error is not None:
            raise last_error

    @property
    def buffered_bytes(self) -> int:
        try:
            with self._lock:
                return int(len(self._playback_buffer))
        except Exception:
            return 0

    @property
    def pending_chunk_count(self) -> int:
        try:
            return int(self._capture_queue.qsize())
        except Exception:
            return 0

    @property
    def queue(self) -> "queue.Queue[CapturedAudioChunk]":
        return self._capture_queue

    def start_mic(self) -> None:
        self._ensure_stream()

    def stop_mic(self) -> None:
        self.stop()

    def stop(self) -> None:
        with self._lock:
            self._closed = True
            self._playback_buffer.clear()
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
        self._running = False
        self._closed = False
        self._level = 0.0
        self._lip_sync.reset()
        self._echo_reference_chunks.clear()
        self._echo_reference_enqueued_samples = 0
        self._echo_reference_played_samples = 0
        self._last_callback_mono_ns = 0
        self._last_render_mono_ns = 0
        self._last_capture_mono_ns = 0
        self._echo_reference_played_end_mono_ns = 0
        self._captured_samples = 0
        self._input_prev = np.zeros((0,), dtype=np.int16)
        try:
            while True:
                self._capture_queue.get_nowait()
        except Exception:
            pass

    def enqueue_pcm16(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self._ensure_stream()
        stream_pcm = pcm_bytes
        if self.stream_samplerate != self.samplerate:
            try:
                source = np.frombuffer(pcm_bytes, dtype=np.int16)
                stream_pcm = _resample_pcm16_mono(source, self.samplerate, self.stream_samplerate).tobytes()
            except Exception:
                pass
        with self._lock:
            combined = bytes(self._playback_buffer) + bytes(stream_pcm)
            if len(combined) > self._max_playback_buffer_bytes:
                combined = combined[-self._max_playback_buffer_bytes :]
            self._playback_buffer = bytearray(combined)
            retained_pcm16 = bytes(self._playback_buffer if self.stream_samplerate == self.samplerate else pcm_bytes[-min(len(pcm_bytes), self._max_playback_buffer_bytes) :])
            retained_samples = len(retained_pcm16) // 2
            self._echo_reference_enqueued_samples = self._echo_reference_played_samples + retained_samples
            self._echo_reference_chunks.clear()
            if retained_samples > 0:
                self._echo_reference_chunks.append((self._echo_reference_enqueued_samples, retained_pcm16))

    def build_render_frame(
        self,
        *,
        frame_index: int,
        mono_ns: Optional[int],
        pcm16: bytes,
        tts_active: bool,
        prompted_by_assistant_turn: Optional[str] = None,
    ) -> RenderFrame:
        return RenderFrame(
            frame_index=int(frame_index),
            mono_ns=int(mono_ns if mono_ns is not None else time.monotonic_ns()),
            pcm16=bytes(pcm16),
            tts_active=bool(tts_active),
            prompted_by_assistant_turn=prompted_by_assistant_turn,
        )

    def get_level(self) -> float:
        try:
            return float(self._level)
        except Exception:
            return 0.0

    def get_echo_reference(self, max_samples: int = 4096) -> np.ndarray:
        return self.get_echo_reference_for_capture(max_samples=max_samples, captured_at_mono_ns=None)

    def get_echo_reference_for_capture(self, *, max_samples: int = 4096, captured_at_mono_ns: Optional[int]) -> np.ndarray:
        try:
            need_samples = max(1, int(max_samples))
            with self._lock:
                if not self._echo_reference_chunks:
                    return np.zeros((0,), dtype=np.int16)
                played_end = self._echo_reference_played_samples
                if captured_at_mono_ns is not None and self._echo_reference_played_end_mono_ns > 0:
                    future_ns = max(0, int(self._echo_reference_played_end_mono_ns) - int(captured_at_mono_ns))
                    if future_ns > 0:
                        future_samples = int(round(future_ns * (self.samplerate / 1_000_000_000.0)))
                        played_end = max(0, played_end - future_samples)
                if played_end <= 0:
                    return np.zeros((0,), dtype=np.int16)
                start_sample = max(0, played_end - need_samples)
                chunks: list[bytes] = []
                cursor = played_end
                for end_sample, payload in reversed(self._echo_reference_chunks):
                    payload_samples = len(payload) // 2
                    chunk_start = end_sample - payload_samples
                    overlap_start = max(chunk_start, start_sample)
                    overlap_end = min(end_sample, played_end)
                    if overlap_end <= overlap_start:
                        continue
                    offset_start = overlap_start - chunk_start
                    offset_end = overlap_end - chunk_start
                    payload_view = memoryview(payload)[offset_start * 2 : offset_end * 2]
                    chunks.append(bytes(payload_view))
                    cursor = overlap_start
                    if cursor <= start_sample:
                        break
                tail = b"".join(reversed(chunks))
            if not tail:
                return np.zeros((0,), dtype=np.int16)
            array = np.frombuffer(tail, dtype=np.int16)
            if array.size > need_samples:
                array = array[-need_samples:]
            return array.copy()
        except Exception:
            return np.zeros((0,), dtype=np.int16)

    def get_echo_reference_stats(self) -> dict[str, int]:
        try:
            with self._lock:
                available_samples = sum(len(payload) // 2 for _, payload in self._echo_reference_chunks)
                return {
                    "available_samples": int(max(0, min(available_samples, self._echo_reference_played_samples))),
                    "total_samples": int(self._echo_reference_enqueued_samples),
                    "played_samples": int(self._echo_reference_played_samples),
                    "callback_age_ms": int((time.monotonic_ns() - self._last_callback_mono_ns) / 1_000_000) if self._last_callback_mono_ns else -1,
                    "played_end_mono_ns": int(self._echo_reference_played_end_mono_ns),
                }
        except Exception:
            return {"available_samples": 0, "total_samples": 0, "played_samples": 0, "callback_age_ms": -1, "played_end_mono_ns": 0}

    def get_lipsync_snapshot(self) -> dict[str, object]:
        try:
            snap = self._lip_sync.snapshot()
            return {
                "pose": snap.pose,
                "openness": snap.openness,
                "energy": snap.energy,
                "weights": dict(snap.weights),
            }
        except Exception:
            return {
                "pose": "closed",
                "openness": 0.0,
                "energy": 0.0,
                "weights": {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0},
            }

    def get_runtime_state(self) -> dict[str, object]:
        reference_stats = self.get_echo_reference_stats()
        now_ns = time.monotonic_ns()
        last_render_ns = int(reference_stats.get("played_end_mono_ns", 0) or 0)
        last_capture_ns = int(self._last_capture_mono_ns or 0)
        return {
            "started_at_mono_ns": int(self.started_at_mono_ns),
            "clock_mode": "unified_duplex",
            "stream_samplerate": int(self.stream_samplerate),
            "buffered_bytes": int(self.buffered_bytes),
            "pending_chunk_count": int(self.pending_chunk_count),
            "render_age_ms": int((now_ns - last_render_ns) / 1_000_000) if last_render_ns > 0 else -1,
            "capture_age_ms": int((now_ns - last_capture_ns) / 1_000_000) if last_capture_ns > 0 else -1,
            "rendered_samples": int(reference_stats.get("played_samples", 0) or 0),
            "captured_samples": int(self._captured_samples),
            "reference_available_samples": int(reference_stats.get("available_samples", 0) or 0),
            "reference_callback_age_ms": int(reference_stats.get("callback_age_ms", -1) or -1),
        }


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
