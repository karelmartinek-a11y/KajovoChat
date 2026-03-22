from __future__ import annotations

import io
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# Optional but strongly recommended for good resampling quality.
# SciPy is added to requirements in this branch.
from scipy import signal
import math

from .lip_sync_engine import LipSyncEngine


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
class AudioCalibrationResult:
    input_device: Optional[int]
    output_device: Optional[int]
    ambient_rms: float
    playback_rms: float
    bleed_ratio: float
    similarity: float
    recommended_profile: dict[str, float]
    notes: list[str]


def list_audio_devices() -> dict:
    """List audio devices for UI selection.

    This is best-effort: if enumeration fails, returns empty lists.

    Returns:
        {"inputs": [{"index": int, "name": str, "max_channels": int}],
         "outputs": [{"index": int, "name": str, "max_channels": int}]}
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return {"inputs": [], "outputs": []}

    inputs = []
    outputs = []
    for idx, d in enumerate(devices or []):
        name = str(d.get("name", f"Device {idx}"))
        mi = int(d.get("max_input_channels", 0) or 0)
        mo = int(d.get("max_output_channels", 0) or 0)
        if mi > 0:
            inputs.append({"index": idx, "name": name, "max_channels": mi})
        if mo > 0:
            outputs.append({"index": idx, "name": name, "max_channels": mo})

    return {"inputs": inputs, "outputs": outputs}


def _device_valid(index: Optional[int], kind: str) -> bool:
    if index is None:
        return True
    try:
        info = sd.query_devices(index, kind)
    except Exception:
        return False
    if kind == "input":
        return int(info.get("max_input_channels", 0) or 0) > 0
    if kind == "output":
        return int(info.get("max_output_channels", 0) or 0) > 0
    return False


def _score_name(name: str, kind: str) -> int:
    """Heuristic scoring to prefer built-in laptop mic/speakers.

    This is intentionally conservative and cross-platform-ish.
    """
    n = (name or "").lower()
    score = 0
    # Built-in / internal tends to be what users want for "NB mic/speakers".
    if any(k in n for k in ["built-in", "builtin", "internal", "integro", "notebook", "laptop"]):
        score += 40
    if kind == "input":
        if any(k in n for k in ["microphone", "mic", "array", "input"]):
            score += 25
        if any(k in n for k in ["usb", "webcam", "camera"]):
            score -= 10  # many users don't want these by default
    else:
        if any(k in n for k in ["speaker", "speakers", "output", "headphone", "headphones"]):
            score += 25
        if any(k in n for k in ["bluetooth", "bt"]):
            score -= 5

    # Common Windows drivers for internal audio
    if any(k in n for k in ["realtek", "conexant", "intel"]):
        score += 8
    # Avoid obvious "monitor"/"virtual"/"loopback" devices.
    if any(k in n for k in ["loopback", "virtual", "monitor", "cable", "vb-audio", "blackhole"]):
        score -= 30
    return score


def pick_audio_device(kind: str, preferred: Optional[int]) -> tuple[Optional[int], str]:
    """Pick a usable device index.

    Order:
      1) preferred (if valid)
      2) system default (if valid)
      3) best-effort heuristic match (built-in mic/speakers)

    Returns: (device_index_or_None, note)
    """
    kind = "input" if kind == "input" else "output"

    if preferred is not None and _device_valid(preferred, kind):
        return int(preferred), "selected:settings"

    # sounddevice default is either a scalar or a (in,out) pair.
    try:
        default = sd.default.device
        if isinstance(default, (list, tuple)) and len(default) >= 2:
            default_idx = default[0] if kind == "input" else default[1]
        else:
            default_idx = default
        if default_idx is not None and int(default_idx) >= 0 and _device_valid(int(default_idx), kind):
            return int(default_idx), "selected:system_default"
    except Exception:
        pass

    try:
        devices = sd.query_devices() or []
    except Exception:
        return None, "selected:none"

    best_idx: Optional[int] = None
    best_score = -10**9
    for idx, d in enumerate(devices):
        name = str(d.get("name", ""))
        mi = int(d.get("max_input_channels", 0) or 0)
        mo = int(d.get("max_output_channels", 0) or 0)
        if kind == "input" and mi <= 0:
            continue
        if kind == "output" and mo <= 0:
            continue
        s = _score_name(name, kind)
        # Slightly prefer devices that look like "default" in name.
        if "default" in name.lower():
            s += 5
        if s > best_score:
            best_score = s
            best_idx = idx

    if best_idx is not None and _device_valid(best_idx, kind):
        return int(best_idx), "selected:heuristic"
    return None, "selected:none"


def format_device_help() -> str:
    """User-facing device dump for error messages."""
    devs = list_audio_devices()
    lines = ["Dostupná audio zařízení (index: název):"]
    ins = devs.get("inputs", [])
    outs = devs.get("outputs", [])
    if ins:
        lines.append("Vstupy:")
        for d in ins[:30]:
            lines.append(f"  {d['index']}: {d['name']}")
    else:
        lines.append("Vstupy: (nenalezeno)")
    if outs:
        lines.append("Výstupy:")
        for d in outs[:30]:
            lines.append(f"  {d['index']}: {d['name']}")
    else:
        lines.append("Výstupy: (nenalezeno)")
    lines.append("Tip: Aplikace používá systémová výchozí zařízení, případně interní heuristiku pro vestavěný mikrofon a reproduktory.")
    return "\n".join(lines)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _normalized_similarity(reference: np.ndarray, recorded: np.ndarray, max_shift_samples: int) -> float:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    rec = np.asarray(recorded, dtype=np.float32).reshape(-1)
    if ref.size < 64 or rec.size < 64:
        return 0.0

    ref = ref - float(np.mean(ref))
    rec = rec - float(np.mean(rec))
    ref_norm = float(np.linalg.norm(ref) + 1e-6)
    rec_norm = float(np.linalg.norm(rec) + 1e-6)
    if ref_norm <= 1e-6 or rec_norm <= 1e-6:
        return 0.0

    best = 0.0
    max_shift_samples = max(0, int(max_shift_samples))
    for shift in range(0, max_shift_samples + 1, max(1, max_shift_samples // 16 or 1)):
        usable = min(ref.size, rec.size - shift)
        if usable < 64:
            continue
        ref_seg = ref[:usable]
        rec_seg = rec[shift : shift + usable]
        corr = abs(float(np.dot(ref_seg, rec_seg)) / (float(np.linalg.norm(ref_seg) + 1e-6) * float(np.linalg.norm(rec_seg) + 1e-6)))
        if corr > best:
            best = corr
    return float(max(0.0, min(1.0, best)))


def suppress_echo_from_pcm16(
    mic_pcm: bytes,
    reference: np.ndarray,
    *,
    max_shift_samples: int = 960,
) -> tuple[bytes, float]:
    """Zkusí odečíst referenční playback z mikrofonního chunku."""
    if not mic_pcm:
        return b"", 0.0

    mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
    ref = np.asarray(reference, dtype=np.int16).astype(np.float32).reshape(-1)
    if mic.size < 120 or ref.size < mic.size:
        return mic_pcm, 0.0

    best_similarity = 0.0
    best_shift = 0
    best_segment: Optional[np.ndarray] = None
    max_shift_samples = max(0, int(max_shift_samples))
    mic_centered = mic - float(np.mean(mic))
    mic_norm = float(np.linalg.norm(mic_centered) + 1e-6)
    if mic_norm <= 1e-6:
        return mic_pcm, 0.0

    for shift in range(0, max_shift_samples + 1, 120):
        segment = ref[ref.size - mic.size - shift : ref.size - shift if shift > 0 else ref.size]
        if segment.size != mic.size:
            continue
        segment_centered = segment - float(np.mean(segment))
        seg_norm = float(np.linalg.norm(segment_centered) + 1e-6)
        if seg_norm <= 1e-6:
            continue
        similarity = abs(float(np.dot(mic_centered, segment_centered)) / (mic_norm * seg_norm))
        if similarity > best_similarity:
            best_similarity = similarity
            best_shift = shift
            best_segment = segment

    if best_segment is None or best_similarity < 0.08:
        return mic_pcm, float(best_similarity)

    ref_centered = best_segment - float(np.mean(best_segment))
    denom = float(np.dot(ref_centered, ref_centered) + 1e-6)
    gain = float(np.dot(mic_centered, ref_centered) / denom)
    gain = max(0.0, min(1.15, gain))
    cleaned = mic - (ref_centered * gain)
    if best_similarity >= 0.28:
        cleaned -= ref_centered * min(0.18, best_similarity * 0.22)
    cleaned = np.clip(cleaned, -32768.0, 32767.0).astype(np.int16)
    return cleaned.tobytes(), float(best_similarity)


def _playrec_with_fallbacks(
    playback_buffer: np.ndarray,
    *,
    samplerate: int,
    input_device: Optional[int],
    output_device: Optional[int],
) -> np.ndarray:
    attempts = [
        {"device": (input_device, output_device)},
        {"device": (None, output_device)},
        {"device": None},
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return np.asarray(
                sd.playrec(
                    playback_buffer.reshape(-1, 1),
                    samplerate=samplerate,
                    channels=1,
                    dtype="float32",
                    device=attempt["device"],
                    blocking=True,
                ),
                dtype=np.float32,
            ).reshape(-1)
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Audio kalibraci se nepodařilo spustit.")


def calibrate_audio_devices(
    *,
    input_device: Optional[int],
    output_device: Optional[int],
    samplerate: int = 24000,
    playback_seconds: float = 1.8,
    playback_gain: float = 0.28,
) -> AudioCalibrationResult:
    """Automaticky změří bleed mezi reproduktorem a mikrofonem a navrhne guard profil."""
    samplerate = int(samplerate)
    ambient_frames = max(1, int(samplerate * 0.45))
    playback_frames = max(1, int(samplerate * max(1.2, playback_seconds)))
    total_frames = ambient_frames + playback_frames

    probe_time = np.linspace(0.0, playback_frames / samplerate, playback_frames, endpoint=False, dtype=np.float32)
    chirp = signal.chirp(probe_time, f0=180.0, f1=4200.0, t1=max(0.1, playback_seconds), method="logarithmic")
    envelope = np.hanning(playback_frames).astype(np.float32)
    pulse = np.sin(2.0 * np.pi * 42.0 * probe_time).astype(np.float32)
    playback_signal = ((chirp * 0.72) + (pulse * 0.28)) * envelope * float(playback_gain)
    playback_buffer = np.concatenate(
        [
            np.zeros((ambient_frames,), dtype=np.float32),
            playback_signal.astype(np.float32),
        ]
    )

    recorded = _playrec_with_fallbacks(
        playback_buffer,
        samplerate=samplerate,
        input_device=input_device,
        output_device=output_device,
    )
    if recorded.size < total_frames:
        padded = np.zeros((total_frames,), dtype=np.float32)
        padded[: recorded.size] = recorded
        recorded = padded

    ambient = recorded[:ambient_frames]
    captured = recorded[ambient_frames : ambient_frames + playback_frames]

    ambient_rms = _rms(ambient)
    playback_rms = _rms(captured)
    bleed_ratio = float(playback_rms / max(ambient_rms, 1e-4))
    similarity = _normalized_similarity(playback_signal, captured, max_shift_samples=int(samplerate * 0.12))

    recommended_profile = {
        "server_vad_threshold": float(np.clip(0.72 + max(0.0, similarity - 0.45) * 0.18 + max(0.0, min(0.18, ambient_rms)) * 0.55, 0.68, 0.9)),
        "playback_activity_level": float(np.clip(max(0.028, playback_rms * 0.45), 0.028, 0.16)),
        "echo_similarity_drop": float(np.clip(0.78 + similarity * 0.14, 0.78, 0.96)),
        "echo_similarity_soft": float(np.clip(0.6 + similarity * 0.12, 0.58, 0.86)),
        "barge_in_min_input_level": float(np.clip(max(0.05, ambient_rms * 4.2, playback_rms * 0.3), 0.05, 0.22)),
        "barge_in_output_ratio": float(np.clip(1.2 + min(bleed_ratio, 8.0) * 0.06, 1.2, 1.8)),
    }
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"ambient_rms={ambient_rms:.4f}",
        f"playback_rms={playback_rms:.4f}",
        f"bleed_ratio={bleed_ratio:.2f}",
        f"similarity={similarity:.3f}",
    ]
    return AudioCalibrationResult(
        input_device=input_device,
        output_device=output_device,
        ambient_rms=ambient_rms,
        playback_rms=playback_rms,
        bleed_ratio=bleed_ratio,
        similarity=similarity,
        recommended_profile=recommended_profile,
        notes=notes,
    )


def calibrate_audio_devices_advanced(
    *,
    input_device: Optional[int],
    output_device: Optional[int],
    samplerate: int = 24000,
) -> AudioCalibrationResult:
    """Víceprůchodová kalibrace s různou délkou a hlasitostí testovacího signálu."""
    samplerates = [int(samplerate), 48000, 44100]
    passes = [
        {"playback_seconds": 1.4, "playback_gain": 0.18},
        {"playback_seconds": 1.8, "playback_gain": 0.24},
        {"playback_seconds": 2.2, "playback_gain": 0.30},
    ]
    results: list[AudioCalibrationResult] = []
    errors: list[str] = []
    for current_samplerate in samplerates:
        for current in passes:
            try:
                results.append(
                    calibrate_audio_devices(
                        input_device=input_device,
                        output_device=output_device,
                        samplerate=int(current_samplerate),
                        playback_seconds=float(current["playback_seconds"]),
                        playback_gain=float(current["playback_gain"]),
                    )
                )
            except Exception as exc:
                errors.append(f"{current_samplerate}Hz/{current['playback_gain']:.2f}: {exc}")

    if not results:
        raise RuntimeError("Pokročilá audio kalibrace selhala: " + " | ".join(errors[:4]))

    if len(results) > 4:
        # Drž jen reprezentativní podmnožinu, aby nebyl profil přehnaně rozkolísaný.
        results = results[:4]

    ambient_values = np.asarray([item.ambient_rms for item in results], dtype=np.float32)
    playback_values = np.asarray([item.playback_rms for item in results], dtype=np.float32)
    bleed_values = np.asarray([item.bleed_ratio for item in results], dtype=np.float32)
    similarity_values = np.asarray([item.similarity for item in results], dtype=np.float32)

    recommended_profile = {
        "server_vad_threshold": float(max(item.recommended_profile["server_vad_threshold"] for item in results)),
        "playback_activity_level": float(np.median([item.recommended_profile["playback_activity_level"] for item in results])),
        "echo_similarity_drop": float(max(item.recommended_profile["echo_similarity_drop"] for item in results)),
        "echo_similarity_soft": float(np.median([item.recommended_profile["echo_similarity_soft"] for item in results])),
        "barge_in_min_input_level": float(max(item.recommended_profile["barge_in_min_input_level"] for item in results)),
        "barge_in_output_ratio": float(max(item.recommended_profile["barge_in_output_ratio"] for item in results)),
    }
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"passes={len(results)}",
        f"ambient_med={float(np.median(ambient_values)):.4f}",
        f"playback_med={float(np.median(playback_values)):.4f}",
        f"bleed_peak={float(np.max(bleed_values)):.2f}",
        f"similarity_peak={float(np.max(similarity_values)):.3f}",
    ]
    return AudioCalibrationResult(
        input_device=input_device,
        output_device=output_device,
        ambient_rms=float(np.median(ambient_values)),
        playback_rms=float(np.median(playback_values)),
        bleed_ratio=float(np.max(bleed_values)),
        similarity=float(np.max(similarity_values)),
        recommended_profile=recommended_profile,
        notes=notes,
    )


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
        self._echo_reference = bytearray()
        self._echo_reference_max_bytes = int(self.target_samplerate * 2 * 0.5)

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
                self._lip_sync.consume_playback_pcm16(played.tobytes(), samplerate=self.samplerate)
                echo_ref = played
                if self.samplerate != self.target_samplerate:
                    echo_ref = _resample_pcm16_mono(played, self.samplerate, self.target_samplerate)
                with self._lock:
                    self._echo_reference.extend(echo_ref.tobytes())
                    if len(self._echo_reference) > self._echo_reference_max_bytes:
                        del self._echo_reference[: len(self._echo_reference) - self._echo_reference_max_bytes]
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
            self._echo_reference.clear()
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
        try:
            need_bytes = max(1, int(max_samples)) * 2
            with self._lock:
                tail = bytes(self._echo_reference[-need_bytes:])
            if not tail:
                return np.zeros((0,), dtype=np.int16)
            return np.frombuffer(tail, dtype=np.int16).copy()
        except Exception:
            return np.zeros((0,), dtype=np.int16)

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
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._running = False

    @property
    def queue(self) -> "queue.Queue[bytes]":
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
            try:
                # indata dtype=int16, shape=(frames, 1)
                if not self.using_resampler:
                    self._queue.put_nowait(indata.tobytes())
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

                self._queue.put_nowait(resampled.tobytes())
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
        # best-effort clear
        try:
            while True:
                self._queue.get_nowait()
        except Exception:
            pass


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
