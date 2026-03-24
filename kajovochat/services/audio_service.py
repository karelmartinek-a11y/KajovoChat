from __future__ import annotations

import hashlib
import io
import queue
import threading
import time
from collections import deque
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
from .voice_features import estimate_voice_likelihood_from_pcm16
from .windows_native_aec import WindowsNativeAECBackend, probe_windows_native_aec
from ..settings import DEFAULT_AUDIO_AEC_MODE, normalize_audio_aec_mode

try:
    from aec_audio_processing import AudioProcessor as WebRTCAudioProcessor
except Exception:
    WebRTCAudioProcessor = None


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
    latency_samples: int = 0
    preferred_frame_size: int = 480
    filter_length: int = 256
    device_fingerprint: str = "unknown"
    audio_mode: str = "notebook_builtin"


@dataclass
class CapturedAudioChunk:
    pcm_bytes: bytes
    captured_at_mono_ns: int


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


def _device_name(index: Optional[int], kind: str) -> str:
    try:
        if index is None:
            return "default"
        info = sd.query_devices(index, kind)
        return str(info.get("name", f"{kind}:{index}"))
    except Exception:
        return f"{kind}:{index if index is not None else 'default'}"


def _infer_audio_mode(input_name: str, output_name: str) -> str:
    combined = f"{input_name} {output_name}".lower()
    if any(token in combined for token in ("bluetooth", "airpods", "buds", "hands-free")):
        return "bluetooth_headset"
    if any(token in combined for token in ("headphone", "headphones", "headset", "earbuds", "usb audio")):
        return "wired_headset"
    if any(token in combined for token in ("speaker", "speakers", "monitor", "hdmi", "display audio", "dock")) and not any(
        token in combined for token in ("built-in", "builtin", "internal", "laptop", "notebook")
    ):
        return "external_speakers"
    return "notebook_builtin"


def _build_device_fingerprint(input_device: Optional[int], output_device: Optional[int], samplerate: int) -> str:
    payload = "|".join([
        _device_name(input_device, "input"),
        _device_name(output_device, "output"),
        str(int(samplerate)),
    ])
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def build_device_fingerprint(input_device: Optional[int], output_device: Optional[int], samplerate: int = 24000) -> str:
    """Vrati stabilni fingerprint aktualni dvojice vstup/vystup."""
    return _build_device_fingerprint(input_device, output_device, samplerate)


def _extract_reference_segment(reference: np.ndarray, mic_size: int, shift: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    end = ref.size - max(0, int(shift))
    start = end - int(mic_size)
    if start < 0 or end > ref.size or end <= start:
        return np.zeros((0,), dtype=np.float32)
    return ref[start:end].copy()


def _extract_reference_context(reference: np.ndarray, mic_size: int, shift: int, filter_length: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    taps = max(1, int(filter_length))
    need = int(mic_size) + taps - 1
    end = ref.size - max(0, int(shift))
    start = end - need
    if end <= 0:
        return np.zeros((need,), dtype=np.float32)
    if start >= 0 and end <= ref.size:
        return ref[start:end].copy()

    context = np.zeros((need,), dtype=np.float32)
    src_start = max(0, start)
    src_end = min(ref.size, end)
    if src_end <= src_start:
        return context
    dst_start = max(0, -start)
    context[dst_start : dst_start + (src_end - src_start)] = ref[src_start:src_end]
    return context


def _candidate_shifts(max_shift_samples: int, expected_shift: Optional[int]) -> list[int]:
    limit = max(0, int(max_shift_samples))
    if expected_shift is None:
        shifts = list(range(0, limit + 1, 24))
        if shifts[-1] != limit:
            shifts.append(limit)
        return shifts
    expected = max(0, min(limit, int(expected_shift)))
    window = max(120, min(limit, 240))
    shifts = list(range(max(0, expected - window), min(limit, expected + window) + 1, 8))
    if expected not in shifts:
        shifts.append(expected)
    return sorted(set(shifts))


def _find_best_alignment(
    mic: np.ndarray,
    reference: np.ndarray,
    *,
    max_shift_samples: int,
    expected_shift: Optional[int] = None,
) -> tuple[np.ndarray, float, int]:
    near = np.asarray(mic, dtype=np.float32).reshape(-1)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    limit = max(0, int(max_shift_samples))
    if near.size < 64 or ref.size < near.size:
        return np.zeros((0,), dtype=np.float32), 0.0, 0
    if limit <= 4096:
        return _find_best_alignment_exhaustive(
            near,
            ref,
            max_shift_samples=limit,
            expected_shift=expected_shift,
        )

    near_centered = near - float(np.mean(near))
    near_norm = float(np.linalg.norm(near_centered) + 1e-6)
    best_similarity = 0.0
    best_shift = 0
    best_segment = np.zeros((0,), dtype=np.float32)
    for shift in _candidate_shifts(max_shift_samples, expected_shift):
        segment = _extract_reference_segment(ref, near.size, shift)
        if segment.size != near.size:
            continue
        segment_centered = segment - float(np.mean(segment))
        seg_norm = float(np.linalg.norm(segment_centered) + 1e-6)
        if seg_norm <= 1e-6:
            continue
        similarity = abs(float(np.dot(near_centered, segment_centered)) / (near_norm * seg_norm))
        if similarity > best_similarity:
            best_similarity = similarity
            best_shift = int(shift)
            best_segment = segment

    if best_segment.size == 0:
        return np.zeros((0,), dtype=np.float32), 0.0, 0

    for shift in range(max(0, best_shift - 12), min(max(0, int(max_shift_samples)), best_shift + 12) + 1):
        segment = _extract_reference_segment(ref, near.size, shift)
        if segment.size != near.size:
            continue
        segment_centered = segment - float(np.mean(segment))
        seg_norm = float(np.linalg.norm(segment_centered) + 1e-6)
        if seg_norm <= 1e-6:
            continue
        similarity = abs(float(np.dot(near_centered, segment_centered)) / (near_norm * seg_norm))
        if similarity > best_similarity:
            best_similarity = similarity
            best_shift = int(shift)
            best_segment = segment
    if best_similarity < 0.72 and limit <= 8192:
        exhaustive_segment, exhaustive_similarity, exhaustive_shift = _find_best_alignment_exhaustive(
            near,
            ref,
            max_shift_samples=limit,
            expected_shift=expected_shift,
        )
        if exhaustive_similarity > best_similarity + 0.02:
            return exhaustive_segment, exhaustive_similarity, exhaustive_shift
    return best_segment, float(max(0.0, min(1.0, best_similarity))), int(best_shift)


def _find_best_alignment_exhaustive(
    mic: np.ndarray,
    reference: np.ndarray,
    *,
    max_shift_samples: int,
    expected_shift: Optional[int] = None,
) -> tuple[np.ndarray, float, int]:
    near = np.asarray(mic, dtype=np.float32).reshape(-1)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    limit = max(0, int(max_shift_samples))
    if near.size < 64 or ref.size < near.size:
        return np.zeros((0,), dtype=np.float32), 0.0, 0

    search_size = min(ref.size, near.size + limit)
    search = ref[-search_size:]
    if search.size < near.size:
        return np.zeros((0,), dtype=np.float32), 0.0, 0

    near_centered = near - float(np.mean(near))
    near_norm = float(np.linalg.norm(near_centered) + 1e-6)
    if near_norm <= 1e-6:
        return np.zeros((0,), dtype=np.float32), 0.0, 0

    windows = np.lib.stride_tricks.sliding_window_view(search, near.size)
    if windows.size == 0:
        return np.zeros((0,), dtype=np.float32), 0.0, 0
    centered = windows - np.mean(windows, axis=1, keepdims=True)
    dots = centered @ near_centered
    norms = np.linalg.norm(centered, axis=1) * near_norm + 1e-6
    waveform_similarity = np.abs(dots) / norms

    # Pomaleji se měnící obálka pomáhá v praxi i tam, kde waveform korelace
    # skáče kvůli barvě zvuku, kompresi nebo odražené cestě.
    envelope_win = max(4, min(48, near.size // 32))
    kernel = np.ones((envelope_win,), dtype=np.float32) / float(envelope_win)
    near_envelope = np.convolve(np.abs(near_centered), kernel, mode="same")
    near_envelope -= float(np.mean(near_envelope))
    near_envelope_norm = float(np.linalg.norm(near_envelope) + 1e-6)
    window_envelopes = np.apply_along_axis(lambda row: np.convolve(np.abs(row), kernel, mode="same"), 1, centered)
    window_envelopes = window_envelopes - np.mean(window_envelopes, axis=1, keepdims=True)
    envelope_norms = np.linalg.norm(window_envelopes, axis=1) * near_envelope_norm + 1e-6
    envelope_similarity = np.abs(window_envelopes @ near_envelope) / envelope_norms

    near_slope = np.diff(near_centered, prepend=near_centered[:1])
    near_slope -= float(np.mean(near_slope))
    near_slope_norm = float(np.linalg.norm(near_slope) + 1e-6)
    window_slopes = np.diff(centered, axis=1, prepend=centered[:, :1])
    window_slopes = window_slopes - np.mean(window_slopes, axis=1, keepdims=True)
    slope_norms = np.linalg.norm(window_slopes, axis=1) * near_slope_norm + 1e-6
    slope_similarity = np.abs(window_slopes @ near_slope) / slope_norms

    raw_similarity = (waveform_similarity * 0.56) + (envelope_similarity * 0.24) + (slope_similarity * 0.20)
    shifts = np.arange(windows.shape[0] - 1, -1, -1, dtype=np.int32)
    score = raw_similarity.copy()
    if expected_shift is not None and limit > 0:
        expected = int(max(0, min(limit, int(expected_shift))))
        prior_width = float(max(64, min(limit, 256)))
        distance = np.abs(shifts.astype(np.float32) - float(expected))
        score += np.clip(1.0 - (distance / prior_width), 0.0, 1.0) * 0.08

    best_index = int(np.argmax(score))
    best_similarity = float(
        max(
            raw_similarity[best_index],
            waveform_similarity[best_index],
            envelope_similarity[best_index] * 0.96,
            slope_similarity[best_index] * 0.96,
        )
    )
    best_shift = int(windows.shape[0] - 1 - best_index)
    return windows[best_index].copy(), float(max(0.0, min(1.0, best_similarity))), best_shift


class _WebRTCAECBackend:
    """Volitelný backend nad webrtc audio processing knihovnou."""

    def __init__(self, *, input_samplerate: int) -> None:
        if WebRTCAudioProcessor is None:
            raise RuntimeError("WebRTC AEC backend není dostupný.")
        self.input_samplerate = int(input_samplerate)
        self.backend_samplerate = 16000
        self.frame_samples = 160  # 10 ms @ 16 kHz
        self._processor = WebRTCAudioProcessor(
            enable_aec=True,
            enable_ns=False,
            enable_agc=False,
            enable_vad=False,
        )
        self._processor.set_stream_format(self.backend_samplerate, 1, self.backend_samplerate, 1)
        self._processor.set_reverse_stream_format(self.backend_samplerate, 1)

    def process(self, *, mic_pcm: bytes, reference_pcm: np.ndarray, delay_ms: int) -> bytes:
        mic = np.frombuffer(mic_pcm, dtype=np.int16)
        ref = np.asarray(reference_pcm, dtype=np.int16).reshape(-1)
        if mic.size == 0 or ref.size == 0:
            return mic_pcm

        mic_16 = _resample_pcm16_mono(mic, self.input_samplerate, self.backend_samplerate)
        ref_16 = _resample_pcm16_mono(ref, self.input_samplerate, self.backend_samplerate)
        frame = int(self.frame_samples)
        if mic_16.size == 0 or ref_16.size == 0:
            return mic_pcm

        frame_count = max(1, int(math.ceil(mic_16.size / float(frame))))
        mic_pad = frame_count * frame - mic_16.size
        ref_pad = frame_count * frame - ref_16.size
        if mic_pad > 0:
            mic_16 = np.pad(mic_16, (0, mic_pad))
        if ref_pad > 0:
            ref_16 = np.pad(ref_16, (0, ref_pad))

        self._processor.set_stream_delay(max(0, int(delay_ms)))
        cleaned_frames: list[bytes] = []
        for index in range(frame_count):
            start = index * frame
            end = start + frame
            reverse_frame = ref_16[start:end].astype(np.int16, copy=False).tobytes()
            stream_frame = mic_16[start:end].astype(np.int16, copy=False).tobytes()
            self._processor.process_reverse_stream(reverse_frame)
            cleaned_frames.append(self._processor.process_stream(stream_frame))

        cleaned_16 = np.frombuffer(b"".join(cleaned_frames), dtype=np.int16)
        if cleaned_16.size > mic_16.size:
            cleaned_16 = cleaned_16[: mic_16.size]
        cleaned = _resample_pcm16_mono(cleaned_16, self.backend_samplerate, self.input_samplerate)
        if cleaned.size > mic.size:
            cleaned = cleaned[: mic.size]
        elif cleaned.size < mic.size:
            cleaned = np.pad(cleaned, (0, mic.size - cleaned.size))
        return cleaned.astype(np.int16, copy=False).tobytes()


class AdaptiveEchoCanceller:
    """Lehký vícecestný adaptivní canceler nad playback referencí."""

    def __init__(
        self,
        *,
        samplerate: int = 24000,
        max_shift_samples: int = 960,
        branch_offsets: Optional[tuple[int, ...]] = None,
        filter_length: int = 448,
        ridge: float = 1e-3,
    ) -> None:
        self.samplerate = int(samplerate)
        self.max_shift_samples = int(max_shift_samples)
        self._filter_length = max(256, int(filter_length))
        self.ridge = float(ridge)
        self._mu = 0.45
        self._leakage = 0.9992
        self._weights = np.zeros((self._filter_length,), dtype=np.float32)
        self._last_shift = 0
        self._delay_lock_shift = 0
        self._delay_lock_votes = 0
        self._last_double_talk = False
        self._windows_native_probe = probe_windows_native_aec()
        self._windows_native_backend: Optional[WindowsNativeAECBackend] = None
        self._windows_native_backend_attempted = False
        self._external_backend: Optional[_WebRTCAECBackend] = None
        if WebRTCAudioProcessor is not None:
            try:
                self._external_backend = _WebRTCAECBackend(input_samplerate=self.samplerate)
            except Exception:
                self._external_backend = None

    @property
    def last_shift(self) -> int:
        return int(self._last_shift)

    @property
    def filter_length(self) -> int:
        return int(self._filter_length)

    def configure(self, *, max_shift_samples: Optional[int] = None, filter_length: Optional[int] = None) -> None:
        if max_shift_samples is not None:
            self.max_shift_samples = max(0, int(max_shift_samples))
        if filter_length is not None:
            self._filter_length = max(256, int(filter_length))
            self._weights = np.zeros((self._filter_length,), dtype=np.float32)
            self._windows_native_backend = None
            self._windows_native_backend_attempted = False

    def reset(self) -> None:
        self._weights = np.zeros((self._filter_length,), dtype=np.float32)
        self._last_shift = 0
        self._delay_lock_shift = 0
        self._delay_lock_votes = 0
        self._last_double_talk = False
        self._windows_native_backend = None
        self._windows_native_backend_attempted = False

    @staticmethod
    def _build_tapped_matrix(context: np.ndarray, taps: int) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(context, taps)
        return np.ascontiguousarray(windows[:, ::-1], dtype=np.float32)

    def _nlms_candidate(
        self,
        design: np.ndarray,
        mic: np.ndarray,
        *,
        iterations: int,
        initial_weights: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if initial_weights is None:
            weights = self._weights.astype(np.float32, copy=True)
        else:
            weights = np.asarray(initial_weights, dtype=np.float32).copy()
        target = mic.astype(np.float32, copy=False)
        predicted = np.zeros_like(target, dtype=np.float32)
        for _ in range(max(1, iterations)):
            weights *= self._leakage
            for index in range(target.size):
                taps = design[index]
                estimate = float(np.dot(weights, taps))
                predicted[index] = estimate
                error = float(target[index] - estimate)
                norm = float(np.dot(taps, taps) + 1e-6)
                weights += (self._mu * error / norm) * taps
            weights = np.clip(weights, -1.2, 1.2)
        return weights, predicted

    def _ridge_candidate(
        self,
        design: np.ndarray,
        mic: np.ndarray,
        *,
        ridge: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(mic, dtype=np.float32).reshape(-1)
        matrix = np.asarray(design, dtype=np.float32)
        sample_count = int(target.size)
        if sample_count <= 0:
            weights = self._weights.astype(np.float32, copy=True)
            return weights, np.zeros((0,), dtype=np.float32)
        regularization = float(self.ridge if ridge is None else ridge)
        regularization = max(1e-5, regularization)
        try:
            gram = matrix @ matrix.T
            gram.flat[:: sample_count + 1] += regularization
            dual = np.linalg.solve(gram, target)
            weights = matrix.T @ dual
            weights = np.clip(weights.astype(np.float32, copy=False), -1.2, 1.2)
            predicted = matrix @ weights
            return weights, predicted.astype(np.float32, copy=False)
        except Exception:
            return self._nlms_candidate(matrix, target, iterations=2)

    def _ensure_windows_native_backend(self) -> Optional[WindowsNativeAECBackend]:
        if self._windows_native_backend is not None:
            return self._windows_native_backend
        if self._windows_native_backend_attempted:
            return None
        self._windows_native_backend_attempted = True
        if self._windows_native_probe.available:
            try:
                self._windows_native_backend = WindowsNativeAECBackend(
                    input_samplerate=self.samplerate,
                    filter_length=self._filter_length,
                    max_shift_samples=self.max_shift_samples,
                )
            except Exception:
                self._windows_native_backend = None
        return self._windows_native_backend

    @staticmethod
    def _detect_double_talk(
        *,
        similarity: float,
        mic_level: float,
        ref_level: float,
        predicted_level: float,
        residual_level: float,
        improvement_ratio: float,
        voice_likelihood: float,
        previous_double_talk: bool,
    ) -> bool:
        residual_ratio = residual_level / max(mic_level, 1e-4)
        near_excess = mic_level > max(0.02, predicted_level * 1.02)
        residual_excess = residual_level > max(0.016, predicted_level * 0.3)
        speech_hint = voice_likelihood >= (0.44 if previous_double_talk else 0.52)
        likely = (
            0.26 <= similarity <= 0.82
            and ref_level >= 0.01
            and near_excess
            and residual_excess
            and residual_ratio >= (0.28 if previous_double_talk else 0.29)
            and improvement_ratio <= (0.9 if previous_double_talk else 0.82)
            and speech_hint
        )
        fallback_nearend = bool(
            similarity >= 0.72
            and ref_level >= 0.01
            and voice_likelihood >= (0.5 if previous_double_talk else 0.54)
            and mic_level > max(0.03, ref_level * 1.14)
        )
        if not likely and previous_double_talk:
            return bool(
                fallback_nearend
                or (
                    similarity >= 0.26
                    and ref_level >= 0.008
                    and residual_ratio >= 0.32
                    and improvement_ratio <= 0.84
                    and voice_likelihood >= 0.42
                )
            )
        return bool(likely or fallback_nearend)

    def _update_delay_tracker(self, best_shift: int, similarity: float) -> None:
        target = int(best_shift)
        max_lock_drift = max(96, self._filter_length // 6)
        if self._last_shift <= 0:
            self._last_shift = target
            self._delay_lock_shift = target
            self._delay_lock_votes = 1
            return
        if self._delay_lock_shift <= 0:
            self._delay_lock_shift = self._last_shift
            self._delay_lock_votes = 1
        if abs(target - self._delay_lock_shift) <= 32:
            self._delay_lock_votes = min(6, self._delay_lock_votes + 1)
            lock_target = target
        elif similarity >= 0.88 and abs(target - self._delay_lock_shift) <= max_lock_drift:
            self._delay_lock_shift = target
            self._delay_lock_votes = min(4, self._delay_lock_votes + 1)
            lock_target = target
        else:
            self._delay_lock_votes = max(0, self._delay_lock_votes - 1)
            lock_target = self._delay_lock_shift
        self._delay_lock_shift = int(lock_target)
        delta = abs(target - self._last_shift)
        if self._delay_lock_votes >= 3:
            target = self._delay_lock_shift
            delta = abs(target - self._last_shift)
        if delta <= 24:
            alpha = 0.2
        elif delta <= 96:
            alpha = 0.12 if similarity >= 0.8 else 0.05
        elif similarity >= 0.92:
            alpha = 0.1
        elif similarity >= 0.8 and self._delay_lock_votes >= 2:
            alpha = 0.04
        else:
            alpha = 0.01
        self._last_shift = int(round((self._last_shift * (1.0 - alpha)) + (target * alpha)))

    def process(
        self,
        mic_pcm: bytes,
        reference: np.ndarray,
        *,
        max_shift_samples: Optional[int] = None,
        expected_shift: Optional[int] = None,
        aec_mode: str = DEFAULT_AUDIO_AEC_MODE,
    ) -> dict[str, object]:
        aec_mode = normalize_audio_aec_mode(aec_mode)
        prefer_native_mode = bool(aec_mode == "windows_native_preferred")
        prefer_webrtc_mode = bool(aec_mode == "webrtc_preferred")
        custom_only_mode = bool(aec_mode == "custom_only")
        if not mic_pcm:
            return {
                "pcm": b"",
                "similarity": 0.0,
                "delay_samples": int(self._last_shift),
                "double_talk": False,
                "residual_level": 0.0,
                "mic_level": 0.0,
                "aec_quality": 0.0,
                "webrtc_success": False,
                "voice_likelihood": 0.0,
            }

        mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
        ref = np.asarray(reference, dtype=np.int16).astype(np.float32).reshape(-1)
        voice_likelihood = float(estimate_voice_likelihood_from_pcm16(mic_pcm))
        anchor_shift = self._last_shift if expected_shift is None else int(expected_shift)
        if mic.size < 120 or ref.size < mic.size:
            level = _rms(mic / 32768.0)
            return {
                "pcm": mic_pcm,
                "similarity": 0.0,
                "delay_samples": int(self._last_shift),
                "double_talk": False,
                "residual_level": float(level),
                "mic_level": float(level),
                "aec_quality": 0.0,
                "webrtc_success": False,
                "voice_likelihood": voice_likelihood,
            }

        shift_limit = self.max_shift_samples if max_shift_samples is None else int(max_shift_samples)
        segment, similarity, best_shift = _find_best_alignment(
            mic,
            ref,
            max_shift_samples=shift_limit,
            expected_shift=anchor_shift,
        )
        if segment.size != mic.size or similarity < 0.04:
            level = _rms(mic / 32768.0)
            return {
                "pcm": mic_pcm,
                "similarity": float(similarity),
                "delay_samples": int(best_shift),
                "double_talk": False,
                "residual_level": float(level),
                "mic_level": float(level),
                "aec_quality": 0.0,
                "webrtc_success": False,
                "voice_likelihood": voice_likelihood,
            }

        taps = int(self._filter_length)
        context = _extract_reference_context(ref, mic.size, best_shift, taps)
        if context.size != mic.size + taps - 1:
            level = _rms(mic / 32768.0)
            return {
                "pcm": mic_pcm,
                "similarity": float(similarity),
                "delay_samples": int(best_shift),
                "double_talk": False,
                "residual_level": float(level),
                "mic_level": float(level),
                "aec_quality": 0.0,
                "webrtc_success": False,
                "voice_likelihood": voice_likelihood,
            }

        mic_centered = (mic / 32768.0).astype(np.float32)
        mic_centered -= float(np.mean(mic_centered))
        context_norm = (context / 32768.0).astype(np.float32)
        context_norm -= float(np.mean(context_norm))
        design = self._build_tapped_matrix(context_norm, taps)
        active_weights = self._weights.astype(np.float32, copy=True)
        predicted_before = design @ active_weights

        mic_level = _rms(mic / 32768.0)
        ref_level = _rms(segment / 32768.0)
        predicted_level_before = _rms(predicted_before)
        residual_before = mic_centered - predicted_before
        residual_level_before = _rms(residual_before)

        probe_weights, predicted_probe = self._nlms_candidate(design, mic_centered, iterations=1)
        residual_probe = mic_centered - predicted_probe
        residual_level_probe = _rms(residual_probe)
        predicted_level_probe = _rms(predicted_probe)
        improvement_ratio = 1.0 - min(1.0, residual_level_probe / max(mic_level, 1e-4))
        stable_delay_lock = bool(
            self._delay_lock_votes >= 3
            and self._delay_lock_shift > 0
            and abs(int(best_shift) - int(self._delay_lock_shift)) <= max(64, taps // 6)
        )
        shift_error = abs(int(best_shift) - int(anchor_shift or 0)) if anchor_shift > 0 else 0
        custom_anchor_guard = bool(anchor_shift > 0 and shift_error > max(176, taps // 3) and similarity < 0.78)
        double_talk = self._detect_double_talk(
            similarity=similarity,
            mic_level=mic_level,
            ref_level=ref_level,
            predicted_level=predicted_level_probe,
            residual_level=residual_level_probe,
            improvement_ratio=improvement_ratio,
            voice_likelihood=voice_likelihood,
            previous_double_talk=self._last_double_talk,
        )
        adapt_allowed = bool(
            similarity >= 0.45
            and ref_level >= 0.012
            and predicted_level_probe >= 0.015
            and not double_talk
            and not custom_anchor_guard
            and (anchor_shift <= 0 or shift_error <= max(176, taps // 3) or stable_delay_lock or similarity >= 0.88)
        )

        if adapt_allowed:
            iterations = 2 if similarity >= 0.72 else 1
            nlms_weights, _predicted_after = self._nlms_candidate(
                design,
                mic_centered,
                iterations=iterations,
                initial_weights=probe_weights,
            )
            candidate_weights = nlms_weights
            if similarity >= 0.45:
                ridge_weights, ridge_predicted = self._ridge_candidate(
                    design,
                    mic_centered,
                    ridge=max(self.ridge, 2.5e-3),
                )
                ridge_residual = mic_centered - ridge_predicted
                ridge_residual_level = _rms(ridge_residual)
                ridge_predicted_level = _rms(ridge_predicted)
                ridge_improvement = 1.0 - min(1.0, ridge_residual_level / max(mic_level, 1e-4))
                if ridge_predicted_level >= predicted_level_probe * 0.8 and ridge_improvement >= improvement_ratio - 0.04:
                    candidate_weights = (nlms_weights * 0.4) + (ridge_weights * 0.6)
            if np.any(self._weights):
                blend = 0.18
                if similarity >= 0.72 and improvement_ratio >= 0.12:
                    blend = 0.42
                elif similarity >= 0.55 and improvement_ratio >= 0.06:
                    blend = 0.28
                self._weights = (self._weights * (1.0 - blend)) + (candidate_weights * blend)
            else:
                self._weights = candidate_weights
            active_weights = self._weights
            predicted = design @ active_weights
            residual = mic_centered - predicted
        else:
            predicted = predicted_before
            residual = residual_before

        post_similarity = _normalized_similarity(segment.astype(np.int16), (residual * 32767.0).astype(np.int16), max_shift_samples=0)
        if similarity >= 0.24 and post_similarity >= 0.1 and not double_talk:
            suppress = min(0.2, max(0.0, similarity - post_similarity) * 0.3 + 0.06)
            residual -= context_norm[taps - 1 :] * suppress
        cleaned = np.clip((residual + float(np.mean(mic / 32768.0))) * 32767.0, -32768.0, 32767.0).astype(np.int16)
        residual_level = _rms(cleaned.astype(np.float32) / 32768.0)
        predicted_level = _rms(predicted)
        improvement_ratio = max(0.0, 1.0 - min(1.0, residual_level / max(mic_level, 1e-4)))
        backend_used = "custom"
        webrtc_success = False
        native_attempted = False
        native_selected = False
        selection_reason = "custom_fallback"
        probe_similarity = float(similarity)
        probe_shift = int(best_shift)
        probe_shift_error = int(shift_error)
        if (
            similarity >= 0.2
            and backend_used == "custom"
            and improvement_ratio < 0.03
            and not double_talk
        ):
            similarity = min(float(similarity), 0.08)
            predicted_level = 0.0
            residual_level = float(mic_level)
            best_shift = 0
            cleaned = mic.astype(np.int16, copy=False)
            selection_reason = "custom_low_gain"
        if self._windows_native_probe.available and prefer_native_mode and segment.size == mic.size and not double_talk:
            try:
                native_backend = self._ensure_windows_native_backend()
                if native_backend is not None:
                    native_attempted = True
                    native_pcm = native_backend.process(
                        mic_pcm=mic_pcm,
                        reference_pcm=segment.astype(np.int16, copy=False),
                        delay_ms=int(round(probe_shift * (1000.0 / float(self.samplerate)))),
                    )
                    native_cleaned = np.frombuffer(native_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    native_residual_level = _rms(native_cleaned)
                    native_improvement = max(0.0, 1.0 - min(1.0, native_residual_level / max(mic_level, 1e-4)))
                    native_predicted_level = max(0.0, mic_level - native_residual_level)
                    native_strong = bool(
                        native_improvement >= 0.2
                        and native_residual_level <= max(0.0035, mic_level * 0.12)
                    )
                    prefer_native = bool(
                        native_improvement >= improvement_ratio + 0.02
                        or (
                            probe_similarity >= 0.3
                            and native_residual_level <= residual_level * 1.05
                            and native_improvement >= max(0.02, improvement_ratio - 0.01)
                        )
                        or (
                            stable_delay_lock
                            and native_improvement >= max(0.08, improvement_ratio + 0.015)
                            and native_residual_level <= residual_level * 0.96
                        )
                    )
                    if prefer_native_mode:
                        prefer_native = bool(
                            prefer_native
                            or (
                                native_improvement >= max(0.08, improvement_ratio + 0.02)
                                and native_residual_level <= residual_level * 1.0
                                and (
                                    probe_similarity >= 0.28
                                    or (stable_delay_lock and ref_level >= 0.01 and mic_level >= 0.01)
                                )
                            )
                            or (
                                backend_used == "custom"
                                and native_improvement >= 0.12
                                and native_residual_level <= residual_level * 0.98
                            )
                        )
                    if anchor_shift > 0 and probe_shift_error > max(240, taps // 3) and voice_likelihood >= 0.55:
                        prefer_native = False
                    if prefer_native:
                        cleaned = np.frombuffer(native_pcm, dtype=np.int16).copy()
                        similarity = probe_similarity
                        residual_level = float(native_residual_level)
                        improvement_ratio = float(native_improvement)
                        predicted_level = max(float(predicted_level), float(native_predicted_level))
                        best_shift = probe_shift
                        backend_used = "windows_native"
                        webrtc_success = native_strong
                        native_selected = True
                        selection_reason = "windows_native_preferred"
                        if (
                            probe_similarity >= 0.35
                            and native_improvement >= 0.05
                            and (
                                self._delay_lock_shift <= 0
                                or abs(int(probe_shift) - int(self._delay_lock_shift)) <= max(128, self._filter_length // 4)
                            )
                        ):
                            self._delay_lock_shift = int(probe_shift)
                            self._delay_lock_votes = max(self._delay_lock_votes, 3)
            except Exception:
                pass
        allow_webrtc_probe = bool(
            (
                probe_similarity >= 0.18
                and (anchor_shift <= 0 or probe_shift_error <= max(288, taps // 2) or probe_similarity >= 0.62)
            )
            or (
                stable_delay_lock
                and ref_level >= 0.012
                and mic_level >= 0.01
                and predicted_level_probe >= 0.003
            )
        )
        if custom_only_mode:
            allow_webrtc_probe = False
        elif prefer_webrtc_mode:
            allow_webrtc_probe = bool(
                allow_webrtc_probe
                or (
                    stable_delay_lock
                    and ref_level >= 0.008
                    and mic_level >= 0.008
                    and probe_similarity >= 0.14
                )
            )
        if self._external_backend is not None and allow_webrtc_probe and segment.size == mic.size and not double_talk:
            try:
                delay_ms = int(round(probe_shift * (1000.0 / float(self.samplerate))))
                external_pcm = self._external_backend.process(
                    mic_pcm=mic_pcm,
                    reference_pcm=segment.astype(np.int16, copy=False),
                    delay_ms=delay_ms,
                )
                external_cleaned = np.frombuffer(external_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                external_residual_level = _rms(external_cleaned)
                external_improvement = max(0.0, 1.0 - min(1.0, external_residual_level / max(mic_level, 1e-4)))
                external_predicted_level = max(0.0, mic_level - external_residual_level)
                strong_external = bool(
                    external_improvement >= 0.18
                    and external_residual_level <= max(0.0035, mic_level * 0.12)
                )
                prefer_external = bool(
                    external_improvement >= improvement_ratio + 0.02
                    or (
                        probe_similarity >= 0.35
                        and external_residual_level <= residual_level * 1.08
                        and external_improvement >= max(0.02, improvement_ratio - 0.01)
                    )
                    or (
                        probe_similarity >= 0.5 and external_residual_level <= residual_level * 1.02
                    )
                    or (
                        stable_delay_lock
                        and external_improvement >= max(0.08, improvement_ratio + 0.015)
                        and external_residual_level <= residual_level * 0.96
                    )
                    or (
                        probe_similarity >= 0.18
                        and improvement_ratio < 0.02
                        and external_improvement >= 0.05
                        and external_residual_level <= residual_level * 0.96
                    )
                )
                if anchor_shift > 0 and probe_shift_error > max(320, taps // 2) and probe_similarity < 0.68:
                    prefer_external = False
                if anchor_shift > 0 and probe_shift_error > max(240, taps // 3) and voice_likelihood >= 0.55:
                    prefer_external = False
                if prefer_native_mode and backend_used == "windows_native":
                    prefer_external = bool(
                        external_improvement >= improvement_ratio + 0.35
                        or (
                            strong_external
                            and external_improvement >= improvement_ratio + 0.18
                            and external_residual_level <= residual_level * 0.75
                            and probe_similarity >= 0.6
                        )
                    )
                if not strong_external:
                    if external_improvement < improvement_ratio + 0.12:
                        prefer_external = False
                    if voice_likelihood >= 0.55 and external_improvement < 0.45:
                        prefer_external = False
                if prefer_external:
                    cleaned = np.frombuffer(external_pcm, dtype=np.int16).copy()
                    similarity = probe_similarity
                    residual_level = float(external_residual_level)
                    improvement_ratio = float(external_improvement)
                    predicted_level = max(float(predicted_level), float(external_predicted_level))
                    best_shift = probe_shift
                    backend_used = "webrtc"
                    webrtc_success = strong_external
                    native_selected = False
                    selection_reason = "webrtc_override"
                    if (
                        probe_similarity >= 0.4
                        and external_improvement >= 0.05
                        and (
                            self._delay_lock_shift <= 0
                            or abs(int(probe_shift) - int(self._delay_lock_shift)) <= max(128, self._filter_length // 4)
                        )
                        ):
                            self._delay_lock_shift = int(probe_shift)
                            self._delay_lock_votes = max(self._delay_lock_votes, 3)
            except Exception:
                pass
        quality = float(
            max(
                0.0,
                min(
                    1.0,
                    similarity
                    * improvement_ratio
                    * min(1.0, predicted_level / max(ref_level, 1e-4) * 1.2),
                ),
            )
        )
        self._update_delay_tracker(best_shift, similarity)
        self._last_double_talk = bool(double_talk)
        return {
            "pcm": cleaned.tobytes(),
            "similarity": float(similarity),
            "delay_samples": int(best_shift),
            "double_talk": bool(double_talk),
            "residual_level": float(residual_level),
            "mic_level": float(mic_level),
            "aec_quality": quality,
            "predicted_level": float(predicted_level),
            "improvement_ratio": float(improvement_ratio),
            "backend": backend_used,
            "webrtc_success": bool(webrtc_success),
            "native_attempted": bool(native_attempted),
            "native_selected": bool(native_selected),
            "selection_reason": selection_reason,
            "voice_likelihood": voice_likelihood,
        }


def suppress_echo_from_pcm16(
    mic_pcm: bytes,
    reference: np.ndarray,
    *,
    max_shift_samples: int = 960,
) -> tuple[bytes, float]:
    """Kompatibilní wrapper pro jednorázové potlačení echo v testech."""
    result = AdaptiveEchoCanceller(max_shift_samples=max_shift_samples).process(
        mic_pcm,
        reference,
        max_shift_samples=max_shift_samples,
    )
    return bytes(result["pcm"]), float(result["similarity"])


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
    _aligned_segment, similarity, latency_samples = _find_best_alignment(
        captured.astype(np.float32),
        playback_signal.astype(np.float32),
        max_shift_samples=int(samplerate * 0.12),
    )
    input_name = _device_name(input_device, "input")
    output_name = _device_name(output_device, "output")
    audio_mode = _infer_audio_mode(input_name, output_name)
    filter_length = int(np.clip(256 + max(0, latency_samples) * 2, 256, 2048))
    preferred_frame_size = 960 if audio_mode in {"notebook_builtin", "external_speakers"} else 480

    recommended_profile = {
        "server_vad_threshold": float(np.clip(0.72 + max(0.0, similarity - 0.45) * 0.18 + max(0.0, min(0.18, ambient_rms)) * 0.55, 0.68, 0.9)),
        "playback_activity_level": float(np.clip(max(0.028, playback_rms * 0.45), 0.028, 0.16)),
        "echo_similarity_drop": float(np.clip(0.78 + similarity * 0.14, 0.78, 0.96)),
        "echo_similarity_soft": float(np.clip(0.6 + similarity * 0.12, 0.58, 0.86)),
        "barge_in_min_input_level": float(np.clip(max(0.05, ambient_rms * 4.2, playback_rms * 0.3), 0.05, 0.22)),
        "barge_in_output_ratio": float(np.clip(1.2 + min(bleed_ratio, 8.0) * 0.06 + (0.08 if audio_mode == "notebook_builtin" else 0.0), 1.18, 1.9)),
    }
    if audio_mode in {"wired_headset", "bluetooth_headset"}:
        recommended_profile["echo_similarity_drop"] = float(np.clip(recommended_profile["echo_similarity_drop"] - 0.08, 0.68, 0.96))
        recommended_profile["echo_similarity_soft"] = float(np.clip(recommended_profile["echo_similarity_soft"] - 0.06, 0.54, 0.86))
        recommended_profile["playback_activity_level"] = float(np.clip(recommended_profile["playback_activity_level"] * 0.82, 0.02, 0.12))
    elif audio_mode in {"external_speakers", "notebook_builtin"}:
        recommended_profile["server_vad_threshold"] = float(np.clip(recommended_profile["server_vad_threshold"] + 0.01, 0.68, 0.9))
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"ambient_rms={ambient_rms:.4f}",
        f"playback_rms={playback_rms:.4f}",
        f"bleed_ratio={bleed_ratio:.2f}",
        f"similarity={similarity:.3f}",
        f"latency_samples={latency_samples}",
        f"audio_mode={audio_mode}",
        f"frame_size={preferred_frame_size}",
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
        latency_samples=int(latency_samples),
        preferred_frame_size=int(preferred_frame_size),
        filter_length=int(filter_length),
        device_fingerprint=_build_device_fingerprint(input_device, output_device, samplerate),
        audio_mode=audio_mode,
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
    latency_samples = int(round(float(np.median([getattr(item, "latency_samples", 0) for item in results]))))
    preferred_frame_size = int(round(float(np.median([getattr(item, "preferred_frame_size", 480) for item in results]))))
    filter_length = int(round(float(np.median([getattr(item, "filter_length", 256) for item in results]))))
    mode_counts: dict[str, int] = {}
    for item in results:
        mode = getattr(item, "audio_mode", "notebook_builtin")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    audio_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "notebook_builtin"
    device_fingerprint = getattr(results[0], "device_fingerprint", "unknown")
    if recommended_profile["echo_similarity_soft"] >= recommended_profile["echo_similarity_drop"]:
        recommended_profile["echo_similarity_soft"] = round(recommended_profile["echo_similarity_drop"] - 0.05, 3)

    notes = [
        f"passes={len(results)}",
        f"ambient_med={float(np.median(ambient_values)):.4f}",
        f"playback_med={float(np.median(playback_values)):.4f}",
        f"bleed_peak={float(np.max(bleed_values)):.2f}",
        f"similarity_peak={float(np.max(similarity_values)):.3f}",
        f"latency_med={latency_samples}",
        f"audio_mode={audio_mode}",
        f"frame_size={preferred_frame_size}",
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
        latency_samples=int(latency_samples),
        preferred_frame_size=int(preferred_frame_size),
        filter_length=int(filter_length),
        device_fingerprint=device_fingerprint,
        audio_mode=audio_mode,
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
        self._echo_reference_chunks: "deque[tuple[int, bytes]]" = deque()
        self._echo_reference_enqueued_samples = 0
        self._echo_reference_played_samples = 0
        self._echo_reference_max_samples = int(self.target_samplerate * 2.0)
        self._last_callback_mono_ns = 0
        self._echo_reference_played_end_mono_ns = 0

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
