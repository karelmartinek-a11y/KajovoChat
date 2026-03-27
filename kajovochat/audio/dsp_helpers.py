from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

from .aec_backends import (
    AecBackendContext,
    WebRtcApmBackendRunner,
    WindowsSystemAecBackendRunner,
)
from .aec_native import WindowsNativeAecResources
from .aec_orchestration import process_adaptive_echo
from .io.common import _resample_pcm16_mono
from ..services.voice_features import estimate_voice_likelihood_from_pcm16
from ..settings import DEFAULT_AUDIO_AEC_MODE, normalize_audio_aec_mode

try:
    from aec_audio_processing import AudioProcessor as WebRTCAudioProcessor
except Exception:
    WebRTCAudioProcessor = None

@dataclass
class ReferencePrepResult:
    mic: np.ndarray
    mic_centered: np.ndarray
    context_norm: np.ndarray
    design: np.ndarray
    segment: np.ndarray
    similarity: float
    best_shift: int
    mic_level: float
    ref_level: float
    anchor_shift: int
    shift_error: int
    stable_delay_lock: bool
    voice_likelihood: float

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
        self._native_aec_resources = WindowsNativeAecResources(
            samplerate=self.samplerate,
            filter_length=self._filter_length,
            max_shift_samples=self.max_shift_samples,
        )
        self._windows_native_probe = self._native_aec_resources.probe
        self._external_backend: Optional[_WebRTCAECBackend] = None
        if WebRTCAudioProcessor is not None:
            try:
                self._external_backend = _WebRTCAECBackend(input_samplerate=self.samplerate)
            except Exception:
                self._external_backend = None
        self._windows_system_runner = WindowsSystemAecBackendRunner(self)
        self._webrtc_runner = WebRtcApmBackendRunner(self)

    @property
    def last_shift(self) -> int:
        return int(self._last_shift)

    @property
    def _windows_native_backend(self):
        return self._native_aec_resources.backend

    @_windows_native_backend.setter
    def _windows_native_backend(self, value) -> None:
        self._native_aec_resources._backend = value

    @property
    def _windows_native_session(self):
        return self._native_aec_resources.session

    @_windows_native_session.setter
    def _windows_native_session(self, value) -> None:
        self._native_aec_resources._session = value

    @property
    def _windows_native_backend_attempted(self) -> bool:
        return bool(self._native_aec_resources._backend_attempted)

    @_windows_native_backend_attempted.setter
    def _windows_native_backend_attempted(self, value: bool) -> None:
        self._native_aec_resources._backend_attempted = bool(value)

    @property
    def filter_length(self) -> int:
        return int(self._filter_length)

    def configure(self, *, max_shift_samples: Optional[int] = None, filter_length: Optional[int] = None) -> None:
        if max_shift_samples is not None:
            self.max_shift_samples = max(0, int(max_shift_samples))
        if filter_length is not None:
            self._filter_length = max(256, int(filter_length))
            self._weights = np.zeros((self._filter_length,), dtype=np.float32)
            self._native_aec_resources.reconfigure(
                filter_length=self._filter_length,
                max_shift_samples=self.max_shift_samples,
            )
            self._windows_native_probe = self._native_aec_resources.probe

    def reset(self) -> None:
        self._weights = np.zeros((self._filter_length,), dtype=np.float32)
        self._last_shift = 0
        self._delay_lock_shift = 0
        self._delay_lock_votes = 0
        self._last_double_talk = False
        self._native_aec_resources.reset()
        self._windows_native_probe = self._native_aec_resources.probe

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

    def _ensure_windows_native_backend(self):
        return self._native_aec_resources.ensure_backend()

    def _ensure_windows_native_session(self):
        return self._native_aec_resources.ensure_session()

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

    def _prepare_reference_window(
        self,
        *,
        mic_pcm: bytes,
        reference: np.ndarray,
        expected_shift: Optional[int],
        max_shift_samples: Optional[int],
    ) -> tuple[Optional[ReferencePrepResult], Optional[dict[str, object]]]:
        mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
        ref = np.asarray(reference, dtype=np.int16).astype(np.float32).reshape(-1)
        voice_likelihood = float(estimate_voice_likelihood_from_pcm16(mic_pcm))
        if mic.size < 120 or ref.size < mic.size:
            level = _rms(mic / 32768.0)
            return None, {
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

        anchor_shift = self._last_shift if expected_shift is None else int(expected_shift)
        shift_limit = self.max_shift_samples if max_shift_samples is None else int(max_shift_samples)
        segment, similarity, best_shift = _find_best_alignment(
            mic,
            ref,
            max_shift_samples=shift_limit,
            expected_shift=anchor_shift,
        )
        if segment.size != mic.size or similarity < 0.04:
            level = _rms(mic / 32768.0)
            return None, {
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
            return None, {
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
        mic_level = _rms(mic / 32768.0)
        ref_level = _rms(segment / 32768.0)
        shift_error = abs(int(best_shift) - int(anchor_shift or 0)) if anchor_shift > 0 else 0
        stable_delay_lock = bool(
            self._delay_lock_votes >= 3
            and self._delay_lock_shift > 0
            and abs(int(best_shift) - int(self._delay_lock_shift)) <= max(64, taps // 6)
        )
        return ReferencePrepResult(
            mic=mic,
            mic_centered=mic_centered,
            context_norm=context_norm,
            design=design,
            segment=segment,
            similarity=float(similarity),
            best_shift=int(best_shift),
            mic_level=float(mic_level),
            ref_level=float(ref_level),
            anchor_shift=int(anchor_shift),
            shift_error=int(shift_error),
            stable_delay_lock=bool(stable_delay_lock),
            voice_likelihood=float(voice_likelihood),
        ), None

    def _process_windows_system_capture(self, mic_pcm: bytes) -> dict[str, object]:
        mic = np.frombuffer(mic_pcm, dtype=np.int16).astype(np.float32)
        mic_level = _rms(mic / 32768.0) if mic.size else 0.0
        voice_likelihood = float(estimate_voice_likelihood_from_pcm16(mic_pcm)) if mic_pcm else 0.0
        return {
            "pcm": mic_pcm,
            "similarity": 0.0,
            "delay_samples": 0,
            "double_talk": False,
            "residual_level": float(mic_level),
            "mic_level": float(mic_level),
            "aec_quality": 0.16,
            "predicted_level": 0.0,
            "improvement_ratio": 0.0,
            "backend": "windows_system_aec",
            "backend_policy": "windows_system_aec",
            "webrtc_success": False,
            "native_attempted": True,
            "native_selected": True,
            "selection_reason": "windows_system_aec",
            "voice_likelihood": voice_likelihood,
            "system_capture_processed": True,
        }

    def _process_webrtc_only(
        self,
        *,
        mic_pcm: bytes,
        reference: np.ndarray,
        expected_shift: Optional[int],
        max_shift_samples: Optional[int],
    ) -> dict[str, object]:
        prep, early_result = self._prepare_reference_window(
            mic_pcm=mic_pcm,
            reference=reference,
            expected_shift=expected_shift,
            max_shift_samples=max_shift_samples,
        )
        if early_result is not None:
            result = dict(early_result)
            result.update(
                {
                    "predicted_level": 0.0,
                    "improvement_ratio": 0.0,
                    "backend": "webrtc",
                    "backend_policy": "webrtc_apm",
                    "native_attempted": False,
                    "native_selected": False,
                    "selection_reason": "webrtc_reference_unavailable",
                }
            )
            return result
        assert prep is not None
        mic_level = float(prep.mic_level)
        double_talk = self._detect_double_talk(
            similarity=prep.similarity,
            mic_level=prep.mic_level,
            ref_level=prep.ref_level,
            predicted_level=0.0,
            residual_level=prep.mic_level,
            improvement_ratio=0.0,
            voice_likelihood=prep.voice_likelihood,
            previous_double_talk=self._last_double_talk,
        )
        self._last_double_talk = bool(double_talk)
        if double_talk:
            return {
                "pcm": mic_pcm,
                "similarity": float(prep.similarity),
                "delay_samples": int(prep.best_shift),
                "double_talk": True,
                "residual_level": mic_level,
                "mic_level": mic_level,
                "aec_quality": 0.0,
                "predicted_level": 0.0,
                "improvement_ratio": 0.0,
                "backend": "webrtc",
                "backend_policy": "webrtc_apm",
                "webrtc_success": False,
                "native_attempted": False,
                "native_selected": False,
                "selection_reason": "webrtc_double_talk_passthrough",
                "voice_likelihood": float(prep.voice_likelihood),
            }
        webrtc_result = self._run_webrtc_apm(
            prep=prep,
            mic_pcm=mic_pcm,
            residual_level=mic_level,
            improvement_ratio=0.0,
            predicted_level=0.0,
            prefer_native_mode=False,
            backend_used="webrtc",
        )
        if webrtc_result is not None:
            quality = float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(webrtc_result["similarity"])
                        * float(webrtc_result["improvement_ratio"])
                        * min(1.0, float(webrtc_result["predicted_level"]) / max(prep.ref_level, 1e-4) * 1.2),
                    ),
                )
            )
            self._update_delay_tracker(int(webrtc_result["delay_samples"]), float(webrtc_result["similarity"]))
            return {
                "pcm": webrtc_result["pcm"].tobytes(),
                "similarity": float(webrtc_result["similarity"]),
                "delay_samples": int(webrtc_result["delay_samples"]),
                "double_talk": False,
                "residual_level": float(webrtc_result["residual_level"]),
                "mic_level": mic_level,
                "aec_quality": quality,
                "predicted_level": float(webrtc_result["predicted_level"]),
                "improvement_ratio": float(webrtc_result["improvement_ratio"]),
                "backend": "webrtc",
                "backend_policy": "webrtc_apm",
                "webrtc_success": bool(webrtc_result["webrtc_success"]),
                "native_attempted": False,
                "native_selected": False,
                "selection_reason": "webrtc_apm",
                "voice_likelihood": float(prep.voice_likelihood),
            }
        return {
            "pcm": mic_pcm,
            "similarity": float(prep.similarity),
            "delay_samples": int(prep.best_shift),
            "double_talk": False,
            "residual_level": mic_level,
            "mic_level": mic_level,
            "aec_quality": 0.0,
            "predicted_level": 0.0,
            "improvement_ratio": 0.0,
            "backend": "webrtc",
            "backend_policy": "webrtc_apm",
            "webrtc_success": False,
            "native_attempted": False,
            "native_selected": False,
            "selection_reason": "webrtc_no_gain",
            "voice_likelihood": float(prep.voice_likelihood),
        }

    def _run_windows_system_aec(
        self,
        *,
        prep: ReferencePrepResult,
        mic_pcm: bytes,
        residual_level: float,
        improvement_ratio: float,
        predicted_level: float,
        backend_used: str,
    ) -> Optional[dict[str, object]]:
        result = self._windows_system_runner.run(
            context=AecBackendContext(
                prep=prep,
                mic_pcm=mic_pcm,
                residual_level=residual_level,
                improvement_ratio=improvement_ratio,
                predicted_level=predicted_level,
                backend_used=backend_used,
            )
        )
        if result is None:
            return None
        if hasattr(result, "as_dict"):
            return result.as_dict()
        return result

    def _run_webrtc_apm(
        self,
        *,
        prep: ReferencePrepResult,
        mic_pcm: bytes,
        residual_level: float,
        improvement_ratio: float,
        predicted_level: float,
        prefer_native_mode: bool,
        backend_used: str,
    ) -> Optional[dict[str, object]]:
        result = self._webrtc_runner.run(
            context=AecBackendContext(
                prep=prep,
                mic_pcm=mic_pcm,
                residual_level=residual_level,
                improvement_ratio=improvement_ratio,
                predicted_level=predicted_level,
                backend_used=backend_used,
                prefer_native_mode=prefer_native_mode,
            )
        )
        if result is None:
            return None
        return result.as_dict()

    def process(
        self,
        mic_pcm: bytes,
        reference: np.ndarray,
        *,
        max_shift_samples: Optional[int] = None,
        expected_shift: Optional[int] = None,
        aec_mode: str = "custom_lab",
    ) -> dict[str, object]:
        return process_adaptive_echo(
            self,
            mic_pcm,
            reference,
            max_shift_samples=max_shift_samples,
            expected_shift=expected_shift,
            aec_mode=aec_mode,
        )


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

