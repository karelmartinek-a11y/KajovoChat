from __future__ import annotations

import sys
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import sounddevice as sd

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QLineEdit,
    QStyle,
)

from .settings import (
    AppSettings,
    DEFAULT_AUDIO_GUARD_PROFILE,
    build_system_prompt,
    normalize_audio_aec_mode,
)
from .dialogs.settings_dialog import SettingsDialog
from .services.audio_service import (
    AdaptiveEchoCanceller,
    AudioPlayer,
    RealtimeMicStream,
    build_device_fingerprint,
    pick_audio_device,
    format_device_help,
    list_audio_devices,
    calibrate_audio_devices_advanced,
    suppress_echo_from_pcm16,
)
from .services.windows_native_aec import probe_windows_native_aec
from .services.realtime_service import RealtimeConfig, RealtimeService
from .services.log_service import RealtimeLogWriter
from .services.app_logging import install_app_logging
from .services.guard_adaptation import GuardAdaptor
from .services.guard_replay import append_guard_replay_metrics
from .services.guard_telemetry import GuardTelemetry
from .services.voice_features import estimate_voice_likelihood_from_pcm16
from .resources.assets import verify_asset_manifest
from .widgets.head_widget import HeadWidget
from .theme import Theme, app_stylesheet


_ALLOWED_LANGS = {"cs", "en", "de", "sk", "fr"}
_STATE_IDLE = "idle"
_STATE_CONNECTING = "connecting"
_STATE_LISTENING = "listening"
_STATE_TRANSCRIBING = "transcribing"
_STATE_THINKING = "thinking"
_STATE_SPEAKING = "speaking"
_STATE_RECONNECTING = "reconnecting"
_STATE_ERROR = "error"
_VALID_STATES = {
    _STATE_IDLE,
    _STATE_CONNECTING,
    _STATE_LISTENING,
    _STATE_TRANSCRIBING,
    _STATE_THINKING,
    _STATE_SPEAKING,
    _STATE_RECONNECTING,
    _STATE_ERROR,
}

_REALTIME_MODEL = "gpt-realtime"
_TTS_VOICE = "alloy"
_TTS_SPEED = 1.0
_NOISE_REDUCTION = "far_field"
_SERVER_VAD_SILENCE_MS = 900
_SERVER_VAD_PREFIX_MS = 300
_SERVER_VAD_THRESHOLD = 0.72
_PLAYBACK_ACTIVITY_LEVEL = 0.035
_ECHO_TRAILING_HOLD_S = 0.28
_ECHO_SIMILARITY_DROP = 0.82
_ECHO_SIMILARITY_SOFT = 0.68
_BARGE_IN_MIN_INPUT_LEVEL = 0.06
_BARGE_IN_OUTPUT_RATIO = 1.35


def _audio_guard_profile(settings: AppSettings) -> dict[str, float]:
    try:
        return settings.normalized_audio_guard_profile()
    except Exception:
        return dict(DEFAULT_AUDIO_GUARD_PROFILE)


def _closed_pose_snapshot() -> dict[str, object]:
    return {
        "pose": "closed",
        "openness": 0.0,
        "energy": 0.0,
        "weights": {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0},
    }


def _sanitize_text(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_OPENAI_KEY]", text)
    return text


def _pcm16_echo_similarity(mic_pcm: bytes, reference: object) -> float:
    try:
        import numpy as _np

        mic = _np.frombuffer(mic_pcm, dtype=_np.int16).astype(_np.float32)
        ref = _np.asarray(reference, dtype=_np.int16).astype(_np.float32).reshape(-1)
        if mic.size < 120 or ref.size < mic.size:
            return 0.0

        mic = mic - float(_np.mean(mic))
        mic_norm = float(_np.linalg.norm(mic) + 1e-6)
        if mic_norm <= 1e-6:
            return 0.0

        best = 0.0
        max_shift = min(max(0, ref.size - mic.size), 960)
        for shift in range(0, max_shift + 1, 120):
            segment = ref[ref.size - mic.size - shift : ref.size - shift if shift > 0 else ref.size]
            if segment.size != mic.size:
                continue
            segment = segment - float(_np.mean(segment))
            seg_norm = float(_np.linalg.norm(segment) + 1e-6)
            if seg_norm <= 1e-6:
                continue
            corr = abs(float(_np.dot(mic, segment)) / (mic_norm * seg_norm))
            if corr > best:
                best = corr
        return float(max(0.0, min(1.0, best)))
    except Exception:
        return 0.0


def _should_drop_mic_chunk(
    *,
    mode: str,
    guard_active: bool,
    playback_active: bool,
    similarity: float,
    input_level: float,
    output_level: float,
    profile: Optional[dict[str, float]] = None,
    residual_level: Optional[float] = None,
    voice_likelihood: float = 0.0,
    double_talk: bool = False,
    aec_quality: float = 0.0,
) -> tuple[bool, str]:
    active_profile = dict(DEFAULT_AUDIO_GUARD_PROFILE)
    if profile:
        active_profile.update(profile)
    echo_similarity_drop = float(active_profile["echo_similarity_drop"])
    echo_similarity_soft = float(active_profile["echo_similarity_soft"])
    barge_in_min_input_level = float(active_profile["barge_in_min_input_level"])
    barge_in_output_ratio = float(active_profile["barge_in_output_ratio"])
    residual = float(input_level if residual_level is None else residual_level)

    if mode != "handsfree" or not guard_active:
        return False, ""

    strong_user = (
        input_level >= barge_in_min_input_level
        and input_level >= max(barge_in_min_input_level, output_level * barge_in_output_ratio)
    )
    if double_talk and (voice_likelihood >= 0.42 or strong_user):
        return False, ""
    if similarity >= echo_similarity_drop and not strong_user and residual <= max(0.08, output_level * 1.05):
        return True, "echo_similarity"
    if playback_active and aec_quality < 0.04 and similarity >= max(0.66, echo_similarity_soft) and not strong_user and voice_likelihood < 0.5:
        return True, "echo_similarity_fallback"
    if playback_active and similarity >= echo_similarity_soft and residual <= max(0.045, output_level * (0.98 if aec_quality > 0.2 else 1.08)):
        return True, "echo_residual"
    if playback_active and aec_quality < 0.05 and output_level >= 0.06 and residual <= 0.022 and voice_likelihood < 0.26:
        return True, "quiet_bleed"
    return False, ""


def run_audio_guard_selftest() -> dict[str, object]:
    """Lehký lokální selftest audio guardu a dostupnosti zařízení."""
    checks: list[dict[str, object]] = []
    profile = dict(DEFAULT_AUDIO_GUARD_PROFILE)

    drop_echo, reason_echo = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.94,
        input_level=0.03,
        output_level=0.09,
        profile=profile,
    )
    checks.append(
        {
            "name": "echo_drop",
            "ok": drop_echo and reason_echo == "echo_similarity",
            "detail": f"dropped={drop_echo}, reason={reason_echo or '-'}",
        }
    )

    keep_voice, reason_voice = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.21,
        input_level=0.19,
        output_level=0.05,
        profile=profile,
    )
    checks.append(
        {
            "name": "voice_pass",
            "ok": (not keep_voice) and reason_voice == "",
            "detail": f"dropped={keep_voice}, reason={reason_voice or '-'}",
        }
    )

    input_device, input_note = pick_audio_device("input", None)
    output_device, output_note = pick_audio_device("output", None)
    devices = list_audio_devices()
    checks.append(
        {
            "name": "devices_present",
            "ok": input_device is not None and output_device is not None,
            "detail": (
                f"in={input_device if input_device is not None else 'none'} ({input_note}), "
                f"out={output_device if output_device is not None else 'none'} ({output_note}), "
                f"inputs={len(devices.get('inputs', []))}, outputs={len(devices.get('outputs', []))}"
            ),
        }
    )

    if input_device is not None and output_device is not None:
        try:
            calibration = calibrate_audio_devices_advanced(input_device=input_device, output_device=output_device)
            strong_playback_capture = calibration.playback_rms >= max(
                calibration.ambient_rms * 2.4,
                calibration.ambient_rms + 0.006,
            )
            strong_bleed_evidence = calibration.bleed_ratio >= 2.8
            correlation_detected = calibration.similarity >= 0.03
            auto_ok = strong_playback_capture and (strong_bleed_evidence or correlation_detected)
            checks.append(
                {
                    "name": "auto_calibration",
                    "ok": auto_ok,
                    "detail": "; ".join(calibration.notes),
                    "profile": calibration.recommended_profile,
                    "calibration": {
                        "latency_samples": getattr(calibration, "latency_samples", 0),
                        "preferred_frame_size": getattr(calibration, "preferred_frame_size", 480),
                        "filter_length": getattr(calibration, "filter_length", 256),
                        "audio_mode": getattr(calibration, "audio_mode", "notebook_builtin"),
                        "device_fingerprint": getattr(calibration, "device_fingerprint", "unknown"),
                    },
                    "non_blocking": strong_playback_capture,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "auto_calibration",
                    "ok": False,
                    "detail": _sanitize_text(str(exc)),
                    "profile": dict(profile),
                    "non_blocking": False,
                }
            )

    overall_ok = True
    for item in checks:
        if item["ok"]:
            continue
        if item.get("non_blocking"):
            continue
        overall_ok = False
        break

    return {
        "ok": overall_ok,
        "checks": checks,
        "profile": next((dict(item.get("profile", {})) for item in reversed(checks) if item.get("profile")), dict(profile)),
        "calibration": next((dict(item.get("calibration", {})) for item in reversed(checks) if item.get("calibration")), {}),
    }


class ConversationWorker(QObject):
    """Realtime speech-to-speech conversation (WebSocket).

    The UI has two modes:
    - Hands-free: continuous mic streaming, server-side VAD triggers responses.
    - Push-to-talk: mic streams only while button is pressed; on release we commit+response.
    """

    state_changed = Signal(str)        # idle/listening/transcribing/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message
    guard_debug_updated = Signal(object)

    # Realtime levely pro animaci hlavy (0..1).
    input_level = Signal(float)
    output_level = Signal(float)
    output_pose = Signal(object)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._guard_profile = _audio_guard_profile(settings)
        self._guard_calibration = dict(settings.audio_guard_calibration or {})
        self._guard_telemetry = GuardTelemetry()
        self._guard_adaptor = GuardAdaptor()
        self._guard_last_adapt_at = 0.0
        self._guard_learning_until = 0.0
        self._guard_aec_aware = False
        self._aec_mode = normalize_audio_aec_mode(self.settings.audio_aec_mode)
        self._native_aec_probe = probe_windows_native_aec()
        self._session_log_dir: Optional[Path] = None
        self._session_name: str = ""
        self._input_device_name = "default"
        self._output_device_name = "default"
        self._audio_mode = "notebook_builtin"

        self._stop_all = threading.Event()

        self._captions = ""
        self._logger: Optional[RealtimeLogWriter] = None
        self._player: Optional[AudioPlayer] = None
        self._aec = AdaptiveEchoCanceller(samplerate=24000)
        self._resolved_input_device: Optional[int] = None
        self._resolved_output_device: Optional[int] = None

        self._rt: Optional[RealtimeService] = None
        self._rt_loop_stop = threading.Event()
        self._rt_loop_thread: Optional[threading.Thread] = None

        self._mic: Optional[RealtimeMicStream] = None
        self._mic_enabled = threading.Event()

        self._mode: str = "idle"  # "handsfree" | "ptt" | "idle"
        self._resolved_lang = "cs"
        self._rt_turn_mode = "server_vad"
        self._reconnect_attempts = 0
        self._next_reconnect_at = 0.0
        self._response_started_at: Optional[float] = None
        self._response_first_audio_at: Optional[float] = None
        self._speech_stopped_at: Optional[float] = None
        self._last_server_activity_at = time.monotonic()
        self._last_backlog_log_at = 0.0
        self._last_player_progress_at = time.monotonic()
        self._last_player_buffer_bytes = 0
        self._mic_suppressed_until = 0.0
        self._echo_drop_count = 0
        self._barge_in_chunk_count = 0
        self._last_echo_stat_log_at = 0.0
        self._last_aec_diag_log_at = 0.0
        self._last_aec_success_log_at = 0.0
        self._last_echo_drop_reported = 0
        self._last_barge_in_reported = 0
        self._aec_diag_stats = self._empty_aec_diag_stats()
        self._playback_reference_armed = False
        self._reference_warmup_until = 0.0
        self._cached_echo_reference: bytes = b""
        self._cached_reference_at = 0.0
        self._latency_candidate_samples = 0
        self._latency_candidate_hits = 0
        self._latency_last_committed = int(self._guard_calibration.get("latency_samples", 0) or 0)

        # Level signals are throttled to avoid saturating the Qt event loop.
        self._last_in_level: float = 0.0
        self._last_out_level: float = 0.0
        self._last_level_emit_t: float = 0.0

        # True while waiting for server transcription completion.
        self._awaiting_transcript = False

        # Best-effort current UI state.
        self._ui_state = _STATE_IDLE

    @Slot()
    def reload_guard_profile(self) -> None:
        self._guard_profile = _audio_guard_profile(self.settings)
        self._emit_guard_debug()

    def _emit_guard_debug(self) -> None:
        snapshot = self._guard_telemetry.snapshot(window_s=15.0)
        payload = {
            "state": self._guard_adaptor.state,
            "profile": dict(self._guard_profile),
            "telemetry": snapshot,
            "audio_mode": self._audio_mode,
            "aec_aware": self._guard_aec_aware,
            "learning_mode": time.monotonic() < self._guard_learning_until,
            "native_aec_available": self._native_aec_probe.available,
            "native_aec_reason": self._native_aec_probe.reason,
            "calibration": dict(self._guard_calibration),
            "input_device_name": self._input_device_name,
            "output_device_name": self._output_device_name,
        }
        self.guard_debug_updated.emit(payload)

    @staticmethod
    def _empty_aec_diag_stats() -> dict[str, float]:
        return {
            "samples": 0.0,
            "double_talk": 0.0,
            "low_quality": 0.0,
            "reference_miss": 0.0,
            "reference_ready": 0.0,
            "aligned": 0.0,
            "strong_aligned": 0.0,
            "residual_sum": 0.0,
            "quality_sum": 0.0,
            "aligned_residual_sum": 0.0,
            "aligned_quality_sum": 0.0,
            "delay_error_sum": 0.0,
            "max_delay_error": 0.0,
        }

    def _record_aec_diag_sample(
        self,
        *,
        residual_level: float,
        aec_quality: float,
        double_talk: bool,
        delay_samples: int,
        similarity: float,
        reference_miss: bool,
    ) -> None:
        stats = self._aec_diag_stats
        stats["samples"] += 1.0
        stats["residual_sum"] += float(residual_level)
        stats["quality_sum"] += float(aec_quality)
        if double_talk:
            stats["double_talk"] += 1.0
        if aec_quality < 0.18:
            stats["low_quality"] += 1.0
        if reference_miss:
            stats["reference_miss"] += 1.0
        else:
            stats["reference_ready"] += 1.0
        if similarity >= 0.2:
            stats["aligned"] += 1.0
            stats["aligned_residual_sum"] += float(residual_level)
            stats["aligned_quality_sum"] += float(aec_quality)
        if similarity >= 0.5:
            stats["strong_aligned"] += 1.0
        calibration_latency = int(self._guard_calibration.get("latency_samples", 0) or 0)
        delay_error = abs(int(delay_samples) - calibration_latency) if calibration_latency > 0 and delay_samples > 0 else 0
        stats["delay_error_sum"] += float(delay_error)
        stats["max_delay_error"] = max(float(stats["max_delay_error"]), float(delay_error))

    def _build_aec_summary(self) -> dict[str, float]:
        stats = dict(self._aec_diag_stats)
        samples = max(1.0, float(stats.get("samples", 0.0) or 0.0))
        aligned = max(1.0, float(stats.get("aligned", 0.0) or 0.0))
        return {
            "samples": int(stats.get("samples", 0.0) or 0.0),
            "avg_residual": float(stats.get("residual_sum", 0.0) or 0.0) / samples,
            "avg_quality": float(stats.get("quality_sum", 0.0) or 0.0) / samples,
            "double_talk_ratio": float(stats.get("double_talk", 0.0) or 0.0) / samples,
            "low_quality_ratio": float(stats.get("low_quality", 0.0) or 0.0) / samples,
            "reference_miss_ratio": float(stats.get("reference_miss", 0.0) or 0.0) / samples,
            "reference_ready_ratio": float(stats.get("reference_ready", 0.0) or 0.0) / samples,
            "aligned_ratio": float(stats.get("aligned", 0.0) or 0.0) / samples,
            "strong_alignment_ratio": float(stats.get("strong_aligned", 0.0) or 0.0) / samples,
            "avg_quality_when_aligned": float(stats.get("aligned_quality_sum", 0.0) or 0.0) / aligned,
            "avg_residual_when_aligned": float(stats.get("aligned_residual_sum", 0.0) or 0.0) / aligned,
            "avg_delay_error": float(stats.get("delay_error_sum", 0.0) or 0.0) / samples,
            "max_delay_error": float(stats.get("max_delay_error", 0.0) or 0.0),
        }

    def _set_state(self, s: str) -> None:
        normalized = (s or _STATE_IDLE).strip().lower()
        if normalized not in _VALID_STATES:
            normalized = _STATE_IDLE
        self._ui_state = normalized
        self.state_changed.emit(normalized)

    @staticmethod
    def _pcm16_level(pcm: bytes) -> float:
        """Quick 0..1 loudness estimate from PCM16 mono bytes."""
        if not pcm:
            return 0.0
        try:
            import numpy as _np

            x = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32)
            if x.size == 0:
                return 0.0
            x = x / 32768.0
            rms = float(_np.sqrt(_np.mean(x * x) + 1e-12))
            peak = float(_np.max(_np.abs(x)))
            lvl = max(rms * 2.2, peak * 1.1)
            return float(max(0.0, min(1.0, lvl)))
        except Exception:
            return 0.0

    def _preferred_frame_size(self) -> int:
        value = int(self._guard_calibration.get("preferred_frame_size", 480) or 480)
        return max(240, min(1920, value))

    def _current_device_fingerprint(self) -> str:
        return build_device_fingerprint(self._resolved_input_device, self._resolved_output_device, 24000)

    def _configure_aec_from_calibration(self) -> None:
        calibration = self._guard_calibration or {}
        filter_length = int(calibration.get("filter_length", self._aec.filter_length) or self._aec.filter_length)
        latency = int(calibration.get("latency_samples", self._aec.last_shift) or self._aec.last_shift)
        self._aec.configure(
            filter_length=filter_length,
            max_shift_samples=max(960, latency + max(240, filter_length // 2)),
        )
        if latency > 0:
            self._aec._last_shift = latency
            self._latency_last_committed = latency

    def _consider_runtime_latency_update(
        self,
        *,
        delay_samples: int,
        similarity: float,
        aec_quality: float,
        improvement_ratio: float,
        backend: str,
        double_talk: bool,
        prefer_webrtc: bool = False,
    ) -> None:
        if delay_samples <= 0 or double_talk:
            self._latency_candidate_hits = 0
            return
        committed = int(self._guard_calibration.get("latency_samples", self._latency_last_committed) or self._latency_last_committed or 0)
        if committed <= 0:
            committed = int(delay_samples)
        webrtc_close_window = max(96, self._aec.filter_length // (4 if prefer_webrtc else 5))
        stable_candidate = bool(
            (
                backend == "webrtc"
                and similarity >= (0.42 if prefer_webrtc else 0.5)
                and improvement_ratio >= (0.08 if prefer_webrtc else 0.12)
                and aec_quality >= (0.04 if prefer_webrtc else 0.05)
                and abs(int(delay_samples) - committed) <= webrtc_close_window
            )
            or (similarity >= 0.55 and aec_quality >= 0.08)
        )
        if not stable_candidate:
            self._latency_candidate_hits = max(0, self._latency_candidate_hits - 1)
            return

        max_step = max(160, self._aec.filter_length // 2)
        if backend == "webrtc" and committed > 0 and abs(int(delay_samples) - committed) > max_step + 96:
            self._latency_candidate_hits = max(0, self._latency_candidate_hits - 1)
            return
        if abs(int(delay_samples) - committed) > max_step:
            target = committed + max_step if delay_samples > committed else committed - max_step
        else:
            target = int(delay_samples)

        if self._latency_candidate_samples <= 0 or abs(int(target) - int(self._latency_candidate_samples)) > 48:
            self._latency_candidate_samples = int(target)
            self._latency_candidate_hits = 1
            return

        self._latency_candidate_samples = int(round((self._latency_candidate_samples * 0.6) + (int(target) * 0.4)))
        self._latency_candidate_hits += 1
        required_hits = 3 if (backend == "webrtc" and prefer_webrtc) else (4 if backend == "webrtc" else 3)
        if self._latency_candidate_hits < required_hits:
            return

        new_latency = int(self._latency_candidate_samples)
        self._guard_calibration["latency_samples"] = new_latency
        self._latency_last_committed = new_latency
        self._latency_candidate_hits = 0

    def _reference_needed_samples(self, chunk_bytes: int) -> int:
        return max(int(chunk_bytes) // 2 + max(96, self._aec.filter_length // 6), 640)

    def _is_reference_ready(
        self,
        *,
        now_monotonic: float,
        reference_needed: int,
        available_samples: int,
        played_samples: int,
        callback_age_ms: int,
    ) -> bool:
        if not self._playback_reference_armed:
            return False
        enough_headroom = available_samples >= reference_needed + max(240, reference_needed // 4)
        warmup_done = bool(
            now_monotonic >= self._reference_warmup_until
            or played_samples >= max(240, reference_needed // 4)
            or enough_headroom
        )
        return bool(
            warmup_done
            and available_samples >= reference_needed
            and (callback_age_ms >= 0 or played_samples >= max(240, reference_needed // 4))
        )

    def _can_use_cached_reference(self, *, now_monotonic: float, reference_needed: int, cached_samples: int) -> bool:
        if cached_samples < max(448, reference_needed - 448):
            return False
        return bool((now_monotonic - self._cached_reference_at) <= 0.5)

    def _ensure_player(self) -> None:
        if self._player is not None:
            return
        self._player = AudioPlayer(
            samplerate=24000,
            device=self._resolved_output_device,
            blocksize=self._preferred_frame_size(),
        )

    def _device_calibration_matches(self) -> bool:
        calibration = self._guard_calibration or {}
        fingerprint = str(calibration.get("device_fingerprint", "")).strip().lower()
        if fingerprint and fingerprint != "unknown":
            return fingerprint == self._current_device_fingerprint().strip().lower()
        input_name = str(calibration.get("input_device_name", "")).strip().lower()
        output_name = str(calibration.get("output_device_name", "")).strip().lower()
        if not input_name or not output_name:
            return False
        return input_name == self._input_device_name.strip().lower() and output_name == self._output_device_name.strip().lower()

    def _apply_device_class_policy(self) -> None:
        profile = dict(self._guard_profile)
        if self._audio_mode in {"wired_headset", "bluetooth_headset", "external_headphones"}:
            profile["echo_similarity_soft"] = max(0.54, profile["echo_similarity_soft"] - 0.05)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] - 0.06)
            profile["playback_activity_level"] = max(0.02, profile["playback_activity_level"] * 0.82)
        elif self._audio_mode in {"external_speakers", "notebook_builtin"}:
            profile["server_vad_threshold"] = min(0.9, profile["server_vad_threshold"] + 0.01)
            profile["barge_in_output_ratio"] = min(1.9, profile["barge_in_output_ratio"] + 0.06)
        self._guard_profile = profile

    def _apply_aec_mode_policy(self) -> None:
        profile = dict(self._guard_profile)
        if self._aec_mode == "webrtc_preferred":
            profile["echo_similarity_soft"] = max(0.58, profile["echo_similarity_soft"] - 0.03)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] - 0.025)
            profile["playback_activity_level"] = max(0.025, profile["playback_activity_level"] * 0.92)
            profile["barge_in_output_ratio"] = max(1.18, profile["barge_in_output_ratio"] * 0.97)
        elif self._aec_mode == "custom_only":
            profile["echo_similarity_soft"] = min(0.76, profile["echo_similarity_soft"] + 0.02)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] + 0.01)
            profile["playback_activity_level"] = min(0.08, profile["playback_activity_level"] * 1.03)
        self._guard_profile = profile

    def _apply_saved_calibration(self) -> None:
        if not self._device_calibration_matches():
            self._apply_device_class_policy()
            self._apply_aec_mode_policy()
            return
        saved_profile = self._guard_calibration.get("profile")
        if isinstance(saved_profile, dict) and saved_profile:
            merged = dict(self.settings.normalized_audio_guard_profile())
            merged.update({key: float(value) for key, value in saved_profile.items() if key in DEFAULT_AUDIO_GUARD_PROFILE})
            self._guard_profile = merged
        self._configure_aec_from_calibration()
        self._apply_aec_mode_policy()

    def _resolve_audio_devices(self) -> None:
        """Pick stable defaults for laptop mic + speakers.

        Produkt běží na systémových výchozích zařízeních, případně na interní
        heuristice, pokud default selže.
        """
        in_dev, in_note = pick_audio_device("input", None)
        out_dev, out_note = pick_audio_device("output", None)
        self._resolved_input_device = in_dev
        self._resolved_output_device = out_dev

        # Best-effort show chosen device names (for troubleshooting).
        try:
            in_name = sd.query_devices(in_dev, "input")["name"] if in_dev is not None else "default"
        except Exception:
            in_name = "(neznámé)"
        try:
            out_name = sd.query_devices(out_dev, "output")["name"] if out_dev is not None else "default"
        except Exception:
            out_name = "(neznámé)"
        self._input_device_name = str(in_name)
        self._output_device_name = str(out_name)
        combined_names = f"{self._input_device_name} {self._output_device_name}".lower()
        self._audio_mode = "notebook_builtin"
        if any(token in combined_names for token in ("bluetooth", "airpods", "buds", "hands-free")):
            self._audio_mode = "bluetooth_headset"
        elif any(token in combined_names for token in ("headphone", "headphones", "headset", "earbuds")):
            self._audio_mode = "wired_headset"
        elif any(token in combined_names for token in ("speaker", "speakers", "monitor", "hdmi", "display audio", "dock")) and not any(token in combined_names for token in ("built-in", "builtin", "internal", "laptop", "notebook")):
            self._audio_mode = "external_speakers"
        self._apply_saved_calibration()
        self._append_caption(f"Mic: {in_dev if in_dev is not None else 'Default'} – {in_name}")
        self._append_caption(f"Spk: {out_dev if out_dev is not None else 'Default'} – {out_name}")

        # Surface auto-selection in captions so troubleshooting is simple.
        notes = []
        if in_note != "selected:settings":
            notes.append(f"mic:{in_note}")
        if out_note != "selected:settings":
            notes.append(f"spk:{out_note}")
        if notes:
            self._append_caption("Audio: " + ", ".join(notes))

    def _append_caption(self, line: str) -> None:
        self._captions = (self._captions + "\n" + line).strip()
        self._captions = "\n".join(self._captions.splitlines()[-12:])
        self.captions_updated.emit(self._captions)

    def _log_event(self, record_type: str, **extra) -> None:
        if not self._logger:
            return
        payload = {"type": record_type}
        payload.update(extra)
        if record_type == "aec_diag" and "text_line" not in payload:
            payload["text_line"] = (
                "AEC"
                f" sim={float(payload.get('similarity', 0.0) or 0.0):.3f}"
                f" residual={float(payload.get('residual_level', 0.0) or 0.0):.4f}"
                f" q={float(payload.get('aec_quality', 0.0) or 0.0):.3f}"
                f" pred={float(payload.get('predicted_level', 0.0) or 0.0):.4f}"
                f" improve={float(payload.get('improvement_ratio', 0.0) or 0.0):.3f}"
                f" backend={str(payload.get('backend', 'custom') or 'custom')}"
                f" sel={str(payload.get('selection_reason', 'custom_fallback') or 'custom_fallback')}"
                f" ws={'on' if bool(payload.get('webrtc_success')) else 'off'}"
                f" native={'on' if bool(payload.get('native_selected')) else 'off'}"
                f" natry={'on' if bool(payload.get('native_attempted')) else 'off'}"
                f" dt={'on' if bool(payload.get('double_talk')) else 'off'}"
                f" voice={float(payload.get('voice_likelihood', 0.0) or 0.0):.3f}"
                f" delay={int(payload.get('delay_samples', 0) or 0)}"
                f" calib={int(payload.get('calibration_latency', 0) or 0)}"
                f" ref={int(payload.get('reference_available', 0) or 0)}"
                f" age_ms={int(payload.get('reference_callback_age_ms', -1) or -1)}"
                f" miss={'on' if bool(payload.get('reference_miss')) else 'off'}"
            )
        self._logger.append(payload)
        if self._logger.last_error:
            self._append_caption(f"Logování: {self._logger.last_error}")

    def _log_conversation_text(self, record_type: str, text: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        self._log_event(record_type, chars=len(normalized))

    def _set_caption_preview(self, prefix: str, text: str) -> None:
        base = self._captions.splitlines()[-11:]
        preview = (text or "").replace("\n", " ").strip()
        self.captions_updated.emit("\n".join(base + [f"{prefix}: {preview}"]))

    def _start_session_if_needed(self) -> None:
        if self._logger:
            return
        self._captions = ""
        self.captions_updated.emit(self._captions)
        self._reconnect_attempts = 0
        self._next_reconnect_at = 0.0
        self._response_started_at = None
        self._response_first_audio_at = None
        self._speech_stopped_at = None
        self._last_server_activity_at = time.monotonic()
        self._last_backlog_log_at = 0.0
        self._last_player_progress_at = time.monotonic()
        self._last_player_buffer_bytes = 0
        self._echo_drop_count = 0
        self._barge_in_chunk_count = 0
        self._last_echo_stat_log_at = 0.0
        self._last_aec_diag_log_at = 0.0
        self._last_aec_success_log_at = 0.0
        self._last_echo_drop_reported = 0
        self._last_barge_in_reported = 0
        self._aec_diag_stats = self._empty_aec_diag_stats()
        self._cached_echo_reference = b""
        self._cached_reference_at = 0.0
        self._guard_telemetry = GuardTelemetry()
        self._aec.reset()
        self._guard_learning_until = time.monotonic() + 30.0
        log_dir = self.settings.validate_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        self._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)
        self._session_log_dir = log_dir
        self._session_name = session_name
        self._append_caption(f"Log: {str(self._logger.jsonl_path)}")

        self._log_event(
            "session_start",
            settings={
                "openai_base_url": "wss://api.openai.com/v1/realtime",
                "realtime_model": _REALTIME_MODEL,
                "answer_language_mode": self.settings.answer_language_mode,
                "fixed_answer_language": self.settings.fixed_answer_language,
                "response_style": self.settings.response_style,
                "audio_aec_mode": self._aec_mode,
                "tts_voice": _TTS_VOICE,
                "tts_speed": _TTS_SPEED,
                "audio": {
                    "input_device": self._resolved_input_device,
                    "output_device": self._resolved_output_device,
                    "input_device_name": self._input_device_name,
                    "output_device_name": self._output_device_name,
                    "audio_mode": self._audio_mode,
                    "calibration": dict(self._guard_calibration),
                },
            },
        )
        self._append_caption(
            f"Relace: model={_REALTIME_MODEL}, hlas={_TTS_VOICE}, jazyk={self.settings.answer_language_mode}, styl={self.settings.response_style}"
        )

    def _end_session(self) -> None:
        if not self._logger:
            return
        telemetry = self._guard_telemetry.snapshot(window_s=60.0)
        aec_summary = self._build_aec_summary()
        self._log_event(
            "session_end_guard",
            profile=self._guard_profile,
            telemetry=telemetry,
            guard_state=self._guard_adaptor.state,
            aec_aware=self._guard_aec_aware,
            aec_summary=aec_summary,
        )
        self._log_event("session_end", dropped_records=self._logger.dropped_records, last_error=self._logger.last_error)
        if int(aec_summary.get("samples", 0) or 0) > 0:
            self._log_event(
                "aec_summary",
                **{
                    key: (round(float(value), 4) if isinstance(value, float) else value)
                    for key, value in aec_summary.items()
                },
            )
            self._append_caption(
                "AEC summary: "
                f"samples={int(aec_summary['samples'])} "
                f"q={float(aec_summary['avg_quality']):.3f} "
                f"residual={float(aec_summary['avg_residual']):.4f} "
                f"dt={float(aec_summary['double_talk_ratio']):.3f} "
                f"delay_err={float(aec_summary['avg_delay_error']):.1f}"
            )
        try:
            self.settings.audio_guard_profile = self.settings.normalized_audio_guard_profile() | {
                key: float(value) for key, value in self._guard_profile.items() if key in DEFAULT_AUDIO_GUARD_PROFILE
            }
            self.settings.audio_guard_calibration = dict(self._guard_calibration)
            self.settings.save()
        except Exception:
            pass
        try:
            if self._session_log_dir is not None:
                append_guard_replay_metrics(
                    self._session_log_dir,
                    {
                        "session": self._session_name,
                        "audio_mode": self._audio_mode,
                        "guard_state": self._guard_adaptor.state,
                        "aec_aware": self._guard_aec_aware,
                        "profile": {key: round(float(value), 5) for key, value in self._guard_profile.items()},
                        "telemetry": telemetry,
                    },
                )
        except Exception:
            pass
        try:
            self._logger.close()
        except Exception:
            pass
        self._logger = None
        self._session_log_dir = None
        self._session_name = ""

    def _ensure_realtime(self, turn_mode: str) -> RealtimeService:
        if not self.settings.openai_api_key:
            raise ValueError("Chybí API key")
        self._rt_turn_mode = turn_mode

        self._resolved_lang = self.settings.fixed_answer_language if self.settings.fixed_answer_language in _ALLOWED_LANGS else "cs"
        instructions = build_system_prompt(self.settings, self._resolved_lang)

        cfg = RealtimeConfig(
            api_key=self.settings.openai_api_key,
            model=_REALTIME_MODEL,
            instructions=instructions,
            voice=_TTS_VOICE,
            language_hint="auto",
            turn_mode=turn_mode,
            auto_interrupt=True,
            noise_reduction=_NOISE_REDUCTION,
            output_speed=_TTS_SPEED,
            server_vad_silence_ms=_SERVER_VAD_SILENCE_MS,
            server_vad_prefix_ms=_SERVER_VAD_PREFIX_MS,
            server_vad_threshold=float(self._guard_profile["server_vad_threshold"]),
        )

        if self._rt is None or not self._rt.is_connected:
            # Znovu vytvorit websocket po odpojeni.
            self._rt = RealtimeService(cfg)
            self._wire_realtime_callbacks(self._rt)
            self._set_state(_STATE_CONNECTING if self._reconnect_attempts == 0 else _STATE_RECONNECTING)
            self._rt.connect()
            self._reconnect_attempts = 0
            self._next_reconnect_at = 0.0
            self._last_server_activity_at = time.monotonic()
            return self._rt

        # Same websocket; update session settings.
        # Update extra audio/session knobs as well (update_session only touches a subset).
        self._rt.cfg.noise_reduction = _NOISE_REDUCTION
        self._rt.cfg.output_speed = _TTS_SPEED
        self._rt.cfg.server_vad_silence_ms = _SERVER_VAD_SILENCE_MS
        self._rt.cfg.server_vad_prefix_ms = _SERVER_VAD_PREFIX_MS
        self._rt.cfg.server_vad_threshold = float(self._guard_profile["server_vad_threshold"])
        self._rt.update_session(
            instructions=instructions,
            voice=_TTS_VOICE,
            language_hint="auto",
            turn_mode=turn_mode,
        )
        return self._rt

    def _wire_realtime_callbacks(self, rt: RealtimeService) -> None:
        def _status(msg: str) -> None:
            self._last_server_activity_at = time.monotonic()
            self._append_caption(msg)

        rt.on_status = _status

        def _is_recoverable_realtime_error(msg: str) -> bool:
            text = (msg or "").lower()
            markers = ("timed out", "timeout", "connection", "socket", "reset", "closed", "disconnect", "broken pipe")
            return any(marker in text for marker in markers)

        def _err(msg: str) -> None:
            safe_msg = _sanitize_text(msg)
            self._log_event("error", message=safe_msg)
            if self._mode != "idle" and _is_recoverable_realtime_error(msg):
                self._schedule_reconnect(safe_msg)
                return
            self._stop_realtime_session()
            self._set_state(_STATE_ERROR)
            self.error.emit(safe_msg)

        rt.on_error = _err

        def _user(t: str) -> None:
            self._last_server_activity_at = time.monotonic()
            self._append_caption(f"Ty: {t}")
            self._log_conversation_text("user", t)

            # Transition from "transcribing" to "thinking" once we have a transcript.
            self._awaiting_transcript = False
            if self._ui_state not in {_STATE_SPEAKING, _STATE_ERROR}:
                self._set_state(_STATE_THINKING)
            self._response_started_at = time.monotonic()

        rt.on_user_transcript = _user

        rt.on_assistant_text_delta = lambda d: self._set_caption_preview("AI", d)

        def _ai_done(t: str) -> None:
            self._last_server_activity_at = time.monotonic()
            self._append_caption(f"AI: {t}")
            self._log_conversation_text("assistant", t)

        rt.on_assistant_text_done = _ai_done

        def _audio(pcm: bytes) -> None:
            # Audio deltas arrive faster than realtime; enqueue and let the player drain.
            self._last_server_activity_at = time.monotonic()
            self._set_state(_STATE_SPEAKING)
            if self._response_first_audio_at is None:
                self._response_first_audio_at = time.monotonic()
                latency_ms = None
                if self._response_started_at is not None:
                    latency_ms = int((self._response_first_audio_at - self._response_started_at) * 1000)
                self._log_event("assistant_audio_first_delta", latency_ms=latency_ms, bytes=len(pcm))
            try:
                self._ensure_player()
                if self._player:
                    self._player.enqueue_pcm16(pcm)
            except Exception as e:
                # If playback fails (wrong output device), surface a helpful error.
                self._log_event("error", message=str(e))
                self._stop_realtime_session()
                self._set_state(_STATE_ERROR)
                self.error.emit(_sanitize_text(str(e)) + "\n\n" + format_device_help())

        rt.on_assistant_audio_delta = _audio

        def _speech_started() -> None:
            # Barge-in: stop local playback immediately.
            try:
                if self._player:
                    self._player.stop()
            except Exception:
                pass
            self._awaiting_transcript = False
            self._set_state(_STATE_LISTENING)
            self._response_started_at = None
            self._response_first_audio_at = None
            self._speech_stopped_at = None
            self._last_server_activity_at = time.monotonic()
            self._log_event("speech_started")

        rt.on_vad_speech_started = _speech_started

        def _speech_stopped() -> None:
            # Server will emit input_audio_transcription.completed afterwards.
            self._awaiting_transcript = True
            # In handsfree, the server will auto-create the response (create_response=True).
            self._set_state(_STATE_TRANSCRIBING)
            self._speech_stopped_at = time.monotonic()
            self._last_server_activity_at = time.monotonic()
            self._log_event("speech_stopped")

        rt.on_vad_speech_stopped = _speech_stopped

        def _resp_done() -> None:
            # In handsfree mode we keep listening; in PTT return to idle.
            total_latency_ms = None
            if self._speech_stopped_at is not None:
                total_latency_ms = int((time.monotonic() - self._speech_stopped_at) * 1000)
            self._log_event("response_done", total_latency_ms=total_latency_ms)
            if self._mode == "handsfree":
                self._set_state(_STATE_LISTENING)
            else:
                self._set_state(_STATE_IDLE)
            self._awaiting_transcript = False
            self._response_started_at = None
            self._response_first_audio_at = None
            self._speech_stopped_at = None
            self._last_server_activity_at = time.monotonic()

        rt.on_response_done = _resp_done

    def _schedule_reconnect(self, reason: str) -> None:
        self._reconnect_attempts += 1
        delay = min(8.0, 0.8 * (2 ** max(0, self._reconnect_attempts - 1)))
        self._next_reconnect_at = time.monotonic() + delay
        self._append_caption(f"Realtime: plánuju reconnect za {delay:.1f} s")
        self._log_event("reconnect_scheduled", reason=reason, attempt=self._reconnect_attempts, delay_s=delay)
        try:
            if self._rt:
                self._rt.close()
        except Exception:
            pass
        self._rt = None
        self._set_state(_STATE_RECONNECTING)

    def _attempt_reconnect_if_needed(self) -> None:
        if self._mode == "idle":
            return
        if self._rt is not None and self._rt.is_connected:
            return
        if self._next_reconnect_at and time.monotonic() < self._next_reconnect_at:
            return
        try:
            self._append_caption("Realtime: obnovuji spojení…")
            self._ensure_realtime(self._rt_turn_mode)
            self._log_event("reconnect_ok", attempt=self._reconnect_attempts)
            if self._mode == "handsfree" and self._mic and not self._mic_enabled.is_set():
                self._mic_enabled.set()
            if self._mode == "handsfree":
                self._set_state(_STATE_LISTENING)
        except Exception as exc:
            safe_exc = _sanitize_text(str(exc))
            self._log_event("reconnect_failed", message=safe_exc, attempt=self._reconnect_attempts)
            if self._reconnect_attempts >= 5:
                self._stop_realtime_session()
                self._set_state(_STATE_ERROR)
                self.error.emit(f"Realtime se nepodařilo obnovit: {safe_exc}")
                return
            self._schedule_reconnect(safe_exc)

    def _check_runtime_health(self) -> None:
        now = time.monotonic()
        pending_events = self._rt.pending_event_count if self._rt else 0
        pending_mic = self._mic.pending_chunk_count if self._mic else 0
        pending_player_bytes = self._player.buffered_bytes if self._player else 0

        if (
            now - self._last_backlog_log_at >= 5.0
            and (pending_events > 0 or pending_mic > 0 or pending_player_bytes > 0)
        ):
            self._log_event(
                "backlog",
                rt_events=pending_events,
                mic_chunks=pending_mic,
                player_bytes=pending_player_bytes,
            )
            self._last_backlog_log_at = now

        if (
            now - self._last_echo_stat_log_at >= 5.0
            and (
                self._echo_drop_count != self._last_echo_drop_reported
                or self._barge_in_chunk_count != self._last_barge_in_reported
            )
        ):
            self._log_event(
                "echo_guard",
                dropped_echo_chunks=self._echo_drop_count,
                barge_in_chunks=self._barge_in_chunk_count,
            )
            self._last_echo_stat_log_at = now
            self._last_echo_drop_reported = self._echo_drop_count
            self._last_barge_in_reported = self._barge_in_chunk_count

        if self._player:
            if pending_player_bytes != self._last_player_buffer_bytes:
                self._last_player_progress_at = now
                self._last_player_buffer_bytes = pending_player_bytes
            elif pending_player_bytes > 0 and now - self._last_player_progress_at > 8.0:
                self._log_event("watchdog", message="audio playback stagnuje", buffered_bytes=pending_player_bytes)
                try:
                    self._player.stop()
                except Exception:
                    pass
                self._last_player_progress_at = now
                self._last_player_buffer_bytes = 0

        if (
            self._mode != "idle"
            and self._rt is not None
            and self._rt.is_connected
            and self._ui_state in {_STATE_CONNECTING, _STATE_RECONNECTING, _STATE_TRANSCRIBING, _STATE_THINKING}
            and now - self._last_server_activity_at > 25.0
        ):
            self._log_event("watchdog", message="realtime bez aktivity", state=self._ui_state)
            self._schedule_reconnect("watchdog: realtime bez aktivity")

    def _start_rt_loop(self) -> None:
        if self._rt_loop_thread and self._rt_loop_thread.is_alive():
            return
        self._rt_loop_stop.clear()

        def loop() -> None:
            while not self._rt_loop_stop.is_set():
                self._attempt_reconnect_if_needed()
                if self._rt:
                    self._rt.pump_events()
                self._check_runtime_health()
                if time.monotonic() - self._guard_last_adapt_at >= 1.2:
                    telemetry_snapshot = self._guard_telemetry.snapshot(window_s=15.0)
                    self._guard_aec_aware = (
                        float(telemetry_snapshot.get("playback_ratio", 0.0) or 0.0) > 0.22
                        and float(telemetry_snapshot.get("avg_output", 0.0) or 0.0) > 0.028
                        and float(telemetry_snapshot.get("avg_similarity", 0.0) or 0.0) < 0.08
                    )
                    adaptation = self._guard_adaptor.adapt(
                        self._guard_profile,
                        telemetry_snapshot,
                        learning_mode=time.monotonic() < self._guard_learning_until,
                        aec_aware=self._guard_aec_aware,
                    )
                    self._guard_profile = adaptation.profile
                    self._guard_last_adapt_at = time.monotonic()
                    self._emit_guard_debug()

                now_monotonic = time.monotonic()
                current_out_level = 0.0
                is_playing_out = False
                if self._player is not None:
                    try:
                        current_out_level = float(self._player.get_level())
                        buffered = self._player.buffered_bytes
                    except Exception:
                        current_out_level = 0.0
                        buffered = 0
                    is_playing_out = (
                        buffered > 0
                        or current_out_level > float(self._guard_profile["playback_activity_level"])
                        or self._ui_state == _STATE_SPEAKING
                    )
                    if is_playing_out:
                        if not self._playback_reference_armed:
                            self._playback_reference_armed = True
                            self._reference_warmup_until = now_monotonic + 0.03
                        self._mic_suppressed_until = max(self._mic_suppressed_until, now_monotonic + _ECHO_TRAILING_HOLD_S)
                    else:
                        self._playback_reference_armed = False
                        self._reference_warmup_until = 0.0
                guard_active = (
                    self._mode == "handsfree"
                    and (is_playing_out or now_monotonic < self._mic_suppressed_until)
                )

                # Mic streaming + input level.
                if self._mic_enabled.is_set() and self._mic is not None and self._rt is not None:
                    # Drain a few chunks per tick to reduce backlog.
                    for _ in range(6):
                        try:
                            chunk_item = self._mic.queue.get_nowait()
                        except queue.Empty:
                            break
                        mic_captured_at_mono_ns = 0
                        if hasattr(chunk_item, "pcm_bytes"):
                            chunk = bytes(chunk_item.pcm_bytes)
                            mic_captured_at_mono_ns = int(getattr(chunk_item, "captured_at_mono_ns", 0) or 0)
                        else:
                            chunk = chunk_item
                        if chunk:
                            processed_chunk = chunk
                            aec_result: dict[str, object] = {}
                            similarity = 0.0
                            residual_level = self._pcm16_level(chunk)
                            aec_quality = 0.0
                            double_talk = False
                            raw_voice_likelihood = 0.0
                            predicted_level = 0.0
                            improvement_ratio = 0.0
                            aec_backend = "custom"
                            webrtc_success = False
                            reference_miss = False
                            if guard_active and self._player is not None:
                                try:
                                    previous_calibration_latency = int(self._guard_calibration.get("latency_samples", 0) or 0)
                                    calibration_latency = previous_calibration_latency
                                    reference = self._player.get_echo_reference_for_capture(
                                        max_samples=max(8192, len(chunk) // 2 + 1920),
                                        captured_at_mono_ns=mic_captured_at_mono_ns or None,
                                    )
                                    reference_stats = self._player.get_echo_reference_stats()
                                except Exception:
                                    previous_calibration_latency = 0
                                    calibration_latency = 0
                                    reference = b""
                                    reference_stats = {"available_samples": 0, "total_samples": 0, "played_samples": 0, "callback_age_ms": -1}
                                reference_needed = self._reference_needed_samples(len(chunk))
                                callback_age_ms = int(reference_stats.get("callback_age_ms", -1) or -1)
                                played_samples = int(reference_stats.get("played_samples", 0) or 0)
                                available_samples = int(reference_stats.get("available_samples", 0) or 0)
                                reference_ready = self._is_reference_ready(
                                    now_monotonic=time.monotonic(),
                                    reference_needed=reference_needed,
                                    available_samples=available_samples,
                                    played_samples=played_samples,
                                    callback_age_ms=callback_age_ms,
                                )
                                if not reference_ready and self._cached_echo_reference:
                                    cached_samples = len(self._cached_echo_reference) // 2
                                    if self._can_use_cached_reference(
                                        now_monotonic=time.monotonic(),
                                        reference_needed=reference_needed,
                                        cached_samples=cached_samples,
                                    ):
                                        reference = np.frombuffer(self._cached_echo_reference, dtype=np.int16).copy()
                                        reference_ready = True
                                        available_samples = cached_samples
                                        reference_stats = dict(reference_stats)
                                        reference_stats["available_samples"] = cached_samples
                                        reference_stats["callback_age_ms"] = max(0, callback_age_ms)
                                elif (
                                    not reference_ready
                                    and available_samples <= 0
                                    and self._cached_echo_reference
                                    and played_samples > 0
                                    and (time.monotonic() - self._cached_reference_at) <= 0.35
                                ):
                                    cached_samples = len(self._cached_echo_reference) // 2
                                    if cached_samples >= max(640, len(chunk) // 2):
                                        reference = np.frombuffer(self._cached_echo_reference, dtype=np.int16).copy()
                                        reference_ready = True
                                        available_samples = cached_samples
                                        reference_stats = dict(reference_stats)
                                        reference_stats["available_samples"] = cached_samples
                                        reference_stats["callback_age_ms"] = max(0, callback_age_ms)
                                reference_miss = not reference_ready
                                if reference_ready:
                                    try:
                                        self._cached_echo_reference = np.asarray(reference, dtype=np.int16).astype(np.int16, copy=False).tobytes()
                                        self._cached_reference_at = time.monotonic()
                                    except Exception:
                                        pass
                                    try:
                                        aec_result = self._aec.process(
                                            chunk,
                                            reference,
                                            max_shift_samples=max(960, calibration_latency + max(240, self._aec.filter_length // 2)),
                                            expected_shift=calibration_latency or None,
                                            aec_mode=self._aec_mode,
                                        )
                                        processed_chunk = bytes(aec_result.get("pcm", chunk))
                                        similarity = float(aec_result.get("similarity", 0.0) or 0.0)
                                        residual_level = float(aec_result.get("residual_level", self._pcm16_level(processed_chunk)) or 0.0)
                                        aec_quality = float(aec_result.get("aec_quality", 0.0) or 0.0)
                                        double_talk = bool(aec_result.get("double_talk", False))
                                        raw_voice_likelihood = float(aec_result.get("voice_likelihood", 0.0) or 0.0)
                                        predicted_level = float(aec_result.get("predicted_level", 0.0) or 0.0)
                                        improvement_ratio = float(aec_result.get("improvement_ratio", 0.0) or 0.0)
                                        aec_backend = str(aec_result.get("backend", "custom") or "custom")
                                        webrtc_success = bool(aec_result.get("webrtc_success", False))
                                        native_attempted = bool(aec_result.get("native_attempted", False))
                                        native_selected = bool(aec_result.get("native_selected", False))
                                        selection_reason = str(aec_result.get("selection_reason", "custom_fallback") or "custom_fallback")
                                        delay_samples = int(aec_result.get("delay_samples", 0) or 0)
                                        self._record_aec_diag_sample(
                                            residual_level=residual_level,
                                            aec_quality=aec_quality,
                                            double_talk=double_talk,
                                            delay_samples=delay_samples,
                                            similarity=similarity,
                                            reference_miss=False,
                                        )
                                        can_refresh_latency = bool(
                                            delay_samples > 0
                                            and (
                                                (similarity >= 0.55 and aec_quality >= 0.08)
                                                or (aec_backend == "webrtc" and similarity >= 0.4 and improvement_ratio >= 0.05)
                                            )
                                            and not double_talk
                                            and reference_stats.get("available_samples", 0) >= max(len(chunk) // 2 + 256, 1024)
                                        )
                                        if can_refresh_latency:
                                            self._consider_runtime_latency_update(
                                                delay_samples=delay_samples,
                                                similarity=similarity,
                                                aec_quality=aec_quality,
                                                improvement_ratio=improvement_ratio,
                                                backend=aec_backend,
                                                double_talk=double_talk,
                                                prefer_webrtc=self._aec_mode == "webrtc_preferred",
                                            )
                                        elif webrtc_success and delay_samples > 0 and not double_talk:
                                            self._consider_runtime_latency_update(
                                                delay_samples=delay_samples,
                                                similarity=max(0.4, similarity),
                                                aec_quality=max(0.08, aec_quality),
                                                improvement_ratio=improvement_ratio,
                                                backend=aec_backend,
                                                double_talk=False,
                                                prefer_webrtc=self._aec_mode == "webrtc_preferred",
                                            )
                                    except Exception:
                                        processed_chunk = chunk
                                        similarity = 0.0
                                        residual_level = self._pcm16_level(chunk)
                                        aec_quality = 0.0
                                        double_talk = False
                                        raw_voice_likelihood = 0.0
                                        predicted_level = 0.0
                                        improvement_ratio = 0.0
                                        aec_backend = "custom"
                                        webrtc_success = False
                                        native_attempted = False
                                        native_selected = False
                                        selection_reason = "custom_fallback"
                                        previous_calibration_latency = calibration_latency
                                        reference_stats = {"available_samples": 0, "total_samples": 0, "callback_age_ms": -1}
                                        reference_miss = True
                                else:
                                    predicted_level = 0.0
                                    improvement_ratio = 0.0
                                    aec_backend = "custom"
                                    webrtc_success = False
                                    native_attempted = False
                                    native_selected = False
                                    selection_reason = "custom_fallback"
                                    self._record_aec_diag_sample(
                                        residual_level=residual_level,
                                        aec_quality=0.0,
                                        double_talk=False,
                                        delay_samples=0,
                                        similarity=0.0,
                                        reference_miss=True,
                                    )
                                if reference_ready:
                                    pass
                            # Update last input VU level.
                            in_level = self._pcm16_level(processed_chunk)
                            processed_voice_likelihood = estimate_voice_likelihood_from_pcm16(processed_chunk)
                            if is_playing_out:
                                playback_safe_raw_voice = raw_voice_likelihood * (0.45 if (double_talk or aec_quality > 0.22) else 0.2)
                                voice_likelihood = max(processed_voice_likelihood, playback_safe_raw_voice)
                            else:
                                voice_likelihood = max(raw_voice_likelihood, processed_voice_likelihood)
                            effective_aec_quality = max(aec_quality, 0.1 if webrtc_success else 0.0)
                            drop_chunk, drop_reason = _should_drop_mic_chunk(
                                mode=self._mode,
                                guard_active=guard_active,
                                playback_active=is_playing_out,
                                similarity=similarity,
                                input_level=max(in_level, voice_likelihood * 0.45),
                                output_level=current_out_level,
                                profile=self._guard_profile,
                                residual_level=residual_level,
                                voice_likelihood=voice_likelihood,
                                double_talk=double_talk,
                                aec_quality=effective_aec_quality,
                            )
                            delay_drift = abs(int(self._guard_calibration.get("latency_samples", 0) or 0) - int(aec_result.get("delay_samples", 0) or 0))
                            now_for_diag = time.monotonic()
                            diag_interval_s = 0.8 if (reference_miss or similarity >= 0.4 or aec_backend == "webrtc" or double_talk) else 5.0
                            should_log_problem_diag = bool(
                                guard_active
                                and now_for_diag - self._last_aec_diag_log_at >= diag_interval_s
                                and (
                                    reference_miss
                                    or similarity >= 0.45
                                    or webrtc_success
                                    or aec_quality < 0.18
                                    or double_talk
                                    or delay_drift > 96
                                )
                            )
                            should_log_success_diag = bool(
                                guard_active
                                and not should_log_problem_diag
                                and now_for_diag - self._last_aec_success_log_at >= 2.0
                                and (
                                    aec_backend == "webrtc"
                                    or webrtc_success
                                    or (not reference_miss and similarity >= 0.2)
                                    or (not reference_miss and predicted_level > 0.0 and improvement_ratio > 0.0)
                                )
                            )
                            if should_log_problem_diag or should_log_success_diag:
                                self._log_event(
                                    "aec_diag",
                                    similarity=round(similarity, 3),
                                    residual_level=round(residual_level, 4),
                                    aec_quality=round(aec_quality, 3),
                                    predicted_level=round(predicted_level, 4),
                                    improvement_ratio=round(improvement_ratio, 3),
                                    backend=aec_backend,
                                    webrtc_success=bool(webrtc_success),
                                    native_attempted=bool(native_attempted),
                                    native_selected=bool(native_selected),
                                    selection_reason=selection_reason,
                                    double_talk=bool(double_talk),
                                    voice_likelihood=round(voice_likelihood, 3),
                                    delay_samples=int(aec_result.get("delay_samples", 0) or 0) if guard_active and self._player is not None else 0,
                                    calibration_latency=int(previous_calibration_latency if guard_active and self._player is not None else self._guard_calibration.get("latency_samples", 0) or 0),
                                    reference_available=int(reference_stats.get("available_samples", 0) or 0),
                                    reference_callback_age_ms=int(reference_stats.get("callback_age_ms", -1) or -1),
                                    reference_miss=bool(reference_miss),
                                )
                                if should_log_problem_diag:
                                    self._last_aec_diag_log_at = now_for_diag
                                else:
                                    self._last_aec_success_log_at = now_for_diag
                            barge_in_candidate = (
                                is_playing_out
                                and voice_likelihood >= 0.42
                                and not drop_chunk
                                and in_level >= max(
                                    float(self._guard_profile["barge_in_min_input_level"]) * 0.8,
                                    current_out_level * (float(self._guard_profile["barge_in_output_ratio"]) * 0.72),
                                )
                            )
                            self._guard_telemetry.add_sample(
                                input_level=in_level,
                                output_level=current_out_level,
                                similarity=similarity,
                                voice_likelihood=voice_likelihood,
                                dropped=drop_chunk,
                                playback_active=is_playing_out,
                                reason=drop_reason,
                                barge_in_candidate=barge_in_candidate,
                                residual_level=residual_level,
                                aec_quality=aec_quality,
                                double_talk=double_talk,
                            )
                            if drop_chunk:
                                self._echo_drop_count += 1
                                self._last_in_level = 0.0
                                if self._echo_drop_count <= 3:
                                    self._log_event(
                                        "echo_drop",
                                        reason=drop_reason,
                                        similarity=round(similarity, 3),
                                        input_level=round(in_level, 3),
                                        output_level=round(current_out_level, 3),
                                        voice_likelihood=round(voice_likelihood, 3),
                                    )
                                continue
                            self._last_in_level = in_level
                            if barge_in_candidate:
                                self._barge_in_chunk_count += 1
                            rt = self._rt
                            if rt is None:
                                continue
                            try:
                                rt.append_audio_pcm16(processed_chunk)
                            except AttributeError:
                                continue

                # Output level from the audio callback (reflects actual playback).
                if self._player is not None:
                    try:
                        self._last_out_level = current_out_level
                        out_pose = self._player.get_lipsync_snapshot()
                    except Exception:
                        self._last_out_level = 0.0
                        out_pose = _closed_pose_snapshot()
                else:
                    self._last_out_level = 0.0
                    out_pose = _closed_pose_snapshot()

                # Throttle signals to ~60Hz.
                now = time.time()
                if now - self._last_level_emit_t >= 0.016:
                    in_lvl = self._last_in_level if self._mic_enabled.is_set() else 0.0
                    out_lvl = self._last_out_level
                    try:
                        self.input_level.emit(float(in_lvl))
                        self.output_level.emit(float(out_lvl))
                        self.output_pose.emit(out_pose)
                    except RuntimeError:
                        self._rt_loop_stop.set()
                        break
                    self._last_level_emit_t = now
                time.sleep(0.005)

        self._rt_loop_thread = threading.Thread(target=loop, daemon=True)
        self._rt_loop_thread.start()

    def _stop_rt_loop(self, *, timeout_s: float = 1.0) -> None:
        self._rt_loop_stop.set()
        t = self._rt_loop_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=timeout_s)
        self._rt_loop_thread = None

    def _stop_realtime_session(self) -> None:
        self._mic_enabled.clear()
        self._stop_rt_loop()
        try:
            if self._mic:
                self._mic.stop()
        except Exception:
            pass
        self._mic = None
        try:
            if self._player:
                self._player.stop()
        except Exception:
            pass
        self._player = None
        try:
            if self._rt:
                self._rt.close()
        except Exception:
            pass
        self._rt = None
        self._mode = "idle"
        self._awaiting_transcript = False
        self._reconnect_attempts = 0
        self._next_reconnect_at = 0.0
        self._mic_suppressed_until = 0.0
        self._echo_drop_count = 0
        self._barge_in_chunk_count = 0
        self._end_session()

    @Slot()
    def request_stop(self) -> None:
        self._stop_all.set()
        self._mode = "idle"
        self._mic_enabled.clear()
        self._stop_rt_loop()
        try:
            if self._mic:
                self._mic.stop()
        except Exception:
            pass
        try:
            if self._player:
                self._player.stop()
        except Exception:
            pass
        self._player = None
        try:
            if self._rt:
                self._rt.close()
        except Exception:
            pass
        self._rt = None
        self._reconnect_attempts = 0
        self._next_reconnect_at = 0.0
        self._mic_suppressed_until = 0.0
        self._echo_drop_count = 0
        self._barge_in_chunk_count = 0
        self.input_level.emit(0.0)
        self.output_level.emit(0.0)
        self.output_pose.emit(_closed_pose_snapshot())
        self._set_state(_STATE_IDLE)
        self._emit_guard_debug()
        self._end_session()

    # -------- Hands-free mode --------

    @Slot()
    def start_handsfree(self) -> None:
        try:
            self._mode = "handsfree"
            self._resolve_audio_devices()
            self._start_session_if_needed()
            if self._resolved_input_device is None or self._resolved_output_device is None:
                raise RuntimeError("Nenalezen mikrofon nebo výstupní zařízení.\n\n" + format_device_help())
            self._ensure_player()
            self._set_state(_STATE_CONNECTING)
            rt = self._ensure_realtime("server_vad")
            self._start_rt_loop()
            self._mic = RealtimeMicStream(
                samplerate=24000,
                device=self._resolved_input_device,
                blocksize=self._preferred_frame_size(),
            )
            self._mic.start()
            if getattr(self._mic, "using_resampler", False):
                self._append_caption(
                    f"Mikrofon jede na {self._mic.input_samplerate} Hz, resampluji na 24000 Hz."
                )
            self._mic_enabled.set()
            self._set_state(_STATE_LISTENING)
            self._append_caption("Hands-free: Realtime aktivní (server VAD).")
        except Exception as e:
            self._set_state(_STATE_ERROR)
            self.error.emit(_sanitize_text(str(e)))


    @Slot()
    def ptt_pressed(self) -> None:
        if self._mode == "handsfree":
            return
        try:
            self._mode = "ptt"
            self._resolve_audio_devices()
            self._start_session_if_needed()
            if self._resolved_input_device is None or self._resolved_output_device is None:
                raise RuntimeError("Nenalezen mikrofon nebo výstupní zařízení.\n\n" + format_device_help())
            self._ensure_player()
            self._set_state(_STATE_CONNECTING)
            rt = self._ensure_realtime("ptt")
            self._start_rt_loop()
            rt.clear_input_audio()
            self._mic = RealtimeMicStream(
                samplerate=24000,
                device=self._resolved_input_device,
                blocksize=self._preferred_frame_size(),
            )
            self._mic.start()
            if getattr(self._mic, "using_resampler", False):
                self._append_caption(
                    f"Mikrofon jede na {self._mic.input_samplerate} Hz, resampluji na 24000 Hz."
                )
            self._mic_enabled.set()
            self._set_state(_STATE_LISTENING)
            self._append_caption("PTT: poslouchám…")
        except Exception as e:
            self._set_state(_STATE_ERROR)
            self.error.emit(_sanitize_text(str(e)))


    @Slot()
    def ptt_released(self) -> None:
        if self._mode != "ptt":
            return
        if not self._rt:
            return
        self._mic_enabled.clear()
        try:
            if self._mic:
                self._mic.stop()
        except Exception:
            pass
        # Commit input audio and ask for a response.
        # We show "transcribing" until the server emits the transcript.
        self._awaiting_transcript = True
        self._set_state(_STATE_TRANSCRIBING)
        self._rt.commit_input_audio()
        self._rt.request_response()
        self._append_caption("PTT: čekám na odpověď…")


class MainWindow(QMainWindow):
    sig_start_handsfree = Signal()
    sig_request_stop = Signal()
    sig_reload_guard_profile = Signal()

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self._handsfree_running = False

        self._thread = QThread(self)
        self.worker = ConversationWorker(self.settings)
        self.worker.moveToThread(self._thread)
        self._thread.start()

        self.sig_start_handsfree.connect(self.worker.start_handsfree)
        self.sig_request_stop.connect(self.worker.request_stop)
        self.sig_reload_guard_profile.connect(self.worker.reload_guard_profile)

        self._theme = Theme()

        self.setWindowTitle("Chatbot Kája")
        try:
            assets_dir = Path(__file__).resolve().parent / "resources" / "assets"
            icon_path = assets_dir / "logo_chatbot_kaja.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass

        self._build_ui()
        self._wire()
        self._sync_start_stop_button()
        QTimer.singleShot(0, self._report_render_backend)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout()
        outer.setContentsMargins(18, 14, 18, 16)
        outer.setSpacing(14)
        root.setLayout(outer)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        logo = QLabel()
        logo.setFixedSize(56, 56)
        try:
            assets_dir = Path(__file__).resolve().parent / "resources" / "assets"
            logo_path = assets_dir / "logo_chatbot_kaja.png"
            if logo_path.exists():
                pm = QPixmap(str(logo_path))
                logo.setPixmap(pm.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(0)
        title = QLabel("Chatbot Kája")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"QLabel {{ color: {self._theme.text}; }}")
        subtitle = QLabel("Hlasový asistent s hands-free EKG vizualizací")
        subtitle.setStyleSheet(f"QLabel {{ color: {self._theme.text_muted}; font-size: 12px; }}")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)

        header.addWidget(logo)
        header.addSpacing(12)
        header.addLayout(title_wrap)
        header.addStretch(1)

        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("OpenAI API klíč")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setFixedWidth(250)
        if self.settings.openai_api_key:
            self.api_key_input.setText(self.settings.openai_api_key)

        self.btn_save_key = QPushButton("Uložit klíč")
        self.btn_delete_key = QPushButton("Smazat klíč")
        self.btn_audio_test = QPushButton("Audio test")
        self.btn_audio_test.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        self.btn_start_stop = QPushButton("Start")
        self.btn_settings = QPushButton("Nastavení")
        self.btn_save = QPushButton("Uložit")
        self.btn_clear = QPushButton("Vyčistit relaci")
        self.btn_exit = QPushButton("Konec")

        self.btn_start_stop.setProperty("variant", "primary")
        self.btn_settings.setProperty("variant", "primary")
        self.btn_exit.setProperty("variant", "danger")

        header.addWidget(self.api_key_input)
        header.addWidget(self.btn_save_key)
        header.addWidget(self.btn_delete_key)
        header.addWidget(self.btn_audio_test)
        header.addWidget(self.btn_start_stop)
        header.addWidget(self.btn_settings)
        header.addWidget(self.btn_save)
        header.addWidget(self.btn_clear)
        header.addSpacing(8)
        header.addWidget(self.btn_exit)

        outer.addLayout(header)

        self.status_label = QLabel("Připraveno")
        self.status_label.setStyleSheet(
            "QLabel {"
            "  padding: 10px 14px;"
            "  border-radius: 12px;"
            "  background-color: rgba(255,255,255,8);"
            "  border: 1px solid rgba(255,255,255,16);"
            "  font-size: 12px;"
            "}"
        )
        outer.addWidget(self.status_label)

        self.guard_debug_label = QLabel("")
        self.guard_debug_label.setStyleSheet(
            "QLabel {"
            "  padding: 10px 12px;"
            "  border-radius: 12px;"
            "  background-color: rgba(7,22,18,180);"
            "  border: 1px solid rgba(124,255,141,45);"
            "  color: #9BFFC3;"
            "  font-family: Consolas;"
            "  font-size: 11px;"
            "}"
        )
        self.guard_debug_label.setText(
            "GUARD: čekám na telemetrii\n"
            "stav=-  drop_rate=0.000  similarity=0.000  voice=0.000\n"
            "echo_soft=-  echo_drop=-  barge_in=-  playback=-\n"
            "mode=notebook_builtin  aec=off  learning=off"
        )
        outer.addWidget(self.guard_debug_label)

        head_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "head_photo.png")
        self.head = HeadWidget(head_path)
        self.head.setMinimumSize(520, 520)
        outer.addWidget(self.head, 1)

        if not self.settings.openai_api_key:
            self.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte.")

    def _wire(self) -> None:
        self.btn_exit.clicked.connect(lambda _=False: self.close())
        self.btn_save_key.clicked.connect(lambda _=False: self._save_api_key())
        self.btn_delete_key.clicked.connect(lambda _=False: self._delete_api_key())
        self.btn_audio_test.clicked.connect(lambda _=False: self._run_audio_test())
        self.btn_start_stop.clicked.connect(lambda _=False: self._toggle_handsfree())
        self.btn_settings.clicked.connect(lambda _=False: self._open_settings_dialog())
        self.btn_save.clicked.connect(lambda _=False: self._save_defaults())
        self.btn_clear.clicked.connect(lambda _=False: self._clear_session())

        self.worker.state_changed.connect(self._on_state)
        self.worker.captions_updated.connect(self._on_captions)
        self.worker.error.connect(self._on_error)
        self.worker.input_level.connect(self._on_input_level)
        self.worker.output_level.connect(self._on_output_level)
        self.worker.output_pose.connect(self._on_output_pose)
        self.worker.guard_debug_updated.connect(self._on_guard_debug)

    def _report_render_backend(self) -> None:
        try:
            summary = self.head.render_backend_summary()
        except Exception:
            return
        prefix = "Renderer: GPU" if self.head.is_gpu_renderer_active() else "Renderer: fallback"
        self._append_terminal_line(f"SYS: {prefix} | {summary}")

    def _open_settings_dialog(self) -> None:
        try:
            d = SettingsDialog(self.settings, parent=self)
            if d.exec():
                d.apply()
                self.settings.save()
        except Exception:
            import logging

            logging.getLogger("kajovochat").exception("settings_dialog_failed")
            try:
                QMessageBox.critical(self, "Nastavení", "Nepodařilo se otevřít nastavení. Podrobnosti jsou v logu.")
            except Exception:
                pass

    def _save_api_key(self) -> None:
        key = self.api_key_input.text().strip()
        if not key:
            QMessageBox.warning(self, "API klíč", "Nejdřív zadejte OpenAI API klíč.")
            return
        self.settings.openai_api_key = key
        self.settings.save()
        self._append_terminal_line("SYS: OpenAI API klíč byl bezpečně uložen lokálně.")
        QMessageBox.information(
            self,
            "API klíč",
            "Klíč byl uložen lokálně. Ve Windows se chrání přes DPAPI, na ostatních platformách přes systémový keyring, pokud je dostupný.",
        )

    def _delete_api_key(self) -> None:
        self.api_key_input.clear()
        self.settings.openai_api_key = ""
        self.settings.save()
        self._append_terminal_line("SYS: OpenAI API klíč byl smazán.")
        QMessageBox.information(self, "API klíč", "OpenAI API klíč byl smazán.")

    def _apply_audio_profile(self, profile: dict[str, float], *, source: str) -> None:
        self.settings.audio_guard_profile = {
            key: float(value)
            for key, value in profile.items()
            if key in DEFAULT_AUDIO_GUARD_PROFILE
        }
        self.settings.audio_guard_profile = self.settings.normalized_audio_guard_profile()
        self.settings.save()
        self.sig_reload_guard_profile.emit()
        self._append_terminal_line(
            f"SELFTEST APPLY [{source}]: "
            + ", ".join(f"{key}={value:.3f}" for key, value in self.settings.audio_guard_profile.items())
        )

    def _calibrate_audio_guard(self, *, trigger: str, show_dialog: bool, restart_after: bool) -> dict[str, object]:
        self.status_label.setText("Probíhá automatická kalibrace reproduktoru a mikrofonu.")
        self._append_terminal_line(f"SELFTEST START [{trigger}]")
        result = run_audio_guard_selftest()

        lines = []
        for item in result["checks"]:
            mark = "OK" if item["ok"] else ("WARN" if item.get("non_blocking") else "FAIL")
            lines.append(f"{mark} {item['name']}: {item['detail']}")
            self._append_terminal_line(f"SELFTEST {mark}: {item['name']} | {item['detail']}")

        profile = result.get("profile")
        calibration = result.get("calibration") if isinstance(result.get("calibration"), dict) else {}
        if isinstance(profile, dict) and profile:
            self._apply_audio_profile(profile, source=trigger)
        if calibration:
            in_dev, _ = pick_audio_device("input", None)
            out_dev, _ = pick_audio_device("output", None)
            try:
                input_name = str(sd.query_devices(in_dev, "input").get("name", "default")) if in_dev is not None else "default"
            except Exception:
                input_name = "default"
            try:
                output_name = str(sd.query_devices(out_dev, "output").get("name", "default")) if out_dev is not None else "default"
            except Exception:
                output_name = "default"
            self.settings.audio_guard_calibration = {
                **calibration,
                "profile": dict(profile) if isinstance(profile, dict) else {},
                "input_device_name": input_name,
                "output_device_name": output_name,
                "device_fingerprint": calibration.get("device_fingerprint") or build_device_fingerprint(in_dev, out_dev, 24000),
            }
            self.settings.save()

        if restart_after and result.get("ok"):
            self._handsfree_running = True
            self.head.set_running(True)
            self.status_label.setText("Kalibrace hotová, startuji chat.")
            self._sync_start_stop_button()
            self.sig_start_handsfree.emit()
        else:
            self.status_label.setText("Audio kalibrace doběhla.")

        if show_dialog:
            QMessageBox.information(self, "Audio selftest", "Audio guard selftest\n" + "\n".join(lines))
        return result

    def _save_defaults(self) -> None:
        self.settings.save()
        self._append_terminal_line("SYS: Aktuální nastavení bylo uloženo.")
        QMessageBox.information(self, "SAVE", "Aktuální nastavení bylo uloženo jako výchozí.")

    def _clear_session(self) -> None:
        try:
            self.sig_request_stop.emit()
        except Exception:
            pass
        self._handsfree_running = False
        self._sync_start_stop_button()
        self.head.set_running(False)
        self.head.set_error_text("")
        self.head.set_lipsync_snapshot(_closed_pose_snapshot())
        self.head.set_terminal_text("")
        self.status_label.setText("Relace byla vyčištěna.")
        if not self.settings.openai_api_key:
            self.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte.")

    @Slot()
    def _toggle_handsfree(self) -> None:
        if self._handsfree_running:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.head.set_running(False)
            self.status_label.setText("Hlasový chat zastaven.")
            self._append_terminal_line("SYS: Hlasový chat byl zastaven.")
            self._sync_start_stop_button()
            return

        if not self.settings.openai_api_key and self.api_key_input.text().strip():
            self.settings.openai_api_key = self.api_key_input.text().strip()
            self.settings.save()
            self._append_terminal_line("SYS: OpenAI API klíč byl uložen před startem relace.")

        if not self.settings.openai_api_key:
            self.status_label.setText("Nejdřív uložte OpenAI API klíč.")
            QMessageBox.warning(self, "API klíč", "Chybí OpenAI API klíč.")
            return

        if not isinstance(self.settings.audio_guard_calibration, dict) or not self.settings.audio_guard_calibration:
            self._append_terminal_line("SYS: Chybí uložený audio profil, spouštím krátký preflight.")
            self._calibrate_audio_guard(trigger="preflight", show_dialog=False, restart_after=False)

        self._handsfree_running = True
        self.head.set_running(True)
        self.status_label.setText("Hands-free relace se spouští.")
        self._append_terminal_line("SYS: Hands-free relace se spouští.")
        self._sync_start_stop_button()
        self.sig_start_handsfree.emit()

    @Slot(float)
    def _on_input_level(self, lvl: float) -> None:
        self.head.set_input_level(lvl)

    @Slot(float)
    def _on_output_level(self, lvl: float) -> None:
        self.head.set_output_level(lvl)

    @Slot(object)
    def _on_output_pose(self, snapshot: object) -> None:
        self.head.set_lipsync_snapshot(snapshot)

    @Slot(str)
    def _on_state(self, s: str) -> None:
        self.head.set_state(s)
        if s == "error":
            self._handsfree_running = False
            self.head.set_running(False)
            self._sync_start_stop_button()
        else:
            self.head.set_error_text("")
        self.status_label.setText(f"Stav relace: {s}")

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.head.set_terminal_text(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.head.set_error_text(msg)
        self._handsfree_running = False
        self.head.set_running(False)
        self._sync_start_stop_button()
        self.status_label.setText("Došlo k chybě relace.")
        self._append_terminal_line(f"SYS: ERROR {msg}")

    @Slot(object)
    def _on_guard_debug(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        telemetry = data.get("telemetry", {}) if isinstance(data.get("telemetry"), dict) else {}
        profile = data.get("profile", {}) if isinstance(data.get("profile"), dict) else {}
        state = str(data.get("state", "-"))
        audio_mode = str(data.get("audio_mode", "notebook_builtin"))
        aec_aware = "on" if bool(data.get("aec_aware")) else "off"
        learning_mode = "on" if bool(data.get("learning_mode")) else "off"
        input_name = str(data.get("input_device_name", "-"))
        output_name = str(data.get("output_device_name", "-"))
        self.guard_debug_label.setText(
            "GUARD: živá telemetrie\n"
            f"stav={state}  samples={int(telemetry.get('samples', 0) or 0)}  "
            f"drop_rate={float(telemetry.get('drop_rate', 0.0) or 0.0):.3f}  "
            f"similarity={float(telemetry.get('avg_similarity', 0.0) or 0.0):.3f}  "
            f"voice={float(telemetry.get('avg_voice_likelihood', 0.0) or 0.0):.3f}  "
            f"aec={float(telemetry.get('avg_aec_quality', 0.0) or 0.0):.3f}\n"
            f"echo_soft={float(profile.get('echo_similarity_soft', 0.0) or 0.0):.3f}  "
            f"echo_drop={float(profile.get('echo_similarity_drop', 0.0) or 0.0):.3f}  "
            f"barge_in={float(profile.get('barge_in_min_input_level', 0.0) or 0.0):.3f}  "
            f"playback={float(profile.get('playback_activity_level', 0.0) or 0.0):.3f}\n"
            f"mode={audio_mode}  aec={aec_aware}  learning={learning_mode}  latency={int(data.get('calibration', {}).get('latency_samples', 0) or 0)}\n"
            f"mic={input_name[:42]} | spk={output_name[:42]}"
        )

    def _run_audio_test(self) -> None:
        restart_after = self._handsfree_running
        if restart_after:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.head.set_running(False)
            self._sync_start_stop_button()
            self._append_terminal_line("SYS: Chat byl dočasně zastaven kvůli ruční překalibraci.")
        result = self._calibrate_audio_guard(trigger="manual", show_dialog=True, restart_after=restart_after)
        if restart_after and not result.get("ok"):
            self._handsfree_running = True
            self.head.set_running(True)
            self._sync_start_stop_button()
            self.sig_start_handsfree.emit()
            self._append_terminal_line("SYS: Překalibrace nebyla ideální, chat byl obnoven s posledním profilem.")

    def _sync_start_stop_button(self) -> None:
        self.btn_start_stop.setText("Stop" if self._handsfree_running else "Start")

    def _append_terminal_line(self, line: str) -> None:
        current = list(getattr(self.head, "_terminal_lines", []))
        current.append((line or "").strip())
        self.head.set_terminal_text("\n".join(current[-10:]))

    def closeEvent(self, event) -> None:
        try:
            self.worker.request_stop()
        except Exception:
            pass
        try:
            self._thread.quit()
            self._thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    settings = AppSettings.load()
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        install_app_logging(log_dir=settings.ensure_log_dir(), session_tag=session_tag)
    except Exception:
        pass
    asset_issues = verify_asset_manifest()
    if asset_issues:
        raise RuntimeError("Integrita assetů selhala: " + "; ".join(asset_issues))

    app = QApplication(sys.argv)
    app.setStyleSheet(app_stylesheet())
    w = MainWindow(settings)
    if not settings.openai_api_key:
        QTimer.singleShot(0, lambda: w.head.set_terminal_text("SYS: Chybí OpenAI API klíč. Zadejte ho nahoře a uložte."))
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
