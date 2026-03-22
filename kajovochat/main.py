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
)
from .dialogs.settings_dialog import SettingsDialog
from .services.audio_service import (
    AudioPlayer,
    RealtimeMicStream,
    pick_audio_device,
    format_device_help,
    list_audio_devices,
    calibrate_audio_devices_advanced,
    suppress_echo_from_pcm16,
)
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
) -> tuple[bool, str]:
    active_profile = dict(DEFAULT_AUDIO_GUARD_PROFILE)
    if profile:
        active_profile.update(profile)
    echo_similarity_drop = float(active_profile["echo_similarity_drop"])
    echo_similarity_soft = float(active_profile["echo_similarity_soft"])
    barge_in_min_input_level = float(active_profile["barge_in_min_input_level"])
    barge_in_output_ratio = float(active_profile["barge_in_output_ratio"])

    if mode != "handsfree" or not guard_active:
        return False, ""

    strong_user = (
        input_level >= barge_in_min_input_level
        and input_level >= max(barge_in_min_input_level, output_level * barge_in_output_ratio)
    )
    if similarity >= echo_similarity_drop and not strong_user:
        return True, "echo_similarity"
    if playback_active and similarity >= echo_similarity_soft and input_level <= max(0.045, output_level * 1.10):
        return True, "echo_during_playback"
    if playback_active and output_level >= 0.06 and input_level <= 0.025:
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
        self._guard_telemetry = GuardTelemetry()
        self._guard_adaptor = GuardAdaptor()
        self._guard_last_adapt_at = 0.0
        self._guard_learning_until = 0.0
        self._guard_aec_aware = False
        self._session_log_dir: Optional[Path] = None
        self._session_name: str = ""
        self._input_device_name = "default"
        self._output_device_name = "default"
        self._audio_mode = "notebook_builtin"

        self._stop_all = threading.Event()

        self._captions = ""
        self._logger: Optional[RealtimeLogWriter] = None
        self._player: Optional[AudioPlayer] = None
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
        self._last_echo_drop_reported = 0
        self._last_barge_in_reported = 0

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
            "input_device_name": self._input_device_name,
            "output_device_name": self._output_device_name,
        }
        self.guard_debug_updated.emit(payload)

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

    def _ensure_player(self) -> None:
        if self._player is not None:
            return
        self._player = AudioPlayer(samplerate=24000, device=self._resolved_output_device)

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
        self._audio_mode = "notebook_builtin"
        if any(token in self._output_device_name.lower() for token in ("headphone", "headset", "bluetooth", "airpods")):
            self._audio_mode = "external_headphones"
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
        self._last_echo_drop_reported = 0
        self._last_barge_in_reported = 0
        self._guard_telemetry = GuardTelemetry()
        self._guard_learning_until = time.monotonic() + 30.0
        log_dir = self.settings.validate_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        self._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)
        self._session_log_dir = log_dir
        self._session_name = session_name

        self._log_event(
            "session_start",
            settings={
                "openai_base_url": "wss://api.openai.com/v1/realtime",
                "realtime_model": _REALTIME_MODEL,
                "answer_language_mode": self.settings.answer_language_mode,
                "fixed_answer_language": self.settings.fixed_answer_language,
                "response_style": self.settings.response_style,
                "tts_voice": _TTS_VOICE,
                "tts_speed": _TTS_SPEED,
                "audio": {
                    "input_device": self._resolved_input_device,
                    "output_device": self._resolved_output_device,
                    "input_device_name": self._input_device_name,
                    "output_device_name": self._output_device_name,
                    "audio_mode": self._audio_mode,
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
        self._log_event(
            "session_end_guard",
            profile=self._guard_profile,
            telemetry=telemetry,
            guard_state=self._guard_adaptor.state,
            aec_aware=self._guard_aec_aware,
        )
        self._log_event("session_end", dropped_records=self._logger.dropped_records, last_error=self._logger.last_error)
        try:
            self.settings.audio_guard_profile = self.settings.normalized_audio_guard_profile() | {
                key: float(value) for key, value in self._guard_profile.items() if key in DEFAULT_AUDIO_GUARD_PROFILE
            }
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
                        self._mic_suppressed_until = max(self._mic_suppressed_until, now_monotonic + _ECHO_TRAILING_HOLD_S)
                guard_active = (
                    self._mode == "handsfree"
                    and (is_playing_out or now_monotonic < self._mic_suppressed_until)
                )

                # Mic streaming + input level.
                if self._mic_enabled.is_set() and self._mic is not None and self._rt is not None:
                    # Drain a few chunks per tick to reduce backlog.
                    for _ in range(6):
                        try:
                            chunk = self._mic.queue.get_nowait()
                        except queue.Empty:
                            break
                        if chunk:
                            processed_chunk = chunk
                            similarity = 0.0
                            if guard_active and self._player is not None:
                                try:
                                    reference = self._player.get_echo_reference(max_samples=max(4096, len(chunk) // 2 + 960))
                                except Exception:
                                    reference = b""
                                try:
                                    processed_chunk, similarity = suppress_echo_from_pcm16(
                                        chunk,
                                        reference,
                                        max_shift_samples=960,
                                    )
                                except Exception:
                                    processed_chunk = chunk
                                    similarity = 0.0
                            # Update last input VU level.
                            in_level = self._pcm16_level(processed_chunk)
                            voice_likelihood = estimate_voice_likelihood_from_pcm16(processed_chunk)
                            drop_chunk, drop_reason = _should_drop_mic_chunk(
                                mode=self._mode,
                                guard_active=guard_active,
                                playback_active=is_playing_out,
                                similarity=similarity,
                                input_level=max(in_level, voice_likelihood * 0.45),
                                output_level=current_out_level,
                                profile=self._guard_profile,
                            )
                            barge_in_candidate = (
                                is_playing_out
                                and voice_likelihood >= 0.42
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
                            self._rt.append_audio_pcm16(processed_chunk)

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
        self._stop_rt_loop()
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
        self._stop_rt_loop()
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
            self._mic = RealtimeMicStream(samplerate=24000, device=self._resolved_input_device)
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
            self._mic = RealtimeMicStream(samplerate=24000, device=self._resolved_input_device)
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
        if isinstance(profile, dict) and profile:
            self._apply_audio_profile(profile, source=trigger)

        if restart_after and result.get("ok"):
            self._handsfree_running = True
            self.head.set_running(True)
            self.status_label.setText("Kalibrace hotová, startuji chat.")
            self._sync_start_stop_button()
            self.sig_start_handsfree.emit()
        else:
            self.status_label.setText("Audio kalibrace doběhla.")

        if show_dialog:
            QMessageBox.information(self, "Audio selftest", "Audio guard selftest:\n" + "\n".join(lines))
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
            f"voice={float(telemetry.get('avg_voice_likelihood', 0.0) or 0.0):.3f}\n"
            f"echo_soft={float(profile.get('echo_similarity_soft', 0.0) or 0.0):.3f}  "
            f"echo_drop={float(profile.get('echo_similarity_drop', 0.0) or 0.0):.3f}  "
            f"barge_in={float(profile.get('barge_in_min_input_level', 0.0) or 0.0):.3f}  "
            f"playback={float(profile.get('playback_activity_level', 0.0) or 0.0):.3f}\n"
            f"mode={audio_mode}  aec={aec_aware}  learning={learning_mode}\n"
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
