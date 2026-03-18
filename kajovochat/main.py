from __future__ import annotations

import sys
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
    QSizePolicy,
)

from .settings import (
    AppSettings,
    build_system_prompt,
    LANG_TO_PREFERRED_VOICES,
    TTS_VOICES,
)
from .dialogs.settings_dialog import SettingsDialog
from .dialogs.openai_dialog import OpenAIDialog
from .services.openai_service import OpenAIService
from .services.audio_service import (
    AudioPlayer,
    RealtimeMicStream,
    pick_audio_device,
    format_device_help,
)
from .services.realtime_service import RealtimeConfig, RealtimeService
from .services.log_service import RealtimeLogWriter
from .services.app_logging import install_app_logging
from .resources.assets import verify_asset_manifest
from .widgets.head_widget import HeadWidget
from .widgets.globe_button import GlobeButton
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


def _closed_pose_snapshot() -> dict[str, object]:
    return {
        "pose": "closed",
        "openness": 0.0,
        "energy": 0.0,
        "weights": {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0},
    }


def _resolve_tts_voice(lang: str, preferred: str) -> Tuple[str, Optional[str]]:
    """Return (voice, fallback_reason)."""
    preferred = (preferred or "").strip()
    if preferred in TTS_VOICES:
        allowed = LANG_TO_PREFERRED_VOICES.get(lang)
        if allowed and preferred not in allowed:
            fallback = allowed[0]
            return fallback, f"fallback_voice:{preferred}->{fallback}"
        return preferred, None

    allowed = LANG_TO_PREFERRED_VOICES.get(lang, [])
    fallback = allowed[0] if allowed else (TTS_VOICES[0] if TTS_VOICES else "alloy")
    return fallback, f"fallback_voice:{preferred}->{fallback}"


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


class ConversationWorker(QObject):
    """Realtime speech-to-speech conversation (WebSocket).

    The UI has two modes:
    - Hands-free: continuous mic streaming, server-side VAD triggers responses.
    - Push-to-talk: mic streams only while button is pressed; on release we commit+response.
    """

    state_changed = Signal(str)        # idle/listening/transcribing/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message

    # Realtime levely pro animaci hlavy (0..1).
    input_level = Signal(float)
    output_level = Signal(float)
    output_pose = Signal(object)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

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

        # Level signals are throttled to avoid saturating the Qt event loop.
        self._last_in_level: float = 0.0
        self._last_out_level: float = 0.0
        self._last_level_emit_t: float = 0.0

        # True while waiting for server transcription completion.
        self._awaiting_transcript = False

        # Best-effort current UI state.
        self._ui_state = _STATE_IDLE

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

        Users can override in Settings. If Settings points to an invalid device,
        we fall back to system default, then heuristic choice.
        """
        in_dev, in_note = pick_audio_device("input", self.settings.input_device)
        out_dev, out_note = pick_audio_device("output", self.settings.output_device)
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
        if self.settings.log_conversations:
            self._log_event(record_type, text=normalized)
        else:
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
        if not self.settings.write_logs:
            self._append_caption(
                f"Relace: model={self.settings.realtime_model}, hlas={self.settings.tts_voice}, log obsahu=ne"
            )
            return
        log_dir = self.settings.validate_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        self._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)

        self._log_event(
            "session_start",
            settings={
                "openai_base_url": "wss://api.openai.com/v1/realtime",
                "realtime_model": self.settings.realtime_model,
                "language": self.settings.language,
                "tts_voice": self.settings.tts_voice,
                "tts_speed": self.settings.tts_speed,
                "write_logs": bool(self.settings.write_logs),
                "log_conversations": bool(self.settings.log_conversations),
                "audio": {
                    "input_device": self.settings.input_device,
                    "output_device": self.settings.output_device,
                },
            },
        )
        self._append_caption(
            f"Relace: model={self.settings.realtime_model}, hlas={self.settings.tts_voice}, log obsahu={'ano' if self.settings.log_conversations else 'ne'}"
        )

    def _end_session(self) -> None:
        if not self._logger:
            return
        self._log_event("session_end", dropped_records=self._logger.dropped_records, last_error=self._logger.last_error)
        try:
            self._logger.close()
        except Exception:
            pass
        self._logger = None

    def _ensure_realtime(self, turn_mode: str) -> RealtimeService:
        if not self.settings.openai_api_key:
            raise ValueError("Chybí API key")
        self._rt_turn_mode = turn_mode

        # Odpovídej podle skutečně slyšeného jazyka; pevné nastavení ber jen jako fallback.
        configured_lang = self.settings.language if self.settings.language in _ALLOWED_LANGS else "auto"
        resolved = "auto"
        self._resolved_lang = configured_lang if configured_lang != "auto" else "cs"

        instructions = build_system_prompt(self.settings, self._resolved_lang)
        voice, fallback_reason = _resolve_tts_voice(self._resolved_lang, self.settings.tts_voice)

        # Keep within current Realtime constraints (speed max is 1.5).
        try:
            speed = float(self.settings.tts_speed)
        except Exception:
            speed = 1.0
        speed = max(0.25, min(1.5, speed))

        cfg = RealtimeConfig(
            api_key=self.settings.openai_api_key,
            model=self.settings.realtime_model,
            instructions=instructions,
            voice=voice,
            language_hint="auto",
            turn_mode=turn_mode,
            auto_interrupt=True,
            noise_reduction="far_field",
            output_speed=speed,
            server_vad_silence_ms=int(self.settings.vad_silence_ms or 900),
            server_vad_prefix_ms=300,
            server_vad_threshold=0.72,
        )

        if fallback_reason:
            self._append_caption(f"Hlas: {fallback_reason}")

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
        self._rt.cfg.noise_reduction = "far_field"
        self._rt.cfg.output_speed = speed
        self._rt.cfg.server_vad_silence_ms = int(self.settings.vad_silence_ms or 900)
        self._rt.cfg.server_vad_prefix_ms = 300
        self._rt.cfg.server_vad_threshold = 0.72
        self._rt.update_session(
            instructions=instructions,
            voice=voice,
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
                    is_playing_out = buffered > 0 or current_out_level > 0.035 or self._ui_state == _STATE_SPEAKING
                    if is_playing_out:
                        self._mic_suppressed_until = max(self._mic_suppressed_until, now_monotonic + 0.35)
                suppress_mic = (
                    self._mode == "handsfree"
                    and now_monotonic < self._mic_suppressed_until
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
                            # Update last input VU level.
                            in_level = self._pcm16_level(chunk)
                            if suppress_mic:
                                self._last_in_level = 0.0
                                continue
                            self._last_in_level = in_level
                            self._rt.append_audio_pcm16(chunk)

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
                    in_lvl = self._last_in_level if (self._mic_enabled.is_set() and not suppress_mic) else 0.0
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
        self.input_level.emit(0.0)
        self.output_level.emit(0.0)
        self.output_pose.emit(_closed_pose_snapshot())
        self._set_state(_STATE_IDLE)
        self._end_session()

    # -------- Hands-free mode --------

    @Slot()
    def start_handsfree(self) -> None:
        try:
            self._start_session_if_needed()
            self._mode = "handsfree"
            self._resolve_audio_devices()
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
            self._start_session_if_needed()
            self._mode = "ptt"
            self._resolve_audio_devices()
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
    sig_ptt_pressed = Signal()
    sig_ptt_released = Signal()

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
        self.sig_ptt_pressed.connect(self.worker.ptt_pressed)
        self.sig_ptt_released.connect(self.worker.ptt_released)

        self._theme = Theme()

        # Window branding
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

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout()
        outer.setContentsMargins(18, 14, 18, 16)
        root.setLayout(outer)

        # --- Header (brand + quick actions) ---
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

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
        tf = QFont()
        tf.setPointSize(22)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet(f"QLabel {{ color: {self._theme.text}; }}")
        subtitle = QLabel("Hlasový asistent (hands‑free / push‑to‑talk)")
        subtitle.setStyleSheet(f"QLabel {{ color: {self._theme.text_muted}; font-size: 12px; }}")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)

        header.addWidget(logo)
        header.addSpacing(12)
        header.addLayout(title_wrap)
        header.addStretch(1)

        self.btn_settings = QPushButton("Nastavení")
        self.btn_openai = QPushButton("OpenAI")
        self.btn_save = QPushButton("Uložit")
        self.btn_clear = QPushButton("Vyčistit relaci")
        self.btn_exit = QPushButton("Konec")

        self.btn_settings.setProperty("variant", "primary")
        self.btn_exit.setProperty("variant", "danger")

        header.addWidget(self.btn_openai)
        header.addWidget(self.btn_settings)
        header.addWidget(self.btn_save)
        header.addWidget(self.btn_clear)
        header.addSpacing(8)
        header.addWidget(self.btn_exit)

        outer.addLayout(header)

        # Captions/status panel
        self.captions = QLabel("")
        self.captions.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.captions.setWordWrap(True)
        self.captions.setMinimumHeight(120)
        self.captions.setStyleSheet(
            "QLabel {"
            "  padding: 12px 14px;"
            "  border-radius: 14px;"
            "  background-color: rgba(255,255,255,6);"
            "  border: 1px solid rgba(255,255,255,16);"
            "  font-size: 13px;"
            "  line-height: 1.2;"
            "}"
        )
        outer.addWidget(self.captions)

        center = QVBoxLayout()
        center.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        head_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "head_photo.png")
        earth_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "earth_hd.png")
        earth_clouds_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "earth_clouds_hd.png")

        self.head = HeadWidget(head_path)
        self.head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.head.setMinimumSize(520, 520)

        self.globe = GlobeButton(earth_path, earth_clouds_path)

        center.addStretch(1)
        center.addWidget(self.head, 0, Qt.AlignHCenter)
        center.addSpacing(14)
        center.addWidget(self.globe, 0, Qt.AlignHCenter)
        center.addStretch(2)

        outer.addLayout(center, 1)

        # App-wide stylesheet is installed on QApplication; keep per-widget overrides minimal.

    def _wire(self) -> None:
        self.btn_exit.clicked.connect(lambda _=False: self.close())
        self.btn_openai.clicked.connect(lambda _=False: self._open_openai_dialog())
        self.btn_settings.clicked.connect(lambda _=False: self._open_settings_dialog())
        self.btn_save.clicked.connect(lambda _=False: self._save_defaults())
        self.btn_clear.clicked.connect(lambda _=False: self._clear_session())

        self.head.orb_clicked.connect(self._on_orb_click)
        self.head.reset_clicked.connect(self._on_orb_reset)
        self.globe.ptt_pressed.connect(self._on_globe_press)
        self.globe.ptt_released.connect(self._on_globe_release)

        self.worker.state_changed.connect(self._on_state)
        self.worker.captions_updated.connect(self._on_captions)
        self.worker.error.connect(self._on_error)
        self.worker.input_level.connect(self._on_input_level)
        self.worker.output_level.connect(self._on_output_level)
        self.worker.output_pose.connect(self._on_output_pose)

    def _open_openai_dialog(self) -> None:
        d = OpenAIDialog(self.settings, self)
        d.exec()
        self.settings.save()

    def _load_models(self) -> List[str]:
        if not self.settings.openai_api_key:
            return []
        svc = OpenAIService(self.settings.openai_api_key)
        try:
            models = svc.list_models()
            realtime = [m for m in models if "realtime" in m.lower()]
            if realtime:
                return sorted(set(realtime))
            return OpenAIService.filter_chat_models(models)
        finally:
            svc.close()

    def _open_settings_dialog(self) -> None:
        try:
            d = SettingsDialog(
                self.settings,
                load_models_fn=self._load_models if self.settings.openai_api_key else None,
                parent=self,
            )
            if d.exec():
                d.apply()
                self.settings.save()
        except Exception:
            import logging, traceback
            logging.getLogger("kajovochat").exception("settings_dialog_failed")
            try:
                QMessageBox.critical(self, "Nastavení", "Nepodařilo se otevřít nastavení. Podrobnosti jsou v logu.")
            except Exception:
                pass

    def _save_defaults(self) -> None:
        self.settings.save()
        QMessageBox.information(self, "SAVE", "Aktuální nastavení bylo uloženo jako výchozí.")

    def _clear_session(self) -> None:
        try:
            self.sig_request_stop.emit()
        except Exception:
            pass
        self.captions.setText("")
        self._handsfree_running = False
        self.globe.setEnabled(True)
        self.head.set_running(False)
        self.head.set_error_text("")
        self.head.set_lipsync_snapshot(_closed_pose_snapshot())
        if not self.settings.openai_api_key:
            self.captions.setText("Chybí OpenAI API key. Otevřete dialog OpenAI a vložte klíč.")

    @Slot()
    def _on_orb_click(self) -> None:
        # Hlava přepíná hands-free režim.
        if self._handsfree_running:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.head.set_running(False)
            self.globe.setEnabled(True)
            return

        self.globe.setEnabled(False)
        self._handsfree_running = True
        self.head.set_running(True)
        self.captions.setText("Hands-free: aktivní")  # immediate UI feedback
        self.sig_start_handsfree.emit()

    @Slot()
    def _on_orb_reset(self) -> None:
        # Reset brings the app back to a clean idle state.
        try:
            self.sig_request_stop.emit()
        except Exception:
            pass
        self._handsfree_running = False
        self.globe.setEnabled(True)
        self.head.set_running(False)
        self.head.set_error_text("")
        self.head.set_lipsync_snapshot(_closed_pose_snapshot())

    @Slot(float)
    def _on_input_level(self, lvl: float) -> None:
        self.head.set_input_level(lvl)

    @Slot(float)
    def _on_output_level(self, lvl: float) -> None:
        self.head.set_output_level(lvl)

    @Slot(object)
    def _on_output_pose(self, snapshot: object) -> None:
        self.head.set_lipsync_snapshot(snapshot)

    @Slot()
    def _on_globe_press(self) -> None:
        # In hands-free we ignore globe (disabled anyway).
        if self._handsfree_running:
            return
        self.sig_ptt_pressed.emit()

    @Slot()
    def _on_globe_release(self) -> None:
        if self._handsfree_running:
            return
        self.sig_ptt_released.emit()

    @Slot(str)
    def _on_state(self, s: str) -> None:
        self.head.set_state(s)
        if s == "error":
            self._handsfree_running = False
            self.globe.setEnabled(True)
            self.head.set_running(False)
        elif s in {_STATE_CONNECTING, _STATE_RECONNECTING}:
            self.globe.setEnabled(False)
        else:
            # Clear stale error message.
            self.head.set_error_text("")

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.captions.setText(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        # Chybu zobrazíme přímo ve widgetu hlavy a zachováme captions.
        self.head.set_error_text(msg)
        self._handsfree_running = False
        self.globe.setEnabled(True)
        self.head.set_running(False)

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
        QTimer.singleShot(0, lambda: w.captions.setText("Chybí OpenAI API key. Otevřete dialog OpenAI a vložte klíč."))
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
