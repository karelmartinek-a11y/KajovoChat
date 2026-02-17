from __future__ import annotations

import sys
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import sounddevice as sd

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
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
from .widgets.orb_widget import OrbWidget
from .widgets.globe_button import GlobeButton
from .theme import Theme, app_stylesheet


_ALLOWED_LANGS = {"cs", "en", "de", "sk", "fr"}


def _resolve_language(settings: AppSettings, stt_lang: Optional[str], last_lang: Optional[str]) -> str:
    """Resolved language used for LLM+TTS.

    - If settings.language != auto => fixed.
    - If auto => use stt_lang when in allowed set, else last_lang, else cs.
    """
    if settings.language != "auto":
        return settings.language
    if stt_lang in _ALLOWED_LANGS:
        return stt_lang
    return last_lang if last_lang in _ALLOWED_LANGS else "cs"


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


class ConversationWorker(QObject):
    """Realtime speech-to-speech conversation (WebSocket).

    The UI has two modes:
    - Hands-free: continuous mic streaming, server-side VAD triggers responses.
    - Push-to-talk: mic streams only while button is pressed; on release we commit+response.
    """

    state_changed = Signal(str)        # idle/listening/transcribing/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message

    # Real-time levels for orb animation (0..1).
    input_level = Signal(float)
    output_level = Signal(float)

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

        # Level signals are throttled to avoid saturating the Qt event loop.
        self._last_in_level: float = 0.0
        self._last_out_level: float = 0.0
        self._last_level_emit_t: float = 0.0

        # True while waiting for server transcription completion.
        self._awaiting_transcript = False

        # Best-effort current UI state.
        self._ui_state = "idle"

    def _set_state(self, s: str) -> None:
        self._ui_state = s
        self.state_changed.emit(s)

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

    def _set_caption_preview(self, prefix: str, text: str) -> None:
        base = self._captions.splitlines()[-11:]
        preview = (text or "").replace("\n", " ").strip()
        self.captions_updated.emit("\n".join(base + [f"{prefix}: {preview}"]))

    def _start_session_if_needed(self) -> None:
        if self._logger:
            return
        log_dir = self.settings.ensure_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        self._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)
        self._captions = ""
        self.captions_updated.emit(self._captions)

        self._logger.append(
            {
                "type": "session_start",
                "settings": {
                    "openai_base_url": "wss://api.openai.com/v1/realtime",
                    "realtime_model": self.settings.realtime_model,
                    "language": self.settings.language,
                    "tts_voice": self.settings.tts_voice,
                    "audio": {
                        "input_device": self.settings.input_device,
                        "output_device": self.settings.output_device,
                    },
                },
            }
        )

    def _end_session(self) -> None:
        if not self._logger:
            return
        self._logger.append({"type": "session_end"})
        try:
            self._logger.close()
        except Exception:
            pass
        self._logger = None

    def _ensure_realtime(self, turn_mode: str) -> RealtimeService:
        if not self.settings.openai_api_key:
            raise ValueError("Chybí API key")

        # Language used for instructions + transcription hint.
        resolved = self.settings.language if self.settings.language in _ALLOWED_LANGS else "cs"
        if resolved == "auto":
            resolved = "cs"
        self._resolved_lang = resolved

        instructions = build_system_prompt(self.settings, resolved)

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
            voice=self.settings.tts_voice,
            language_hint=resolved if self.settings.language != "auto" else "auto",
            turn_mode=turn_mode,
            auto_interrupt=True,
            noise_reduction="far_field",
            output_speed=speed,
            server_vad_silence_ms=int(self.settings.vad_silence_ms or 900),
            server_vad_prefix_ms=300,
            server_vad_threshold=0.60,
        )

        if self._rt is None or not self._rt.is_connected:
            # Znovu vytvorit websocket po odpojeni.
            self._rt = RealtimeService(cfg)
            self._wire_realtime_callbacks(self._rt)
            self._rt.connect()
            return self._rt

        # Same websocket; update session settings.
        # Update extra audio/session knobs as well (update_session only touches a subset).
        self._rt.cfg.noise_reduction = "far_field"
        self._rt.cfg.output_speed = speed
        self._rt.cfg.server_vad_silence_ms = int(self.settings.vad_silence_ms or 900)
        self._rt.cfg.server_vad_prefix_ms = 300
        self._rt.cfg.server_vad_threshold = 0.60
        self._rt.update_session(
            instructions=instructions,
            voice=self.settings.tts_voice,
            language_hint=cfg.language_hint,
            turn_mode=turn_mode,
        )
        return self._rt

    def _wire_realtime_callbacks(self, rt: RealtimeService) -> None:
        rt.on_status = lambda s: self._append_caption(s)

        def _err(msg: str) -> None:
            if self._logger:
                self._logger.append({"type": "error", "message": msg})
            self._stop_realtime_session()
            self._set_state("error")
            self.error.emit(msg)

        rt.on_error = _err

        def _user(t: str) -> None:
            self._append_caption(f"Ty: {t}")
            if self._logger:
                self._logger.append({"type": "user", "text": t})

            # Transition from "transcribing" to "thinking" once we have a transcript.
            self._awaiting_transcript = False
            if self._ui_state not in {"speaking", "error"}:
                self._set_state("thinking")

        rt.on_user_transcript = _user

        rt.on_assistant_text_delta = lambda d: self._set_caption_preview("AI", d)

        def _ai_done(t: str) -> None:
            self._append_caption(f"AI: {t}")
            if self._logger:
                self._logger.append({"type": "assistant", "text": t})

        rt.on_assistant_text_done = _ai_done

        def _audio(pcm: bytes) -> None:
            # Audio deltas arrive faster than realtime; enqueue and let the player drain.
            self._set_state("speaking")
            try:
                self._ensure_player()
                if self._player:
                    self._player.enqueue_pcm16(pcm)
            except Exception as e:
                # If playback fails (wrong output device), surface a helpful error.
                _err(str(e) + "\n\n" + format_device_help())

        rt.on_assistant_audio_delta = _audio

        def _speech_started() -> None:
            # Barge-in: stop local playback immediately.
            try:
                if self._player:
                    self._player.stop()
            except Exception:
                pass
            self._awaiting_transcript = False
            self._set_state("listening")

        rt.on_vad_speech_started = _speech_started

        def _speech_stopped() -> None:
            # Server will emit input_audio_transcription.completed afterwards.
            self._awaiting_transcript = True
            # In handsfree, the server will auto-create the response (create_response=True).
            self._set_state("transcribing")

        rt.on_vad_speech_stopped = _speech_stopped

        def _resp_done() -> None:
            # In handsfree mode we keep listening; in PTT return to idle.
            if self._mode == "handsfree":
                self._set_state("listening")
            else:
                self._set_state("idle")
            self._awaiting_transcript = False

        rt.on_response_done = _resp_done

    def _start_rt_loop(self) -> None:
        if self._rt_loop_thread and self._rt_loop_thread.is_alive():
            return
        self._rt_loop_stop.clear()

        def loop() -> None:
            while not self._rt_loop_stop.is_set():
                if self._rt:
                    self._rt.pump_events()

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
                            self._last_in_level = self._pcm16_level(chunk)
                            self._rt.append_audio_pcm16(chunk)

                # Output level from the audio callback (reflects actual playback).
                if self._player is not None:
                    try:
                        self._last_out_level = float(self._player.get_level())
                    except Exception:
                        self._last_out_level = 0.0
                else:
                    self._last_out_level = 0.0

                # Throttle signals to ~60Hz.
                now = time.time()
                if now - self._last_level_emit_t >= 0.016:
                    in_lvl = self._last_in_level if self._mic_enabled.is_set() else 0.0
                    out_lvl = self._last_out_level
                    self.input_level.emit(float(in_lvl))
                    self.output_level.emit(float(out_lvl))
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
        self.input_level.emit(0.0)
        self.output_level.emit(0.0)
        self._set_state("idle")
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
            rt = self._ensure_realtime("server_vad")
            self._start_rt_loop()
            self._mic = RealtimeMicStream(samplerate=24000, device=self._resolved_input_device)
            self._mic.start()
            if getattr(self._mic, "using_resampler", False):
                self._append_caption(
                    f"Mikrofon jede na {self._mic.input_samplerate} Hz, resampluji na 24000 Hz."
                )
            self._mic_enabled.set()
            self._set_state("listening")
            self._append_caption("Hands-free: Realtime aktivní (server VAD).")
        except Exception as e:
            self._set_state("error")
            self.error.emit(str(e))


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
            self._set_state("listening")
            self._append_caption("PTT: poslouchám…")
        except Exception as e:
            self._set_state("error")
            self.error.emit(str(e))


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
        self._set_state("transcribing")
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
        self.showMaximized()

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
        self.btn_exit = QPushButton("Konec")

        self.btn_settings.setProperty("variant", "primary")
        self.btn_exit.setProperty("variant", "danger")

        header.addWidget(self.btn_openai)
        header.addWidget(self.btn_settings)
        header.addWidget(self.btn_save)
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

        moon_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "moon_hd.png")
        earth_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "earth_hd.png")
        earth_clouds_path = str(Path(__file__).resolve().parent / "resources" / "assets" / "earth_clouds_hd.png")

        self.orb = OrbWidget(moon_path)
        self.orb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orb.setMinimumSize(520, 520)

        self.globe = GlobeButton(earth_path, earth_clouds_path)

        center.addStretch(1)
        center.addWidget(self.orb, 0, Qt.AlignHCenter)
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

        self.orb.orb_clicked.connect(self._on_orb_click)
        self.orb.reset_clicked.connect(self._on_orb_reset)
        self.globe.ptt_pressed.connect(self._on_globe_press)
        self.globe.ptt_released.connect(self._on_globe_release)

        self.worker.state_changed.connect(self._on_state)
        self.worker.captions_updated.connect(self._on_captions)
        self.worker.error.connect(self._on_error)
        self.worker.input_level.connect(self._on_input_level)
        self.worker.output_level.connect(self._on_output_level)

    def _open_openai_dialog(self) -> None:
        d = OpenAIDialog(self.settings, self)
        d.exec()
        self.settings.save()

    def _load_models(self) -> List[str]:
        if not self.settings.openai_api_key:
            return []
        svc = OpenAIService(self.settings.openai_api_key)
        models = svc.list_models()
        realtime = [m for m in models if "realtime" in m.lower()]
        if realtime:
            return sorted(set(realtime))
        return OpenAIService.filter_chat_models(models)

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

    @Slot()
    def _on_orb_click(self) -> None:
        # Orb toggles hands-free mode.
        if self._handsfree_running:
            self.sig_request_stop.emit()
            self._handsfree_running = False
            self.orb.set_running(False)
            self.globe.setEnabled(True)
            return

        self.globe.setEnabled(False)
        self._handsfree_running = True
        self.orb.set_running(True)
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
        self.orb.set_running(False)
        self.orb.set_error_text("")

    @Slot(float)
    def _on_input_level(self, lvl: float) -> None:
        self.orb.set_input_level(lvl)

    @Slot(float)
    def _on_output_level(self, lvl: float) -> None:
        self.orb.set_output_level(lvl)

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
        self.orb.set_state(s)
        if s == "error":
            self._handsfree_running = False
            self.globe.setEnabled(True)
            self.orb.set_running(False)
        else:
            # Clear stale error message.
            self.orb.set_error_text("")

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.captions.setText(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        # Show error in the orb (with Reset). Keep captions as-is.
        self.orb.set_error_text(msg)
        self._handsfree_running = False
        self.globe.setEnabled(True)
        self.orb.set_running(False)

    def closeEvent(self, event) -> None:
        try:
            self.sig_request_stop.emit()
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

    app = QApplication(sys.argv)
    app.setStyleSheet(app_stylesheet())
    w = MainWindow(settings)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
