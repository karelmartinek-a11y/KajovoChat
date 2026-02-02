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
from PySide6.QtGui import QFont
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
from .widgets.orb_widget import OrbWidget
from .widgets.globe_button import GlobeButton


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

    state_changed = Signal(str)        # idle/listening/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message

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

        if self._rt is None:
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
            self.state_changed.emit("error")
            self.error.emit(msg)

        rt.on_error = _err

        def _user(t: str) -> None:
            self._append_caption(f"Ty: {t}")
            if self._logger:
                self._logger.append({"type": "user", "text": t})

        rt.on_user_transcript = _user

        rt.on_assistant_text_delta = lambda d: self._set_caption_preview("AI", d)

        def _ai_done(t: str) -> None:
            self._append_caption(f"AI: {t}")
            if self._logger:
                self._logger.append({"type": "assistant", "text": t})

        rt.on_assistant_text_done = _ai_done

        def _audio(pcm: bytes) -> None:
            # Audio deltas arrive faster than realtime; enqueue and let the player drain.
            self.state_changed.emit("speaking")
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
            self.state_changed.emit("listening")

        rt.on_vad_speech_started = _speech_started

        def _resp_done() -> None:
            # In handsfree mode we keep listening; in PTT return to idle.
            if self._mode == "handsfree":
                self.state_changed.emit("listening")
            else:
                self.state_changed.emit("idle")

        rt.on_response_done = _resp_done

    def _start_rt_loop(self) -> None:
        if self._rt_loop_thread and self._rt_loop_thread.is_alive():
            return
        self._rt_loop_stop.clear()

        def loop() -> None:
            while not self._rt_loop_stop.is_set():
                if self._rt:
                    self._rt.pump_events()
                if self._mic_enabled.is_set() and self._mic is not None and self._rt is not None:
                    # Drain a few chunks per tick to reduce backlog.
                    for _ in range(6):
                        try:
                            chunk = self._mic.queue.get_nowait()
                        except queue.Empty:
                            break
                        if chunk:
                            self._rt.append_audio_pcm16(chunk)
                time.sleep(0.005)

        self._rt_loop_thread = threading.Thread(target=loop, daemon=True)
        self._rt_loop_thread.start()

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
        self._rt_loop_stop.set()
        self.state_changed.emit("idle")
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
            self.state_changed.emit("listening")
            self._append_caption("Hands-free: Realtime aktivní (server VAD).")
        except Exception as e:
            self.state_changed.emit("error")
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
            self.state_changed.emit("listening")
            self._append_caption("PTT: poslouchám…")
        except Exception as e:
            self.state_changed.emit("error")
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
        self.state_changed.emit("thinking")
        self._rt.commit_input_audio()
        self._rt.request_response()
        self._append_caption("PTT: čekám na odpověď…")


class MainWindow(QMainWindow):
    sig_start_handsfree = Signal()
    sig_request_stop = Signal()
    sig_ptt_pressed = Signal()
    sig_ptt_released = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self._handsfree_running = False

        self._thread = QThread(self)
        self.worker = ConversationWorker(self.settings)
        self.worker.moveToThread(self._thread)
        self._thread.start()

        self.sig_start_handsfree.connect(self.worker.start_handsfree)
        self.sig_request_stop.connect(self.worker.request_stop)
        self.sig_ptt_pressed.connect(self.worker.ptt_pressed)
        self.sig_ptt_released.connect(self.worker.ptt_released)

        self.setWindowTitle("KájovoChat")
        self._build_ui()
        self._wire()
        self.showMaximized()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout()
        outer.setContentsMargins(16, 12, 16, 12)
        root.setLayout(outer)

        title = QLabel("KájovoChat")
        f = QFont()
        f.setPointSize(22)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(title)

        row = QHBoxLayout()
        self.btn_settings = QPushButton("NASTAVENÍ")
        self.btn_openai = QPushButton("OPEN AI")
        self.btn_save = QPushButton("SAVE")
        self.btn_exit = QPushButton("EXIT")

        row.addWidget(self.btn_settings)
        row.addWidget(self.btn_openai)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        row.addWidget(self.btn_exit)
        outer.addLayout(row)

        self.captions = QLabel("")
        self.captions.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.captions.setWordWrap(True)
        self.captions.setStyleSheet("QLabel { color: rgba(245,245,245,210); font-size: 14px; }")
        self.captions.setMinimumHeight(110)
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

        root.setStyleSheet(
            "QWidget { background-color: #0b0f18; color: #f2f2f2; }"
            " QLabel { color: rgba(245,245,245,210); }"
            " QGroupBox { border: 1px solid rgba(255,255,255,35); border-radius: 8px; margin-top: 10px; }"
            " QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: rgba(245,245,245,220); }"
            " QLineEdit, QComboBox { background-color: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,55); border-radius: 6px; padding: 6px 8px; color: #f2f2f2; }"
            " QComboBox::drop-down { border: none; }"
            " QPushButton { padding: 8px 14px; color: #f2f2f2; background-color: rgba(255,255,255,22); border: 1px solid rgba(255,255,255,65); border-radius: 6px; }"
            " QPushButton:hover { background-color: rgba(255,255,255,30); }"
            " QPushButton:pressed { background-color: rgba(255,255,255,40); }"
            " QPushButton:disabled { color: rgba(245,245,245,120); background-color: rgba(255,255,255,10); border-color: rgba(255,255,255,25); }"
        )

    def _wire(self) -> None:
        self.btn_exit.clicked.connect(self.close)
        self.btn_openai.clicked.connect(self._open_openai_dialog)
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        self.btn_save.clicked.connect(self._save_defaults)

        self.orb.orb_clicked.connect(self._on_orb_click)
        self.globe.ptt_pressed.connect(self._on_globe_press)
        self.globe.ptt_released.connect(self._on_globe_release)

        self.worker.state_changed.connect(self._on_state)
        self.worker.captions_updated.connect(self._on_captions)
        self.worker.error.connect(self._on_error)

    def _open_openai_dialog(self) -> None:
        d = OpenAIDialog(self.settings, self)
        d.exec()
        self.settings.save()

    def _load_models(self) -> List[str]:
        if not self.settings.openai_api_key:
            return []
        svc = OpenAIService(self.settings.openai_api_key)
        models = svc.list_models()
        return OpenAIService.filter_chat_models(models)

    def _open_settings_dialog(self) -> None:
        d = SettingsDialog(
            self.settings,
            load_models_fn=self._load_models if self.settings.openai_api_key else None,
            parent=self,
        )
        if d.exec():
            d.apply()
            self.settings.save()

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

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.captions.setText(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Chyba", msg)
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
    app = QApplication(sys.argv)
    # Ensure buttons/inputs are readable on dark background across all dialogs.
    app.setStyleSheet(
        "QWidget { background-color: #0b0f18; color: #f2f2f2; }"
        " QLabel { color: rgba(245,245,245,210); }"
        " QGroupBox { border: 1px solid rgba(255,255,255,35); border-radius: 8px; margin-top: 10px; }"
        " QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: rgba(245,245,245,220); }"
        " QLineEdit, QComboBox { background-color: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,55); border-radius: 6px; padding: 6px 8px; color: #f2f2f2; }"
        " QComboBox::drop-down { border: none; }"
        " QPushButton { padding: 8px 14px; color: #f2f2f2; background-color: rgba(255,255,255,22); border: 1px solid rgba(255,255,255,65); border-radius: 6px; }"
        " QPushButton:hover { background-color: rgba(255,255,255,30); }"
        " QPushButton:pressed { background-color: rgba(255,255,255,40); }"
        " QPushButton:disabled { color: rgba(245,245,245,120); background-color: rgba(255,255,255,10); border-color: rgba(255,255,255,25); }"
    )
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()