from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QSizePolicy
)

from .settings import AppSettings, STYLE_PROMPTS, LENGTH_PROMPTS, LANGUAGE_PROMPTS
from .dialogs.settings_dialog import SettingsDialog
from .dialogs.openai_dialog import OpenAIDialog
from .services.openai_service import OpenAIService
from .services.audio_service import AudioRecorder, AudioPlayer
from .services.log_service import RealtimeLogWriter
from .widgets.orb_widget import OrbWidget
from .widgets.globe_button import GlobeButton


def _lang_hint(name: str) -> Optional[str]:
    return {
        "česky": "cs",
        "slovensky": "sk",
        "německy": "de",
        "anglicky": "en",
        "francouzsky": "fr",
    }.get(name)


class ConversationWorker(QObject):
    state_changed = Signal(str)
    captions_updated = Signal(str)
    error = Signal(str)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self._stop = threading.Event()
        self._messages: List[dict] = []
        self._captions = ""
        self._logger: Optional[RealtimeLogWriter] = None

    def _system_prompt(self) -> str:
        style = STYLE_PROMPTS.get(self.settings.response_style, STYLE_PROMPTS["věcné"])
        length = LENGTH_PROMPTS.get(self.settings.response_length, LENGTH_PROMPTS["normální"])
        lang = LANGUAGE_PROMPTS.get(self.settings.voice_language, LANGUAGE_PROMPTS["česky"])
        return f"{lang}\n{style}\n{length}\n"

    def _voice(self) -> str:
        return self.settings.tts_voice_female if self.settings.voice_gender == "ženský" else self.settings.tts_voice_male

    @Slot()
    def request_stop(self) -> None:
        # Thread-safe stop request (does not require worker event loop later).
        self._stop.set()

    @Slot()
    def start_nonstop(self) -> None:
        if not self.settings.openai_api_key:
            self.error.emit("Chybí API key. Otevři OPEN AI a vlož API key.")
            return
        self._stop.clear()
        self._start_session()
        self._run_loop(nonstop=True)

    @Slot()
    def start_push_to_talk(self) -> None:
        if not self.settings.openai_api_key:
            self.error.emit("Chybí API key. Otevři OPEN AI a vlož API key.")
            return
        self._stop.clear()
        self._start_session()
        self._run_loop(nonstop=False)

    def _start_session(self) -> None:
        if self._logger:
            return
        log_dir = self.settings.ensure_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        self._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)
        self._messages = []
        self._captions = ""
        self.captions_updated.emit(self._captions)

        self._logger.append({
            "ts": time.time(),
            "type": "session_start",
            "settings": {
                "style": self.settings.response_style,
                "length": self.settings.response_length,
                "voice_language": self.settings.voice_language,
                "voice_gender": self.settings.voice_gender,
                "chat_model": self.settings.chat_model,
                "stt_model": self.settings.stt_model,
                "tts_model": self.settings.tts_model,
            }
        })

    def _end_session(self) -> None:
        if self._logger:
            self._logger.append({"ts": time.time(), "type": "session_end"})
            self._logger.close()
            self._logger = None

    def _append_caption(self, line: str) -> None:
        self._captions = (self._captions + "\n" + line).strip()
        lines = self._captions.splitlines()[-12:]
        self._captions = "\n".join(lines)
        self.captions_updated.emit(self._captions)

    def _run_loop(self, nonstop: bool) -> None:
        svc = OpenAIService(self.settings.openai_api_key)
        recorder = AudioRecorder(
            samplerate=self.settings.input_samplerate,
            device=self.settings.input_device,
            rms_threshold=self.settings.vad_rms_threshold,
            silence_ms=self.settings.vad_silence_ms,
            max_seconds=self.settings.max_record_seconds,
        )
        player = AudioPlayer(
            samplerate=self.settings.tts_samplerate,
            device=self.settings.output_device,
        )

        try:
            while not self._stop.is_set():
                self.state_changed.emit("listening")
                try:
                    rec = recorder.record_once()
                except Exception as e:
                    self.state_changed.emit("error")
                    self.error.emit(f"Nepodařilo se nahrát mikrofon: {e}")
                    break

                if self._logger:
                    self._logger.append({"ts": time.time(), "type": "audio_recorded", "duration_s": rec.duration_s})

                self.state_changed.emit("transcribing")
                try:
                    user_text = svc.transcribe_wav(
                        rec.wav_bytes,
                        model=self.settings.stt_model,
                        language_hint=_lang_hint(self.settings.voice_language),
                    )
                except Exception as e:
                    self.state_changed.emit("error")
                    self.error.emit(f"Přepis (Whisper) selhal: {e}")
                    break

                user_text = (user_text or "").strip()
                if not user_text:
                    if not nonstop:
                        break
                    continue

                self._append_caption(f"Ty: {user_text}")
                if self._logger:
                    self._logger.append({"ts": time.time(), "type": "user_text", "text": user_text, "text_line": f"Ty: {user_text}"})

                self._messages.append({"role": "user", "content": user_text})
                if len(self._messages) > 20:
                    self._messages = self._messages[-20:]

                self.state_changed.emit("thinking")

                assistant_acc = ""

                def on_delta(d: str) -> None:
                    nonlocal assistant_acc
                    assistant_acc += d
                    preview = assistant_acc.replace("\n", " ").strip()
                    self.captions_updated.emit("\n".join(self._captions.splitlines()[-11:] + [f"AI: {preview}"]))

                try:
                    assistant_text = svc.chat_stream(
                        model=self.settings.chat_model,
                        system_prompt=self._system_prompt(),
                        messages=self._messages,
                        on_delta=on_delta,
                    )
                except Exception as e:
                    self.state_changed.emit("error")
                    self.error.emit(f"Chat selhal: {e}")
                    break

                assistant_text = assistant_text.strip()
                if assistant_text:
                    self._messages.append({"role": "assistant", "content": assistant_text})
                    self._append_caption(f"AI: {assistant_text}")
                    if self._logger:
                        self._logger.append({"ts": time.time(), "type": "assistant_text", "text": assistant_text, "text_line": f"AI: {assistant_text}"})

                self.state_changed.emit("speaking")
                try:
                    pcm = svc.tts_pcm16(
                        text=assistant_text,
                        model=self.settings.tts_model,
                        voice=self._voice(),
                        response_format="pcm",
                    )
                    if self._logger:
                        self._logger.append({"ts": time.time(), "type": "tts_bytes", "nbytes": len(pcm)})
                    player.play_pcm16(pcm)
                except Exception as e:
                    self.state_changed.emit("error")
                    self.error.emit(f"TTS / přehrávání selhalo: {e}")
                    break

                self.state_changed.emit("idle")

                if not nonstop:
                    # single turn for push-to-talk
                    break

        finally:
            self.state_changed.emit("idle")
            self._end_session()
            self._stop.clear()


class MainWindow(QMainWindow):
    sig_start_nonstop = Signal()
    sig_start_push_to_talk = Signal()
    sig_request_stop = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.settings = AppSettings.load()
        self._nonstop_running = False

        self._thread = QThread(self)
        self.worker = ConversationWorker(self.settings)
        self.worker.moveToThread(self._thread)
        self._thread.start()

        self.sig_start_nonstop.connect(self.worker.start_nonstop)
        self.sig_start_push_to_talk.connect(self.worker.start_push_to_talk)
        self.sig_request_stop.connect(self.worker.request_stop)

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

        self.orb = OrbWidget(moon_path)
        self.orb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.orb.setMinimumSize(520, 520)

        self.globe = GlobeButton(earth_path)

        center.addStretch(1)
        center.addWidget(self.orb, 0, Qt.AlignHCenter)
        center.addSpacing(14)
        center.addWidget(self.globe, 0, Qt.AlignHCenter)
        center.addStretch(2)

        outer.addLayout(center, 1)

        root.setStyleSheet("QWidget { background-color: #0b0f18; } QPushButton { padding: 8px 14px; }")

    def _wire(self) -> None:
        self.btn_exit.clicked.connect(self.close)
        self.btn_openai.clicked.connect(self._open_openai_dialog)
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        self.btn_save.clicked.connect(self._save_defaults)

        self.orb.orb_clicked.connect(self._on_orb_click)
        self.globe.toggled_on.connect(self._on_globe_toggle)

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
            parent=self
        )
        if d.exec():
            d.apply()
            self.settings.save()

    def _save_defaults(self) -> None:
        self.settings.save()
        QMessageBox.information(self, "SAVE", "Aktuální nastavení bylo uloženo jako výchozí.")

    @Slot()
    def _on_orb_click(self) -> None:
        # Orb toggles NONSTOP mode.
        if self._nonstop_running:
            self.sig_request_stop.emit()
            self._nonstop_running = False
            self.orb.set_running(False)
            self.globe.setEnabled(True)
            return

        # start
        self.globe.setChecked(False)
        self.globe.setEnabled(False)
        self._nonstop_running = True
        self.orb.set_running(True)
        self.sig_start_nonstop.emit()

    @Slot(bool)
    def _on_globe_toggle(self, enabled: bool) -> None:
        # Globe is push-to-talk toggle. When ON, do one turn; when OFF, request stop.
        if enabled:
            if self._nonstop_running:
                self.sig_request_stop.emit()
                self._nonstop_running = False
                self.orb.set_running(False)
            self.sig_start_push_to_talk.emit()
        else:
            self.sig_request_stop.emit()

    @Slot(str)
    def _on_state(self, s: str) -> None:
        self.orb.set_state(s)
        if s == "idle" and not self._nonstop_running:
            self.orb.set_running(False)
        if s == "error":
            self._nonstop_running = False
            self.globe.setEnabled(True)
            self.globe.setChecked(False)

    @Slot(str)
    def _on_captions(self, text: str) -> None:
        self.captions.setText(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Chyba", msg)
        self._nonstop_running = False
        self.globe.setEnabled(True)
        self.globe.setChecked(False)
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
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
