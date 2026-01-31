from __future__ import annotations

import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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
from .services.openai_service import OpenAIService, InvalidApiKeyError, TranscriptionResult
from .services.audio_service import AudioRecorder, AudioPlayer, VADMonitor, RecordResult
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
    """End-to-end voice conversation pipeline.

    Runs in a dedicated QThread. All slow operations (audio + network) are outside GUI thread.
    """

    state_changed = Signal(str)        # idle/listening/transcribing/thinking/speaking/error
    captions_updated = Signal(str)     # full captions text to show
    error = Signal(str)               # safe UI error message

    # PTT recording emits RecordResult from a separate thread
    ptt_record_done = Signal(object)  # RecordResult

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings

        # global stop for current session
        self._stop_all = threading.Event()

        # per-turn cancellation (barge-in / user interrupt)
        self._turn_cancel = threading.Event()

        # PTT recording control
        self._ptt_stop: Optional[threading.Event] = None
        self._ptt_thread: Optional[threading.Thread] = None

        # context
        self._messages: List[dict] = []
        self._captions = ""
        self._logger: Optional[RealtimeLogWriter] = None
        self._last_lang: Optional[str] = None

        # current audio player (so we can stop immediately on barge-in)
        self._player: Optional[AudioPlayer] = None
        self._player_lock = threading.Lock()

        self.ptt_record_done.connect(self._on_ptt_record_done)

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
        self._messages = []
        self._captions = ""
        self.captions_updated.emit(self._captions)

        self._logger.append({
            "type": "session_start",
            "settings": {
                "response_style": self.settings.response_style,
                "response_length": self.settings.response_length,
                "response_detail": self.settings.response_detail,
                "language": self.settings.language,
                "formality": self.settings.formality,
                "chat_model": self.settings.chat_model,
                "stt_model": self.settings.stt_model,  # fixed to whisper-1
                "tts_model": self.settings.tts_model,
                "tts_voice": self.settings.tts_voice,
                "tts_speed": self.settings.tts_speed,
                "temperature": self.settings.temperature,
                "max_output_tokens": self.settings.max_output_tokens,
                "audio": {
                    "input_device": self.settings.input_device,
                    "output_device": self.settings.output_device,
                    "input_samplerate": self.settings.input_samplerate,
                    "tts_samplerate": self.settings.tts_samplerate,
                },
                "vad": {
                    "vad_rms_threshold": self.settings.vad_rms_threshold,
                    "vad_silence_ms": self.settings.vad_silence_ms,
                    "vad_calibration_s": self.settings.vad_calibration_s,
                    "vad_multiplier": self.settings.vad_multiplier,
                    "max_record_seconds": self.settings.max_record_seconds,
                },
            }
        })

    def _end_session(self) -> None:
        if not self._logger:
            return
        self._logger.append({"type": "session_end"})
        try:
            self._logger.close()
        except Exception:
            pass
        self._logger = None

    def _stop_playback_immediately(self) -> None:
        with self._player_lock:
            p = self._player
        if p:
            try:
                p.stop()
            except Exception:
                pass

    def _cancel_current_turn(self, reason: str) -> None:
        self._turn_cancel.set()
        self._stop_playback_immediately()
        if self._logger:
            self._logger.append({"type": "turn_cancel", "reason": reason})

    @Slot()
    def request_stop(self) -> None:
        # stop everything
        self._stop_all.set()
        self._cancel_current_turn("request_stop")
        if self._ptt_stop:
            self._ptt_stop.set()

    # -------- Hands-free mode --------

    @Slot()
    def start_handsfree(self) -> None:
        if not self.settings.openai_api_key:
            self.error.emit("Chybí API key. Otevři OPEN AI a vlož API key.")
            return

        self._stop_all.clear()
        self._turn_cancel.clear()
        self._start_session_if_needed()

        self._append_caption("Hands-free: aktivní.")

        self._run_handsfree_loop()

    def _run_handsfree_loop(self) -> None:
        svc = OpenAIService(self.settings.openai_api_key)
        recorder = AudioRecorder(
            samplerate=self.settings.input_samplerate,
            device=self.settings.input_device,
            rms_threshold=self.settings.vad_rms_threshold,
            silence_ms=self.settings.vad_silence_ms,
            max_seconds=self.settings.max_record_seconds,
        )

        # noise calibration
        threshold = float(self.settings.vad_rms_threshold)
        try:
            noise = recorder.calibrate_noise(seconds=float(self.settings.vad_calibration_s))
            threshold = max(threshold, float(noise) * float(self.settings.vad_multiplier))
            if self._logger:
                self._logger.append({
                    "type": "handsfree_calibration",
                    "noise_rms": float(noise),
                    "vad_threshold": float(threshold),
                })
        except Exception as e:
            if self._logger:
                self._logger.append({"type": "handsfree_calibration_error", "error": str(e)})

        try:
            while not self._stop_all.is_set():
                self._turn_cancel.clear()
                self.state_changed.emit("listening")

                try:
                    rec = recorder.record_handsfree(cancel=self._stop_all, threshold=threshold)
                except Exception as e:
                    self.state_changed.emit("error")
                    self.error.emit(f"Nepodařilo se nahrát mikrofon: {e}")
                    return

                if self._stop_all.is_set():
                    return

                if self._logger:
                    self._logger.append({
                        "type": "audio_recorded",
                        "mode": "handsfree",
                        "duration_s": rec.duration_s,
                        "rms_median": rec.rms_median,
                    })

                ok = self._process_turn(svc=svc, wav_bytes=rec.wav_bytes, mode="handsfree")
                if not ok:
                    # On error/cancel in a single turn, continue listening unless stop_all.
                    if self._stop_all.is_set():
                        return

        finally:
            self.state_changed.emit("idle")
            self._end_session()

    # -------- PTT mode (hold-to-speak) --------

    @Slot()
    def ptt_pressed(self) -> None:
        if not self.settings.openai_api_key:
            self.error.emit("Chybí API key. Otevři OPEN AI a vlož API key.")
            return

        # barge-in by design: start speaking => cancel current output immediately
        self._cancel_current_turn("ptt_pressed")

        self._stop_all.clear()
        self._turn_cancel.clear()
        self._start_session_if_needed()

        # if a recording is already running, ignore
        if self._ptt_thread and self._ptt_thread.is_alive():
            return

        self.state_changed.emit("listening")
        self._append_caption("PTT: mluvte…")

        self._ptt_stop = threading.Event()
        recorder = AudioRecorder(
            samplerate=self.settings.input_samplerate,
            device=self.settings.input_device,
            rms_threshold=self.settings.vad_rms_threshold,
            silence_ms=self.settings.vad_silence_ms,
            max_seconds=self.settings.max_record_seconds,
        )

        def _rec() -> None:
            try:
                rr = recorder.record_ptt(stop_event=self._ptt_stop, cancel=self._stop_all)
            except Exception as e:
                # push error to UI thread via signal
                self.error.emit(f"Nepodařilo se nahrát mikrofon (PTT): {e}")
                self.state_changed.emit("error")
                return
            self.ptt_record_done.emit(rr)

        self._ptt_thread = threading.Thread(target=_rec, name="PTTRecorder", daemon=True)
        self._ptt_thread.start()

    @Slot()
    def ptt_released(self) -> None:
        if self._ptt_stop:
            self._ptt_stop.set()

    @Slot(object)
    def _on_ptt_record_done(self, rec: RecordResult) -> None:
        if self._stop_all.is_set():
            return
        if self._logger:
            self._logger.append({
                "type": "audio_recorded",
                "mode": "ptt",
                "duration_s": rec.duration_s,
                "rms_median": rec.rms_median,
            })
        svc = OpenAIService(self.settings.openai_api_key)
        self._turn_cancel.clear()
        self._process_turn(svc=svc, wav_bytes=rec.wav_bytes, mode="ptt")

    # -------- Core pipeline --------

    def _process_turn(self, *, svc: OpenAIService, wav_bytes: bytes, mode: str) -> bool:
        """Run STT -> LLM(stream) -> TTS -> playback. Returns True if completed."""
        try:
            self.state_changed.emit("transcribing")

            lang_hint = None if self.settings.language == "auto" else self.settings.language
            tr: TranscriptionResult
            try:
                tr = svc.transcribe_wav(wav_bytes, language_hint=lang_hint)
            except InvalidApiKeyError:
                self.state_changed.emit("error")
                self.error.emit("Neplatný API key. Otevři OPEN AI a vlož správný klíč.")
                return False
            except Exception as e:
                self.state_changed.emit("error")
                self.error.emit(f"Přepis (Whisper) selhal: {e}")
                return False

            user_text = (tr.text or "").strip()
            stt_lang = (tr.language or None)
            resolved_lang = _resolve_language(self.settings, stt_lang, self._last_lang)
            self._last_lang = resolved_lang

            if self._logger:
                self._logger.append({
                    "type": "stt_result",
                    "mode": mode,
                    "text": user_text,
                    "language_hint": lang_hint,
                    "detected_language": stt_lang,
                    "resolved_language": resolved_lang,
                })

            if not user_text or self._turn_cancel.is_set() or self._stop_all.is_set():
                return False

            self._append_caption(f"Ty: {user_text}")
            if self._logger:
                self._logger.append({"type": "user_text", "text": user_text, "text_line": f"Ty: {user_text}"})

            self._messages.append({"role": "user", "content": user_text})
            self._messages = self._messages[-20:]

            self.state_changed.emit("thinking")

            assistant_acc = ""

            def on_delta(d: str) -> None:
                nonlocal assistant_acc
                assistant_acc += d
                self._set_caption_preview("AI", assistant_acc)

            # Start barge-in monitor for this turn (thinking + speaking)
            vad_threshold = max(0.012, float(self.settings.vad_rms_threshold) * 1.7)
            monitor = VADMonitor(
                samplerate=self.settings.input_samplerate,
                device=self.settings.input_device,
                threshold=vad_threshold,
                trigger_ms=140,
            )
            monitor.start(lambda rms: self._cancel_current_turn("barge_in"))

            assistant_text = ""
            try:
                assistant_text = svc.chat_stream(
                    model=self.settings.chat_model,
                    system_prompt=build_system_prompt(self.settings, resolved_language=resolved_lang),
                    messages=self._messages,
                    temperature=float(self.settings.temperature),
                    max_output_tokens=int(self.settings.max_output_tokens),
                    on_delta=on_delta,
                    cancel=lambda: self._turn_cancel.is_set() or self._stop_all.is_set(),
                )
            except InvalidApiKeyError:
                monitor.stop()
                self.state_changed.emit("error")
                self.error.emit("Neplatný API key. Otevři OPEN AI a vlož správný klíč.")
                return False
            except Exception as e:
                monitor.stop()
                if self._turn_cancel.is_set():
                    return False
                self.state_changed.emit("error")
                self.error.emit(f"Chat selhal: {e}")
                return False

            if self._turn_cancel.is_set() or self._stop_all.is_set():
                monitor.stop()
                return False

            assistant_text = (assistant_text or "").strip()
            if assistant_text:
                self._messages.append({"role": "assistant", "content": assistant_text})
                self._messages = self._messages[-20:]
                self._append_caption(f"AI: {assistant_text}")
                if self._logger:
                    self._logger.append({"type": "assistant_text", "text": assistant_text, "text_line": f"AI: {assistant_text}"})

            # TTS voice fallback + warning
            voice, fallback_reason = _resolve_tts_voice(resolved_lang, self.settings.tts_voice)
            if fallback_reason:
                self._append_caption(f"(TTS) Použit fallback hlas: {voice}")
                if self._logger:
                    self._logger.append({"type": "tts_voice_fallback", "reason": fallback_reason, "resolved_voice": voice})

            self.state_changed.emit("speaking")

            pcm = b""
            try:
                pcm = svc.tts_pcm16(
                    text=assistant_text,
                    model=self.settings.tts_model,
                    voice=voice,
                    speed=float(self.settings.tts_speed),
                    response_format="pcm",
                )
            except InvalidApiKeyError:
                monitor.stop()
                self.state_changed.emit("error")
                self.error.emit("Neplatný API key. Otevři OPEN AI a vlož správný klíč.")
                return False
            except Exception as e:
                monitor.stop()
                if self._turn_cancel.is_set():
                    return False
                self.state_changed.emit("error")
                self.error.emit(f"TTS selhalo: {e}")
                return False

            if self._logger:
                self._logger.append({"type": "tts_bytes", "nbytes": len(pcm), "voice": voice, "speed": float(self.settings.tts_speed)})

            if self._turn_cancel.is_set() or self._stop_all.is_set():
                monitor.stop()
                return False

            # playback (stable buffering) with cancel
            player = AudioPlayer(samplerate=self.settings.tts_samplerate, device=self.settings.output_device)
            with self._player_lock:
                self._player = player
            try:
                player.play_pcm16(pcm, cancel=self._turn_cancel)
            finally:
                with self._player_lock:
                    self._player = None
                try:
                    player.stop()
                except Exception:
                    pass

            monitor.stop()
            self.state_changed.emit("idle")
            return True

        except Exception as e:
            # No unhandled exceptions should crash the app.
            if self._logger:
                self._logger.append({"type": "unhandled_exception", "error": str(e)})
            self.state_changed.emit("error")
            self.error.emit("Nastala neočekávaná chyba. Zkus to prosím znovu.")
            return False


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

        root.setStyleSheet("QWidget { background-color: #0b0f18; } QPushButton { padding: 8px 14px; }")  # noqa: E501

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
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
