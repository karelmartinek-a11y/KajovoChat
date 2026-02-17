from __future__ import annotations

import logging
import os
import sys
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot, QProcess, QProcessEnvironment
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QLabel,
)

from ..settings import AppSettings, LANGUAGE_CHOICES, TTS_VOICES
from ..services.audio_service import list_audio_devices


_VOICE_SAMPLE_TEXT = {
    "cs": "Ahoj, tady je ukázka vybraného hlasu. Jak vám mohu pomoci?",
    "sk": "Ahoj, toto je ukážka vybraného hlasu. Ako vám môžem pomôcť?",
    "en": "Hello. This is a short preview of the selected voice. How can I help you?",
    "de": "Hallo. Das ist eine kurze Vorschau der ausgewählten Stimme. Wie kann ich helfen?",
    "fr": "Bonjour. Voici un court aperçu de la voix sélectionnée. Comment puis-je aider ?",
}

REALTIME_MODELS = [
    "gpt-realtime",
    "gpt-realtime-mini",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
]

LoadModelsFn = Callable[[], List[str]]


class _ModelLoaderWorker(QObject):
    finished = Signal(list)
    failed = Signal(str)

    def __init__(self, fn: LoadModelsFn) -> None:
        super().__init__()
        self._fn = fn

    @Slot()
    def run(self) -> None:
        try:
            models = self._fn() or []
            self.finished.emit(list(models))
        except Exception as e:
            self.failed.emit(str(e))


class SettingsDialog(QDialog):
    """
    Minimal, robust settings UI:
    - always opens (no blocking network work on UI thread)
    - shows only options that are used by the app runtime
    - voice preview runs in a separate process to avoid crashing the main UI
    """

    def __init__(self, settings: AppSettings, load_models_fn: Optional[LoadModelsFn] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NASTAVENÍ")
        self.setModal(True)

        self._log = logging.getLogger("kajovochat.settings")
        self.settings = settings
        self.load_models_fn = load_models_fn

        # Language
        self._lang = QComboBox()
        for code, lbl in LANGUAGE_CHOICES:
            self._lang.addItem(lbl, userData=code)
        self._set_lang_current(settings.language)

        # Model (chat)
        self._model = QComboBox()
        self._model.setEditable(True)
        if settings.chat_model:
            self._model.addItem(settings.chat_model)
        self._model.setCurrentText(settings.chat_model or "")

        self._refresh_models_btn = QPushButton("Načíst modely")
        self._refresh_models_btn.clicked.connect(lambda _=False: self._refresh_models_async())

        # Realtime model (voice)
        self._realtime_model = QComboBox()
        self._realtime_model.setEditable(True)
        for m in REALTIME_MODELS:
            self._realtime_model.addItem(m)
        if settings.realtime_model and settings.realtime_model not in REALTIME_MODELS:
            self._realtime_model.addItem(settings.realtime_model)
        self._realtime_model.setCurrentText(settings.realtime_model or (REALTIME_MODELS[0] if REALTIME_MODELS else ""))

        # LLM params
        self._temperature = QLineEdit(f"{settings.temperature:.2f}")
        self._temperature.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        self._max_tokens = QLineEdit(str(settings.max_output_tokens))
        self._max_tokens.setValidator(QIntValidator(64, 4096, self))

        # TTS
        self._tts_voice = QComboBox()
        self._tts_voice.addItems(TTS_VOICES)
        if settings.tts_voice in TTS_VOICES:
            self._tts_voice.setCurrentText(settings.tts_voice)
        elif TTS_VOICES:
            self._tts_voice.setCurrentText(TTS_VOICES[0])

        self._tts_speed = QLineEdit(f"{settings.tts_speed:.2f}")
        self._tts_speed.setValidator(QDoubleValidator(0.25, 4.0, 2, self))

        self._tts_preview_btn = QPushButton("Ukázka")
        self._tts_preview_btn.clicked.connect(lambda _=False: self._toggle_tts_preview())
        self._preview_proc: Optional[QProcess] = None

        # Audio devices
        self._input_dev = QComboBox()
        self._output_dev = QComboBox()
        self._refresh_devices_btn = QPushButton("Načíst zařízení")
        self._refresh_devices_btn.clicked.connect(lambda _=False: self._refresh_devices())
        self._refresh_devices()

        # VAD / recording (hands-free)
        self._vad_thr = QLineEdit(f"{settings.vad_rms_threshold:.4f}")
        self._vad_thr.setValidator(QDoubleValidator(0.0, 1.0, 6, self))

        self._vad_silence = QLineEdit(str(settings.vad_silence_ms))
        self._vad_silence.setValidator(QIntValidator(100, 5000, self))

        self._vad_calib = QLineEdit(f"{settings.vad_calibration_s:.2f}")
        self._vad_calib.setValidator(QDoubleValidator(0.2, 5.0, 2, self))

        self._vad_mult = QLineEdit(f"{settings.vad_multiplier:.2f}")
        self._vad_mult.setValidator(QDoubleValidator(1.0, 20.0, 2, self))

        self._max_rec = QLineEdit(str(settings.max_record_seconds))
        self._max_rec.setValidator(QIntValidator(3, 120, self))

        # Logs directory
        self._log_dir = QLineEdit(settings.log_dir)
        self._log_dir.setReadOnly(True)
        pick = QPushButton("Vybrat…")
        pick.clicked.connect(lambda _=False: self._pick_dir())

        # Layout
        form = QFormLayout()
        form.addRow("Jazyk konverzace:", self._lang)

        form.addRow(self._sep("Modely"))
        model_row = QHBoxLayout()
        model_row.addWidget(self._model, 1)
        model_row.addWidget(self._refresh_models_btn)
        form.addRow("Chat model:", model_row)
        form.addRow("Realtime (hlas):", self._realtime_model)

        form.addRow(self._sep("Odpovědi"))
        form.addRow("Temperature (0–1):", self._temperature)
        form.addRow("Max output tokens:", self._max_tokens)

        form.addRow(self._sep("TTS"))
        voice_row = QHBoxLayout()
        voice_row.addWidget(self._tts_voice, 1)
        voice_row.addWidget(self._tts_preview_btn)
        form.addRow("Hlas:", voice_row)
        form.addRow("Rychlost (0.25–4.0):", self._tts_speed)

        form.addRow(self._sep("Audio"))
        in_row = QHBoxLayout()
        in_row.addWidget(self._input_dev, 1)
        in_row.addWidget(self._refresh_devices_btn)
        form.addRow("Vstup (mikrofon):", in_row)
        form.addRow("Výstup (reproduktory):", self._output_dev)

        form.addRow(self._sep("Detekce hlasu (hands-free)"))
        form.addRow("RMS práh (base):", self._vad_thr)
        form.addRow("Ticho pro ukončení (ms):", self._vad_silence)
        form.addRow("Kalibrace šumu (s):", self._vad_calib)
        form.addRow("Násobič kalibrace:", self._vad_mult)
        form.addRow("Max délka záznamu (s):", self._max_rec)

        form.addRow(self._sep("Soubory"))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._log_dir, 1)
        dir_row.addWidget(pick)
        form.addRow("Adresář logů:", dir_row)

        box = QGroupBox("Nastavení")
        box.setLayout(form)

        btns = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Zrušit")
        ok.clicked.connect(lambda _=False: self.accept())
        cancel.clicked.connect(lambda _=False: self.reject())
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)

        layout = QVBoxLayout()
        layout.addWidget(box)
        layout.addLayout(btns)
        self.setLayout(layout)

        # model loading thread state
        self._model_thread: Optional[QThread] = None
        self._model_worker: Optional[_ModelLoaderWorker] = None

    @staticmethod
    def _sep(title: str) -> QLabel:
        return QLabel(f"<b>{title}</b>")

    def _set_lang_current(self, code: str) -> None:
        for i in range(self._lang.count()):
            if self._lang.itemData(i) == code:
                self._lang.setCurrentIndex(i)
                return
        # fallback to auto
        for i in range(self._lang.count()):
            if self._lang.itemData(i) == "auto":
                self._lang.setCurrentIndex(i)
                return

    def _pick_dir(self) -> None:
        try:
            p = QFileDialog.getExistingDirectory(self, "Vyber adresář pro logy", self.settings.log_dir)
            if p:
                self._log_dir.setText(p)
        except Exception:
            self._log.exception("pick_dir_failed")

    def _sample_text(self) -> str:
        lang = str(self._lang.currentData() or self.settings.language or "auto")
        if lang == "auto":
            lang = "cs"
        return _VOICE_SAMPLE_TEXT.get(lang, _VOICE_SAMPLE_TEXT["cs"])

    def _toggle_tts_preview(self) -> None:
        """
        Run preview in a helper process so audio-stack crashes cannot kill the main UI.
        """
        try:
            if self._preview_proc is not None and self._preview_proc.state() == QProcess.Running:
                self._preview_proc.terminate()
                if not self._preview_proc.waitForFinished(800):
                    self._preview_proc.kill()
                return

            voice = (self._tts_voice.currentText() or self.settings.tts_voice or "alloy").strip()
            try:
                speed = float(self._tts_speed.text().replace(",", "."))
            except Exception:
                speed = float(self.settings.tts_speed)
            speed = max(0.25, min(4.0, float(speed)))

            output_dev = self._output_dev.currentData()
            out_arg = str(int(output_dev)) if output_dev is not None else "-1"

            proc = QProcess(self)
            proc.setProgram(sys.executable)
            env = QProcessEnvironment.systemEnvironment()
            # předáme API klíč z nastavení (voice_preview očekává KAJOVOCHAT_API_KEY)
            if self.settings.openai_api_key:
                env.insert("KAJOVOCHAT_API_KEY", self.settings.openai_api_key)
                env.insert("OPENAI_API_KEY", self.settings.openai_api_key)
            proc.setProcessEnvironment(env)
            proc.setArguments(
                [
                    "-m",
                    "kajovochat.tools.voice_preview",
                    "--model",
                    str(self.settings.tts_model),
                    "--voice",
                    str(voice),
                    "--speed",
                    str(speed),
                    "--output_device",  # argparse používá podtržítko
                    out_arg,
                    "--text",
                    self._sample_text(),
                    "--lang",
                    (self._lang.currentData() or "cs"),
                ]
            )
            proc.readyReadStandardError.connect(lambda: self._on_preview_stderr(proc))
            proc.readyReadStandardOutput.connect(lambda: self._on_preview_stdout(proc))
            proc.finished.connect(lambda code, status: self._on_preview_finished(code, status))
            self._preview_proc = proc
            self._tts_preview_btn.setText("Stop")
            proc.start()
        except Exception:
            self._log.exception("voice_preview_start_failed")
            QMessageBox.warning(self, "Ukázka hlasu", "Ukázku se nepodařilo spustit. Podrobnosti jsou v logu.")
            self._tts_preview_btn.setText("Ukázka")

    def _on_preview_stdout(self, proc: QProcess) -> None:
        try:
            data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()
            if data:
                self._log.info("voice_preview_stdout %s", data)
        except Exception:
            pass

    def _on_preview_stderr(self, proc: QProcess) -> None:
        try:
            data = bytes(proc.readAllStandardError()).decode("utf-8", errors="ignore").strip()
            if data:
                self._log.warning("voice_preview_stderr %s", data)
        except Exception:
            pass

    def _on_preview_finished(self, code: int, status) -> None:
        try:
            self._log.info("voice_preview_finished code=%s status=%s", code, status)
        except Exception:
            pass
        self._tts_preview_btn.setText("Ukázka")
        self._preview_proc = None
        if code != 0:
            QMessageBox.warning(self, "Ukázka hlasu", "Ukázka selhala. Podrobnosti jsou v logu.")

    def _refresh_devices(self) -> None:
        try:
            info = list_audio_devices() or {"inputs": [], "outputs": []}
            ins = info.get("inputs") or []
            outs = info.get("outputs") or []

            self._input_dev.clear()
            self._output_dev.clear()

            self._input_dev.addItem("Systémová výchozí", userData=None)
            for d in ins:
                idx = int(d.get("index", -1))
                name = str(d.get("name", f"Input {idx}"))
                self._input_dev.addItem(name, userData=idx)

            self._output_dev.addItem("Systémová výchozí", userData=None)
            for d in outs:
                idx = int(d.get("index", -1))
                name = str(d.get("name", f"Output {idx}"))
                self._output_dev.addItem(name, userData=idx)

            # restore current selection
            self._set_current_device(self._input_dev, self.settings.input_device)
            self._set_current_device(self._output_dev, self.settings.output_device)
        except Exception:
            self._log.exception("refresh_devices_failed")

    @staticmethod
    def _set_current_device(combo: QComboBox, wanted: Optional[int]) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == wanted:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _refresh_models_async(self) -> None:
        if not self.load_models_fn:
            QMessageBox.information(self, "Modely", "API key není nastaven. Otevřete OPEN AI a vložte API key.")
            return

        if self._model_thread is not None and self._model_thread.isRunning():
            return

        self._refresh_models_btn.setEnabled(False)
        self._refresh_models_btn.setText("Načítám…")

        self._model_thread = QThread(self)
        self._model_worker = _ModelLoaderWorker(self.load_models_fn)
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._on_models_loaded)
        self._model_worker.failed.connect(self._on_models_failed)
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.failed.connect(self._model_thread.quit)
        self._model_thread.finished.connect(lambda: self._refresh_models_btn.setEnabled(True))
        self._model_thread.finished.connect(lambda: self._refresh_models_btn.setText("Načíst modely"))
        self._model_thread.start()

    def _on_models_loaded(self, models: list) -> None:
        try:
            cur = self._model.currentText().strip() or self.settings.chat_model
            self._model.clear()
            if cur:
                self._model.addItem(cur)
            for m in models:
                if m and m != cur:
                    self._model.addItem(str(m))
            if cur:
                self._model.setCurrentText(cur)
        except Exception:
            self._log.exception("models_loaded_failed")

    def _on_models_failed(self, msg: str) -> None:
        self._log.warning("model_refresh_failed %s", msg)
        QMessageBox.warning(self, "Modely", "Nepodařilo se načíst modely. Podrobnosti jsou v logu.")

    def apply(self) -> None:
        # language
        self.settings.language = str(self._lang.currentData() or "auto")

        # numeric fields
        try:
            t = float(self._temperature.text().replace(",", "."))
        except Exception:
            t = self.settings.temperature
        self.settings.temperature = min(1.0, max(0.0, t))

        try:
            mt = int(self._max_tokens.text())
        except Exception:
            mt = self.settings.max_output_tokens
        self.settings.max_output_tokens = max(64, min(4096, mt))

        # models
        self.settings.chat_model = self._model.currentText().strip() or self.settings.chat_model
        self.settings.realtime_model = self._realtime_model.currentText().strip() or self.settings.realtime_model

        # TTS
        chosen_voice = (self._tts_voice.currentText() or self.settings.tts_voice).strip()
        if TTS_VOICES and chosen_voice not in TTS_VOICES:
            chosen_voice = TTS_VOICES[0]
        self.settings.tts_voice = chosen_voice

        try:
            sp = float(self._tts_speed.text().replace(",", "."))
        except Exception:
            sp = self.settings.tts_speed
        self.settings.tts_speed = max(0.25, min(4.0, float(sp)))

        # audio devices
        self.settings.input_device = self._input_dev.currentData()
        self.settings.output_device = self._output_dev.currentData()

        # VAD/recording
        try:
            thr = float(self._vad_thr.text().replace(",", "."))
        except Exception:
            thr = self.settings.vad_rms_threshold
        self.settings.vad_rms_threshold = max(0.0, min(1.0, float(thr)))

        try:
            sm = int(self._vad_silence.text())
        except Exception:
            sm = self.settings.vad_silence_ms
        self.settings.vad_silence_ms = max(100, min(5000, sm))

        try:
            cs = float(self._vad_calib.text().replace(",", "."))
        except Exception:
            cs = self.settings.vad_calibration_s
        self.settings.vad_calibration_s = max(0.2, min(5.0, float(cs)))

        try:
            vm = float(self._vad_mult.text().replace(",", "."))
        except Exception:
            vm = self.settings.vad_multiplier
        self.settings.vad_multiplier = max(1.0, min(20.0, float(vm)))

        try:
            mx = int(self._max_rec.text())
        except Exception:
            mx = self.settings.max_record_seconds
        self.settings.max_record_seconds = max(3, min(120, mx))

        # log dir
        self.settings.log_dir = self._log_dir.text().strip() or self.settings.log_dir
        try:
            self.settings.ensure_log_dir()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        # stop preview process if still running
        try:
            if self._preview_proc is not None and self._preview_proc.state() == QProcess.Running:
                self._preview_proc.kill()
        except Exception:
            pass
        super().closeEvent(event)
