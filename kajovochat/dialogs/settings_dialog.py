from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QFileDialog, QLineEdit, QMessageBox, QGroupBox, QFormLayout, QLabel
)

from ..settings import AppSettings, LANGUAGE_CHOICES, TTS_VOICES, language_label
from ..services.audio_service import list_audio_devices


REALTIME_MODELS = [
    "gpt-realtime",
    "gpt-realtime-mini",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
]


LoadModelsFn = Callable[[], List[str]]


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, load_models_fn: Optional[LoadModelsFn] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NASTAVENÍ")
        self.setModal(True)
        self.settings = settings
        self.load_models_fn = load_models_fn

        # Response
        self._style = QComboBox()
        self._style.addItems(["obsáhlé", "věcné", "exaktní", "strohé"])
        self._style.setCurrentText(settings.response_style)

        self._length = QComboBox()
        self._length.addItems(["krátké", "normální", "dlouhé"])
        self._length.setCurrentText(settings.response_length)

        self._detail = QComboBox()
        self._detail.addItems(["stručná", "detailní"])
        self._detail.setCurrentText(settings.response_detail)

        # Language + formality
        self._lang = QComboBox()
        for code, lbl in LANGUAGE_CHOICES:
            self._lang.addItem(lbl, userData=code)
        self._set_lang_current(settings.language)

        self._formality = QComboBox()
        self._formality.addItems(["vykání", "tykání"])
        self._formality.setCurrentText(settings.formality)

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
        else:
            self._tts_voice.setCurrentText(TTS_VOICES[0] if TTS_VOICES else "")

        self._tts_speed = QLineEdit(f"{settings.tts_speed:.2f}")
        self._tts_speed.setValidator(QDoubleValidator(0.25, 4.0, 2, self))

        # Audio devices
        self._input_dev = QComboBox()
        self._output_dev = QComboBox()
        self._refresh_devices_btn = QPushButton("Načíst zařízení")
        self._refresh_devices_btn.clicked.connect(self._refresh_devices)
        self._refresh_devices()  # initial fill

        # VAD / recording
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

        # Logs
        self._log_dir = QLineEdit(settings.log_dir)
        self._log_dir.setReadOnly(True)
        pick = QPushButton("Vybrat…")
        pick.clicked.connect(self._pick_dir)

        # Model
        self._model = QComboBox()
        self._model.setEditable(False)
        self._model.addItem(settings.chat_model)
        self._model.setCurrentText(settings.chat_model)

        # Realtime (voice)
        self._realtime_model = QComboBox()
        self._realtime_model.setEditable(False)
        for m in REALTIME_MODELS:
            self._realtime_model.addItem(m)
        if settings.realtime_model and settings.realtime_model not in REALTIME_MODELS:
            self._realtime_model.addItem(settings.realtime_model)
        self._realtime_model.setCurrentText(settings.realtime_model)

        refresh = QPushButton("Načíst modely")
        refresh.clicked.connect(self._refresh_models)

        form = QFormLayout()
        form.addRow("Jazyk konverzace:", self._lang)
        form.addRow("Tykání/Vykání:", self._formality)

        form.addRow(self._sep("Odpovědi"))
        form.addRow("Styl:", self._style)
        form.addRow("Délka:", self._length)
        form.addRow("Režim:", self._detail)
        form.addRow("Temperature (0–1):", self._temperature)
        form.addRow("Max output tokens:", self._max_tokens)

        form.addRow(self._sep("TTS"))
        form.addRow("Hlas:", self._tts_voice)
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

        model_row = QHBoxLayout()
        model_row.addWidget(self._model, 1)
        model_row.addWidget(refresh)
        form.addRow("Model (chat):", model_row)

        form.addRow("Model (hlas / realtime):", self._realtime_model)

        box = QGroupBox("Předvolby")
        box.setLayout(form)

        btns = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Zrušit")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(cancel)
        btns.addWidget(ok)

        layout = QVBoxLayout()
        layout.addWidget(box)
        layout.addLayout(btns)
        self.setLayout(layout)

    @staticmethod
    def _sep(title: str) -> QLabel:
        lbl = QLabel(f"<b>{title}</b>")
        return lbl

    def _set_lang_current(self, code: str) -> None:
        for i in range(self._lang.count()):
            if self._lang.itemData(i) == code:
                self._lang.setCurrentIndex(i)
                return
        # fallback
        for i in range(self._lang.count()):
            if self._lang.itemData(i) == "auto":
                self._lang.setCurrentIndex(i)
                return

    def _pick_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Vyber adresář pro logy", self.settings.log_dir)
        if p:
            self._log_dir.setText(p)

    def _refresh_models(self) -> None:
        if not self.load_models_fn:
            QMessageBox.information(self, "Modely", "Načtení modelů není dostupné (chybí API key).")
            return
        try:
            models = self.load_models_fn()
        except Exception as e:
            QMessageBox.critical(self, "Chyba", f"Načtení modelů selhalo:\n{e}")
            return
        if not models:
            QMessageBox.information(self, "Modely", "Nebyl nalezen žádný model.")
            return
        current = self._model.currentText() or self.settings.chat_model
        self._model.clear()
        self._model.addItems(models)
        if current in models:
            self._model.setCurrentText(current)

    def _refresh_devices(self) -> None:
        devs = list_audio_devices()

        def _fill(box: QComboBox, items: list, current: Optional[int]) -> None:
            box.blockSignals(True)
            box.clear()
            box.addItem("Default", userData=None)
            for d in items:
                idx = int(d["index"])
                name = str(d["name"])
                ch = int(d.get("max_channels", 0) or 0)
                box.addItem(f"{idx}: {name} ({ch}ch)", userData=idx)
            # restore selection
            if current is None:
                box.setCurrentIndex(0)
            else:
                for i in range(box.count()):
                    if box.itemData(i) == current:
                        box.setCurrentIndex(i)
                        break
            box.blockSignals(False)

        _fill(self._input_dev, devs.get("inputs", []), self.settings.input_device)
        _fill(self._output_dev, devs.get("outputs", []), self.settings.output_device)

    def apply(self) -> None:
        self.settings.response_style = self._style.currentText()
        self.settings.response_length = self._length.currentText()
        self.settings.response_detail = self._detail.currentText()

        self.settings.language = str(self._lang.currentData() or "auto")
        self.settings.formality = self._formality.currentText()

        # parse numeric fields safely
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

        chosen_voice = self._tts_voice.currentText() or self.settings.tts_voice
        self.settings.tts_voice = chosen_voice if chosen_voice in TTS_VOICES else (TTS_VOICES[0] if TTS_VOICES else chosen_voice)

        try:
            sp = float(self._tts_speed.text().replace(",", "."))
        except Exception:
            sp = self.settings.tts_speed
        self.settings.tts_speed = max(0.25, min(4.0, sp))

        # audio devices
        self.settings.input_device = self._input_dev.currentData()
        self.settings.output_device = self._output_dev.currentData()

        # VAD/recording
        try:
            thr = float(self._vad_thr.text().replace(",", "."))
        except Exception:
            thr = self.settings.vad_rms_threshold
        self.settings.vad_rms_threshold = max(0.0, min(1.0, thr))

        try:
            sm = int(self._vad_silence.text())
        except Exception:
            sm = self.settings.vad_silence_ms
        self.settings.vad_silence_ms = max(100, min(5000, sm))

        try:
            cs = float(self._vad_calib.text().replace(",", "."))
        except Exception:
            cs = self.settings.vad_calibration_s
        self.settings.vad_calibration_s = max(0.2, min(5.0, cs))

        try:
            vm = float(self._vad_mult.text().replace(",", "."))
        except Exception:
            vm = self.settings.vad_multiplier
        self.settings.vad_multiplier = max(1.0, min(20.0, vm))

        try:
            mx = int(self._max_rec.text())
        except Exception:
            mx = self.settings.max_record_seconds
        self.settings.max_record_seconds = max(3, min(120, mx))

        self.settings.log_dir = self._log_dir.text().strip() or self.settings.log_dir
        self.settings.chat_model = self._model.currentText() or self.settings.chat_model
        self.settings.realtime_model = self._realtime_model.currentText() or self.settings.realtime_model
        self.settings.ensure_log_dir()
