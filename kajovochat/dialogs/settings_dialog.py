from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QFileDialog, QLineEdit, QMessageBox, QGroupBox, QFormLayout, QLabel
)

from ..settings import AppSettings, LANGUAGE_CHOICES, TTS_VOICES, language_label


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
            self._tts_voice.setCurrentText("nova" if "nova" in TTS_VOICES else TTS_VOICES[0])

        self._tts_speed = QLineEdit(f"{settings.tts_speed:.2f}")
        self._tts_speed.setValidator(QDoubleValidator(0.25, 4.0, 2, self))

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

        form.addRow(self._sep("Soubory"))
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._log_dir, 1)
        dir_row.addWidget(pick)
        form.addRow("Adresář logů:", dir_row)

        model_row = QHBoxLayout()
        model_row.addWidget(self._model, 1)
        model_row.addWidget(refresh)
        form.addRow("Model (chat):", model_row)

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

        self.settings.tts_voice = self._tts_voice.currentText() or self.settings.tts_voice

        try:
            sp = float(self._tts_speed.text().replace(",", "."))
        except Exception:
            sp = self.settings.tts_speed
        self.settings.tts_speed = max(0.25, min(4.0, sp))

        self.settings.log_dir = self._log_dir.text().strip() or self.settings.log_dir
        self.settings.chat_model = self._model.currentText() or self.settings.chat_model
        self.settings.ensure_log_dir()
