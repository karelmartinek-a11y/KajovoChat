from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QFileDialog, QLineEdit, QMessageBox, QGroupBox, QFormLayout
)

from ..settings import AppSettings


LoadModelsFn = Callable[[], List[str]]


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, load_models_fn: Optional[LoadModelsFn] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NASTAVENÍ")
        self.setModal(True)
        self.settings = settings
        self.load_models_fn = load_models_fn

        self._style = QComboBox()
        self._style.addItems(["obsáhlé", "věcné", "exaktní", "strohé"])
        self._style.setCurrentText(settings.response_style)

        self._length = QComboBox()
        self._length.addItems(["krátké", "normální", "dlouhé"])
        self._length.setCurrentText(settings.response_length)

        self._voice_lang = QComboBox()
        self._voice_lang.addItems(["česky", "slovensky", "německy", "anglicky", "francouzsky"])
        self._voice_lang.setCurrentText(settings.voice_language)

        self._voice_gender = QComboBox()
        self._voice_gender.addItems(["ženský", "mužský"])
        self._voice_gender.setCurrentText(settings.voice_gender)

        self._log_dir = QLineEdit(settings.log_dir)
        self._log_dir.setReadOnly(True)
        pick = QPushButton("Vybrat…")
        pick.clicked.connect(self._pick_dir)

        self._model = QComboBox()
        self._model.setEditable(False)
        self._model.addItem(settings.chat_model)
        self._model.setCurrentText(settings.chat_model)

        refresh = QPushButton("Načíst modely")
        refresh.clicked.connect(self._refresh_models)

        form = QFormLayout()
        form.addRow("Styl odpovědi:", self._style)
        form.addRow("Délka odpovědi:", self._length)
        form.addRow("Jazyk hlasu:", self._voice_lang)
        form.addRow("Druh hlasu:", self._voice_gender)

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
        self.settings.voice_language = self._voice_lang.currentText()
        self.settings.voice_gender = self._voice_gender.currentText()
        self.settings.log_dir = self._log_dir.text().strip() or self.settings.log_dir
        self.settings.chat_model = self._model.currentText() or self.settings.chat_model
        self.settings.ensure_log_dir()
