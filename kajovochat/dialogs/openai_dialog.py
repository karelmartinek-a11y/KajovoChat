from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox, QGroupBox, QFormLayout
)

from ..settings import AppSettings


class OpenAIDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("OPEN AI")
        self.setModal(True)
        self.settings = settings

        self._key = QLineEdit()
        self._key.setEchoMode(QLineEdit.Password)
        self._key.setPlaceholderText("Zadej API key…")
        if settings.openai_api_key:
            self._key.setText(settings.openai_api_key)

        self._show = QPushButton("Zobrazit")
        self._show.setCheckable(True)
        self._show.toggled.connect(self._toggle_show)

        self._save = QPushButton("Uložit")
        self._save.clicked.connect(self._save_key)

        self._delete = QPushButton("Smazat")
        self._delete.clicked.connect(self._delete_key)

        row = QHBoxLayout()
        row.addWidget(self._key, 1)
        row.addWidget(self._show)
        row.addWidget(self._save)
        row.addWidget(self._delete)

        form = QFormLayout()
        form.addRow("API key:", row)

        box = QGroupBox("Klíč")
        box.setLayout(form)

        btns = QHBoxLayout()
        ok = QPushButton("Zavřít")
        ok.clicked.connect(self.accept)
        btns.addStretch(1)
        btns.addWidget(ok)

        layout = QVBoxLayout()
        layout.addWidget(box)
        layout.addLayout(btns)
        self.setLayout(layout)

    def _toggle_show(self, on: bool) -> None:
        self._key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password)
        self._show.setText("Skrýt" if on else "Zobrazit")

    def _save_key(self) -> None:
        key = self._key.text().strip()
        if not key:
            QMessageBox.warning(self, "API key", "API key je prázdný.")
            return
        self.settings.openai_api_key = key
        QMessageBox.information(self, "API key", "API key uložen do nastavení (lokálně v PC).")

    def _delete_key(self) -> None:
        self._key.setText("")
        self.settings.openai_api_key = ""
        QMessageBox.information(self, "API key", "API key byl smazán.")
