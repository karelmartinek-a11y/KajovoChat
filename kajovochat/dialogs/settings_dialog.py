from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..settings import (
    ANSWER_LANGUAGE_MODE_CHOICES,
    AppSettings,
    LANGUAGE_CHOICES,
    RESPONSE_STYLE_CHOICES,
)


class SettingsDialog(QDialog):
    """Minimální produktové nastavení bez technických voleb."""

    def __init__(self, settings: AppSettings, load_models_fn=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nastavení")
        self.setModal(True)
        self.settings = settings
        self.load_models_fn = load_models_fn

        self._answer_language_mode = QComboBox()
        for code, label in ANSWER_LANGUAGE_MODE_CHOICES:
            self._answer_language_mode.addItem(label, userData=code)
        self._set_current(self._answer_language_mode, settings.answer_language_mode)
        self._answer_language_mode.currentIndexChanged.connect(self._sync_visibility)

        self._fixed_answer_language = QComboBox()
        for code, label in LANGUAGE_CHOICES:
            self._fixed_answer_language.addItem(label, userData=code)
        self._set_current(self._fixed_answer_language, settings.fixed_answer_language)

        self._response_style = QComboBox()
        for code, label in RESPONSE_STYLE_CHOICES:
            self._response_style.addItem(label, userData=code)
        self._set_current(self._response_style, settings.response_style)

        self._fixed_language_label = QLabel("Jazyk odpovědi:")
        self._fixed_language_hint = QLabel(
            "Audio profil, model, hlas, VAD i výběr zařízení jsou řízené interně a v běžném UI se nenastavují."
        )
        self._fixed_language_hint.setWordWrap(True)
        self._fixed_language_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        form = QFormLayout()
        form.addRow("Režim jazyka odpovědi:", self._answer_language_mode)
        form.addRow(self._fixed_language_label, self._fixed_answer_language)
        form.addRow("Styl odpovědi:", self._response_style)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Zrušit")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self._fixed_language_hint)
        layout.addLayout(buttons)
        self.setLayout(layout)
        self._sync_visibility()

    @staticmethod
    def _set_current(combo: QComboBox, wanted: str) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == wanted:
                combo.setCurrentIndex(idx)
                return
        combo.setCurrentIndex(0)

    def _sync_visibility(self) -> None:
        fixed_mode = (self._answer_language_mode.currentData() or "follow_input") == "fixed"
        self._fixed_language_label.setVisible(fixed_mode)
        self._fixed_answer_language.setVisible(fixed_mode)

    def apply(self) -> None:
        self.settings.answer_language_mode = str(self._answer_language_mode.currentData() or "follow_input")
        self.settings.fixed_answer_language = str(self._fixed_answer_language.currentData() or "cs")
        self.settings.response_style = str(self._response_style.currentData() or "normální")
