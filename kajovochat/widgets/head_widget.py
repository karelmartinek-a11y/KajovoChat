from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..orb.widget import LivingOrbWidget


class HeadWidget(QWidget):
    """Kompatibilní wrapper nad GPU living orb widgetem."""

    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, image_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._orb = LivingOrbWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._orb)

        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._reset_rect = QRectF()
        self._mouth_energy = 0.0
        self._aurora_bias = 0.0

        self._orb.orb_clicked.connect(lambda: self.orb_clicked.emit())
        self._orb.reset_clicked.connect(lambda: self.reset_clicked.emit())

    def _sync_reset_rect(self) -> None:
        if self._orb._reset_rect.width() > 0.0:
            self._reset_rect = self._orb._reset_rect
            return
        if self._state == "error":
            w = float(max(1, self.width()))
            h = float(max(1, self.height()))
            self._reset_rect = QRectF(w * 0.5 - 75.0, h * 0.78 + 34.0, 150.0, 38.0)
        else:
            self._reset_rect = QRectF()

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        self._orb.set_state(self._state)
        self._sync_reset_rect()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self._orb.set_running(self._running)

    def set_input_level(self, level: float) -> None:
        self._orb.set_input_level(level)

    def set_output_level(self, level: float) -> None:
        self._orb.set_output_level(level)

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        self._orb.set_lipsync_snapshot(snapshot)
        self._mouth_energy = self._orb._mouth_energy
        self._aurora_bias = self._orb._aurora_bias

    def set_error_text(self, msg: str) -> None:
        self._error_text = (msg or "").strip()
        self._orb.set_error_text(self._error_text)
        self._sync_reset_rect()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        self._sync_reset_rect()

    def closeEvent(self, event) -> None:
        try:
            self._orb.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
