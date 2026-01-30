from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPixmap, QPainterPath, QRadialGradient, QColor, QTransform
from PySide6.QtWidgets import QPushButton


class GlobeButton(QPushButton):
    toggled_on = Signal(bool)

    def __init__(self, earth_texture_path: str, parent: Optional[QPushButton] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(94, 94)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")

        self._pix = QPixmap(earth_texture_path)
        self._angle = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self.toggled.connect(self._emit)

    def _emit(self, checked: bool) -> None:
        self.toggled_on.emit(checked)
        self.update()

    def _tick(self) -> None:
        if self.isChecked():
            self._angle = (self._angle + 0.9) % 360.0
            self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        cx, cy = w/2.0, h/2.0
        r = min(w, h) * 0.45

        # subtle glow only when active
        if self.isChecked():
            g = QRadialGradient(QPointF(cx, cy), r*1.35)
            g.setColorAt(0.0, QColor(160, 220, 255, 80))
            g.setColorAt(0.7, QColor(160, 220, 255, 25))
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(g)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), r*1.25, r*1.25)

        circle = QPainterPath()
        circle.addEllipse(QPointF(cx, cy), r, r)
        p.setClipPath(circle)

        if not self._pix.isNull():
            pix = self._pix
            target = QRectF(cx - r, cy - r, r*2.0, r*2.0)
            tr = QTransform()
            tr.translate(cx, cy)
            tr.rotate(self._angle)
            tr.translate(-cx, -cy)
            p.setTransform(tr, True)
            p.drawPixmap(target, pix, QRectF(0, 0, pix.width(), pix.height()))
            p.resetTransform()

        # border
        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(255,255,255, 90))
        p.drawEllipse(QPointF(cx, cy), r, r)
