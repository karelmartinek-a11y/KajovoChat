from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QPixmap, QPainterPath, QRadialGradient, QColor, QTransform, QFont
from PySide6.QtWidgets import QWidget


class OrbWidget(QWidget):
    orb_clicked = Signal()

    def __init__(self, moon_texture_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._pix = QPixmap(moon_texture_path)
        self._state = "idle"
        self._running = False  # nonstop mode on/off

        self._angle = 0.0
        self._breathe_phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.orb_clicked.emit()
        super().mousePressEvent(event)

    def _tick(self) -> None:
        dt = 0.016
        base_rot = 0.15 if self._running else 0.0
        if self._state == "speaking":
            base_rot += 0.35
        elif self._state == "listening":
            base_rot += 0.20

        self._angle = (self._angle + base_rot) % 360.0
        self._breathe_phase += dt * (1.6 if self._running else 0.7)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        size = min(w, h) * 0.66
        cx, cy = w / 2.0, h / 2.0
        r = size / 2.0

        t = self._breathe_phase
        breathe = 1.0 + 0.018 * math.sin(t)
        pulse = 0.0
        if self._state == "speaking":
            pulse = 0.020 * math.sin(t * 2.8)
        elif self._state == "listening":
            pulse = 0.015 * math.sin(t * 3.6)
        elif self._state == "thinking":
            pulse = 0.010 * math.sin(t * 1.3)

        scale = breathe + pulse
        rr = r * scale

        glow = 0.10
        if self._state == "listening":
            glow = 0.25
        elif self._state == "thinking":
            glow = 0.16
        elif self._state == "speaking":
            glow = 0.30
        elif self._state == "error":
            glow = 0.22

        grad = QRadialGradient(QPointF(cx, cy), rr * 1.35)
        grad.setColorAt(0.0, QColor(255, 255, 255, int(55 * glow)))
        grad.setColorAt(0.6, QColor(255, 255, 255, int(35 * glow)))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), rr * 1.35, rr * 1.35)

        shadow = QRadialGradient(QPointF(cx + rr*0.12, cy + rr*0.18), rr * 1.2)
        shadow.setColorAt(0.0, QColor(0, 0, 0, 110))
        shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(shadow)
        p.drawEllipse(QPointF(cx + rr*0.08, cy + rr*0.12), rr * 1.05, rr * 1.05)

        circle = QPainterPath()
        circle.addEllipse(QPointF(cx, cy), rr, rr)
        p.setClipPath(circle)

        if not self._pix.isNull():
            pix = self._pix
            target = QRectF(cx - rr, cy - rr, rr * 2.0, rr * 2.0)
            tr = QTransform()
            tr.translate(cx, cy)
            tr.rotate(self._angle)
            tr.translate(-cx, -cy)
            p.setTransform(tr, True)
            p.drawPixmap(target, pix, QRectF(0, 0, pix.width(), pix.height()))
            p.resetTransform()

        # vignette
        vign = QRadialGradient(QPointF(cx - rr*0.25, cy - rr*0.25), rr * 1.35)
        vign.setColorAt(0.0, QColor(255, 255, 255, 28))
        vign.setColorAt(0.55, QColor(255, 255, 255, 10))
        vign.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.setClipping(False)
        p.setBrush(vign)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), rr, rr)

        if not self._running:
            p.setPen(QColor(220, 220, 220, 190))
            f = QFont()
            f.setPointSize(12)
            p.setFont(f)
            msg = "Klikni na Měsíc pro nonstop režim" if self._state == "idle" else ""
            if msg:
                p.drawText(QRectF(0, cy + rr + 10, w, 30), Qt.AlignHCenter | Qt.AlignTop, msg)
