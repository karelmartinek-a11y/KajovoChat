from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QImage, QPainterPath, QRadialGradient, QColor, QFont
from PySide6.QtWidgets import QWidget

from .sphere_renderer import SphereRenderer


class OrbWidget(QWidget):
    orb_clicked = Signal()

    def __init__(self, moon_texture_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        moon_img = QImage(moon_texture_path)
        self._renderer = SphereRenderer(moon_img)

        # Render cache (avoid realloc churn).
        self._last_img: Optional[QImage] = None
        self._last_size: int = 0
        self._last_angle_q: int = -10**9
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

        # --- Moon (physically shaded sphere) ---
        target = QRectF(cx - rr, cy - rr, rr * 2.0, rr * 2.0)
        img_size = max(64, int(rr * 2.0))

        # Quantize angle to reduce unnecessary rerenders.
        angle_q = int(self._angle * 2.0)  # 0.5° steps
        if self._last_img is None or self._last_size != img_size or self._last_angle_q != angle_q:
            self._last_img = self._renderer.render_moon(img_size, angle_q / 2.0)
            self._last_size = img_size
            self._last_angle_q = angle_q

        if self._last_img is not None and not self._last_img.isNull():
            p.drawImage(target, self._last_img)

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
