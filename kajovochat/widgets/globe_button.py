from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QImage, QPainterPath, QRadialGradient, QColor
from PySide6.QtWidgets import QPushButton

from .sphere_renderer import SphereRenderer


class GlobeButton(QPushButton):
    """Push-to-talk button (hold to speak)."""

    ptt_pressed = Signal()
    ptt_released = Signal()

    def __init__(
        self,
        earth_texture_path: str,
        earth_clouds_path: Optional[str] = None,
        parent: Optional[QPushButton] = None,
    ) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(94, 94)
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")

        base = QImage(earth_texture_path)
        clouds = QImage(earth_clouds_path) if earth_clouds_path else None
        self._renderer = SphereRenderer(base, clouds_texture=clouds)
        self._angle = 0.0
        self._cloud_angle = 0.0
        self._last_img = None
        self._last_size = 0
        self._last_angle_q = -10**9

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        # Subtle idle spin; faster while pressed.
        self._angle = (self._angle + (0.18 if not self.isChecked() else 1.05)) % 360.0
        self._cloud_angle = (self._cloud_angle + (0.35 if not self.isChecked() else 1.65)) % 360.0
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.setChecked(True)
            self.ptt_pressed.emit()
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.isChecked():
            self.setChecked(False)
            self.ptt_released.emit()
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        # Safety: if user drags cursor away while holding, treat as release.
        if self.isChecked():
            self.setChecked(False)
            self.ptt_released.emit()
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        r = min(w, h) * 0.45

        # subtle glow only when active
        if self.isChecked():
            g = QRadialGradient(QPointF(cx, cy), r * 1.35)
            g.setColorAt(0.0, QColor(160, 220, 255, 80))
            g.setColorAt(0.7, QColor(160, 220, 255, 25))
            g.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setBrush(g)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), r * 1.25, r * 1.25)

        circle = QPainterPath()
        circle.addEllipse(QPointF(cx, cy), r, r)
        p.setClipPath(circle)

        # Earth sphere render.
        target = QRectF(cx - r, cy - r, r * 2.0, r * 2.0)
        img_size = max(48, int(r * 2.0))

        angle_q = int(self._angle * 2.0)  # 0.5° steps
        if self._last_img is None or self._last_size != img_size or self._last_angle_q != angle_q:
            self._last_img = self._renderer.render_earth(
                img_size,
                angle_q / 2.0,
                clouds_angle_deg=self._cloud_angle,
            )
            self._last_size = img_size
            self._last_angle_q = angle_q

        if self._last_img is not None and not self._last_img.isNull():
            p.drawImage(target, self._last_img)

        # border
        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(255, 255, 255, 90))
        p.drawEllipse(QPointF(cx, cy), r, r)
