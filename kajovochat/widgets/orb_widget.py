from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QTimer, Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QImage,
    QPainterPath,
    QRadialGradient,
    QConicalGradient,
    QPen,
    QColor,
    QFont,
)
from PySide6.QtWidgets import QWidget

from .sphere_renderer import SphereRenderer


class OrbWidget(QWidget):
    orb_clicked = Signal()
    reset_clicked = Signal()

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

        # Audio reactive levels (0..1). Set by MainWindow.
        self._in_level_target = 0.0
        self._out_level_target = 0.0
        self._in_level = 0.0
        self._out_level = 0.0

        # Error UI
        self._error_text: str = ""
        self._error_t0 = 0.0
        self._reset_rect = QRectF()

        self._angle = 0.0
        self._breathe_phase = 0.0
        self._t0 = time.perf_counter()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_state(self, state: str) -> None:
        state = (state or "idle").lower().strip()
        self._state = state
        if state == "error":
            self._error_t0 = time.perf_counter()
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = running
        self.update()

    def set_input_level(self, level: float) -> None:
        try:
            self._in_level_target = max(0.0, min(1.0, float(level)))
        except Exception:
            self._in_level_target = 0.0

    def set_output_level(self, level: float) -> None:
        try:
            self._out_level_target = max(0.0, min(1.0, float(level)))
        except Exception:
            self._out_level_target = 0.0

    def set_error_text(self, msg: str) -> None:
        self._error_text = (msg or "").strip()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # In error state we also provide a reset button.
            if self._state == "error" and self._reset_rect.contains(event.position()):
                self.reset_clicked.emit()
                event.accept()
                return
            self.orb_clicked.emit()
        super().mousePressEvent(event)

    @staticmethod
    def _smooth_exp(current: float, target: float, dt: float, tau_up: float, tau_down: float) -> float:
        if dt <= 0.0:
            return current
        tau = tau_up if target > current else tau_down
        tau = max(0.001, float(tau))
        a = 1.0 - math.exp(-dt / tau)
        return current + (target - current) * a

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, min(0.05, now - self._t0))
        self._t0 = now

        # State-dependent rotation (eccentric but smooth).
        base_rot = 0.10 if self._running else 0.0
        if self._state == "idle":
            base_rot += 0.04
        elif self._state == "listening":
            base_rot += 0.28
        elif self._state == "transcribing":
            base_rot += 0.18
        elif self._state == "thinking":
            base_rot += 0.22
        elif self._state == "speaking":
            base_rot += 0.36
        elif self._state == "error":
            base_rot += 0.06

        self._angle = (self._angle + base_rot) % 360.0

        # Main animation phase.
        speed = 0.55
        if self._state in {"listening", "speaking"}:
            speed = 1.25
        elif self._state == "thinking":
            speed = 0.95
        elif self._state == "transcribing":
            speed = 0.70
        self._breathe_phase += dt * speed

        # Audio smoothing. Fast attack (<=50ms), slower release.
        if self._state == "listening":
            self._in_level = self._smooth_exp(self._in_level, self._in_level_target, dt, tau_up=0.03, tau_down=0.14)
        else:
            self._in_level = self._smooth_exp(self._in_level, 0.0, dt, tau_up=0.05, tau_down=0.20)

        if self._state == "speaking":
            self._out_level = self._smooth_exp(self._out_level, self._out_level_target, dt, tau_up=0.04, tau_down=0.18)
        else:
            # Keep a short tail so fade-out is visible.
            self._out_level = self._smooth_exp(self._out_level, 0.0, dt, tau_up=0.05, tau_down=0.25)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        size = min(w, h) * 0.66
        cx, cy = w / 2.0, h / 2.0
        r = size / 2.0

        t = self._breathe_phase

        # --- Eccentric state motion (but continuous) ---
        breathe_amp = 0.028
        breathe_freq = 0.85
        extra = 0.0

        if self._state == "idle":
            breathe_amp = 0.022
            breathe_freq = 0.70
        elif self._state == "listening":
            breathe_amp = 0.018
            breathe_freq = 1.70
            extra += 0.36 * self._in_level
        elif self._state == "transcribing":
            breathe_amp = 0.032
            breathe_freq = 0.92
            extra += 0.08 * math.sin(t * 1.2)
        elif self._state == "thinking":
            breathe_amp = 0.060
            breathe_freq = 1.08
            extra += 0.06 * math.sin(t * 2.0)
        elif self._state == "speaking":
            breathe_amp = 0.020
            breathe_freq = 1.60
            extra += 0.40 * self._out_level
        elif self._state == "error":
            breathe_amp = 0.016
            breathe_freq = 0.70

        breathe = 1.0 + breathe_amp * math.sin(t * breathe_freq)
        micro = 0.014 * math.sin(t * 4.2 + 0.8) + 0.010 * math.sin(t * 6.0)
        scale = max(0.80, min(1.65, breathe + micro + extra))
        rr = r * scale

        # Error shake (short, decaying).
        dx = dy = 0.0
        if self._state == "error" and self._error_t0 > 0.0:
            e = max(0.0, time.perf_counter() - self._error_t0)
            amp = 18.0 * math.exp(-e / 0.75)
            dx = amp * (math.sin(e * 36.0) + 0.5 * math.sin(e * 71.0))
            dy = amp * (math.sin(e * 44.0 + 0.8) + 0.5 * math.sin(e * 67.0 + 0.2))
            if e > 1.2:
                self._error_t0 = 0.0
                dx = dy = 0.0

        glow = 0.10
        if self._state == "listening":
            glow = 0.25
        elif self._state == "thinking":
            glow = 0.16
        elif self._state == "speaking":
            glow = 0.30
        elif self._state == "error":
            glow = 0.22

        grad = QRadialGradient(QPointF(cx + dx, cy + dy), rr * 1.35)
        grad.setColorAt(0.0, QColor(255, 255, 255, int(55 * glow)))
        grad.setColorAt(0.6, QColor(255, 255, 255, int(35 * glow)))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx + dx, cy + dy), rr * 1.35, rr * 1.35)

        shadow = QRadialGradient(QPointF(cx + dx + rr*0.12, cy + dy + rr*0.18), rr * 1.2)
        shadow.setColorAt(0.0, QColor(0, 0, 0, 110))
        shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(shadow)
        p.drawEllipse(QPointF(cx + dx + rr*0.08, cy + dy + rr*0.12), rr * 1.05, rr * 1.05)

        # --- Moon (physically shaded sphere) ---
        target = QRectF(cx + dx - rr, cy + dy - rr, rr * 2.0, rr * 2.0)
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
        vign = QRadialGradient(QPointF(cx + dx - rr*0.25, cy + dy - rr*0.25), rr * 1.35)
        vign.setColorAt(0.0, QColor(255, 255, 255, 28))
        vign.setColorAt(0.55, QColor(255, 255, 255, 10))
        vign.setColorAt(1.0, QColor(0, 0, 0, 160))
        p.setClipping(False)
        p.setBrush(vign)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx + dx, cy + dy), rr, rr)

        # --- State overlays ---
        # Clip overlays to the orb.
        clip = QPainterPath()
        clip.addEllipse(QPointF(cx + dx, cy + dy), rr, rr)
        p.save()
        p.setClipPath(clip)

        if self._state == "thinking":
            # Subtle moving conical gradient, clearly distinct from transcribing.
            cg = QConicalGradient(QPointF(cx + dx, cy + dy), (self._angle * 2.0) % 360.0)
            cg.setColorAt(0.0, QColor(255, 255, 255, 0))
            cg.setColorAt(0.20, QColor(200, 230, 255, 18))
            cg.setColorAt(0.50, QColor(160, 210, 255, 30))
            cg.setColorAt(0.80, QColor(200, 230, 255, 16))
            cg.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setBrush(cg)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx + dx, cy + dy), rr, rr)

        elif self._state == "transcribing":
            # Gentle internal flow band.
            cg = QConicalGradient(QPointF(cx + dx, cy + dy), (self._angle * 1.4) % 360.0)
            cg.setColorAt(0.0, QColor(255, 255, 255, 0))
            cg.setColorAt(0.30, QColor(255, 255, 255, 12))
            cg.setColorAt(0.55, QColor(255, 255, 255, 22))
            cg.setColorAt(0.75, QColor(255, 255, 255, 10))
            cg.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setBrush(cg)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx + dx, cy + dy), rr, rr)

        p.restore()

        # Outer ring cues (not clipped).
        if self._state in {"listening", "speaking"}:
            lvl = self._in_level if self._state == "listening" else self._out_level
            ring_r = rr * (1.10 + 0.12 * lvl + 0.02 * math.sin(t * 2.4))
            ring_alpha = int(70 + 140 * lvl)
            pen = QPen(QColor(235, 245, 255, ring_alpha))
            pen.setWidthF(max(2.0, rr * 0.02))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            # Slight rotation effect via dashed pattern.
            pen.setDashPattern([rr * 0.12, rr * 0.06])
            pen.setDashOffset((self._angle * 0.6) % 10.0)
            p.setPen(pen)
            p.drawEllipse(QPointF(cx + dx, cy + dy), ring_r, ring_r)

        if self._state == "transcribing":
            # Rotating processing ring (stable, distinct from thinking).
            ring_r = rr * 1.18
            pen = QPen(QColor(240, 240, 245, 170))
            pen.setWidthF(max(3.0, rr * 0.024))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            # Draw two arcs.
            rect = QRectF(cx + dx - ring_r, cy + dy - ring_r, ring_r * 2.0, ring_r * 2.0)
            start = int((-self._angle * 4.0) * 16)
            span = int(70 * 16)
            p.drawArc(rect, start, span)
            p.drawArc(rect, start + int(160 * 16), span)

        if self._state == "error":
            # Error ring: slightly warmer and thicker.
            ring_r = rr * 1.15
            pen = QPen(QColor(255, 150, 150, 180))
            pen.setWidthF(max(3.0, rr * 0.028))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx + dx, cy + dy), ring_r, ring_r)

        if self._state == "error":
            # Error message + Reset button under the orb.
            p.setPen(QColor(255, 210, 210, 230))
            f = QFont()
            f.setPointSize(12)
            f.setBold(True)
            p.setFont(f)
            msg = (self._error_text or "Došlo k chybě.").splitlines()[0].strip()
            p.drawText(QRectF(0, cy + rr + 10, w, 26), Qt.AlignHCenter | Qt.AlignTop, msg)

            bw, bh = 150.0, 38.0
            bx = cx - bw / 2.0
            by = cy + rr + 42
            self._reset_rect = QRectF(bx, by, bw, bh)
            p.setPen(QColor(255, 200, 200, 200))
            p.setBrush(QColor(255, 120, 120, 40))
            p.drawRoundedRect(self._reset_rect, 10, 10)
            f2 = QFont()
            f2.setPointSize(12)
            f2.setBold(True)
            p.setFont(f2)
            p.setPen(QColor(255, 235, 235, 235))
            p.drawText(self._reset_rect, Qt.AlignCenter, "Reset")

        elif not self._running:
            p.setPen(QColor(220, 220, 220, 190))
            f = QFont()
            f.setPointSize(12)
            p.setFont(f)
            msg = "Klikni na Měsíc pro nonstop režim" if self._state == "idle" else ""
            if msg:
                p.drawText(QRectF(0, cy + rr + 10, w, 30), Qt.AlignHCenter | Qt.AlignTop, msg)
