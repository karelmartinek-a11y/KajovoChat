from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..theme import Theme
from .sphere_renderer import SphereRenderer


def _blend_color(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class HeadWidget(QWidget):
    """Cinematic planetary avatar widget.

    Původní photo-head widget byl nahrazen hotovou planetární animací postavenou
    nad existujícími assety Země, oblačnosti a Měsíce. Widget zachovává stejné
    veřejné API jako původní head widget, takže zbytek aplikace nemusí řešit,
    jestli se zrovna zobrazuje obličej nebo stylizovaný avatar.
    """

    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, image_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._theme = Theme()
        assets_dir = Path(image_path).resolve().parent
        earth_img = QImage(str(assets_dir / "earth_hd.png"))
        clouds_img = QImage(str(assets_dir / "earth_clouds_hd.png"))
        moon_img = QImage(str(assets_dir / "moon_hd.png"))
        self._planet_renderer = SphereRenderer(earth_img, clouds_img)
        self._moon_renderer = SphereRenderer(moon_img)

        self._planet_img: Optional[QImage] = None
        self._moon_img: Optional[QImage] = None
        self._last_planet_size = 0
        self._last_planet_angle_q = -10**9
        self._last_cloud_angle_q = -10**9
        self._last_moon_size = 0
        self._last_moon_angle_q = -10**9

        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._error_t0 = 0.0
        self._reset_rect = QRectF()

        self._in_level_target = 0.0
        self._out_level_target = 0.0
        self._in_level = 0.0
        self._out_level = 0.0
        self._pose_weights = {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0}
        self._mouth_energy = 0.0
        self._aurora_bias = 0.0
        self._t0 = time.perf_counter()
        self._anim_t = 0.0
        self._planet_angle = 0.0
        self._cloud_angle = 0.0
        self._moon_angle = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @staticmethod
    def _smooth_exp(current: float, target: float, dt: float, tau_up: float, tau_down: float) -> float:
        if dt <= 0.0:
            return current
        tau = tau_up if target > current else tau_down
        tau = max(0.001, float(tau))
        alpha = 1.0 - math.exp(-dt / tau)
        return current + (target - current) * alpha

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        if self._state == "error":
            self._error_t0 = time.perf_counter()
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.update()

    def set_input_level(self, level: float) -> None:
        self._in_level_target = max(0.0, min(1.0, float(level or 0.0)))

    def set_output_level(self, level: float) -> None:
        self._out_level_target = max(0.0, min(1.0, float(level or 0.0)))

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            return
        weights = snapshot.get("weights")
        if not isinstance(weights, dict):
            return
        total = 0.0
        for key in self._pose_weights:
            value = max(0.0, min(1.0, float(weights.get(key, 0.0))))
            self._pose_weights[key] = value
            total += value
        if total > 1e-6:
            for key in self._pose_weights:
                self._pose_weights[key] /= total
        self._mouth_energy = max(
            0.0,
            min(
                1.0,
                self._pose_weights["small"] * 0.20
                + self._pose_weights["aa"] * 1.00
                + self._pose_weights["ee"] * 0.48
                + self._pose_weights["oo"] * 0.72,
            ),
        )
        self._aurora_bias = max(
            0.0,
            min(1.0, self._pose_weights["ee"] * 0.35 + self._pose_weights["oo"] * 0.85 + self._pose_weights["aa"] * 0.20),
        )

    def set_error_text(self, msg: str) -> None:
        self._error_text = (msg or "").strip()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._state == "error" and self._reset_rect.contains(event.position()):
                self.reset_clicked.emit()
                event.accept()
                return
            self.orb_clicked.emit()
        super().mousePressEvent(event)

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, min(0.05, now - self._t0))
        self._t0 = now
        self._anim_t += dt

        if self._state == "listening":
            self._in_level = self._smooth_exp(self._in_level, self._in_level_target, dt, 0.03, 0.16)
        else:
            self._in_level = self._smooth_exp(self._in_level, 0.0, dt, 0.05, 0.20)

        if self._state == "speaking":
            target = max(self._out_level_target, self._mouth_energy * 0.85)
            self._out_level = self._smooth_exp(self._out_level, target, dt, 0.025, 0.18)
        else:
            self._out_level = self._smooth_exp(self._out_level, 0.0, dt, 0.05, 0.26)

        spin = 4.6 if self._running else 0.0
        if self._state == "idle":
            spin += 1.8
        elif self._state == "connecting":
            spin += 8.2
        elif self._state == "listening":
            spin += 5.5 + self._in_level * 2.8
        elif self._state == "transcribing":
            spin += 3.2
        elif self._state == "thinking":
            spin += 6.2
        elif self._state == "speaking":
            spin += 10.5 + self._out_level * 5.2
        elif self._state == "reconnecting":
            spin += 7.0
        elif self._state == "error":
            spin += 1.2
        self._planet_angle = (self._planet_angle + spin * dt) % 360.0
        self._cloud_angle = (self._cloud_angle + (spin * 1.18 + 2.4) * dt) % 360.0
        self._moon_angle = (self._moon_angle + (14.0 + self._anim_t * 0.1) * dt) % 360.0
        self.update()

    def _state_colors(self) -> tuple[QColor, QColor, QColor]:
        glow = QColor(70, 165, 255, 125)
        rim = QColor(228, 247, 255, 170)
        pulse = QColor(96, 195, 255, 190)
        if self._state in {"connecting", "reconnecting"}:
            glow = QColor(255, 188, 94, 135)
            rim = QColor(255, 232, 180, 180)
            pulse = QColor(255, 214, 132, 210)
        elif self._state == "listening":
            glow = QColor(70, 227, 176, 145)
            rim = QColor(204, 255, 236, 190)
            pulse = QColor(98, 255, 210, 215)
        elif self._state in {"thinking", "transcribing"}:
            glow = QColor(136, 140, 255, 140)
            rim = QColor(222, 226, 255, 180)
            pulse = QColor(177, 185, 255, 215)
        elif self._state == "speaking":
            glow = QColor(62, 206, 255, 160)
            rim = QColor(224, 248, 255, 210)
            pulse = QColor(103, 232, 255, 230)
        elif self._state == "error":
            glow = QColor(255, 94, 94, 145)
            rim = QColor(255, 212, 212, 215)
            pulse = QColor(255, 132, 132, 220)
        return glow, rim, pulse

    def _ensure_planet(self, size: int) -> Optional[QImage]:
        size = max(64, int(size))
        angle_q = int(self._planet_angle * 2.0)
        cloud_angle_q = int(self._cloud_angle * 2.0)
        if (
            self._planet_img is None
            or self._last_planet_size != size
            or self._last_planet_angle_q != angle_q
            or self._last_cloud_angle_q != cloud_angle_q
        ):
            self._planet_img = self._planet_renderer.render_earth(size, angle_q / 2.0, cloud_angle_q / 2.0)
            self._last_planet_size = size
            self._last_planet_angle_q = angle_q
            self._last_cloud_angle_q = cloud_angle_q
        return self._planet_img

    def _ensure_moon(self, size: int) -> Optional[QImage]:
        size = max(28, int(size))
        angle_q = int(self._moon_angle * 2.0)
        if self._moon_img is None or self._last_moon_size != size or self._last_moon_angle_q != angle_q:
            self._moon_img = self._moon_renderer.render_moon(size, angle_q / 2.0)
            self._last_moon_size = size
            self._last_moon_angle_q = angle_q
        return self._moon_img

    def _draw_background_glow(self, p: QPainter, cx: float, cy: float, radius: float, glow: QColor) -> None:
        outer = QRadialGradient(QPointF(cx, cy), radius * 1.42)
        outer.setColorAt(0.0, QColor(glow.red(), glow.green(), glow.blue(), int(glow.alpha() * 0.46)))
        outer.setColorAt(0.44, QColor(glow.red(), glow.green(), glow.blue(), int(glow.alpha() * 0.20)))
        outer.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(outer)
        p.drawEllipse(QPointF(cx, cy), radius * 1.42, radius * 1.42)

    def _draw_signal_rings(self, p: QPainter, cx: float, cy: float, radius: float, rim: QColor, pulse: QColor) -> None:
        t = self._anim_t
        p.setBrush(Qt.NoBrush)
        if self._state in {"listening", "speaking"}:
            level = self._in_level if self._state == "listening" else max(self._out_level, self._mouth_energy)
            base = radius * (1.04 + 0.06 * level)
            for idx in range(2):
                alpha = 0.55 - idx * 0.18
                pen = QPen(_blend_color(rim, pulse, 0.35 + idx * 0.28))
                pen.setWidthF(max(2.2, radius * (0.014 - idx * 0.002)))
                pen.setDashPattern([radius * 0.11, radius * 0.07])
                pen.setDashOffset((t * (36.0 + idx * 10.0)) % max(1.0, radius * 0.48))
                color = pen.color()
                color.setAlphaF(max(0.0, min(1.0, alpha)))
                pen.setColor(color)
                p.setPen(pen)
                ring_r = base + radius * idx * 0.07 + math.sin(t * (2.0 + idx * 0.7)) * radius * 0.008
                p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)
        elif self._state in {"thinking", "transcribing", "connecting", "reconnecting"}:
            pen = QPen(rim)
            pen.setWidthF(max(3.0, radius * 0.016))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            rect = QRectF(cx - radius * 1.08, cy - radius * 1.08, radius * 2.16, radius * 2.16)
            start = int((-t * 160.0) * 16)
            p.drawArc(rect, start, int(72 * 16))
            p.drawArc(rect, start + int(162 * 16), int(54 * 16))
            p.drawArc(rect, start + int(286 * 16), int(36 * 16))
        elif self._state == "error":
            pen = QPen(rim)
            pen.setWidthF(max(3.6, radius * 0.018))
            p.setPen(pen)
            p.drawEllipse(QPointF(cx, cy), radius * 1.05, radius * 1.05)

    def _draw_voice_aurora(self, p: QPainter, target: QRectF, pulse: QColor) -> None:
        energy = max(self._out_level, self._mouth_energy)
        if energy <= 0.02 and self._state != "speaking":
            return
        band_h = target.height() * (0.10 + 0.12 * energy)
        band_rect = QRectF(
            target.left() - target.width() * 0.03,
            target.center().y() + target.height() * 0.13 - band_h * 0.5,
            target.width() * 1.06,
            band_h,
        )
        p.save()
        p.setClipRect(target.adjusted(0, target.height() * 0.05, 0, -target.height() * 0.03))
        steps = 10
        for idx in range(steps):
            phase = self._anim_t * (3.0 + idx * 0.25) + idx * 0.55
            wobble = math.sin(phase) * band_rect.height() * (0.10 + 0.08 * self._aurora_bias)
            alpha = int((22 + idx * 8) * (0.45 + energy * 0.9))
            color = _blend_color(pulse, QColor(255, 255, 255, 220), idx / max(1, steps - 1))
            color.setAlpha(max(0, min(255, alpha)))
            pen = QPen(color)
            pen.setWidthF(max(2.0, band_rect.height() * (0.18 - idx * 0.01)))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            y = band_rect.center().y() + wobble + (idx - steps / 2.0) * band_rect.height() * 0.045
            x1 = band_rect.left() + band_rect.width() * 0.12
            x2 = band_rect.right() - band_rect.width() * 0.12
            curve = band_rect.height() * (0.12 + 0.30 * energy)
            p.drawArc(
                QRectF(x1, y - curve, x2 - x1, curve * 2.0),
                int((16 + idx * 3) * 16),
                int((148 + energy * 42.0) * 16),
            )
        p.restore()

    def _draw_moon(self, p: QPainter, cx: float, cy: float, radius: float, glow: QColor) -> None:
        moon_r = radius * 0.24
        orbit_rx = radius * 1.18
        orbit_ry = radius * 0.54
        angle = self._moon_angle * math.pi / 180.0
        mx = cx + math.cos(angle) * orbit_rx
        my = cy + math.sin(angle) * orbit_ry - radius * 0.22
        moon = self._ensure_moon(int(moon_r * 2.0))
        if moon is None or moon.isNull():
            return

        orbit_pen = QPen(QColor(glow.red(), glow.green(), glow.blue(), 34))
        orbit_pen.setWidthF(max(1.3, radius * 0.006))
        p.setPen(orbit_pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy - radius * 0.22), orbit_rx, orbit_ry)

        halo = QRadialGradient(QPointF(mx, my), moon_r * 1.45)
        halo.setColorAt(0.0, QColor(255, 255, 255, 56))
        halo.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QPointF(mx, my), moon_r * 1.45, moon_r * 1.45)
        p.drawImage(QRectF(mx - moon_r, my - moon_r, moon_r * 2.0, moon_r * 2.0), moon)

    def _draw_error_ui(self, p: QPainter, cx: float, cy: float, radius: float, rim: QColor) -> None:
        if self._state != "error":
            self._reset_rect = QRectF()
            return
        msg = self._error_text or "Klikni pro reset"
        text_rect = QRectF(cx - radius * 0.92, cy + radius * 0.86, radius * 1.84, radius * 0.36)
        p.setPen(rim)
        font = QFont(self.font())
        font.setPointSizeF(max(9.0, radius * 0.075))
        font.setBold(True)
        p.setFont(font)
        p.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, msg)

        button_rect = QRectF(cx - radius * 0.40, cy + radius * 1.16, radius * 0.80, radius * 0.22)
        self._reset_rect = button_rect
        p.setPen(QPen(rim, max(1.8, radius * 0.01)))
        p.setBrush(QColor(255, 255, 255, 18))
        p.drawRoundedRect(button_rect, radius * 0.08, radius * 0.08)
        font.setPointSizeF(max(8.5, radius * 0.060))
        p.setFont(font)
        p.drawText(button_rect, Qt.AlignCenter, "RESET")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.285
        t = self._anim_t
        glow, rim, pulse = self._state_colors()

        if self._state == "error" and self._error_t0 > 0.0:
            e = max(0.0, time.perf_counter() - self._error_t0)
            amp = 9.0 * math.exp(-e / 0.75)
            cx += amp * (math.sin(e * 34.0) + 0.4 * math.sin(e * 57.0))
            cy += amp * (math.sin(e * 40.0 + 0.7) + 0.4 * math.sin(e * 62.0 + 0.4))
            if e > 1.3:
                self._error_t0 = 0.0

        breathe = 1.0 + 0.020 * math.sin(t * 0.95) + 0.010 * math.sin(t * 2.15 + 0.6)
        if self._state == "thinking":
            breathe += 0.018 * math.sin(t * 1.45)
        elif self._state == "listening":
            breathe += self._in_level * 0.030
        elif self._state == "speaking":
            breathe += max(self._out_level, self._mouth_energy) * 0.050
        radius *= max(0.88, min(1.28, breathe))

        self._draw_background_glow(p, cx, cy, radius, glow)
        self._draw_signal_rings(p, cx, cy, radius, rim, pulse)

        shadow = QRadialGradient(QPointF(cx + radius * 0.16, cy + radius * 0.20), radius * 1.20)
        shadow.setColorAt(0.0, QColor(0, 0, 0, 110))
        shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(shadow)
        p.drawEllipse(QPointF(cx + radius * 0.10, cy + radius * 0.13), radius * 1.06, radius * 1.06)

        planet = self._ensure_planet(int(radius * 2.0))
        target = QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
        if planet is not None and not planet.isNull():
            p.drawImage(target, planet)

        vign = QRadialGradient(QPointF(cx - radius * 0.24, cy - radius * 0.28), radius * 1.34)
        vign.setColorAt(0.0, QColor(255, 255, 255, 30))
        vign.setColorAt(0.52, QColor(255, 255, 255, 8))
        vign.setColorAt(1.0, QColor(0, 0, 0, 115))
        p.setBrush(vign)
        p.setPen(Qt.NoPen)
        p.drawEllipse(target)

        atmosphere = QRadialGradient(QPointF(cx - radius * 0.30, cy - radius * 0.32), radius * 1.16)
        atmosphere.setColorAt(0.0, QColor(255, 255, 255, 54))
        atmosphere.setColorAt(0.46, QColor(110, 208, 255, 26))
        atmosphere.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(atmosphere)
        p.drawEllipse(QPointF(cx, cy), radius * 1.02, radius * 1.02)

        self._draw_voice_aurora(p, target, pulse)
        self._draw_moon(p, cx, cy, radius, glow)
        self._draw_error_ui(p, cx, cy, radius, rim)
