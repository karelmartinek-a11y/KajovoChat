from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QMarginsF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from ..theme import Theme
from .sphere_renderer import qimage_to_rgba_numpy, rgba_numpy_to_qimage


def _blend_color(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class HeadWidget(QWidget):
    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, image_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._theme = Theme()
        self._source = self._load_head_image(image_path)
        self._head_bbox = self._compute_alpha_bbox(self._source)
        self._mouth_rect = self._derive_mouth_rect(self._head_bbox)

        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._error_t0 = 0.0
        self._reset_rect = QRectF()

        self._in_level_target = 0.0
        self._out_level_target = 0.0
        self._in_level = 0.0
        self._out_level = 0.0
        self._target_weights = {"closed": 1.0, "small": 0.0, "aa": 0.0, "ee": 0.0, "oo": 0.0}
        self._weights = dict(self._target_weights)
        self._mouth_openness = 0.0
        self._t0 = time.perf_counter()
        self._anim_t = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _load_head_image(self, image_path: str) -> QImage:
        raw = QImage(image_path)
        if raw.isNull():
            return QImage()
        rgba = qimage_to_rgba_numpy(raw)
        if rgba.size == 0:
            return raw

        rgb = rgba[..., :3].astype("int16")
        samples = [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1], rgb[min(10, rgb.shape[0] - 1), min(10, rgb.shape[1] - 1)]]
        bg = sum(samples) / float(len(samples))
        distance = ((rgb - bg) ** 2).sum(axis=2) ** 0.5
        bright = rgb.mean(axis=2)
        alpha = rgba[..., 3].astype("uint8")
        mask = (distance < 22.0) | ((distance < 34.0) & (bright > 220.0))
        alpha[mask] = 0
        edge = 12
        alpha[:edge, :] = 0
        alpha[-edge:, :] = 0
        alpha[:, :edge] = 0
        alpha[:, -edge:] = 0
        rgba[..., 3] = alpha
        return rgba_numpy_to_qimage(rgba.astype("uint8"))

    @staticmethod
    def _compute_alpha_bbox(image: QImage) -> QRectF:
        if image.isNull():
            return QRectF(0, 0, 1, 1)
        rgba = qimage_to_rgba_numpy(image)
        if rgba.size == 0:
            return QRectF(0, 0, image.width(), image.height())
        alpha = rgba[..., 3]
        ys, xs = (alpha > 8).nonzero()
        if xs.size == 0 or ys.size == 0:
            return QRectF(0, 0, image.width(), image.height())
        return QRectF(float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))

    @staticmethod
    def _derive_mouth_rect(bbox: QRectF) -> QRectF:
        return QRectF(
            bbox.left() + bbox.width() * 0.435,
            bbox.top() + bbox.height() * 0.472,
            bbox.width() * 0.17,
            bbox.height() * 0.026,
        )

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
        for pose in self._target_weights:
            self._target_weights[pose] = max(0.0, min(1.0, float(weights.get(pose, 0.0))))

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
            self._in_level = self._smooth_exp(self._in_level, self._in_level_target, dt, 0.03, 0.15)
        else:
            self._in_level = self._smooth_exp(self._in_level, 0.0, dt, 0.04, 0.20)

        if self._state == "speaking":
            self._out_level = self._smooth_exp(self._out_level, self._out_level_target, dt, 0.025, 0.17)
        else:
            self._out_level = self._smooth_exp(self._out_level, 0.0, dt, 0.04, 0.24)

        for pose, target in self._target_weights.items():
            self._weights[pose] = self._smooth_exp(self._weights[pose], target, dt, 0.028, 0.12)
        total = sum(max(0.0, value) for value in self._weights.values()) or 1.0
        for pose in self._weights:
            self._weights[pose] = max(0.0, self._weights[pose]) / total

        self._mouth_openness = (
            self._weights["small"] * 0.18
            + self._weights["aa"] * 1.00
            + self._weights["ee"] * 0.48
            + self._weights["oo"] * 0.62
        )
        self.update()

    def _state_colors(self) -> tuple[QColor, QColor]:
        accent = QColor(80, 182, 220, 160)
        rim = QColor(230, 242, 250, 180)
        if self._state in {"connecting", "reconnecting"}:
            accent = QColor(249, 205, 92, 170)
            rim = QColor(255, 228, 152, 180)
        elif self._state == "listening":
            accent = QColor(75, 212, 178, 180)
            rim = QColor(190, 255, 238, 190)
        elif self._state in {"thinking", "transcribing"}:
            accent = QColor(104, 165, 255, 175)
            rim = QColor(220, 234, 255, 180)
        elif self._state == "speaking":
            accent = QColor(63, 176, 219, 200)
            rim = QColor(227, 247, 255, 220)
        elif self._state == "error":
            accent = QColor(255, 104, 104, 180)
            rim = QColor(255, 203, 203, 210)
        return accent, rim

    def _draw_state_overlays(self, p: QPainter, cx: float, cy: float, r: float, accent: QColor, rim: QColor) -> None:
        t = self._anim_t
        if self._state in {"listening", "speaking"}:
            level = self._in_level if self._state == "listening" else self._out_level
            ring_r = r * (1.02 + 0.08 * level + 0.015 * math.sin(t * 2.0))
            pen = QPen(rim)
            pen.setWidthF(max(2.4, r * 0.012))
            pen.setCapStyle(Qt.RoundCap)
            pen.setDashPattern([r * 0.14, r * 0.08])
            pen.setDashOffset((t * 42.0) % max(1.0, r * 0.5))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)
        elif self._state in {"connecting", "reconnecting", "transcribing"}:
            pen = QPen(rim)
            pen.setWidthF(max(3.0, r * 0.014))
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            rect = QRectF(cx - r * 1.02, cy - r * 1.02, r * 2.04, r * 2.04)
            start = int((-t * 170.0) * 16)
            p.drawArc(rect, start, int(82 * 16))
            p.drawArc(rect, start + int(175 * 16), int(58 * 16))
        elif self._state == "thinking":
            p.setPen(Qt.NoPen)
            for idx in range(3):
                phase = t * 1.8 + idx * 2.1
                px = cx + math.cos(phase) * r * 0.98
                py = cy + math.sin(phase) * r * 0.22 - r * 0.72
                p.setBrush(_blend_color(accent, rim, 0.4 + 0.25 * idx))
                p.drawEllipse(QPointF(px, py), r * 0.042, r * 0.042)
        elif self._state == "error":
            pen = QPen(rim)
            pen.setWidthF(max(3.2, r * 0.016))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), r * 1.03, r * 1.03)

    def _draw_mouth_overlay(self, p: QPainter, target: QRectF) -> None:
        src = self._mouth_rect
        tx = target.left() + (src.left() / max(1.0, self._source.width())) * target.width()
        ty = target.top() + (src.top() / max(1.0, self._source.height())) * target.height()
        tw = (src.width() / max(1.0, self._source.width())) * target.width()
        th = (src.height() / max(1.0, self._source.height())) * target.height()
        mouth_rect = QRectF(tx, ty, tw, th)
        aa = self._weights["aa"]
        ee = self._weights["ee"]
        oo = self._weights["oo"]
        small = self._weights["small"]

        width_scale = 1.0 + aa * 0.08 + ee * 0.12 - oo * 0.12 - small * 0.04
        mouth_w = mouth_rect.width() * width_scale
        center_x = mouth_rect.center().x()
        center_y = mouth_rect.center().y()
        lip_y = center_y + mouth_rect.height() * 0.04
        line_thickness = max(1.2, mouth_rect.height() * 0.22)

        # Základní linka rtů sedí přesně na ústech i v idle.
        lip_pen = QPen(QColor(116, 82, 78, 150))
        lip_pen.setWidthF(line_thickness)
        lip_pen.setCapStyle(Qt.RoundCap)
        p.setPen(lip_pen)
        p.setBrush(Qt.NoBrush)
        p.drawLine(
            QPointF(center_x - mouth_w * 0.48, lip_y),
            QPointF(center_x + mouth_w * 0.48, lip_y),
        )

        if self._mouth_openness <= 0.03:
            return

        open_h = mouth_rect.height() * (0.30 + self._mouth_openness * 1.75)
        cavity = QRectF(
            center_x - mouth_w * (0.34 - oo * 0.06),
            lip_y - open_h * 0.42,
            mouth_w * (0.68 - oo * 0.10),
            open_h,
        )
        cavity_color = _blend_color(QColor(52, 10, 18, 175), QColor(100, 30, 44, 220), aa * 0.65 + ee * 0.2)
        grad = QRadialGradient(cavity.center(), max(cavity.width(), cavity.height()) * 0.72)
        grad.setColorAt(0.0, cavity_color)
        grad.setColorAt(0.78, QColor(35, 7, 10, 120))
        grad.setColorAt(1.0, QColor(8, 3, 3, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        radius = cavity.height() * (0.60 if oo > 0.4 else 0.35)
        p.drawRoundedRect(cavity, radius, radius)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
        if self._source.isNull():
            return

        w = float(self.width())
        h = float(self.height())
        accent, rim = self._state_colors()
        t = self._anim_t
        breathe = 1.0 + 0.016 * math.sin(t * (0.8 if self._state == "idle" else 1.1))
        scale = breathe * (1.0 + self._out_level * 0.02)
        target_h = h * 0.78 * scale
        target_w = target_h * (self._source.width() / max(1.0, self._source.height()))
        cx = w * 0.5
        cy = h * 0.47

        dx = 0.0
        dy = math.sin(t * 1.2) * h * 0.007
        if self._state == "listening":
            dx += math.sin(t * 3.0) * w * 0.008 * (0.35 + self._in_level)
        elif self._state == "thinking":
            dx += math.sin(t * 1.6) * w * 0.009
            dy -= abs(math.sin(t * 1.2)) * h * 0.008
        elif self._state == "speaking":
            dx += math.sin(t * 6.0) * w * 0.005 * self._out_level
            dy -= self._out_level * h * 0.008
        elif self._state == "error" and self._error_t0 > 0.0:
            elapsed = max(0.0, time.perf_counter() - self._error_t0)
            amp = 18.0 * math.exp(-elapsed / 0.8)
            dx += amp * (math.sin(elapsed * 34.0) + 0.5 * math.sin(elapsed * 59.0))
            if elapsed > 1.3:
                self._error_t0 = 0.0

        head_rect = QRectF(cx - target_w / 2.0, cy - target_h / 2.0, target_w, target_h)
        radius = max(target_w, target_h) * 0.42

        glow = QRadialGradient(QPointF(cx + dx, cy + dy), radius)
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 88))
        glow.setColorAt(0.55, QColor(accent.red(), accent.green(), accent.blue(), 30))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx + dx, cy + dy), radius, radius)

        shadow = QRadialGradient(QPointF(cx + dx, head_rect.bottom() - target_h * 0.07), target_w * 0.36)
        shadow.setColorAt(0.0, QColor(0, 0, 0, 130))
        shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(shadow)
        p.drawEllipse(QPointF(cx + dx, head_rect.bottom() - target_h * 0.07), target_w * 0.34, target_h * 0.07)

        angle = math.sin(t * 0.85) * 0.9
        if self._state == "listening":
            angle += math.sin(t * 2.8) * (0.8 + self._in_level * 0.8)
        elif self._state == "thinking":
            angle += math.sin(t * 1.4) * 1.6
        elif self._state == "speaking":
            angle += math.sin(t * 6.0) * (0.6 + self._out_level * 0.8)

        p.save()
        p.translate(head_rect.center())
        p.translate(dx, dy)
        p.rotate(angle)
        p.translate(-head_rect.center())
        p.drawImage(head_rect, self._source)
        self._draw_mouth_overlay(p, head_rect)
        p.restore()

        self._draw_state_overlays(p, cx + dx, cy + dy, max(target_w, target_h) * 0.40, accent, rim)

        if self._state == "error":
            p.setPen(QColor(255, 225, 225, 235))
            font = QFont()
            font.setPointSize(12)
            font.setBold(True)
            p.setFont(font)
            msg = (self._error_text or "Došlo k chybě.").splitlines()[0].strip()
            p.drawText(QRectF(0.0, head_rect.bottom() + 10.0, w, 28.0), Qt.AlignHCenter | Qt.AlignTop, msg)

            self._reset_rect = QRectF(cx - 75.0, head_rect.bottom() + 40.0, 150.0, 38.0)
            p.setPen(QColor(255, 200, 200, 200))
            p.setBrush(QColor(255, 120, 120, 42))
            p.drawRoundedRect(self._reset_rect, 10.0, 10.0)
            p.drawText(self._reset_rect, Qt.AlignCenter, "Reset")
        elif not self._running and self._state == "idle":
            p.setPen(QColor(220, 220, 220, 190))
            font = QFont()
            font.setPointSize(12)
            p.setFont(font)
            p.drawText(QRectF(0.0, head_rect.bottom() + 10.0, w, 30.0), Qt.AlignHCenter | Qt.AlignTop, "Klikni na hlavu pro nonstop režim")
