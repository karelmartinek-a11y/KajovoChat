from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QRadialGradient

from ..animation.types import PerformanceFrame
from .rig_layers import RigDefinition, apply_mask, build_pivot_transform, layer_content_bbox, layer_image, layer_target_rect


def _blend(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class TalkingHeadRenderer:
    def render(self, painter: QPainter, rect: QRectF, performance_frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
        frame_rect = self._compute_frame_rect(rect, rig_definition)
        self._render_head(painter, frame_rect, performance_frame, rig_definition)
        self._render_gaze_transform(painter, frame_rect, performance_frame, rig_definition)
        self._render_blink_transform(painter, frame_rect, performance_frame, rig_definition)
        self._render_jaw_transform(painter, frame_rect, performance_frame, rig_definition)
        self._render_lip_transform(painter, frame_rect, performance_frame, rig_definition)
        self._render_mouth_interior_compositing(painter, frame_rect, performance_frame, rig_definition)
        self._render_cheek_shadow_polish(painter, frame_rect, performance_frame, rig_definition)
        self._render_overlay_states(painter, frame_rect, performance_frame, rig_definition)

    def _compute_frame_rect(self, rect: QRectF, rig_definition: RigDefinition) -> QRectF:
        canvas = rig_definition.canvas or {"width": 1, "height": 1}
        aspect = max(0.1, float(canvas.get("width", 1)) / max(1.0, float(canvas.get("height", 1))))
        safe_margin = max(0.02, min(0.16, float(canvas.get("safe_margin", 0.08))))
        usable = rect.adjusted(rect.width() * safe_margin, rect.height() * safe_margin * 0.55, -rect.width() * safe_margin, -rect.height() * safe_margin)
        width = usable.width()
        height = width / aspect
        if height > usable.height():
            height = usable.height()
            width = height * aspect
        if width > usable.width():
            width = usable.width()
            height = width / aspect
        return QRectF(usable.center().x() - width / 2.0, usable.center().y() - height / 2.0, width, height)

    @staticmethod
    def _state_preset(rig_definition: RigDefinition, state: str) -> dict[str, float]:
        preset = rig_definition.state_presets.get(state, rig_definition.state_presets.get("idle", {}))
        return {str(key): float(value) for key, value in preset.items() if isinstance(value, (int, float))}

    def _render_head(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        motion = frame.head_motion
        preset = self._state_preset(rig_definition, frame.state)
        asym_limit = max(0.0, min(0.03, float(rig_definition.fallback.get("asymmetry_limit", 0.018))))
        micro_asym = 0.0
        if frame.state == "speaking":
            micro_asym = max(-asym_limit, min(asym_limit, math.sin(frame.timestamp_s * 7.3) * (0.005 + frame.speech_energy * 0.007)))

        dx = (motion.head_tx + micro_asym) * frame_rect.width() * float(preset.get("head_tx_scale", 0.055))
        dy = motion.head_ty * frame_rect.height() * float(preset.get("head_ty_scale", 0.055))
        rot = (motion.head_rot + micro_asym * 0.55) * float(preset.get("head_rot_deg", 6.2))
        scale = float(preset.get("head_scale", 1.0)) + frame.speech_energy * float(preset.get("speech_scale_boost", 0.010))

        glow = QRadialGradient(frame_rect.center() + QPointF(dx, dy), max(frame_rect.width(), frame_rect.height()) * 0.42)
        glow.setColorAt(0.0, QColor(82, 172, 208, 58 if frame.state != "error" else 72))
        glow.setColorAt(0.65, QColor(82, 172, 208, 18))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(frame_rect.center() + QPointF(dx, dy), frame_rect.width() * 0.43, frame_rect.height() * 0.43)
        painter.restore()

        base_layer = next((layer for layer in rig_definition.active_layers() if layer.role == "head_base" and layer.enabled and layer.exists), None)
        if base_layer is None:
            return

        image = layer_image(base_layer)
        target = layer_target_rect(frame_rect, base_layer)
        painter.save()
        painter.setTransform(build_pivot_transform(target, tx=dx, ty=dy, rot_deg=rot, scale=scale), True)
        painter.setOpacity(max(0.0, min(1.0, base_layer.opacity)))
        painter.drawImage(target, image)
        painter.restore()

    def _mouth_rect(self, frame_rect: QRectF, rig_definition: RigDefinition) -> QRectF:
        mouth = rig_definition.deformation_ranges.get("mouth", {})
        base_layer = next((layer for layer in rig_definition.active_layers() if layer.role == "head_base" and layer.enabled and layer.exists), None)
        if base_layer is None:
            return QRectF(
                frame_rect.left() + frame_rect.width() * float(mouth.get("x", 0.435)),
                frame_rect.top() + frame_rect.height() * float(mouth.get("y", 0.472)),
                frame_rect.width() * float(mouth.get("width", 0.17)),
                frame_rect.height() * float(mouth.get("height", 0.05)),
            )

        image = layer_image(base_layer)
        if image.isNull():
            return QRectF(
                frame_rect.left() + frame_rect.width() * float(mouth.get("x", 0.435)),
                frame_rect.top() + frame_rect.height() * float(mouth.get("y", 0.472)),
                frame_rect.width() * float(mouth.get("width", 0.17)),
                frame_rect.height() * float(mouth.get("height", 0.05)),
            )

        target = layer_target_rect(frame_rect, base_layer)
        content_bbox = layer_content_bbox(base_layer)
        if content_bbox.width() <= 1.0 or content_bbox.height() <= 1.0:
            content_bbox = QRectF(0.0, 0.0, float(image.width()), float(image.height()))

        left = target.left() + (content_bbox.left() + content_bbox.width() * float(mouth.get("x", 0.435))) / max(1.0, float(image.width())) * target.width()
        top = target.top() + (content_bbox.top() + content_bbox.height() * float(mouth.get("y", 0.472))) / max(1.0, float(image.height())) * target.height()
        width = target.width() * (content_bbox.width() * float(mouth.get("width", 0.17)) / max(1.0, float(image.width())))
        height = target.height() * (content_bbox.height() * float(mouth.get("height", 0.05)) / max(1.0, float(image.height())))
        return QRectF(left, top, width, height)

    def _render_gaze_transform(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        if not rig_definition.production_ready or not bool(rig_definition.fallback.get("allows_independent_eyes", False)):
            return
        eyes = rig_definition.deformation_ranges.get("eyes", {})
        for side in ("left_x", "right_x"):
            cx = frame_rect.left() + frame_rect.width() * float(eyes.get(side, 0.5))
            cy = frame_rect.top() + frame_rect.height() * float(eyes.get("y", 0.32))
            radius = frame_rect.width() * float(eyes.get("width", 0.13)) * 0.08
            dx = frame.gaze.gaze_x * frame.gaze.focus_strength * radius * 0.72
            dy = frame.gaze.gaze_y * frame.gaze.focus_strength * radius * 0.64
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(38, 34, 36, 48))
            painter.drawEllipse(QPointF(cx + dx, cy + dy), radius, radius)
            painter.restore()

    def _render_blink_transform(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        blink = frame.blink.blink_amount
        if blink <= 0.01:
            return
        eyes = rig_definition.deformation_ranges.get("eyes", {})
        width = frame_rect.width() * float(eyes.get("width", 0.14))
        height = frame_rect.height() * float(eyes.get("height", 0.052))
        lid_height = max(1.5, height * max(0.10, blink * 0.92))
        for left in (float(eyes.get("left_x", 0.33)), float(eyes.get("right_x", 0.53))):
            eye_rect = QRectF(
                frame_rect.left() + frame_rect.width() * left,
                frame_rect.top() + frame_rect.height() * float(eyes.get("y", 0.31)),
                width,
                height,
            )
            lid = QRectF(eye_rect.left(), eye_rect.center().y() - lid_height / 2.0, eye_rect.width(), lid_height)
            grad = QRadialGradient(lid.center(), max(lid.width(), lid.height()))
            grad.setColorAt(0.0, QColor(126, 91, 84, int(150 * blink)))
            grad.setColorAt(1.0, QColor(126, 91, 84, 0))
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(grad)
            painter.drawRoundedRect(lid, lid.height() * 0.55, lid.height() * 0.55)
            painter.restore()

    def _render_jaw_transform(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        base_layer = next((layer for layer in rig_definition.active_layers() if layer.role == "head_base" and layer.enabled and layer.exists), None)
        if base_layer is None:
            return
        image = layer_image(base_layer)
        if image.isNull():
            return

        mouth = rig_definition.deformation_ranges.get("mouth", {})
        mouth_rect = self._mouth_rect(frame_rect, rig_definition)
        jaw_open = min(float(mouth.get("max_jaw_open", 0.84)), frame.viseme.jaw_open)
        if jaw_open <= 0.01:
            return

        content_bbox = layer_content_bbox(base_layer)
        src_rect = QRectF(
            content_bbox.left() + content_bbox.width() * float(mouth.get("x", 0.435)),
            content_bbox.top() + content_bbox.height() * (float(mouth.get("y", 0.472)) + float(mouth.get("height", 0.05)) * float(mouth.get("jaw_split", 0.54))),
            content_bbox.width() * float(mouth.get("width", 0.17)),
            content_bbox.height() * float(mouth.get("height", 0.05)) * (1.0 - float(mouth.get("jaw_split", 0.54))),
        )
        dst_rect = QRectF(
            mouth_rect.left(),
            mouth_rect.top() + mouth_rect.height() * float(mouth.get("jaw_split", 0.54)) + jaw_open * mouth_rect.height() * float(mouth.get("jaw_drop_scale", 0.58)),
            mouth_rect.width(),
            mouth_rect.height() * (0.55 + jaw_open * 0.56),
        )
        painter.save()
        painter.setOpacity(0.90)
        painter.drawImage(dst_rect, image, src_rect)
        painter.restore()

    def _render_lip_transform(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        mouth = rig_definition.deformation_ranges.get("mouth", {})
        mouth_rect = self._mouth_rect(frame_rect, rig_definition)
        spread = frame.viseme.lip_spread
        roundness = frame.viseme.lip_round
        openness = frame.viseme.mouth_open
        ee_like = max(spread, frame.viseme.cheek_raise * 0.85)
        corner_stretch = ee_like * float(mouth.get("corner_stretch_max", 0.12))
        upper_raise = max(frame.viseme.upper_lip_raise, openness * 0.12) * float(mouth.get("upper_lip_raise_max", 0.18))
        width_scale = 1.0 + corner_stretch - roundness * float(mouth.get("round_width_compensation", 0.13))
        line_y = mouth_rect.center().y() + mouth_rect.height() * (0.05 - upper_raise * 0.22)
        half_width = mouth_rect.width() * 0.48 * width_scale

        painter.save()
        lip_pen = QPen(QColor(122, 84, 79, 168))
        lip_pen.setWidthF(max(1.1, mouth_rect.height() * (0.17 + openness * 0.04)))
        lip_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(lip_pen)
        painter.drawLine(QPointF(mouth_rect.center().x() - half_width, line_y), QPointF(mouth_rect.center().x() + half_width, line_y))

        corner_pen = QPen(QColor(146, 103, 98, int(70 + ee_like * 40)))
        corner_pen.setWidthF(max(0.9, mouth_rect.height() * 0.10))
        corner_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(corner_pen)
        corner_span = mouth_rect.width() * (0.06 + corner_stretch * 0.12)
        painter.drawLine(QPointF(mouth_rect.center().x() - half_width, line_y), QPointF(mouth_rect.center().x() - half_width - corner_span, line_y - mouth_rect.height() * 0.04))
        painter.drawLine(QPointF(mouth_rect.center().x() + half_width, line_y), QPointF(mouth_rect.center().x() + half_width + corner_span, line_y - mouth_rect.height() * 0.04))

        if frame.viseme.lip_press > 0.05:
            press_pen = QPen(QColor(150, 102, 96, int(120 * frame.viseme.lip_press)))
            press_pen.setWidthF(max(1.0, mouth_rect.height() * 0.12))
            press_pen.setCapStyle(Qt.RoundCap)
            painter.setPen(press_pen)
            painter.drawLine(QPointF(mouth_rect.center().x() - half_width * 0.85, line_y + 1.0), QPointF(mouth_rect.center().x() + half_width * 0.85, line_y + 1.0))

        lower_shadow = QRadialGradient(QPointF(mouth_rect.center().x(), line_y + mouth_rect.height() * 0.30), mouth_rect.width() * 0.42)
        lower_shadow.setColorAt(0.0, QColor(42, 16, 18, int(28 + openness * 34)))
        lower_shadow.setColorAt(1.0, QColor(42, 16, 18, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(lower_shadow)
        painter.drawEllipse(QPointF(mouth_rect.center().x(), line_y + mouth_rect.height() * 0.28), mouth_rect.width() * 0.34, mouth_rect.height() * 0.18)
        painter.restore()

    def _render_mouth_interior_compositing(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        openness = frame.viseme.mouth_open
        if openness <= 0.02:
            return
        mouth = rig_definition.deformation_ranges.get("mouth", {})
        darkness_limit = max(0.40, min(0.90, float(rig_definition.fallback.get("mouth_darkening_limit", 0.82))))
        mouth_rect = self._mouth_rect(frame_rect, rig_definition)
        cavity = QRectF(
            mouth_rect.left() + mouth_rect.width() * (0.12 - frame.viseme.lip_round * 0.04),
            mouth_rect.center().y() - mouth_rect.height() * (0.22 + openness * 0.14),
            mouth_rect.width() * (0.76 - frame.viseme.lip_round * 0.10),
            mouth_rect.height() * (0.32 + openness * 1.55),
        )
        cavity_darkening = max(0.0, min(darkness_limit, openness * 0.85 + frame.viseme.jaw_open * 0.25))

        cavity_img = QImage(int(max(2.0, cavity.width())), int(max(2.0, cavity.height())), QImage.Format_ARGB32_Premultiplied)
        cavity_img.fill(0)
        inner = QPainter(cavity_img)
        inner.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
        grad = QRadialGradient(QPointF(cavity_img.width() * 0.5, cavity_img.height() * 0.28), max(cavity_img.width(), cavity_img.height()) * 0.72)
        grad.setColorAt(0.0, QColor(116, 48, 62, int(184 + cavity_darkening * 42)))
        grad.setColorAt(0.45, QColor(58, 18, 24, int(172 + cavity_darkening * 42)))
        grad.setColorAt(1.0, QColor(26, 8, 10, 0))
        inner.setPen(Qt.NoPen)
        inner.setBrush(grad)
        inner.drawRoundedRect(cavity_img.rect(), cavity_img.height() * 0.36, cavity_img.height() * 0.36)
        gum = QColor(162, 98, 108, int(72 + openness * 48))
        inner.setBrush(gum)
        inner.drawRoundedRect(0, 0, cavity_img.width(), max(1, int(cavity_img.height() * 0.18)), cavity_img.height() * 0.16, cavity_img.height() * 0.16)
        inner.setBrush(QColor(34, 10, 12, int(26 + cavity_darkening * 38)))
        inner.drawRoundedRect(
            int(cavity_img.width() * 0.10),
            int(cavity_img.height() * 0.38),
            int(cavity_img.width() * 0.80),
            max(1, int(cavity_img.height() * 0.46)),
            cavity_img.height() * 0.26,
            cavity_img.height() * 0.26,
        )
        inner.end()

        mask = QImage(cavity_img.size(), QImage.Format_ARGB32_Premultiplied)
        mask.fill(0)
        mask_painter = QPainter(mask)
        mask_painter.setRenderHints(QPainter.Antialiasing, True)
        mask_painter.setPen(Qt.NoPen)
        mask_painter.setBrush(QColor(255, 255, 255, 255))
        radius = mask.height() * (0.60 if frame.viseme.lip_round > 0.45 else 0.32)
        mask_painter.drawRoundedRect(mask.rect(), radius, radius)
        mask_painter.end()

        painter.save()
        painter.drawImage(cavity, apply_mask(cavity_img, mask))
        painter.restore()

    def _render_cheek_shadow_polish(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        cheeks = rig_definition.deformation_ranges.get("cheeks", {})
        cheek_y = frame_rect.top() + frame_rect.height() * float(cheeks.get("y", 0.46))
        cheek_w = frame_rect.width() * float(cheeks.get("width", 0.14))
        cheek_h = frame_rect.height() * float(cheeks.get("height", 0.09))
        offset_x = frame_rect.width() * float(cheeks.get("offset_x", 0.155))
        raise_strength = frame.viseme.cheek_raise
        rounded = frame.viseme.lip_round
        compression = rounded * float(cheeks.get("compression_max", 0.11))

        painter.save()
        painter.setPen(Qt.NoPen)
        for sign in (-1.0, 1.0):
            center = QPointF(frame_rect.center().x() + (offset_x - frame_rect.width() * compression * 0.12) * sign, cheek_y + compression * frame_rect.height() * 0.012)
            grad = QRadialGradient(center, max(cheek_w, cheek_h))
            grad.setColorAt(0.0, QColor(208, 112, 118, int(42 + raise_strength * 44)))
            grad.setColorAt(1.0, QColor(208, 112, 118, 0))
            painter.setBrush(grad)
            painter.drawEllipse(center, cheek_w * (1.0 - compression * 0.18), cheek_h * (1.0 - compression * 0.20))
        shadow = QRadialGradient(QPointF(frame_rect.center().x(), frame_rect.bottom() - frame_rect.height() * 0.12), frame_rect.width() * 0.36)
        shadow.setColorAt(0.0, QColor(0, 0, 0, 70))
        shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(shadow)
        painter.drawEllipse(QPointF(frame_rect.center().x(), frame_rect.bottom() - frame_rect.height() * 0.12), frame_rect.width() * 0.34, frame_rect.height() * 0.08)
        painter.restore()

    def _render_overlay_states(self, painter: QPainter, frame_rect: QRectF, frame: PerformanceFrame, rig_definition: RigDefinition) -> None:
        accent = QColor(80, 182, 220, 130)
        rim = QColor(230, 242, 250, 135)
        if frame.state == "listening":
            accent = QColor(75, 212, 178, 138)
            rim = QColor(190, 255, 238, 142)
        elif frame.state in {"thinking", "transcribing"}:
            accent = QColor(104, 165, 255, 138)
            rim = QColor(220, 234, 255, 140)
        elif frame.state == "speaking":
            accent = QColor(63, 176, 219, 144)
            rim = QColor(227, 247, 255, 152)
        elif frame.state == "error":
            accent = QColor(255, 104, 104, 168)
            rim = QColor(255, 203, 203, 190)

        radius = max(frame_rect.width(), frame_rect.height()) * 0.40
        center = frame_rect.center()
        painter.save()
        if frame.state in {"listening", "speaking"}:
            level = frame.input_level if frame.state == "listening" else frame.output_level
            pen = QPen(rim)
            pen.setWidthF(max(1.6, radius * 0.008))
            pen.setCapStyle(Qt.RoundCap)
            pen.setDashPattern([radius * 0.08, radius * 0.13])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius * (0.995 + level * 0.035), radius * (0.995 + level * 0.035))
        elif frame.state == "thinking":
            painter.setPen(Qt.NoPen)
            for idx in range(3):
                angle = frame.timestamp_s * 1.7 + idx * 2.0
                pos = QPointF(center.x() + math.cos(angle) * radius * 0.92, center.y() - radius * 0.72 + math.sin(angle) * radius * 0.18)
                painter.setBrush(_blend(accent, rim, 0.35 + idx * 0.22))
                painter.drawEllipse(pos, radius * 0.04, radius * 0.04)
        elif frame.state == "error":
            pen = QPen(rim)
            pen.setWidthF(max(2.4, radius * 0.014))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, radius * 1.02, radius * 1.02)
        painter.restore()
