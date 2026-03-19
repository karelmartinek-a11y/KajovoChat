from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..animation.performance_driver import PerformanceDriver
from ..animation.types import PerformanceFrame, VisemeFrame
from ..resources.assets import load_talking_head_manifest
from ..theme import Theme
from .rig_layers import RigDefinition, rig_definition_from_manifest
from .talking_head_renderer import TalkingHeadRenderer


class TalkingHeadWidget(QWidget):
    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, image_path: str | None = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._theme = Theme()
        manifest = load_talking_head_manifest(fallback_image_override=image_path) if image_path else load_talking_head_manifest()
        self._rig: RigDefinition = rig_definition_from_manifest(manifest)
        self._renderer = TalkingHeadRenderer()
        self._driver = PerformanceDriver()

        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._error_t0 = 0.0
        self._reset_rect = QRectF()

        self._input_level_target = 0.0
        self._output_level_target = 0.0
        self._input_level = 0.0
        self._output_level = 0.0
        self._latest_viseme = VisemeFrame()
        self._external_performance: PerformanceFrame | None = None
        self._current_frame = PerformanceFrame()
        self._frame_buffer = QImage()
        self._t0 = time.perf_counter()
        self._anim_t = 0.0

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

    @property
    def rig_definition(self) -> RigDefinition:
        return self._rig

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        if self._state == "error":
            self._error_t0 = time.perf_counter()
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.update()

    def set_input_level(self, level: float) -> None:
        self._input_level_target = max(0.0, min(1.0, float(level or 0.0)))

    def set_output_level(self, level: float) -> None:
        self._output_level_target = max(0.0, min(1.0, float(level or 0.0)))

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            return
        if "viseme" in snapshot:
            self._latest_viseme = VisemeFrame.from_dict(snapshot.get("viseme"))
        elif "jaw_open" in snapshot or "legacy_weights" in snapshot or "cluster" in snapshot:
            self._latest_viseme = VisemeFrame.from_dict(snapshot)
        else:
            self._latest_viseme = VisemeFrame.from_legacy_snapshot(snapshot)
        self._external_performance = None

    def set_performance_frame(self, frame: object) -> None:
        if isinstance(frame, PerformanceFrame):
            self._external_performance = frame
            self._latest_viseme = frame.viseme
            return
        if isinstance(frame, dict):
            parsed = PerformanceFrame.from_dict(frame)
            self._external_performance = parsed
            self._latest_viseme = parsed.viseme

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
            self._input_level = self._smooth_exp(self._input_level, self._input_level_target, dt, 0.03, 0.15)
        else:
            self._input_level = self._smooth_exp(self._input_level, 0.0, dt, 0.04, 0.20)

        if self._state == "speaking":
            self._output_level = self._smooth_exp(self._output_level, self._output_level_target, dt, 0.025, 0.17)
        else:
            self._output_level = self._smooth_exp(self._output_level, 0.0, dt, 0.04, 0.24)

        source_viseme = self._external_performance.viseme if self._external_performance is not None else self._latest_viseme
        self._current_frame = self._driver.drive(
            state=self._state,
            input_level=self._input_level,
            output_level=self._output_level,
            lipsync_frame=source_viseme,
            now=now,
        )
        self.update()

    def paintEvent(self, event) -> None:
        rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))
        if rect.width() <= 1.0 or rect.height() <= 1.0:
            return

        if self._frame_buffer.size() != self.size():
            self._frame_buffer = QImage(self.size(), QImage.Format_ARGB32_Premultiplied)

        self._frame_buffer.fill(0)
        buffer_painter = QPainter(self._frame_buffer)
        buffer_painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)

        head_rect = QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.83)
        self._renderer.render(buffer_painter, head_rect, self._current_frame, self._rig)

        if self._state == "error":
            self._draw_error_affordance(buffer_painter, head_rect)
        elif not self._running and self._state == "idle":
            buffer_painter.setPen(QColor(220, 220, 220, 190))
            font = QFont()
            font.setPointSize(12)
            buffer_painter.setFont(font)
            buffer_painter.drawText(QRectF(0.0, head_rect.bottom() + 10.0, rect.width(), 30.0), Qt.AlignHCenter | Qt.AlignTop, "Klikni na hlavu pro nonstop režim")

        buffer_painter.end()

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform, True)
        painter.drawImage(self.rect(), self._frame_buffer)

    def _draw_error_affordance(self, painter: QPainter, head_rect: QRectF) -> None:
        w = float(self.width())
        cx = head_rect.center().x()
        self._reset_rect = QRectF(cx - 75.0, head_rect.bottom() + 40.0, 150.0, 38.0)
        painter.setPen(QColor(255, 200, 200, 200))
        painter.setBrush(QColor(255, 120, 120, 42))
        painter.drawRoundedRect(self._reset_rect, 10.0, 10.0)
        painter.drawText(self._reset_rect, Qt.AlignCenter, "Reset")

        painter.setPen(QColor(255, 225, 225, 235))
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        msg = (self._error_text or "Došlo k chybě.").splitlines()[0].strip()
        painter.drawText(QRectF(0.0, head_rect.bottom() + 10.0, w, 28.0), Qt.AlignHCenter | Qt.AlignTop, msg)
