from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from .config import create_default_config
from .diagnostics import OpenGLProbeResult, probe_opengl_support
from .engine import OrbEngine
from .renderer import LivingOrbRenderer


class _OrbWidgetBase:
    orb_clicked = Signal()
    reset_clicked = Signal()


class OrbFallbackWidget(QWidget):
    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, diagnostics: OpenGLProbeResult, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._reset_rect = QRectF()
        self._mouth_energy = 0.0
        self._aurora_bias = 0.0

    @property
    def diagnostics(self) -> OpenGLProbeResult:
        return self._diagnostics

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.update()

    def set_input_level(self, level: float) -> None:
        _ = level

    def set_output_level(self, level: float) -> None:
        _ = level

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            return
        weights = snapshot.get("weights")
        if not isinstance(weights, dict):
            return
        aa = max(0.0, min(1.0, float(weights.get("aa", 0.0))))
        ee = max(0.0, min(1.0, float(weights.get("ee", 0.0))))
        oo = max(0.0, min(1.0, float(weights.get("oo", 0.0))))
        small = max(0.0, min(1.0, float(weights.get("small", 0.0))))
        self._mouth_energy = max(0.0, min(1.0, small * 0.20 + aa * 1.0 + ee * 0.48 + oo * 0.72))
        self._aurora_bias = max(0.0, min(1.0, ee * 0.35 + oo * 0.85 + aa * 0.20))
        self.update()

    def set_error_text(self, msg: str) -> None:
        self._error_text = (msg or "").strip()
        self.update()

    def shutdown(self) -> None:
        return

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._state == "error" and self._reset_rect.contains(event.position()):
                self.reset_clicked.emit()
                event.accept()
                return
            self.orb_clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()
        cx = rect.center().x()
        cy = rect.center().y()
        radius = min(rect.width(), rect.height()) * 0.26
        grad = QRadialGradient(cx, cy, radius * 1.7)
        grad.setColorAt(0.0, QColor(190, 225, 255, 120))
        grad.setColorAt(0.35, QColor(70, 150, 255, 70))
        grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect.center(), radius * 1.7, radius * 1.7)
        painter.setBrush(QColor(120, 190, 255, 120))
        painter.drawEllipse(rect.center(), radius, radius)
        ring_pen = QPen(QColor(220, 245, 255, 160))
        ring_pen.setWidthF(max(2.0, radius * 0.035))
        painter.setPen(ring_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect.center(), radius * 1.08, radius * 1.08)
        self._paint_overlay(painter)

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._state == "error":
            self._paint_error_ui(painter)
            return
        self._reset_rect = QRectF()
        if not self._running and self._state == "idle":
            painter.setPen(QColor(220, 230, 245, 190))
            font = QFont(self.font())
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(QRectF(0.0, self.height() * 0.82, float(self.width()), 32.0), Qt.AlignHCenter | Qt.AlignTop, "Klikni na orb pro nonstop režim")
        font = QFont(self.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(205, 220, 235, 180))
        painter.drawText(QRectF(0.0, self.height() * 0.06, float(self.width()), 40.0), Qt.AlignHCenter | Qt.AlignTop, "Fallback 2D režim")

    def _paint_error_ui(self, painter: QPainter) -> None:
        w = float(self.width())
        h = float(self.height())
        cx = w * 0.5
        y = h * 0.78
        painter.setPen(QColor(255, 225, 225, 235))
        font = QFont(self.font())
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        msg = (self._error_text or "Došlo k chybě.").splitlines()[0].strip()
        painter.drawText(QRectF(0.0, y, w, 28.0), Qt.AlignHCenter | Qt.AlignTop, msg)
        self._reset_rect = QRectF(cx - 75.0, y + 34.0, 150.0, 38.0)
        painter.setPen(QColor(255, 210, 210, 200))
        painter.setBrush(QColor(255, 120, 120, 42))
        painter.drawRoundedRect(self._reset_rect, 10.0, 10.0)
        painter.drawText(self._reset_rect, Qt.AlignCenter, "Reset")


class LivingOrbWidget(QOpenGLWidget):
    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, diagnostics: OpenGLProbeResult, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._diagnostics = diagnostics
        config = create_default_config()
        self._engine = OrbEngine(config=config, renderer=LivingOrbRenderer(config))
        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._reset_rect = QRectF()
        self._mouth_energy = 0.0
        self._aurora_bias = 0.0
        self._last_out_level = 0.0
        self._last_in_level = 0.0
        self._t0 = time.perf_counter()
        self._gl_failed = False
        self._gl_failure_reason = ""
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    @property
    def diagnostics(self) -> OpenGLProbeResult:
        if self._gl_failed and self._gl_failure_reason:
            return OpenGLProbeResult(
                available=False,
                backend="fallback-2d",
                message=self._gl_failure_reason,
                version=self._diagnostics.version,
            )
        return self._diagnostics

    @property
    def engine(self) -> OrbEngine:
        return self._engine

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        self._engine.set_state(self._map_engine_state(self._state))
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)

    def set_input_level(self, level: float) -> None:
        self._last_in_level = max(0.0, min(1.0, float(level or 0.0)))
        self._engine.set_audio_features(
            {
                "rms": self._last_in_level,
                "loudness": self._last_in_level,
                "low_band": self._last_in_level * 0.35,
                "mid_band": self._last_in_level * 0.20,
                "speaking_gate": 1.0 if self._state == "listening" and self._last_in_level > 0.04 else 0.0,
            }
        )

    def set_output_level(self, level: float) -> None:
        self._last_out_level = max(0.0, min(1.0, float(level or 0.0)))
        self._engine.set_audio_features(
            {
                "rms": self._last_out_level,
                "loudness": self._last_out_level,
                "low_band": self._last_out_level * 0.65,
                "mid_band": self._last_out_level * 0.55,
                "high_band": self._last_out_level * 0.35,
                "speaking_gate": 1.0 if self._state == "speaking" and self._last_out_level > 0.03 else 0.0,
            }
        )

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict):
            return
        weights = snapshot.get("weights")
        if not isinstance(weights, dict):
            return
        aa = max(0.0, min(1.0, float(weights.get("aa", 0.0))))
        ee = max(0.0, min(1.0, float(weights.get("ee", 0.0))))
        oo = max(0.0, min(1.0, float(weights.get("oo", 0.0))))
        small = max(0.0, min(1.0, float(weights.get("small", 0.0))))
        self._mouth_energy = max(0.0, min(1.0, small * 0.20 + aa * 1.0 + ee * 0.48 + oo * 0.72))
        self._aurora_bias = max(0.0, min(1.0, ee * 0.35 + oo * 0.85 + aa * 0.20))
        if self._state == "speaking":
            self._engine.set_audio_features(
                {
                    "rms": max(self._last_out_level, self._mouth_energy * 0.6),
                    "loudness": max(self._last_out_level, self._mouth_energy * 0.8),
                    "mid_band": self._mouth_energy * 0.7,
                    "high_band": self._aurora_bias * 0.6,
                    "spectral_centroid": self._aurora_bias * 0.5,
                    "speaking_gate": 1.0,
                }
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

    def initializeGL(self) -> None:
        try:
            self._engine.renderer.initialize()
            self._engine.resize(self.width(), self.height())
        except Exception as exc:
            self._gl_failed = True
            self._gl_failure_reason = str(exc)

    def resizeGL(self, width: int, height: int) -> None:
        self._engine.resize(width, height)

    def paintGL(self) -> None:
        if self._gl_failed or not self._engine.renderer.is_ready:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            self._paint_overlay(painter)
            painter.end()
            return
        self._engine.render()
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing, True)
        self._paint_overlay(painter)
        painter.end()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = max(0.0, min(0.05, now - self._t0))
        self._t0 = now
        self._engine.update(dt)
        self.update()

    @staticmethod
    def _map_engine_state(state: str) -> str:
        mapping = {
            "idle": "idle",
            "connecting": "thinking",
            "reconnecting": "thinking",
            "listening": "listening",
            "transcribing": "thinking",
            "thinking": "thinking",
            "speaking": "speaking",
            "error": "idle",
        }
        return mapping.get(state, "idle")

    def _paint_overlay(self, painter: QPainter) -> None:
        if self._state == "error":
            self._paint_error_ui(painter)
            return
        self._reset_rect = QRectF()
        if not self._running and self._state == "idle":
            painter.setPen(QColor(220, 230, 245, 190))
            font = QFont(self.font())
            font.setPointSize(12)
            painter.setFont(font)
            painter.drawText(QRectF(0.0, self.height() * 0.82, float(self.width()), 32.0), Qt.AlignHCenter | Qt.AlignTop, "Klikni na orb pro nonstop režim")

    def _paint_error_ui(self, painter: QPainter) -> None:
        w = float(self.width())
        h = float(self.height())
        cx = w * 0.5
        y = h * 0.78
        painter.setPen(QColor(255, 225, 225, 235))
        font = QFont(self.font())
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        msg = (self._error_text or "Došlo k chybě.").splitlines()[0].strip()
        painter.drawText(QRectF(0.0, y, w, 28.0), Qt.AlignHCenter | Qt.AlignTop, msg)
        self._reset_rect = QRectF(cx - 75.0, y + 34.0, 150.0, 38.0)
        painter.setPen(QColor(255, 210, 210, 200))
        painter.setBrush(QColor(255, 120, 120, 42))
        painter.drawRoundedRect(self._reset_rect, 10.0, 10.0)
        painter.drawText(self._reset_rect, Qt.AlignCenter, "Reset")

    def shutdown(self) -> None:
        self._timer.stop()
        self._engine.shutdown()


def create_orb_widget(parent: Optional[QWidget] = None) -> QWidget:
    diagnostics = probe_opengl_support()
    if diagnostics.available:
        return LivingOrbWidget(diagnostics=diagnostics, parent=parent)
    return OrbFallbackWidget(diagnostics=diagnostics, parent=parent)
