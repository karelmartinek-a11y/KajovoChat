from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional

from PySide6.QtCore import QBasicTimer, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class HeadWidget(QWidget):
    """Hlavní vizualizace relace: EKG linka a terminálový přepis."""

    orb_clicked = Signal()
    reset_clicked = Signal()

    def __init__(self, image_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        del image_path

        self._state = "idle"
        self._running = False
        self._error_text = ""
        self._reset_rect = QRectF()
        self._mouth_energy = 0.0
        self._aurora_bias = 0.0
        self._input_level = 0.0
        self._output_level = 0.0
        self._pulse_phase = 0.0
        self._history: Deque[float] = deque([0.0] * 240, maxlen=240)
        self._amplitude_log: Deque[str] = deque(maxlen=8)
        self._current_segment_peak = 0.0
        self._terminal_lines: list[str] = ["[READY] Terminál čeká na relaci."]
        self._timer = QBasicTimer()
        self._timer.start(33, self)

        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMinimumHeight(520)

    def _sync_reset_rect(self) -> None:
        if self._state == "error":
            w = float(max(1, self.width()))
            h = float(max(1, self.height()))
            self._reset_rect = QRectF(w * 0.5 - 88.0, h - 66.0, 176.0, 36.0)
        else:
            self._reset_rect = QRectF()

    def set_state(self, state: str) -> None:
        self._state = (state or "idle").strip().lower()
        self._sync_reset_rect()
        self.update()

    def set_running(self, running: bool) -> None:
        self._running = bool(running)
        self.update()

    def set_input_level(self, level: float) -> None:
        self._input_level = max(0.0, min(1.0, float(level)))

    def set_output_level(self, level: float) -> None:
        self._output_level = max(0.0, min(1.0, float(level)))

    def set_lipsync_snapshot(self, snapshot: object) -> None:
        data = snapshot if isinstance(snapshot, dict) else {}
        weights = data.get("weights", {}) if isinstance(data, dict) else {}
        self._mouth_energy = float(data.get("energy", 0.0) or 0.0)
        self._aurora_bias = float(
            weights.get("aa", 0.0) * 0.9
            + weights.get("ee", 0.0) * 0.7
            + weights.get("oo", 0.0) * 0.8
            + weights.get("small", 0.0) * 0.35
        )

    def set_error_text(self, msg: str) -> None:
        self._error_text = (msg or "").strip()
        self._sync_reset_rect()
        self.update()

    def set_terminal_text(self, text: str) -> None:
        lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
        self._terminal_lines = lines[-10:] if lines else ["[READY] Terminál čeká na relaci."]
        self.update()

    def render_backend_summary(self) -> str:
        return "backend=ekg-2d"

    def is_gpu_renderer_active(self) -> bool:
        return False

    def timerEvent(self, event) -> None:
        if event.timerId() != self._timer.timerId():
            super().timerEvent(event)
            return
        previous_phase = self._pulse_phase
        self._pulse_phase = (self._pulse_phase + 1.0) % 96.0
        offset_ratio = self._current_wave_offset_ratio()
        self._history.append(offset_ratio)
        self._current_segment_peak = max(self._current_segment_peak, abs(offset_ratio))
        if self._pulse_phase < previous_phase:
            self._amplitude_log.append(f"{self._current_segment_peak * 200:.0f}%")
            self._current_segment_peak = 0.0
        self.update()

    def _current_wave_sample(self) -> float:
        base_level = max(self._input_level * 1.7, self._output_level * 1.5, self._mouth_energy * 1.25)
        if self._state in {"connecting", "reconnecting", "transcribing", "thinking"}:
            base_level = max(base_level, 0.22)
        if not self._running and self._state not in {"speaking", "listening"}:
            base_level *= 0.25

        beat = self._beat_shape(self._pulse_phase / 96.0)
        shimmer = (
            math.sin(self._pulse_phase * 0.45)
            + math.sin(self._pulse_phase * 0.19 + 0.8)
            + math.sin(self._pulse_phase * 0.08 + 1.6)
        ) / 3.0
        sample = beat * (0.45 + base_level * 1.65) + shimmer * base_level * 0.95
        return max(-1.0, min(1.0, sample))

    def _current_wave_offset_ratio(self) -> float:
        sample = self._current_wave_sample()
        amp_ratio = 0.24 + max(self._input_level, self._output_level, self._aurora_bias) * 0.68
        return float(max(-0.94, min(0.94, sample * amp_ratio)))

    @staticmethod
    def _beat_shape(position: float) -> float:
        if position < 0.12:
            return -0.08 + position * 0.2
        if position < 0.18:
            return 0.12 + (position - 0.12) * 2.6
        if position < 0.22:
            return 0.28 - (position - 0.18) * 4.0
        if position < 0.25:
            return -0.22 - (position - 0.22) * 8.0
        if position < 0.29:
            return -0.46 + (position - 0.25) * 27.0
        if position < 0.34:
            return 0.62 - (position - 0.29) * 11.8
        if position < 0.42:
            return 0.03 - (position - 0.34) * 0.9
        return math.sin(position * math.tau * 1.15) * 0.03

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        background = QLinearGradient(0.0, 0.0, 0.0, float(rect.height()))
        background.setColorAt(0.0, QColor("#09111A"))
        background.setColorAt(0.45, QColor("#07171D"))
        background.setColorAt(1.0, QColor("#020705"))
        painter.fillRect(rect, background)

        panel_rect = rect.adjusted(18, 18, -18, -18)
        painter.fillRect(panel_rect, QColor(4, 18, 14, 210))
        painter.setPen(QColor(84, 161, 119, 56))
        for row in range(12):
            y = panel_rect.top() + row * (panel_rect.height() / 11.0)
            painter.drawLine(int(panel_rect.left()), int(y), int(panel_rect.right()), int(y))

        wave_rect = QRectF(
            panel_rect.left() + 18.0,
            panel_rect.top() + 18.0,
            panel_rect.width() - 36.0,
            max(120.0, panel_rect.height() * 0.38),
        )
        terminal_rect = QRectF(
            wave_rect.left(),
            wave_rect.bottom() + 18.0,
            wave_rect.width(),
            panel_rect.bottom() - wave_rect.bottom() - 36.0,
        )

        self._paint_header(painter, wave_rect)
        self._paint_wave(painter, wave_rect)
        self._paint_terminal(painter, terminal_rect)
        if self._error_text:
            self._paint_error(painter, panel_rect)

    def _paint_header(self, painter: QPainter, wave_rect: QRectF) -> None:
        label_font = QFont("Consolas", 10)
        painter.setFont(label_font)
        painter.setPen(QColor("#8FFFD0"))
        mode = "RUN" if self._running else "IDLE"
        painter.drawText(
            QRectF(wave_rect.left(), wave_rect.top() - 8.0, wave_rect.width(), 18.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"[{mode}] stav={self._state}  mic={self._input_level:.2f}  out={self._output_level:.2f}",
        )

    def _paint_wave(self, painter: QPainter, wave_rect: QRectF) -> None:
        painter.save()
        painter.fillRect(wave_rect, QColor(6, 12, 10, 180))
        log_width = 92.0
        trace_rect = QRectF(wave_rect.left(), wave_rect.top(), wave_rect.width() - log_width, wave_rect.height())
        log_rect = QRectF(trace_rect.right() + 10.0, wave_rect.top(), log_width - 10.0, wave_rect.height())

        baseline = wave_rect.center().y()
        painter.setPen(QColor(56, 115, 86, 90))
        painter.drawLine(int(trace_rect.left()), int(baseline), int(trace_rect.right()), int(baseline))

        path = QPainterPath()
        samples = list(self._history)
        if not samples:
            painter.restore()
            return

        vertical_margin = 14.0
        amp_scale = max(1.0, (trace_rect.height() * 0.5) - vertical_margin)
        step = trace_rect.width() / max(1, len(samples) - 1)
        for index, value in enumerate(samples):
            x = trace_rect.left() + index * step
            y = baseline - value * amp_scale
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        glow_pen = QPen(QColor(34, 255, 153, 70), 8.0)
        glow_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(glow_pen)
        painter.drawPath(path)

        main_pen = QPen(QColor("#7CFF8D"), 2.3)
        main_pen.setCapStyle(Qt.RoundCap)
        painter.setPen(main_pen)
        painter.drawPath(path)

        self._paint_writer_head(painter, trace_rect, baseline, samples[-1] * amp_scale)
        arrow = QPainterPath()
        arrow.moveTo(log_rect.left() + 8.0, baseline)
        arrow.lineTo(log_rect.left() + 20.0, baseline - 7.0)
        arrow.lineTo(log_rect.left() + 20.0, baseline + 7.0)
        arrow.closeSubpath()
        painter.fillPath(arrow, QColor("#B9FFD5"))

        self._paint_amplitude_log(painter, log_rect)
        painter.restore()

    def _paint_writer_head(self, painter: QPainter, trace_rect: QRectF, baseline: float, offset_y: float) -> None:
        write_x = trace_rect.right() - 2.0
        tip_y = baseline - offset_y

        paper_glow = QLinearGradient(write_x - 42.0, 0.0, write_x + 6.0, 0.0)
        paper_glow.setColorAt(0.0, QColor(0, 255, 170, 0))
        paper_glow.setColorAt(0.72, QColor(130, 255, 180, 36))
        paper_glow.setColorAt(1.0, QColor(0, 255, 170, 0))
        painter.fillRect(trace_rect, paper_glow)

        carriage_top = trace_rect.top() + 10.0
        carriage_bottom = trace_rect.bottom() - 10.0
        painter.setPen(QPen(QColor("#84FFD0"), 1.2))
        painter.drawLine(int(write_x), int(carriage_top), int(write_x), int(carriage_bottom))

        arm_x = write_x + 10.0
        painter.setPen(QPen(QColor("#D6FFF0"), 1.5))
        painter.drawLine(int(arm_x), int(carriage_top + 10.0), int(arm_x), int(tip_y - 6.0))
        painter.drawLine(int(arm_x), int(tip_y - 6.0), int(write_x), int(tip_y))

        painter.setBrush(QColor("#E8FFF6"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(write_x - 3.0), int(tip_y - 3.0), 6, 6)

    def _paint_amplitude_log(self, painter: QPainter, log_rect: QRectF) -> None:
        painter.save()
        painter.fillRect(log_rect, QColor(4, 18, 12, 230))
        painter.setPen(QColor(35, 96, 57, 140))
        rows = 8
        row_height = log_rect.height() / rows
        for index in range(rows + 1):
            y = log_rect.top() + index * row_height
            painter.drawLine(int(log_rect.left()), int(y), int(log_rect.right()), int(y))

        painter.setPen(QColor("#B5FFD0"))
        painter.setFont(QFont("Consolas", 9, QFont.Bold))
        painter.drawText(
            QRectF(log_rect.left() + 6.0, log_rect.top() + 2.0, log_rect.width() - 12.0, 16.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            "AMP",
        )

        painter.setFont(QFont("Consolas", 9))
        values = list(self._amplitude_log)
        while len(values) < rows:
            values.insert(0, "")
        active_row = min(rows - 1, len(self._amplitude_log) % rows)
        for index, value in enumerate(values[-rows:]):
            y = log_rect.top() + index * row_height
            if index == active_row:
                painter.setPen(QColor("#E2FFD8"))
                painter.drawText(
                    QRectF(log_rect.left() + 4.0, y, 10.0, row_height),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    ">"
                )
            painter.setPen(QColor("#7CFF8D"))
            painter.drawText(
                QRectF(log_rect.left() + 18.0, y, log_rect.width() - 22.0, row_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                value,
            )
        painter.restore()

    def _paint_terminal(self, painter: QPainter, terminal_rect: QRectF) -> None:
        painter.save()
        painter.fillRect(terminal_rect, QColor(0, 6, 0, 220))
        painter.setPen(QColor(33, 104, 52, 120))
        rows = 10
        line_height = terminal_rect.height() / rows
        for index in range(rows + 1):
            y = terminal_rect.top() + index * line_height
            painter.drawLine(int(terminal_rect.left()), int(y), int(terminal_rect.right()), int(y))

        title_font = QFont("Consolas", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#70FF84"))
        painter.drawText(
            QRectF(terminal_rect.left() + 12.0, terminal_rect.top() + 8.0, terminal_rect.width() - 24.0, 20.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            "TRANSCRIPT://LIVE_BUFFER",
        )

        text_font = QFont("Consolas", 11)
        painter.setFont(text_font)
        lines = self._terminal_lines[-10:]
        while len(lines) < 10:
            lines.insert(0, "")

        for index, line in enumerate(lines):
            y = terminal_rect.top() + 28.0 + index * line_height
            prefix = f"{index + 1:02d}> "
            text = (prefix + line)[:110]
            painter.drawText(
                QRectF(terminal_rect.left() + 12.0, y, terminal_rect.width() - 24.0, line_height),
                Qt.AlignLeft | Qt.AlignVCenter,
                text,
            )
        painter.restore()

    def _paint_error(self, painter: QPainter, panel_rect: QRectF) -> None:
        painter.save()
        overlay_rect = QRectF(
            panel_rect.left() + 26.0,
            panel_rect.bottom() - 118.0,
            panel_rect.width() - 52.0,
            84.0,
        )
        painter.fillRect(overlay_rect, QColor(62, 8, 8, 220))
        painter.setPen(QColor("#FFB6A3"))
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(
            overlay_rect.adjusted(12.0, 8.0, -12.0, -8.0),
            Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
            f"ERROR:// {self._error_text}",
        )
        painter.restore()
