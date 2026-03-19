from __future__ import annotations

import math
import sys
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from .orb.widget import LivingOrbWidget
from .theme import app_stylesheet


class OrbDemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KajovoChat Living Orb Demo")
        self.resize(980, 820)
        self._states = ["idle", "listening", "thinking", "speaking"]
        self._state_index = 0
        self._t0 = time.perf_counter()

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("Living Orb Demo")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title, 0, Qt.AlignHCenter)

        self._hint = QLabel("Klávesy: 1 idle, 2 listening, 3 thinking, 4 speaking, mezerník scripted cycle")
        layout.addWidget(self._hint, 0, Qt.AlignHCenter)

        self._orb = LivingOrbWidget()
        self._orb.set_running(True)
        self._orb.setMinimumSize(680, 680)
        layout.addWidget(self._orb, 1)

        self._status = QLabel("")
        layout.addWidget(self._status, 0, Qt.AlignHCenter)

        self.setCentralWidget(root)

        self._scripted = True
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._apply_state("idle")

    def _apply_state(self, state: str) -> None:
        self._orb.set_state(state)
        self._status.setText(f"Stav: {state}")

    def _tick(self) -> None:
        t = time.perf_counter() - self._t0
        if self._scripted:
            state_span = 5.5
            state = self._states[int(t / state_span) % len(self._states)]
            self._apply_state(state)
        else:
            state = self._states[self._state_index]
        state = self._status.text().split(": ", 1)[-1] or "idle"
        burst = max(0.0, math.sin(t * 2.6) ** 2 - 0.18)
        gate = 1.0 if burst > 0.10 else 0.0
        features = {
            "loudness": 0.10 + burst * 0.82,
            "rms": 0.08 + burst * 0.66,
            "peak_envelope": 0.12 + burst * 0.78,
            "short_energy": 0.08 + burst * 0.64,
            "low_band": 0.16 + max(0.0, math.sin(t * 1.2 + 0.4)) * 0.58,
            "mid_band": 0.10 + max(0.0, math.sin(t * 2.4 + 1.3)) * 0.54,
            "high_band": 0.06 + max(0.0, math.sin(t * 4.2 + 0.2)) * 0.42,
            "spectral_centroid": 0.22 + max(0.0, math.sin(t * 0.9)) * 0.38,
            "spectral_flux": burst * 0.92,
            "transient_activity": max(0.0, math.sin(t * 7.8)) * burst,
            "speaking_gate": gate if state == "speaking" else 0.0,
        }
        self._orb.engine.set_audio_features(features)

    def keyPressEvent(self, event) -> None:
        mapping = {
            Qt.Key_1: "idle",
            Qt.Key_2: "listening",
            Qt.Key_3: "thinking",
            Qt.Key_4: "speaking",
        }
        if event.key() == Qt.Key_Space:
            self._scripted = not self._scripted
            return
        if event.key() in mapping:
            self._scripted = False
            state = mapping[event.key()]
            self._state_index = self._states.index(state)
            self._apply_state(state)
            return
        super().keyPressEvent(event)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(app_stylesheet())
    window = OrbDemoWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
