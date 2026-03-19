from __future__ import annotations

import math
import random

from .types import GazeFrame


class GazeEngine:
    def __init__(self, *, seed: int = 6789) -> None:
        self._rng = random.Random(seed)
        self._drift_x = 0.0
        self._drift_y = 0.0
        self._target_x = 0.0
        self._target_y = 0.0
        self._next_saccade_at = 0.0

    def update(self, *, now: float, state: str, speech_energy: float, speaking_attack: float) -> GazeFrame:
        if self._next_saccade_at == 0.0:
            self._next_saccade_at = now + self._rng.uniform(0.35, 0.85)

        speaking = state == "speaking"
        if now >= self._next_saccade_at:
            amp = 0.016 if speaking else 0.052
            self._target_x = self._rng.uniform(-amp, amp)
            self._target_y = self._rng.uniform(-amp * 0.7, amp * 0.7)
            self._next_saccade_at = now + self._rng.uniform(0.26, 0.55 if speaking else 1.10)

        idle_drift = 0.018 if state == "idle" else 0.008
        self._target_x += math.sin(now * 0.29) * idle_drift * 0.018
        self._target_y += math.cos(now * 0.24) * idle_drift * 0.013

        stability = 0.24 + (0.56 if speaking else 0.0) + speech_energy * 0.26 + speaking_attack * 0.12
        self._drift_x += (self._target_x - self._drift_x) * max(0.08, min(0.5, stability))
        self._drift_y += (self._target_y - self._drift_y) * max(0.08, min(0.5, stability))

        if speaking:
            lock = max(0.80, 0.90 - speech_energy * 0.10 - speaking_attack * 0.08)
            self._drift_x *= lock
            self._drift_y *= lock

        focus = 0.72
        if state == "idle":
            focus = 0.54
        elif state == "listening":
            focus = 0.82
        elif state == "thinking":
            focus = 0.64
        elif state == "speaking":
            focus = min(1.0, 0.92 + speech_energy * 0.05 + speaking_attack * 0.03)
        elif state == "error":
            focus = 0.95

        return GazeFrame(
            timestamp_s=now,
            gaze_x=max(-1.0, min(1.0, self._drift_x)),
            gaze_y=max(-1.0, min(1.0, self._drift_y)),
            focus_strength=max(0.0, min(1.0, focus)),
        )
