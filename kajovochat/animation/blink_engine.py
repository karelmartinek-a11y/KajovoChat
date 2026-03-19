from __future__ import annotations

import math
import random

from .types import BlinkFrame


class BlinkEngine:
    def __init__(self, *, seed: int = 12345) -> None:
        self._rng = random.Random(seed)
        self._last_time = 0.0
        self._next_blink_at = self._schedule_next(0.0, idle=False)
        self._blink_until = 0.0
        self._blink_started_at = 0.0
        self._idle_accum = 0.0

    def _schedule_next(self, now: float, *, idle: bool) -> float:
        base = self._rng.uniform(2.6, 4.8)
        return now + (base * 0.78 if idle else base)

    def update(
        self,
        *,
        now: float,
        speech_energy: float,
        speaking_attack: float,
        is_idle: bool,
    ) -> BlinkFrame:
        dt = max(0.0, now - self._last_time) if self._last_time else 0.016
        self._last_time = now
        self._idle_accum = self._idle_accum + dt if is_idle else 0.0

        suppressed = speech_energy > 0.14 and speaking_attack > 0.22
        if suppressed and self._next_blink_at <= now + 0.18:
            self._next_blink_at = now + 0.24

        if self._idle_accum >= 5.0 and self._blink_until <= now:
            self._blink_started_at = now
            self._blink_until = now + 0.16
            self._idle_accum = 0.0

        if now >= self._next_blink_at and self._blink_until <= now and not suppressed:
            duration = self._rng.uniform(0.12, 0.18)
            self._blink_started_at = now
            self._blink_until = now + duration
            self._next_blink_at = self._schedule_next(now, idle=is_idle)

        blink_amount = 0.0
        if self._blink_until > now:
            total = max(0.12, self._blink_until - self._blink_started_at)
            phase = (now - self._blink_started_at) / total
            blink_amount = math.sin(max(0.0, min(1.0, phase)) * math.pi)
        elif self._blink_until and now >= self._blink_until:
            self._blink_until = 0.0
            self._blink_started_at = 0.0

        return BlinkFrame(
            timestamp_s=now,
            blink_amount=max(0.0, min(1.0, blink_amount)),
            is_blinking=blink_amount > 0.01,
            speaking_suppressed=suppressed,
        )
