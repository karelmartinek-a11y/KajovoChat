from __future__ import annotations

import time

from .blink_engine import BlinkEngine
from .gaze_engine import GazeEngine
from .head_motion_engine import HeadMotionEngine
from .types import HeadMotionFrame, PerformanceFrame, VisemeFrame


_STATE_PRESETS = {
    "idle": {"motion_scale": 0.75, "speech_boost": 0.0, "blink_idle": True},
    "listening": {"motion_scale": 0.92, "speech_boost": 0.0, "blink_idle": False},
    "thinking": {"motion_scale": 0.84, "speech_boost": 0.0, "blink_idle": False},
    "speaking": {"motion_scale": 1.0, "speech_boost": 0.70, "blink_idle": False},
    "error": {"motion_scale": 0.25, "speech_boost": 0.0, "blink_idle": False},
}


class PerformanceDriver:
    def __init__(self) -> None:
        self._blink = BlinkEngine()
        self._gaze = GazeEngine()
        self._head_motion = HeadMotionEngine()

    def drive(
        self,
        *,
        state: str,
        input_level: float,
        output_level: float,
        lipsync_frame: VisemeFrame | dict[str, object] | None,
        now: float | None = None,
    ) -> PerformanceFrame:
        timestamp = time.perf_counter() if now is None else float(now)
        viseme = lipsync_frame if isinstance(lipsync_frame, VisemeFrame) else VisemeFrame.from_dict(lipsync_frame or {})
        preset = _STATE_PRESETS.get(state, _STATE_PRESETS["idle"])

        speech_energy = viseme.speech_energy if state == "speaking" else max(viseme.speech_energy * 0.45, output_level * 0.25)
        speech_energy = max(0.0, min(1.0, speech_energy))

        blink = self._blink.update(
            now=timestamp,
            speech_energy=speech_energy,
            speaking_attack=viseme.attack if state == "speaking" else 0.0,
            is_idle=bool(preset["blink_idle"]),
        )
        gaze = self._gaze.update(
            now=timestamp,
            state=state,
            speech_energy=speech_energy,
            speaking_attack=viseme.attack,
        )
        head_motion = self._head_motion.update(
            now=timestamp,
            state=state,
            speech_energy=speech_energy,
            speaking_attack=viseme.attack,
        )
        head_motion = self._scale_head_motion(
            head_motion,
            scale=float(preset["motion_scale"]),
            boost=float(preset["speech_boost"]) * speech_energy,
        )

        return PerformanceFrame(
            timestamp_s=timestamp,
            state=state,
            input_level=max(0.0, min(1.0, float(input_level))),
            output_level=max(0.0, min(1.0, float(output_level))),
            speech_energy=speech_energy,
            viseme=viseme,
            blink=blink,
            gaze=gaze,
            head_motion=head_motion,
        )

    @staticmethod
    def _scale_head_motion(head_motion: HeadMotionFrame, *, scale: float, boost: float) -> HeadMotionFrame:
        factor = max(0.0, scale + boost)
        return HeadMotionFrame(
            timestamp_s=head_motion.timestamp_s,
            head_tx=max(-1.0, min(1.0, head_motion.head_tx * factor)),
            head_ty=max(-1.0, min(1.0, head_motion.head_ty * factor)),
            head_rot=max(-1.0, min(1.0, head_motion.head_rot * factor)),
            neck_compensation=max(0.0, min(1.0, head_motion.neck_compensation * min(1.0, 0.75 + factor * 0.35))),
        )

    @staticmethod
    def motion_magnitude(frame: PerformanceFrame) -> float:
        motion = frame.head_motion
        return abs(motion.head_tx) + abs(motion.head_ty) + abs(motion.head_rot) + motion.neck_compensation * 0.25
