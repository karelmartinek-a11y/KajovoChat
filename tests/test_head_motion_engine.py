from __future__ import annotations

from kajovochat.animation.head_motion_engine import HeadMotionEngine


def _magnitude(frame) -> float:
    return abs(frame.head_tx) + abs(frame.head_ty) + abs(frame.head_rot)


def test_head_motion_engine_idle_breathing_drift_changes_over_time() -> None:
    engine = HeadMotionEngine()
    frame_a = engine.update(now=0.0, state="idle", speech_energy=0.0, speaking_attack=0.0)
    frame_b = engine.update(now=2.4, state="idle", speech_energy=0.0, speaking_attack=0.0)
    assert frame_a != frame_b
    assert _magnitude(frame_a) > 0.0 or _magnitude(frame_b) > 0.0


def test_head_motion_engine_speaking_motion_is_stronger_than_idle() -> None:
    engine = HeadMotionEngine()
    idle = engine.update(now=1.1, state="idle", speech_energy=0.0, speaking_attack=0.0)
    speaking = engine.update(now=1.1, state="speaking", speech_energy=0.8, speaking_attack=0.6)
    assert _magnitude(speaking) > _magnitude(idle)
