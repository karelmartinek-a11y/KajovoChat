from __future__ import annotations

from kajovochat.animation.performance_driver import PerformanceDriver
from kajovochat.animation.types import VisemeFrame


def _viseme(**overrides) -> VisemeFrame:
    base = VisemeFrame(
        timestamp_s=0.0,
        cluster="aa",
        pose="aa",
        openness=0.55,
        energy=0.55,
        speech_energy=0.55,
        voicing_confidence=0.7,
        attack=0.0,
        jaw_open=0.58,
        mouth_open=0.54,
        lip_funnel=0.05,
        lip_round=0.06,
        lip_spread=0.18,
        lip_press=0.0,
        upper_lip_raise=0.04,
        lower_lip_drop=0.42,
        cheek_raise=0.08,
        weights={"aa": 1.0},
        legacy_weights={"closed": 0.0, "small": 0.0, "aa": 1.0, "ee": 0.0, "oo": 0.0},
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_performance_driver_speaking_motion_exceeds_idle() -> None:
    driver = PerformanceDriver()
    idle = driver.drive(state="idle", input_level=0.0, output_level=0.0, lipsync_frame=_viseme(speech_energy=0.0), now=1.0)
    speaking = driver.drive(state="speaking", input_level=0.0, output_level=0.7, lipsync_frame=_viseme(speech_energy=0.8, attack=0.5), now=1.0)
    assert driver.motion_magnitude(speaking) > driver.motion_magnitude(idle)


def test_performance_driver_error_motion_is_damped() -> None:
    driver = PerformanceDriver()
    speaking = driver.drive(state="speaking", input_level=0.0, output_level=0.8, lipsync_frame=_viseme(speech_energy=0.9, attack=0.7), now=2.0)
    error = driver.drive(state="error", input_level=0.0, output_level=0.8, lipsync_frame=_viseme(speech_energy=0.9, attack=0.7), now=2.0)
    assert driver.motion_magnitude(error) < driver.motion_magnitude(speaking)


def test_performance_driver_suppresses_blink_during_attack() -> None:
    driver = PerformanceDriver()
    pre = driver.drive(state="speaking", input_level=0.0, output_level=0.6, lipsync_frame=_viseme(speech_energy=0.65, attack=0.0), now=3.0)
    hit = driver.drive(state="speaking", input_level=0.0, output_level=0.6, lipsync_frame=_viseme(speech_energy=0.8, attack=0.9), now=3.1)
    assert hit.blink.speaking_suppressed is True
    assert hit.blink.blink_amount <= pre.blink.blink_amount
