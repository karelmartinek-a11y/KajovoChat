from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from kajovochat.animation.types import BlinkFrame, GazeFrame, HeadMotionFrame, PerformanceFrame, VisemeFrame
from kajovochat.widgets.talking_head_widget import TalkingHeadWidget


def _app() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication(sys.argv)


def test_talking_head_widget_can_be_created_without_production_assets() -> None:
    _app()
    widget = TalkingHeadWidget()
    widget.resize(640, 640)
    assert widget.rig_definition.fallback_mode is True
    assert widget.rig_definition.production_ready is False


def test_talking_head_widget_accepts_legacy_snapshot_dict() -> None:
    _app()
    widget = TalkingHeadWidget()
    widget.set_lipsync_snapshot(
        {
            "pose": "oo",
            "openness": 0.42,
            "energy": 0.37,
            "weights": {"closed": 0.1, "small": 0.1, "aa": 0.1, "ee": 0.1, "oo": 0.6},
        }
    )
    widget._tick()
    assert widget._current_frame.viseme.pose == "oo"


def test_talking_head_widget_accepts_performance_frame() -> None:
    _app()
    widget = TalkingHeadWidget()
    frame = PerformanceFrame(
        state="speaking",
        input_level=0.0,
        output_level=0.6,
        speech_energy=0.6,
        viseme=VisemeFrame(cluster="aa", pose="aa", openness=0.5, energy=0.5, speech_energy=0.5, jaw_open=0.5, mouth_open=0.48),
        blink=BlinkFrame(),
        gaze=GazeFrame(),
        head_motion=HeadMotionFrame(),
    )
    widget.set_performance_frame(frame)
    widget._tick()
    assert widget._current_frame.state == "idle" or widget._current_frame.viseme.pose == "aa"


def test_talking_head_widget_error_state_with_empty_text_does_not_raise() -> None:
    _app()
    widget = TalkingHeadWidget()
    widget.resize(640, 640)
    widget.set_state("error")
    widget.set_error_text("")
    image = widget.grab()
    assert not image.isNull()
