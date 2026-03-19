from __future__ import annotations

from PySide6.QtWidgets import QApplication

from kajovochat.animation.types import BlinkFrame, GazeFrame, HeadMotionFrame, PerformanceFrame, VisemeFrame
from kajovochat.widgets.talking_head_widget import TalkingHeadWidget


def test_talking_head_widget_can_be_created_without_production_assets(qapp: QApplication) -> None:
    widget = TalkingHeadWidget()
    widget.resize(640, 640)
    assert widget.rig_definition.fallback_mode is True
    assert widget.rig_definition.production_ready is False


def test_talking_head_widget_accepts_legacy_snapshot_dict(qapp: QApplication) -> None:
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


def test_talking_head_widget_accepts_performance_frame(qapp: QApplication) -> None:
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
    assert widget._current_frame.state == "speaking"
    assert widget._current_frame.output_level > 0.0
    assert widget._current_frame.viseme.pose == "aa"


def test_talking_head_widget_accepts_performance_frame_dict(qapp: QApplication) -> None:
    widget = TalkingHeadWidget()
    frame = PerformanceFrame(
        state="thinking",
        input_level=0.2,
        output_level=0.0,
        speech_energy=0.2,
        viseme=VisemeFrame(cluster="ih", pose="small", openness=0.18, energy=0.2, speech_energy=0.2, jaw_open=0.12, mouth_open=0.16),
        blink=BlinkFrame(),
        gaze=GazeFrame(),
        head_motion=HeadMotionFrame(),
    )
    widget.set_performance_frame(frame.to_dict())
    widget._tick()
    assert widget._current_frame.state == "thinking"
    assert widget._current_frame.input_level >= 0.19
    assert widget._current_frame.viseme.cluster == "ih"


def test_talking_head_widget_error_state_with_empty_text_does_not_raise(qapp: QApplication) -> None:
    widget = TalkingHeadWidget()
    widget.resize(640, 640)
    widget.set_state("error")
    widget.set_error_text("")
    image = widget.grab()
    assert not image.isNull()
