from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _dict_of_floats(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _float(item) for key, item in value.items()}


@dataclass
class VisemeFrame:
    timestamp_s: float = 0.0
    cluster: str = "sil"
    pose: str = "closed"
    openness: float = 0.0
    energy: float = 0.0
    speech_energy: float = 0.0
    voicing_confidence: float = 0.0
    attack: float = 0.0
    jaw_open: float = 0.0
    mouth_open: float = 0.0
    lip_funnel: float = 0.0
    lip_round: float = 0.0
    lip_spread: float = 0.0
    lip_press: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_drop: float = 0.0
    cheek_raise: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    legacy_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VisemeFrame":
        source = data or {}
        return cls(
            timestamp_s=_float(source.get("timestamp_s")),
            cluster=str(source.get("cluster", "sil")),
            pose=str(source.get("pose", "closed")),
            openness=_clamp01(source.get("openness")),
            energy=_clamp01(source.get("energy")),
            speech_energy=_clamp01(source.get("speech_energy", source.get("energy"))),
            voicing_confidence=_clamp01(source.get("voicing_confidence")),
            attack=_clamp01(source.get("attack")),
            jaw_open=_clamp01(source.get("jaw_open")),
            mouth_open=_clamp01(source.get("mouth_open")),
            lip_funnel=_clamp01(source.get("lip_funnel")),
            lip_round=_clamp01(source.get("lip_round")),
            lip_spread=_clamp01(source.get("lip_spread")),
            lip_press=_clamp01(source.get("lip_press")),
            upper_lip_raise=_clamp01(source.get("upper_lip_raise")),
            lower_lip_drop=_clamp01(source.get("lower_lip_drop")),
            cheek_raise=_clamp01(source.get("cheek_raise")),
            weights=_dict_of_floats(source.get("weights")),
            legacy_weights=_dict_of_floats(source.get("legacy_weights", source.get("weights"))),
        )

    @classmethod
    def from_legacy_snapshot(cls, data: dict[str, Any] | None) -> "VisemeFrame":
        source = data or {}
        weights = _dict_of_floats(source.get("weights"))
        pose = str(source.get("pose", "closed"))
        openness = _clamp01(source.get("openness"))
        energy = _clamp01(source.get("energy"))
        return cls(
            cluster="sil" if pose == "closed" else pose,
            pose=pose,
            openness=openness,
            energy=energy,
            speech_energy=energy,
            mouth_open=openness,
            jaw_open=min(1.0, openness * 0.95),
            lip_round=_clamp01(weights.get("oo")),
            lip_funnel=_clamp01(weights.get("oo", 0.0) * 0.9),
            lip_spread=_clamp01(weights.get("ee")),
            lower_lip_drop=min(1.0, openness * 0.85),
            cheek_raise=_clamp01(weights.get("ee", 0.0) * 0.45),
            weights=weights,
            legacy_weights=weights or {
                "closed": 1.0,
                "small": 0.0,
                "aa": 0.0,
                "ee": 0.0,
                "oo": 0.0,
            },
        )

    def to_legacy_snapshot(self) -> dict[str, Any]:
        weights = dict(self.legacy_weights or {})
        if not weights:
            weights = {
                "closed": 1.0 if self.pose == "closed" else 0.0,
                "small": 0.0,
                "aa": 0.0,
                "ee": 0.0,
                "oo": 0.0,
            }
        return {
            "pose": self.pose,
            "openness": _clamp01(self.openness),
            "energy": _clamp01(self.energy),
            "weights": weights,
        }


@dataclass
class BlinkFrame:
    timestamp_s: float = 0.0
    blink_amount: float = 0.0
    is_blinking: bool = False
    speaking_suppressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BlinkFrame":
        source = data or {}
        amount = _clamp01(source.get("blink_amount"))
        return cls(
            timestamp_s=_float(source.get("timestamp_s")),
            blink_amount=amount,
            is_blinking=bool(source.get("is_blinking", amount > 0.01)),
            speaking_suppressed=bool(source.get("speaking_suppressed", False)),
        )


@dataclass
class GazeFrame:
    timestamp_s: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    focus_strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GazeFrame":
        source = data or {}
        return cls(
            timestamp_s=_float(source.get("timestamp_s")),
            gaze_x=max(-1.0, min(1.0, _float(source.get("gaze_x")))),
            gaze_y=max(-1.0, min(1.0, _float(source.get("gaze_y")))),
            focus_strength=_clamp01(source.get("focus_strength", 1.0)),
        )


@dataclass
class HeadMotionFrame:
    timestamp_s: float = 0.0
    head_tx: float = 0.0
    head_ty: float = 0.0
    head_rot: float = 0.0
    neck_compensation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HeadMotionFrame":
        source = data or {}
        return cls(
            timestamp_s=_float(source.get("timestamp_s")),
            head_tx=max(-1.0, min(1.0, _float(source.get("head_tx")))),
            head_ty=max(-1.0, min(1.0, _float(source.get("head_ty")))),
            head_rot=max(-1.0, min(1.0, _float(source.get("head_rot")))),
            neck_compensation=_clamp01(source.get("neck_compensation", 0.0)),
        )


@dataclass
class PerformanceFrame:
    timestamp_s: float = 0.0
    state: str = "idle"
    input_level: float = 0.0
    output_level: float = 0.0
    speech_energy: float = 0.0
    viseme: VisemeFrame = field(default_factory=VisemeFrame)
    blink: BlinkFrame = field(default_factory=BlinkFrame)
    gaze: GazeFrame = field(default_factory=GazeFrame)
    head_motion: HeadMotionFrame = field(default_factory=HeadMotionFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "state": self.state,
            "input_level": self.input_level,
            "output_level": self.output_level,
            "speech_energy": self.speech_energy,
            "viseme": self.viseme.to_dict(),
            "blink": self.blink.to_dict(),
            "gaze": self.gaze.to_dict(),
            "head_motion": self.head_motion.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PerformanceFrame":
        source = data or {}
        return cls(
            timestamp_s=_float(source.get("timestamp_s")),
            state=str(source.get("state", "idle")),
            input_level=_clamp01(source.get("input_level")),
            output_level=_clamp01(source.get("output_level")),
            speech_energy=_clamp01(source.get("speech_energy")),
            viseme=VisemeFrame.from_dict(source.get("viseme")),
            blink=BlinkFrame.from_dict(source.get("blink")),
            gaze=GazeFrame.from_dict(source.get("gaze")),
            head_motion=HeadMotionFrame.from_dict(source.get("head_motion")),
        )

    @classmethod
    def from_legacy_snapshot(
        cls,
        snapshot: dict[str, Any] | None,
        *,
        state: str = "speaking",
        input_level: float = 0.0,
        output_level: float | None = None,
    ) -> "PerformanceFrame":
        viseme = VisemeFrame.from_legacy_snapshot(snapshot)
        return cls(
            state=state,
            input_level=_clamp01(input_level),
            output_level=_clamp01(viseme.energy if output_level is None else output_level),
            speech_energy=viseme.speech_energy,
            viseme=viseme,
        )
