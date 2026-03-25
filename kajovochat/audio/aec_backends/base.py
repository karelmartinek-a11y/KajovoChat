from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ReferencePrepLike(Protocol):
    mic_level: float
    similarity: float
    best_shift: int
    anchor_shift: int
    shift_error: int
    stable_delay_lock: bool
    voice_likelihood: float
    segment: np.ndarray


@dataclass(frozen=True)
class AecBackendContext:
    prep: ReferencePrepLike
    mic_pcm: bytes
    residual_level: float
    improvement_ratio: float
    predicted_level: float
    backend_used: str
    prefer_native_mode: bool = False


@dataclass(frozen=True)
class AecBackendResult:
    pcm: np.ndarray
    similarity: float
    residual_level: float
    improvement_ratio: float
    predicted_level: float
    delay_samples: int
    backend: str
    selection_reason: str
    webrtc_success: bool = False
    native_attempted: bool = False
    native_selected: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "pcm": self.pcm,
            "similarity": float(self.similarity),
            "residual_level": float(self.residual_level),
            "improvement_ratio": float(self.improvement_ratio),
            "predicted_level": float(self.predicted_level),
            "delay_samples": int(self.delay_samples),
            "backend": self.backend,
            "webrtc_success": bool(self.webrtc_success),
            "native_attempted": bool(self.native_attempted),
            "native_selected": bool(self.native_selected),
            "selection_reason": self.selection_reason,
        }
