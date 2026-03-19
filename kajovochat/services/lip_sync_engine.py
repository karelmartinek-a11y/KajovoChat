from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..animation.types import VisemeFrame
from ..animation.viseme_engine import VisemeEngine


@dataclass(frozen=True)
class LipSyncSnapshot:
    pose: str
    openness: float
    energy: float
    weights: dict[str, float]


class LipSyncEngine:
    """Kompatibilní obálka nad novým audio-driven viseme enginem."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine = VisemeEngine()

    def reset(self) -> None:
        with self._lock:
            self._engine.reset()

    def consume_playback_pcm16(self, pcm_bytes: bytes, samplerate: int) -> None:
        with self._lock:
            self._engine.consume_playback_pcm16(pcm_bytes, samplerate)

    def snapshot(self) -> LipSyncSnapshot:
        with self._lock:
            legacy = self._engine.snapshot().to_legacy_snapshot()
            return LipSyncSnapshot(
                pose=str(legacy["pose"]),
                openness=float(legacy["openness"]),
                energy=float(legacy["energy"]),
                weights=dict(legacy["weights"]),
            )

    def snapshot_dict(self, *, rich: bool = False) -> dict[str, Any]:
        with self._lock:
            return self._engine.snapshot_dict(rich=rich)

    def rich_snapshot(self) -> VisemeFrame:
        with self._lock:
            return self._engine.snapshot()
