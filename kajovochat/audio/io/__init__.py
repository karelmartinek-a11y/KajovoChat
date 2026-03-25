from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AudioPlayer",
    "AudioRecorder",
    "CapturedAudioChunk",
    "DuplexAudioSession",
    "RealtimeMicStream",
    "RecordResult",
    "VADMonitor",
]


def __getattr__(name: str) -> Any:
    if name in {"CapturedAudioChunk", "RecordResult"}:
        module = import_module(".common", __name__)
        return getattr(module, name)
    if name in {"AudioPlayer", "AudioRecorder", "DuplexAudioSession", "RealtimeMicStream", "VADMonitor"}:
        module = import_module(".runtime", __name__)
        return getattr(module, name)
    raise AttributeError(name)
