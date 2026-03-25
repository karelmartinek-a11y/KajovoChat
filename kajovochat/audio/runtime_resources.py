from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..services.audio_service import AudioPlayer, DuplexAudioSession, RealtimeMicStream
from ..services.realtime_service import RealtimeService


@dataclass
class AudioRuntimeResources:
    """Sdílené runtime vlastnictví transportu a audio I/O pro jednu relaci."""

    rt: Optional[RealtimeService] = None
    duplex: Optional[DuplexAudioSession] = None
    mic: Optional[RealtimeMicStream] = None
    player: Optional[AudioPlayer] = None
