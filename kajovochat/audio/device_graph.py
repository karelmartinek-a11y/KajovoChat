from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..services.audio_service import AudioPlayer, DuplexAudioSession, RealtimeMicStream


@dataclass
class DuplexDeviceGraph:
    """Popis aktivní duplexní audio topologie pro jednu relaci."""

    input_device: Optional[int] = None
    output_device: Optional[int] = None
    input_name: str = "default"
    output_name: str = "default"
    audio_mode: str = "notebook_builtin"
    duplex: Optional[DuplexAudioSession] = None
    player: Optional[AudioPlayer] = None
    mic: Optional[RealtimeMicStream] = None

    def stop_io(self) -> None:
        try:
            if self.duplex:
                self.duplex.stop()
        except Exception:
            pass
        self.duplex = None
        try:
            if self.mic:
                self.mic.stop()
        except Exception:
            pass
        self.mic = None
        try:
            if self.player:
                self.player.stop()
        except Exception:
            pass
        self.player = None
