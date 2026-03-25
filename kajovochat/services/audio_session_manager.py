from __future__ import annotations

from ..audio.session_manager import AudioSessionManager
from ..audio.aec_engine import AecEngine, BackendSelectionDecision
from ..audio.session_state import SessionState
from ..audio.io import DuplexAudioSession
from .windows_native_aec import probe_windows_native_aec

try:
    from aec_audio_processing import AudioProcessor as _WebRTCAudioProcessor
except Exception:
    _WebRTCAudioProcessor = None

__all__ = [
    "AecEngine",
    "AudioSessionManager",
    "BackendSelectionDecision",
    "DuplexAudioSession",
    "SessionState",
    "_WebRTCAudioProcessor",
    "probe_windows_native_aec",
]
