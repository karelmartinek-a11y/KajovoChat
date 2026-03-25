from __future__ import annotations

import ast
from pathlib import Path

from kajovochat.audio.devices import build_device_fingerprint
from kajovochat.audio.io import AudioPlayer, DuplexAudioSession
from kajovochat.services.audio_service import (
    AudioPlayer as CompatAudioPlayer,
    DuplexAudioSession as CompatDuplexAudioSession,
    build_device_fingerprint as compat_build_device_fingerprint,
)


def test_audio_service_compat_is_pure_reexport_layer() -> None:
    source = Path("kajovochat/services/audio_service.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    allowed = (ast.ImportFrom, ast.Assign)
    assert all(isinstance(node, allowed) for node in module.body)


def test_audio_service_compat_exports_new_audio_modules() -> None:
    assert CompatAudioPlayer is AudioPlayer
    assert CompatDuplexAudioSession is DuplexAudioSession
    assert compat_build_device_fingerprint is build_device_fingerprint
