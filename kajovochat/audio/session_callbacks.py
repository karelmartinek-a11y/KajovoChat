from __future__ import annotations

from typing import Any


class ConversationAudioCallbacks:
    """Lehká kompatibilní callback vrstva bez runtime ownershipu ani recovery policy."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def handle_user_transcript(self, text: str) -> None:
        self.owner._session_manager.handle_user_transcript(text)

    def handle_assistant_done(self, text: str) -> None:
        self.owner._session_manager.handle_assistant_done(text)

    def handle_assistant_audio(self, pcm: bytes) -> None:
        self.owner._session_manager.handle_assistant_audio(pcm)

    def handle_speech_started(self) -> None:
        self.owner._session_manager.handle_speech_started()

    def handle_speech_stopped(self) -> None:
        self.owner._session_manager.handle_speech_stopped()

    def handle_response_done(self) -> None:
        self.owner._session_manager.handle_response_done()
