from __future__ import annotations

from kajovochat.main import (
    _REALTIME_MODEL,
    _SERVER_VAD_PREFIX_MS,
    _SERVER_VAD_SILENCE_MS,
    _SERVER_VAD_THRESHOLD,
    _TTS_SPEED,
    _TTS_VOICE,
    _should_drop_mic_chunk,
    ConversationWorker,
)
from kajovochat.settings import AppSettings


class _FakeRealtimeService:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.is_connected = False
        self.on_status = None
        self.on_error = None
        self.on_user_transcript = None
        self.on_assistant_text_delta = None
        self.on_assistant_text_done = None
        self.on_assistant_audio_delta = None
        self.on_vad_speech_started = None
        self.on_vad_speech_stopped = None
        self.on_response_done = None

    def connect(self) -> None:
        self.is_connected = True

    def close(self) -> None:
        self.is_connected = False

    def update_session(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.cfg, key, value)


def test_realtime_config_is_hardcoded_and_not_user_tunable(monkeypatch) -> None:
    holder: dict[str, object] = {}

    def _factory(cfg):
        service = _FakeRealtimeService(cfg)
        holder["service"] = service
        return service

    monkeypatch.setattr("kajovochat.main.RealtimeService", _factory)

    settings = AppSettings(answer_language_mode="fixed", fixed_answer_language="fr", response_style="stručný")
    settings.openai_api_key = "sk-test-123"
    worker = ConversationWorker(settings)

    worker._ensure_realtime("server_vad")
    service = holder["service"]

    assert service.cfg.model == _REALTIME_MODEL
    assert service.cfg.voice == _TTS_VOICE
    assert service.cfg.output_speed == _TTS_SPEED
    assert service.cfg.server_vad_silence_ms == _SERVER_VAD_SILENCE_MS
    assert service.cfg.server_vad_prefix_ms == _SERVER_VAD_PREFIX_MS
    assert service.cfg.server_vad_threshold == _SERVER_VAD_THRESHOLD
    assert service.cfg.language_hint == "auto"


def test_echo_chunks_are_dropped_when_similarity_is_high() -> None:
    dropped, reason = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.93,
        input_level=0.05,
        output_level=0.08,
    )

    assert dropped is True
    assert reason == "echo_similarity"


def test_user_voice_passes_when_similarity_is_low() -> None:
    dropped, reason = _should_drop_mic_chunk(
        mode="handsfree",
        guard_active=True,
        playback_active=True,
        similarity=0.18,
        input_level=0.18,
        output_level=0.06,
    )

    assert dropped is False
    assert reason == ""
