from __future__ import annotations

import base64

from kajovochat.services.realtime_service import RealtimeConfig, RealtimeService


def test_realtime_service_parses_callbacks() -> None:
    service = RealtimeService(
        RealtimeConfig(
            api_key="sk-test-123",
            model="gpt-realtime",
            instructions="Test",
            voice="alloy",
        )
    )

    seen: dict[str, object] = {}
    service.on_user_transcript = lambda text: seen.setdefault("user", text)
    service.on_assistant_text_delta = lambda delta: seen.setdefault("delta", delta)
    service.on_assistant_text_done = lambda text: seen.setdefault("assistant", text)
    service.on_assistant_audio_delta = lambda pcm: seen.setdefault("audio", pcm)
    service.on_vad_speech_started = lambda: seen.setdefault("speech_started", True)
    service.on_vad_speech_stopped = lambda: seen.setdefault("speech_stopped", True)
    service.on_response_done = lambda: seen.setdefault("response_done", True)

    service._handle_event({"type": "input_audio_buffer.speech_started"})
    service._handle_event({"type": "input_audio_buffer.speech_stopped"})
    service._handle_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Ahoj"})
    service._handle_event({"type": "response.output_audio_transcript.delta", "delta": "Naz"})
    service._handle_event({"type": "response.output_audio_transcript.delta", "delta": "dar"})
    service._handle_event(
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x01\x02").decode("ascii")}
    )
    service._handle_event({"type": "response.output_audio_transcript.done"})
    service._handle_event({"type": "response.done"})

    assert seen["speech_started"] is True
    assert seen["speech_stopped"] is True
    assert seen["user"] == "Ahoj"
    assert seen["delta"] == "Naz"
    assert seen["assistant"] == "Nazdar"
    assert seen["audio"] == b"\x01\x02"
    assert seen["response_done"] is True
