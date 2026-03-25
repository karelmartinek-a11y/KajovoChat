from __future__ import annotations

import base64
import threading

import pytest

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
    service.on_event = lambda evt: seen.setdefault("event_type", evt.get("type"))

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
    assert seen["event_type"] == "input_audio_buffer.speech_started"
    assert seen["user"] == "Ahoj"
    assert seen["delta"] == "Naz"
    assert seen["assistant"] == "Nazdar"
    assert seen["audio"] == b"\x01\x02"
    assert seen["response_done"] is True


class _ImmediateThread:
    def __init__(self, *, target, daemon: bool = False) -> None:
        self._target = target
        self.daemon = daemon
        self._alive = False

    def start(self) -> None:
        self._alive = True
        try:
            self._target()
        finally:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        del timeout


def test_realtime_service_retries_transient_handshake_failure_before_connecting(monkeypatch) -> None:
    attempts = iter(("fail", "open"))

    class _FakeWebSocketApp:
        def __init__(self, url, header, on_open, on_message, on_error, on_close) -> None:
            del url, header, on_message
            self._mode = next(attempts)
            self._on_open = on_open
            self._on_error = on_error
            self._on_close = on_close

        def run_forever(self, ping_interval=20, ping_timeout=10) -> None:
            del ping_interval, ping_timeout
            if self._mode == "fail":
                self._on_error(self, RuntimeError("Handshake status 504 Gateway Time-out"))
                self._on_close(self, None, "Handshake status 504 Gateway Time-out")
                return
            self._on_open(self)

        def send(self, data: str) -> None:
            del data

        def close(self) -> None:
            return

    service = RealtimeService(
        RealtimeConfig(
            api_key="sk-test-123",
            model="gpt-realtime",
            instructions="Test",
            voice="alloy",
        )
    )
    errors: list[str] = []
    statuses: list[str] = []
    service.on_error = errors.append
    service.on_status = statuses.append

    monkeypatch.setattr("kajovochat.services.realtime_service.websocket.WebSocketApp", _FakeWebSocketApp)
    monkeypatch.setattr(
        "kajovochat.services.realtime_service.threading.Thread",
        lambda target, daemon=True: _ImmediateThread(target=target, daemon=daemon),
    )

    service.connect(timeout_s=0.05, max_attempts=2, retry_delay_s=0.0)

    assert service.is_connected is True
    assert errors == []
    assert statuses.count("Realtime: connected") == 1


def test_realtime_service_append_audio_reports_send_failure() -> None:
    service = RealtimeService(
        RealtimeConfig(
            api_key="sk-test-123",
            model="gpt-realtime",
            instructions="Test",
            voice="alloy",
        )
    )

    class _FailingWs:
        def send(self, data: str) -> None:
            del data
            raise RuntimeError("socket closed")

    errors: list[str] = []
    service.on_error = errors.append
    service._ws = _FailingWs()
    service._connected.set()
    service._closed.clear()

    assert service.append_audio_pcm16(b"\x01\x02") is False
    assert errors == ["socket closed"]


def test_realtime_service_raises_last_handshake_error_after_retry_exhaustion(monkeypatch) -> None:
    class _FakeWebSocketApp:
        def __init__(self, url, header, on_open, on_message, on_error, on_close) -> None:
            del url, header, on_open, on_message
            self._on_error = on_error
            self._on_close = on_close

        def run_forever(self, ping_interval=20, ping_timeout=10) -> None:
            del ping_interval, ping_timeout
            self._on_error(self, RuntimeError("Handshake status 504 Gateway Time-out"))
            self._on_close(self, None, "Handshake status 504 Gateway Time-out")

        def send(self, data: str) -> None:
            del data

        def close(self) -> None:
            return

    service = RealtimeService(
        RealtimeConfig(
            api_key="sk-test-123",
            model="gpt-realtime",
            instructions="Test",
            voice="alloy",
        )
    )
    service.on_error = lambda message: (_ for _ in ()).throw(AssertionError(f"Nemá se volat runtime on_error: {message}"))

    monkeypatch.setattr("kajovochat.services.realtime_service.websocket.WebSocketApp", _FakeWebSocketApp)
    monkeypatch.setattr(
        "kajovochat.services.realtime_service.threading.Thread",
        lambda target, daemon=True: _ImmediateThread(target=target, daemon=daemon),
    )

    with pytest.raises(RuntimeError, match="504 Gateway Time-out"):
        service.connect(timeout_s=0.05, max_attempts=2, retry_delay_s=0.0)
