from __future__ import annotations

import base64
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import websocket


REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?model={model}"


@dataclass
class RealtimeConfig:
    api_key: str
    model: str
    instructions: str
    voice: str
    language_hint: str = "auto"  # ISO-639-1 or "auto"
    turn_mode: str = "server_vad"  # "server_vad" or "ptt"
    auto_interrupt: bool = True

    # Hands-free quality / robustness knobs (server-side)
    # `noise_reduction`: "near_field" for headsets, "far_field" for laptop mics.
    noise_reduction: Optional[str] = "far_field"

    # Output speech speed (0.25–1.5 per API docs). This is post-processed after generation.
    output_speed: float = 1.0

    # Server VAD parameters (only used when turn_mode == "server_vad")
    server_vad_silence_ms: Optional[int] = None
    server_vad_prefix_ms: Optional[int] = None
    server_vad_threshold: Optional[float] = None
    server_vad_idle_timeout_ms: Optional[int] = None


class RealtimeService:
    """Minimal Realtime API (WebSocket) client used by the desktop app.

    - Sends PCM16 @ 24kHz mono via input_audio_buffer.append
    - Receives Base64 audio chunks via response.*audio*.delta
    - Optionally receives input audio transcription events
    """

    def __init__(self, cfg: RealtimeConfig) -> None:
        if not cfg.api_key:
            raise ValueError("Missing OpenAI API key")
        self.cfg = cfg

        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()

        self._events: "queue.Queue[dict]" = queue.Queue()
        self._connected = threading.Event()
        self._closed = threading.Event()

        # Callbacks (set by caller)
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_user_transcript: Optional[Callable[[str], None]] = None
        self.on_assistant_text_delta: Optional[Callable[[str], None]] = None
        self.on_assistant_text_done: Optional[Callable[[str], None]] = None
        self.on_assistant_audio_delta: Optional[Callable[[bytes], None]] = None
        self.on_vad_speech_started: Optional[Callable[[], None]] = None
        self.on_vad_speech_stopped: Optional[Callable[[], None]] = None
        self.on_response_done: Optional[Callable[[], None]] = None

        self._assistant_text_buf = ""

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._closed.is_set()

    def connect(self, timeout_s: float = 10.0) -> None:
        if self._ws_thread and self._ws_thread.is_alive():
            return

        url = REALTIME_WS_URL.format(model=self.cfg.model)
        headers = ["Authorization: Bearer " + self.cfg.api_key]

        def _on_open(ws):
            if self.on_status:
                self.on_status("Realtime: connected")
            self._connected.set()
            # Configure session.
            self._send_session_update()

        def _on_message(ws, message: str):
            try:
                evt = json.loads(message)
            except Exception:
                return
            self._events.put(evt)

        def _on_error(ws, error):
            msg = str(error)
            if self.on_error:
                self.on_error(msg)

        def _on_close(ws, status_code, close_msg):
            self._closed.set()
            self._connected.clear()
            if self.on_status:
                self.on_status("Realtime: disconnected")

        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )

        self._closed.clear()
        self._connected.clear()

        def run():
            # Ping helps keep the socket alive on some networks.
            self._ws.run_forever(ping_interval=20, ping_timeout=10)

        self._ws_thread = threading.Thread(target=run, daemon=True)
        self._ws_thread.start()

        if not self._connected.wait(timeout=timeout_s):
            raise RuntimeError("Failed to connect to Realtime API")

    def close(self) -> None:
        self._closed.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def _send(self, event: dict) -> None:
        if not self._ws or not self.is_connected:
            return
        data = json.dumps(event, ensure_ascii=False)
        with self._send_lock:
            try:
                self._ws.send(data)
            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))

    def _send_session_update(self) -> None:
        # Turn taking configuration.
        # IMPORTANT: Per current Realtime API schema, turn_detection is nested
        # under session.audio.input.turn_detection (not at the session root).
        turn_detection: Any
        if self.cfg.turn_mode == "ptt":
            turn_detection = None
        else:
            td: dict[str, Any] = {
                "type": "server_vad",
                "create_response": True,
                "interrupt_response": bool(self.cfg.auto_interrupt),
            }
            if self.cfg.server_vad_silence_ms is not None:
                td["silence_duration_ms"] = int(self.cfg.server_vad_silence_ms)
            if self.cfg.server_vad_prefix_ms is not None:
                td["prefix_padding_ms"] = int(self.cfg.server_vad_prefix_ms)
            if self.cfg.server_vad_threshold is not None:
                td["threshold"] = float(self.cfg.server_vad_threshold)
            if self.cfg.server_vad_idle_timeout_ms is not None:
                td["idle_timeout_ms"] = int(self.cfg.server_vad_idle_timeout_ms)
            turn_detection = td

        transcription: Any
        if self.cfg.language_hint and self.cfg.language_hint != "auto":
            transcription = {"model": "whisper-1", "language": self.cfg.language_hint}
        else:
            transcription = {"model": "whisper-1"}

        evt = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self.cfg.instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "transcription": transcription,
                        "noise_reduction": (
                            {"type": self.cfg.noise_reduction}
                            if self.cfg.noise_reduction
                            else None
                        ),
                        "turn_detection": turn_detection,
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "voice": self.cfg.voice,
                        "speed": float(self.cfg.output_speed or 1.0),
                    },
                },
                # Per API: output_modalities cannot include both audio and text.
                # When set to ["audio"], the server also emits a transcript.
                "output_modalities": ["audio"],
            },
        }
        self._send(evt)

    def update_session(
        self,
        *,
        instructions: Optional[str] = None,
        voice: Optional[str] = None,
        language_hint: Optional[str] = None,
        turn_mode: Optional[str] = None,
    ) -> None:
        """Best-effort session.update wrapper.

        Note: Realtime sessions cannot update `model`, and `voice` can only be
        updated before the first audio output. We still send the update and let
        the server validate.
        """
        if instructions is not None:
            self.cfg.instructions = instructions
        if voice is not None:
            self.cfg.voice = voice
        if language_hint is not None:
            self.cfg.language_hint = language_hint
        if turn_mode is not None:
            self.cfg.turn_mode = turn_mode
        self._send_session_update()

    # ---- Audio input ----

    def append_audio_pcm16(self, pcm16_bytes: bytes) -> None:
        if not pcm16_bytes:
            return
        b64 = base64.b64encode(pcm16_bytes).decode("ascii")
        self._send({"type": "input_audio_buffer.append", "audio": b64})

    def clear_input_audio(self) -> None:
        self._send({"type": "input_audio_buffer.clear"})

    def commit_input_audio(self) -> None:
        self._send({"type": "input_audio_buffer.commit"})

    def request_response(self) -> None:
        self._send({"type": "response.create"})

    def cancel_response(self) -> None:
        self._send({"type": "response.cancel"})

    # ---- Incoming event pump ----

    def pump_events(self, max_events: int = 50) -> None:
        """Drain some pending server events and invoke callbacks."""
        n = 0
        while n < max_events:
            try:
                evt = self._events.get_nowait()
            except queue.Empty:
                return
            n += 1
            self._handle_event(evt)

    def _handle_event(self, evt: dict) -> None:
        etype = evt.get("type")
        if not etype:
            return

        if etype == "error":
            msg = evt.get("error", {}).get("message") or json.dumps(evt, ensure_ascii=False)
            if self.on_error:
                self.on_error(msg)
            return

        if etype == "session.created":
            # informational
            return

        if etype == "input_audio_buffer.speech_started":
            if self.on_vad_speech_started:
                self.on_vad_speech_started()
            return

        if etype == "input_audio_buffer.speech_stopped":
            if self.on_vad_speech_stopped:
                self.on_vad_speech_stopped()
            return

        if etype == "conversation.item.input_audio_transcription.completed":
            t = evt.get("transcript") or ""
            if t and self.on_user_transcript:
                self.on_user_transcript(t)
            return

        # Assistant text deltas (may come as text output or transcript deltas)
        if etype in {"response.output_text.delta", "response.text.delta", "response.output_audio_transcript.delta"}:
            delta = evt.get("delta") or ""
            if delta:
                self._assistant_text_buf += delta
                if self.on_assistant_text_delta:
                    self.on_assistant_text_delta(delta)
            return

        if etype in {"response.output_text.done", "response.text.done", "response.output_audio_transcript.done"}:
            full = self._assistant_text_buf.strip()
            self._assistant_text_buf = ""
            if full and self.on_assistant_text_done:
                self.on_assistant_text_done(full)
            return

        # Assistant audio deltas
        if etype in {"response.output_audio.delta", "response.audio.delta"}:
            b64 = evt.get("delta")
            if not b64:
                return
            try:
                pcm = base64.b64decode(b64)
            except Exception:
                return
            if self.on_assistant_audio_delta:
                self.on_assistant_audio_delta(pcm)
            return

        # Ignore everything else by default.
        if etype == "response.done":
            if self.on_response_done:
                self.on_response_done()
            return


def drain_with_timeout(service: RealtimeService, stop: threading.Event, timeout_s: float = 0.05) -> None:
    """Utility: pump events for a short time."""
    t_end = time.time() + float(timeout_s)
    while time.time() < t_end and not stop.is_set():
        service.pump_events()
        time.sleep(0.005)
