from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class ConversationAudioTransportState:
    reconnect_attempts: int = 0
    next_reconnect_at: float = 0.0
    response_started_at: float | None = None
    response_first_audio_at: float | None = None
    speech_stopped_at: float | None = None
    last_server_activity_at: float = 0.0
    last_backlog_log_at: float = 0.0
    last_player_progress_at: float = 0.0
    last_player_buffer_bytes: int = 0
    last_echo_stat_log_at: float = 0.0

    def reset(self) -> None:
        self.reconnect_attempts = 0
        self.next_reconnect_at = 0.0
        self.response_started_at = None
        self.response_first_audio_at = None
        self.speech_stopped_at = None
        self.last_server_activity_at = time.monotonic()
        self.last_backlog_log_at = 0.0
        self.last_player_progress_at = self.last_server_activity_at
        self.last_player_buffer_bytes = 0
        self.last_echo_stat_log_at = 0.0


class ConversationAudioCallbacks:
    """Audio callbacky a legacy realtime helpery vytažené z GUI workeru."""

    def __init__(
        self,
        owner: object,
        *,
        realtime_factory,
        realtime_config_cls,
        build_system_prompt_fn: Callable,
        sanitize_text_fn: Callable[[str], str],
        format_device_help_fn: Callable[[], str],
        allowed_langs: set[str],
        realtime_model: str,
        tts_voice: str,
        noise_reduction: str,
        tts_speed: float,
        server_vad_silence_ms: int,
        server_vad_prefix_ms: int,
        state_speaking: str,
        state_error: str,
        state_thinking: str,
        state_listening: str,
        state_transcribing: str,
        state_idle: str,
        state_connecting: str,
        state_reconnecting: str,
    ) -> None:
        self.owner = owner
        self.runtime_state = ConversationAudioTransportState(last_server_activity_at=time.monotonic())
        self._realtime_factory = realtime_factory
        self._realtime_config_cls = realtime_config_cls
        self._build_system_prompt_fn = build_system_prompt_fn
        self._sanitize_text_fn = sanitize_text_fn
        self._format_device_help_fn = format_device_help_fn
        self._allowed_langs = allowed_langs
        self._realtime_model = realtime_model
        self._tts_voice = tts_voice
        self._noise_reduction = noise_reduction
        self._tts_speed = tts_speed
        self._server_vad_silence_ms = server_vad_silence_ms
        self._server_vad_prefix_ms = server_vad_prefix_ms
        self._state_speaking = state_speaking
        self._state_error = state_error
        self._state_thinking = state_thinking
        self._state_listening = state_listening
        self._state_transcribing = state_transcribing
        self._state_idle = state_idle
        self._state_connecting = state_connecting
        self._state_reconnecting = state_reconnecting

    def ensure_realtime(self, turn_mode: str):
        owner = self.owner
        runtime = self.runtime_state
        if not owner.settings.openai_api_key:
            raise ValueError("Chybí API key")
        owner._rt_turn_mode = turn_mode

        owner._resolved_lang = owner.settings.fixed_answer_language if owner.settings.fixed_answer_language in self._allowed_langs else "cs"
        instructions = self._build_system_prompt_fn(owner.settings, owner._resolved_lang)

        cfg = self._realtime_config_cls(
            api_key=owner.settings.openai_api_key,
            model=self._realtime_model,
            instructions=instructions,
            voice=self._tts_voice,
            language_hint="auto",
            turn_mode=turn_mode,
            auto_interrupt=True,
            noise_reduction=self._noise_reduction,
            output_speed=self._tts_speed,
            server_vad_silence_ms=self._server_vad_silence_ms,
            server_vad_prefix_ms=self._server_vad_prefix_ms,
            server_vad_threshold=float(owner._guard_profile["server_vad_threshold"]),
        )

        if owner._rt is None or not owner._rt.is_connected:
            owner._rt = self._realtime_factory(cfg)
            self.wire_realtime_callbacks(owner._rt)
            owner._set_state(self._state_connecting if runtime.reconnect_attempts == 0 else self._state_reconnecting)
            owner._rt.connect()
            runtime.reconnect_attempts = 0
            runtime.next_reconnect_at = 0.0
            runtime.last_server_activity_at = time.monotonic()
            return owner._rt

        owner._rt.cfg.noise_reduction = self._noise_reduction
        owner._rt.cfg.output_speed = self._tts_speed
        owner._rt.cfg.server_vad_silence_ms = self._server_vad_silence_ms
        owner._rt.cfg.server_vad_prefix_ms = self._server_vad_prefix_ms
        owner._rt.cfg.server_vad_threshold = float(owner._guard_profile["server_vad_threshold"])
        owner._rt.update_session(
            instructions=instructions,
            voice=self._tts_voice,
            language_hint="auto",
            turn_mode=turn_mode,
        )
        return owner._rt

    def wire_realtime_callbacks(self, rt) -> None:
        owner = self.owner

        def _status(msg: str) -> None:
            self.runtime_state.last_server_activity_at = time.monotonic()
            owner._append_caption(msg)

        rt.on_status = _status

        def _is_recoverable_realtime_error(msg: str) -> bool:
            text = (msg or "").lower()
            markers = ("timed out", "timeout", "connection", "socket", "reset", "closed", "disconnect", "broken pipe")
            return any(marker in text for marker in markers)

        def _err(msg: str) -> None:
            safe_msg = self._sanitize_text_fn(msg)
            owner._log_event("error", message=safe_msg)
            if owner._mode != "idle" and _is_recoverable_realtime_error(msg):
                self.schedule_reconnect(safe_msg)
                return
            owner._stop_realtime_session()
            owner._set_state(self._state_error)
            owner.error.emit(safe_msg)

        rt.on_error = _err
        rt.on_user_transcript = self.handle_user_transcript
        rt.on_assistant_text_delta = lambda d: owner._set_caption_preview("AI", d)
        rt.on_assistant_text_done = self.handle_assistant_done
        rt.on_assistant_audio_delta = self.handle_assistant_audio
        rt.on_vad_speech_started = self.handle_speech_started
        rt.on_vad_speech_stopped = self.handle_speech_stopped
        rt.on_response_done = self.handle_response_done

    def handle_user_transcript(self, text: str) -> None:
        owner = self.owner
        runtime = self.runtime_state
        runtime.last_server_activity_at = time.monotonic()
        owner._append_caption(f"Ty: {text}")
        owner._log_conversation_text("user", text)
        owner._awaiting_transcript = False
        if owner._ui_state not in {self._state_speaking, self._state_error}:
            owner._set_state(self._state_thinking)
        runtime.response_started_at = time.monotonic()

    def handle_assistant_done(self, text: str) -> None:
        owner = self.owner
        self.runtime_state.last_server_activity_at = time.monotonic()
        owner._append_caption(f"AI: {text}")
        owner._log_conversation_text("assistant", text)

    def handle_assistant_audio(self, pcm: bytes) -> None:
        owner = self.owner
        runtime = self.runtime_state
        runtime.last_server_activity_at = time.monotonic()
        owner._session_manager.note_assistant_rendering()
        if runtime.response_first_audio_at is None:
            runtime.response_first_audio_at = time.monotonic()
            latency_ms = None
            if runtime.response_started_at is not None:
                latency_ms = int((runtime.response_first_audio_at - runtime.response_started_at) * 1000)
            owner._log_event("assistant_audio_first_delta", latency_ms=latency_ms, bytes=len(pcm))
        try:
            owner._session_manager.enqueue_render_pcm(pcm)
        except Exception as exc:
            owner._log_event("error", message=str(exc))
            owner._stop_realtime_session()
            owner._set_state(self._state_error)
            owner.error.emit(self._sanitize_text_fn(str(exc)) + "\n\n" + self._format_device_help_fn())

    def handle_speech_started(self) -> None:
        owner = self.owner
        runtime = self.runtime_state
        try:
            owner._session_manager.stop_render_output()
        except Exception:
            pass
        owner._awaiting_transcript = False
        owner._session_manager.note_speech_started(assistant_rendering=bool(runtime.response_first_audio_at is not None))
        runtime.response_started_at = None
        runtime.response_first_audio_at = None
        runtime.speech_stopped_at = None
        runtime.last_server_activity_at = time.monotonic()
        owner._log_event("speech_started")

    def handle_speech_stopped(self) -> None:
        owner = self.owner
        owner._awaiting_transcript = True
        owner._session_manager.note_barge_in_transition()
        self.runtime_state.speech_stopped_at = time.monotonic()
        self.runtime_state.last_server_activity_at = time.monotonic()
        owner._log_event("speech_stopped")

    def handle_response_done(self) -> None:
        owner = self.owner
        runtime = self.runtime_state
        total_latency_ms = None
        if runtime.speech_stopped_at is not None:
            total_latency_ms = int((time.monotonic() - runtime.speech_stopped_at) * 1000)
        owner._log_event("response_done", total_latency_ms=total_latency_ms)
        owner._session_manager.note_response_done()
        if owner._mode != "handsfree" and owner._session_manager.session_state.value == "ready":
            owner._set_state(self._state_idle)
        owner._awaiting_transcript = False
        runtime.response_started_at = None
        runtime.response_first_audio_at = None
        runtime.speech_stopped_at = None
        runtime.last_server_activity_at = time.monotonic()

    def schedule_reconnect(self, reason: str) -> None:
        owner = self.owner
        runtime = self.runtime_state
        runtime.reconnect_attempts += 1
        delay = min(8.0, 0.8 * (2 ** max(0, runtime.reconnect_attempts - 1)))
        runtime.next_reconnect_at = time.monotonic() + delay
        owner._append_caption(f"Realtime: plánuji reconnect za {delay:.1f} s")
        owner._log_event("reconnect_scheduled", reason=reason, attempt=runtime.reconnect_attempts, delay_s=delay)
        try:
            if owner._rt:
                owner._rt.close()
        except Exception:
            pass
        owner._rt = None
        owner._set_state(self._state_reconnecting)

    def attempt_reconnect_if_needed(self) -> None:
        owner = self.owner
        owner._session_manager.tick()
        owner._rt = owner._session_manager.transport.realtime

    def check_runtime_health(self) -> None:
        owner = self.owner
        runtime = self.runtime_state
        now = time.monotonic()
        pending_snapshot = owner._session_manager.runtime_pending_snapshot()
        pending_events = int(pending_snapshot["pending_events"])
        duplex = owner._active_duplex()
        if duplex is not None:
            duplex_state = duplex.get_runtime_state()
            pending_mic = int(duplex_state.get("pending_chunk_count", 0) or 0)
            pending_player_bytes = int(duplex_state.get("buffered_bytes", 0) or 0)
        else:
            pending_mic = int(pending_snapshot["pending_mic"])
            pending_player_bytes = int(pending_snapshot["pending_player_bytes"])

        if (
            now - runtime.last_backlog_log_at >= 5.0
            and (pending_events > 0 or pending_mic > 0 or pending_player_bytes > 0)
        ):
            owner._log_event(
                "backlog",
                rt_events=pending_events,
                mic_chunks=pending_mic,
                player_bytes=pending_player_bytes,
            )
            runtime.last_backlog_log_at = now

        if (
            now - runtime.last_echo_stat_log_at >= 5.0
            and (
                owner._session_manager.voice_gate_runtime.echo_drop_count != owner._session_manager.voice_gate_runtime.last_echo_drop_reported
                or owner._session_manager.voice_gate_runtime.barge_in_chunk_count != owner._session_manager.voice_gate_runtime.last_barge_in_reported
            )
        ):
            gate_runtime = owner._session_manager.voice_gate_runtime
            owner._log_event(
                "echo_guard",
                dropped_echo_chunks=gate_runtime.echo_drop_count,
                barge_in_chunks=gate_runtime.barge_in_chunk_count,
            )
            runtime.last_echo_stat_log_at = now
            gate_runtime.last_echo_drop_reported = gate_runtime.echo_drop_count
            gate_runtime.last_barge_in_reported = gate_runtime.barge_in_chunk_count

        if duplex is not None or owner._player:
            if pending_player_bytes != runtime.last_player_buffer_bytes:
                runtime.last_player_progress_at = now
                runtime.last_player_buffer_bytes = pending_player_bytes
            elif pending_player_bytes > 0 and now - runtime.last_player_progress_at > 8.0:
                owner._log_event("watchdog", message="audio playback stagnuje", buffered_bytes=pending_player_bytes)
                try:
                    if duplex is not None:
                        duplex.stop()
                    elif owner._player:
                        owner._player.stop()
                except Exception:
                    pass
                runtime.last_player_progress_at = now
                runtime.last_player_buffer_bytes = 0

        if (
            owner._mode != "idle"
            and owner._rt is not None
            and owner._rt.is_connected
            and owner._ui_state in {self._state_connecting, self._state_reconnecting, self._state_transcribing, self._state_thinking}
            and now - self.runtime_state.last_server_activity_at > 25.0
        ):
            owner._log_event("watchdog", message="realtime bez aktivity", state=owner._ui_state)
            self.schedule_reconnect("watchdog: realtime bez aktivity")
