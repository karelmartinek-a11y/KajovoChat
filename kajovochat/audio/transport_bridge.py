from __future__ import annotations

from typing import Callable, Optional

from ..settings import AppSettings, build_system_prompt
from ..services.realtime_service import RealtimeConfig, RealtimeService


class RealtimeTransportBridge:
    def __init__(
        self,
        *,
        settings: AppSettings,
        guard_profile_supplier: Callable[[], dict[str, float]],
        state_sink: Callable[[str], None],
        caption_sink: Callable[[str], None],
        error_sink: Callable[[str], None],
        status_sink: Callable[[str], None],
        user_transcript_sink: Callable[[str], None],
        assistant_preview_sink: Callable[[str], None],
        assistant_done_sink: Callable[[str], None],
        assistant_audio_sink: Callable[[bytes], None],
        speech_started_sink: Callable[[], None],
        speech_stopped_sink: Callable[[], None],
        response_done_sink: Callable[[], None],
        activity_sink: Callable[[], None],
        model: str,
        voice: str,
        noise_reduction: str,
        tts_speed: float,
        server_vad_silence_ms: int,
        server_vad_prefix_ms: int,
        server_vad_threshold: float,
    ) -> None:
        self.settings = settings
        self._guard_profile_supplier = guard_profile_supplier
        self._state_sink = state_sink
        self._caption_sink = caption_sink
        self._error_sink = error_sink
        self._status_sink = status_sink
        self._user_transcript_sink = user_transcript_sink
        self._assistant_preview_sink = assistant_preview_sink
        self._assistant_done_sink = assistant_done_sink
        self._assistant_audio_sink = assistant_audio_sink
        self._speech_started_sink = speech_started_sink
        self._speech_stopped_sink = speech_stopped_sink
        self._response_done_sink = response_done_sink
        self._activity_sink = activity_sink
        self.model = model
        self.voice = voice
        self.noise_reduction = noise_reduction
        self.tts_speed = tts_speed
        self.server_vad_silence_ms = server_vad_silence_ms
        self.server_vad_prefix_ms = server_vad_prefix_ms
        self.server_vad_threshold = server_vad_threshold
        self.realtime: Optional[RealtimeService] = None
        self.turn_mode: str = "server_vad"

    def ensure_connected(self, turn_mode: str, reconnect_attempts: int = 0) -> RealtimeService:
        if not self.settings.openai_api_key:
            raise ValueError("Chybí API key")
        self.turn_mode = turn_mode
        resolved_lang = self.settings.fixed_answer_language if self.settings.fixed_answer_language in {"cs", "en", "de", "sk", "fr"} else "cs"
        cfg = RealtimeConfig(
            api_key=self.settings.openai_api_key,
            model=self.model,
            instructions=build_system_prompt(self.settings, resolved_lang),
            voice=self.voice,
            language_hint="auto",
            turn_mode=turn_mode,
            auto_interrupt=True,
            noise_reduction=self.noise_reduction,
            output_speed=self.tts_speed,
            server_vad_silence_ms=self.server_vad_silence_ms,
            server_vad_prefix_ms=self.server_vad_prefix_ms,
            server_vad_threshold=float(self._guard_profile_supplier()["server_vad_threshold"]),
        )
        if self.realtime is None or not self.realtime.is_connected:
            self.realtime = RealtimeService(cfg)
            self._wire_callbacks(self.realtime)
            self._state_sink("connecting" if reconnect_attempts == 0 else "reconnecting")
            self.realtime.connect()
            self._activity_sink()
            return self.realtime
        self.realtime.cfg.noise_reduction = self.noise_reduction
        self.realtime.cfg.output_speed = self.tts_speed
        self.realtime.cfg.server_vad_silence_ms = self.server_vad_silence_ms
        self.realtime.cfg.server_vad_prefix_ms = self.server_vad_prefix_ms
        self.realtime.cfg.server_vad_threshold = float(self._guard_profile_supplier()["server_vad_threshold"])
        self.realtime.update_session(
            instructions=build_system_prompt(self.settings, resolved_lang),
            voice=self.voice,
            language_hint="auto",
            turn_mode=turn_mode,
        )
        self._activity_sink()
        return self.realtime

    def close(self) -> None:
        try:
            if self.realtime:
                self.realtime.close()
        finally:
            self.realtime = None

    def _wire_callbacks(self, rt: RealtimeService) -> None:
        def _status(message: str) -> None:
            self._activity_sink()
            self._status_sink(message)

        def _error(message: str) -> None:
            self._error_sink(message)

        def _user_transcript(text: str) -> None:
            self._activity_sink()
            self._user_transcript_sink(text)

        def _assistant_preview(text: str) -> None:
            self._activity_sink()
            self._assistant_preview_sink(text)

        def _assistant_done(text: str) -> None:
            self._activity_sink()
            self._assistant_done_sink(text)

        def _assistant_audio(pcm: bytes) -> None:
            self._activity_sink()
            self._assistant_audio_sink(pcm)

        def _speech_started() -> None:
            self._activity_sink()
            self._speech_started_sink()

        def _speech_stopped() -> None:
            self._activity_sink()
            self._speech_stopped_sink()

        def _response_done() -> None:
            self._activity_sink()
            self._response_done_sink()

        rt.on_status = _status
        rt.on_error = _error
        rt.on_user_transcript = _user_transcript
        rt.on_assistant_text_delta = _assistant_preview
        rt.on_assistant_text_done = _assistant_done
        rt.on_assistant_audio_delta = _assistant_audio
        rt.on_vad_speech_started = _speech_started
        rt.on_vad_speech_stopped = _speech_stopped
        rt.on_response_done = _response_done
