from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .session_callbacks import ConversationAudioCallbacks
from .session_lifecycle import ConversationAudioLifecycle
from .session_manager import AudioSessionManager
from .session_observer import ConversationAudioObserver
from .session_policy import ConversationAudioPolicy
from .session_runtime import ConversationAudioRuntimeController
from .worker_controls import ConversationAudioWorkerControls


@dataclass(frozen=True)
class ConversationAudioBootstrap:
    runtime_controller: ConversationAudioRuntimeController
    callbacks: ConversationAudioCallbacks
    lifecycle: ConversationAudioLifecycle
    observer: ConversationAudioObserver
    policy: ConversationAudioPolicy
    controls: ConversationAudioWorkerControls
    session_manager: AudioSessionManager
    mic_enabled: threading.Event


def build_conversation_audio_stack(
    owner: Any,
    *,
    estimate_voice_likelihood_from_pcm16,
    evaluate_capture_gate,
    backend_aware_aec_metrics,
    closed_pose_factory: Callable[[], dict[str, object]],
    build_system_prompt_fn,
    sanitize_text_fn,
    format_device_help_fn,
    realtime_factory,
    realtime_config_cls,
    allowed_langs,
    realtime_model: str,
    tts_voice: str,
    noise_reduction: str,
    tts_speed: float,
    server_vad_silence_ms: int,
    server_vad_prefix_ms: int,
    server_vad_threshold: float,
    state_speaking: str,
    state_error: str,
    state_thinking: str,
    state_listening: str,
    state_transcribing: str,
    state_idle: str,
    state_connecting: str,
    state_reconnecting: str,
    echo_trailing_hold_s: float,
    default_guard_profile: dict[str, float],
    normalize_aec_mode,
) -> ConversationAudioBootstrap:
    """Sestaví celý audio stack mimo worker, aby `main.py` nebootstrapoval audio ručně."""

    runtime_controller = ConversationAudioRuntimeController(
        owner,
        estimate_voice_likelihood_from_pcm16=estimate_voice_likelihood_from_pcm16,
        closed_pose_factory=closed_pose_factory,
        should_drop_mic_chunk=lambda **kwargs: evaluate_capture_gate(
            default_profile=default_guard_profile,
            **kwargs,
        ),
        backend_aware_aec_metrics=backend_aware_aec_metrics,
        state_speaking=state_speaking,
        echo_trailing_hold_s=echo_trailing_hold_s,
    )
    callbacks = ConversationAudioCallbacks(
        owner,
        realtime_factory=realtime_factory,
        realtime_config_cls=realtime_config_cls,
        build_system_prompt_fn=build_system_prompt_fn,
        sanitize_text_fn=sanitize_text_fn,
        format_device_help_fn=format_device_help_fn,
        allowed_langs=allowed_langs,
        realtime_model=realtime_model,
        tts_voice=tts_voice,
        noise_reduction=noise_reduction,
        tts_speed=tts_speed,
        server_vad_silence_ms=server_vad_silence_ms,
        server_vad_prefix_ms=server_vad_prefix_ms,
        state_speaking=state_speaking,
        state_error=state_error,
        state_thinking=state_thinking,
        state_listening=state_listening,
        state_transcribing=state_transcribing,
        state_idle=state_idle,
        state_connecting=state_connecting,
        state_reconnecting=state_reconnecting,
    )
    lifecycle = ConversationAudioLifecycle(owner)
    observer = ConversationAudioObserver(owner)
    policy = ConversationAudioPolicy(owner)
    controls = ConversationAudioWorkerControls(
        owner,
        sanitize_text_fn=sanitize_text_fn,
        state_idle=state_idle,
        state_error=state_error,
        closed_pose_factory=closed_pose_factory,
    )
    session_manager = AudioSessionManager(
        settings=owner.settings,
        mode_supplier=lambda: owner._mode,
        mode_setter=lambda value: setattr(owner, "_mode", value),
        state_sink=owner._set_state,
        caption_sink=owner._append_caption,
        error_sink=lambda msg: owner.error.emit(sanitize_text_fn(str(msg))),
        resolve_devices=owner._resolve_audio_devices,
        ensure_player=owner._ensure_player,
        start_session_if_needed=owner._start_session_if_needed,
        stop_realtime_session=owner._stop_realtime_session,
        start_rt_loop=lambda: owner._start_rt_loop(),
        stop_rt_loop=lambda: owner._stop_rt_loop(),
        preferred_frame_size=owner._preferred_frame_size,
        runtime_resources=owner._runtime_resources,
        input_device_getter=lambda: owner._resolved_input_device,
        output_device_getter=lambda: owner._resolved_output_device,
        guard_profile_supplier=lambda: owner._guard_profile,
        status_sink=lambda msg: (setattr(owner._transport_runtime, "last_server_activity_at", time.monotonic()), owner._append_caption(msg)),
        user_transcript_sink=lambda text: owner._handle_user_transcript(text),
        assistant_preview_sink=lambda text: owner._set_caption_preview("AI", text),
        assistant_done_sink=lambda text: owner._handle_assistant_done(text),
        assistant_audio_sink=lambda pcm: owner._handle_assistant_audio(pcm),
        speech_started_sink=lambda: owner._handle_speech_started(),
        speech_stopped_sink=lambda: owner._handle_speech_stopped(),
        response_done_sink=lambda: owner._handle_response_done(),
        log_sink=lambda record_type, extra: owner._log_event(record_type, **(extra if isinstance(extra, dict) else {"value": extra})),
        aec_mode_setter=lambda value: setattr(owner, "_aec_mode", normalize_aec_mode(value)),
        device_fingerprint_supplier=lambda: owner._current_device_fingerprint(),
        audio_mode_supplier=lambda: owner._audio_mode,
        model=realtime_model,
        voice=tts_voice,
        noise_reduction=noise_reduction,
        tts_speed=tts_speed,
        server_vad_silence_ms=server_vad_silence_ms,
        server_vad_prefix_ms=server_vad_prefix_ms,
        server_vad_threshold=server_vad_threshold,
    )
    return ConversationAudioBootstrap(
        runtime_controller=runtime_controller,
        callbacks=callbacks,
        lifecycle=lifecycle,
        observer=observer,
        policy=policy,
        controls=controls,
        session_manager=session_manager,
        mic_enabled=session_manager.mic_enabled,
    )
