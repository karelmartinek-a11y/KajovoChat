from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from .session_lifecycle import ConversationAudioLifecycle
from .session_manager import AudioSessionManager
from .session_observer import ConversationAudioObserver
from .session_policy import ConversationAudioPolicy
from .session_runtime import ConversationAudioRuntimeController
from .worker_controls import ConversationAudioWorkerControls


@dataclass(frozen=True)
class ConversationAudioBootstrap:
    runtime_controller: ConversationAudioRuntimeController
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
    backend_aware_aec_metrics,
    closed_pose_factory: Callable[[], dict[str, object]],
    sanitize_text_fn,
    format_device_help_fn,
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
    state_idle: str,
    echo_trailing_hold_s: float,
    normalize_aec_mode,
) -> ConversationAudioBootstrap:
    """Sestaví celý audio stack mimo worker, aby `main.py` držel jen UI delegaci."""

    observer = ConversationAudioObserver(owner)
    policy = ConversationAudioPolicy(owner)
    lifecycle = ConversationAudioLifecycle(owner)
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
        start_rt_loop=lambda: owner._start_rt_loop(),
        stop_rt_loop=lambda: owner._stop_rt_loop(),
        preferred_frame_size=owner._preferred_frame_size,
        runtime_resources=owner._runtime_resources,
        input_device_getter=lambda: owner._resolved_input_device,
        output_device_getter=lambda: owner._resolved_output_device,
        guard_profile_supplier=lambda: owner._guard_profile,
        status_sink=owner._append_caption,
        user_transcript_sink=owner._handle_user_transcript,
        assistant_preview_sink=lambda text: owner._set_caption_preview("AI", text),
        assistant_done_sink=owner._handle_assistant_done,
        assistant_audio_sink=owner._handle_assistant_audio,
        speech_started_sink=owner._handle_speech_started,
        speech_stopped_sink=owner._handle_speech_stopped,
        response_done_sink=owner._handle_response_done,
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
    runtime_controller = ConversationAudioRuntimeController(
        owner,
        estimate_voice_likelihood_from_pcm16=estimate_voice_likelihood_from_pcm16,
        closed_pose_factory=closed_pose_factory,
        backend_aware_aec_metrics=backend_aware_aec_metrics,
        state_speaking=state_speaking,
        echo_trailing_hold_s=echo_trailing_hold_s,
    )
    return ConversationAudioBootstrap(
        runtime_controller=runtime_controller,
        lifecycle=lifecycle,
        observer=observer,
        policy=policy,
        controls=controls,
        session_manager=session_manager,
        mic_enabled=session_manager.mic_enabled,
    )
