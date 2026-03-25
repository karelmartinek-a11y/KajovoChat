from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .aec_engine import AecEngine, BackendSelectionDecision
from .device_graph import DuplexDeviceGraph
from .recovery import FailureReason, RecoverySupervisor
from .runtime_resources import AudioRuntimeResources
from .session_state import (
    SessionPresentationState,
    SessionState,
    session_state_to_ui_state,
    validate_session_transition,
)
from .telemetry import AudioTelemetry
from .transport_bridge import RealtimeTransportBridge
from .voice_gate import (
    ReferenceSelectionDecision,
    VoiceGate,
    VoiceGateRuntimeState,
    VoiceGateSnapshot,
)
from ..settings import AppSettings, normalize_audio_aec_mode
from .devices import format_device_help
from .io import DuplexAudioSession
from .windows_system_aec import windows_system_aec_healthcheck

try:
    from aec_audio_processing import AudioProcessor as _WebRTCAudioProcessor
except Exception:
    _WebRTCAudioProcessor = None


class AudioSessionManager:
    def __init__(
        self,
        *,
        settings: AppSettings,
        mode_supplier: Callable[[], str],
        mode_setter: Callable[[str], None],
        state_sink: Callable[[str], None],
        caption_sink: Callable[[str], None],
        error_sink: Callable[[str], None],
        resolve_devices: Callable[[], None],
        ensure_player: Callable[[], None],
        start_session_if_needed: Callable[[], None],
        start_rt_loop: Callable[[], None],
        stop_rt_loop: Callable[[], None],
        preferred_frame_size: Callable[[], int],
        runtime_resources: AudioRuntimeResources,
        input_device_getter: Callable[[], Optional[int]],
        output_device_getter: Callable[[], Optional[int]],
        guard_profile_supplier: Callable[[], dict[str, float]],
        status_sink: Callable[[str], None],
        user_transcript_sink: Callable[[str], None],
        assistant_preview_sink: Callable[[str], None],
        assistant_done_sink: Callable[[str], None],
        assistant_audio_sink: Callable[[bytes], None],
        speech_started_sink: Callable[[], None],
        speech_stopped_sink: Callable[[], None],
        response_done_sink: Callable[[], None],
        log_sink: Callable[[str, object], None],
        aec_mode_setter: Callable[[str], None],
        device_fingerprint_supplier: Callable[[], str],
        audio_mode_supplier: Callable[[], str],
        model: str,
        voice: str,
        noise_reduction: str,
        tts_speed: float,
        server_vad_silence_ms: int,
        server_vad_prefix_ms: int,
        server_vad_threshold: float,
    ) -> None:
        self.settings = settings
        self._mode_supplier = mode_supplier
        self._mode_setter = mode_setter
        self._state_sink = state_sink
        self._caption_sink = caption_sink
        self._error_sink = error_sink
        self._resolve_devices = resolve_devices
        self._ensure_player = ensure_player
        self._start_session_if_needed = start_session_if_needed
        self._start_rt_loop = start_rt_loop
        self._stop_rt_loop = stop_rt_loop
        self._preferred_frame_size = preferred_frame_size
        self._runtime_resources = runtime_resources
        self._input_device_getter = input_device_getter
        self._output_device_getter = output_device_getter
        self._log_sink = log_sink
        self._aec_mode_setter = aec_mode_setter
        self._device_fingerprint_supplier = device_fingerprint_supplier
        self._audio_mode_supplier = audio_mode_supplier
        self._voice_gate = VoiceGate()
        self.telemetry = AudioTelemetry()
        self.session_state = SessionState.IDLE
        self.presentation_state: SessionPresentationState | None = None
        self.aec_engine = AecEngine(settings.audio_aec_mode)
        self.transport = RealtimeTransportBridge(
            settings=settings,
            guard_profile_supplier=guard_profile_supplier,
            state_sink=state_sink,
            caption_sink=caption_sink,
            error_sink=self.handle_transport_error,
            status_sink=status_sink,
            user_transcript_sink=user_transcript_sink,
            assistant_preview_sink=assistant_preview_sink,
            assistant_done_sink=assistant_done_sink,
            assistant_audio_sink=assistant_audio_sink,
            speech_started_sink=speech_started_sink,
            speech_stopped_sink=speech_stopped_sink,
            response_done_sink=response_done_sink,
            activity_sink=self.telemetry.note_server_activity,
            model=model,
            voice=voice,
            noise_reduction=noise_reduction,
            tts_speed=tts_speed,
            server_vad_silence_ms=server_vad_silence_ms,
            server_vad_prefix_ms=server_vad_prefix_ms,
            server_vad_threshold=server_vad_threshold,
        )
        self.recovery = RecoverySupervisor(
            telemetry=self.telemetry,
            transport=self.transport,
            mode_supplier=mode_supplier,
            state_sink=state_sink,
            caption_sink=caption_sink,
            log_sink=log_sink,
            enter_recovering=self.enter_recovering,
            stop_session=self._handle_recovery_exhausted,
            fail_session=lambda message, reason: self.fail(message, reason=reason),
            error_sink=error_sink,
            selected_backend_supplier=lambda: self.telemetry.selected_backend,
            fallback_handler=self._attempt_backend_fallback,
            restore_session_state=self._restore_session_state,
            stop_playback=self.stop_render_output,
        )
        self.device_graph = DuplexDeviceGraph()

    @property
    def mic_enabled(self) -> threading.Event:
        return self._voice_gate.mic_enabled

    @property
    def voice_gate_runtime(self) -> VoiceGateRuntimeState:
        return self._voice_gate.runtime

    def voice_gate_snapshot(self, *, now_monotonic: float | None = None) -> VoiceGateSnapshot:
        return self._voice_gate.snapshot(now_monotonic=now_monotonic, mode=self._mode_supplier())

    @property
    def awaiting_transcript(self) -> bool:
        return self._voice_gate.awaiting_transcript

    @awaiting_transcript.setter
    def awaiting_transcript(self, value: bool) -> None:
        self._voice_gate.awaiting_transcript = bool(value)

    def tick(self) -> None:
        self.recovery.tick()

    def reset_runtime_tracking(self) -> None:
        now = time.monotonic()
        self.telemetry.reset_runtime_watchdog(now_monotonic=now)
        self.telemetry.reset_turn_timing()
        self._last_echo_stat_log_at = 0.0

    def _handle_recovery_exhausted(self) -> None:
        self.shutdown_runtime_resources()
        self._mode_setter("idle")
        self.awaiting_transcript = False
        self.reset_runtime_tracking()
        self.reset_voice_gate_runtime()
        self.telemetry.mark_session_stopped()
        self._set_session_state(SessionState.FAILED, reason=FailureReason.RECOVERY_EXHAUSTED.value)

    def shutdown_runtime_resources(self) -> None:
        self._voice_gate.close()
        try:
            self._stop_rt_loop()
        except Exception:
            pass
        duplex = self._runtime_resources.duplex
        try:
            if duplex is not None:
                duplex.stop()
            elif self._runtime_resources.mic is not None:
                self._runtime_resources.mic.stop()
        except Exception:
            pass
        try:
            if duplex is None and self._runtime_resources.player is not None:
                self._runtime_resources.player.stop()
        except Exception:
            pass
        self.device_graph.stop_io()
        self._runtime_resources.duplex = None
        self._runtime_resources.player = None
        self._runtime_resources.mic = None
        rt = self._runtime_resources.rt
        if rt is not None and rt is not self.transport.realtime:
            try:
                rt.close()
            except Exception:
                pass
        try:
            self.transport.close()
        except Exception:
            pass
        self._runtime_resources.rt = None

    def reset_voice_gate_runtime(self) -> None:
        self._voice_gate.reset_runtime()

    def note_playback_activity(self, *, is_playing_out: bool, now_monotonic: float, trailing_hold_s: float) -> None:
        self._voice_gate.update_playback_reference_state(
            is_playing_out=is_playing_out,
            now_monotonic=now_monotonic,
            trailing_hold_s=trailing_hold_s,
        )

    def is_guard_active(self, *, is_playing_out: bool) -> bool:
        return self._voice_gate.is_guard_active(
            mode=self._mode_supplier(),
            is_playing_out=is_playing_out,
        )

    def cache_reference(self, reference_pcm16: bytes, *, now_monotonic: float) -> None:
        self._voice_gate.note_reference_cache(reference_pcm16, now_monotonic=now_monotonic)

    def select_reference_source(
        self,
        *,
        aec_requires_reference: bool,
        now_monotonic: float,
        reference_needed: int,
        live_reference_pcm16: bytes,
        available_samples: int,
        played_samples: int,
        callback_age_ms: int,
    ) -> ReferenceSelectionDecision:
        return self._voice_gate.select_reference_source(
            aec_requires_reference=aec_requires_reference,
            now_monotonic=now_monotonic,
            reference_needed=reference_needed,
            live_reference_pcm16=live_reference_pcm16,
            available_samples=available_samples,
            played_samples=played_samples,
            callback_age_ms=callback_age_ms,
        )

    def note_tts_rendering(self, *, rendering_active: bool, now_monotonic: float) -> None:
        self._voice_gate.note_tts_window(
            rendering_active=rendering_active,
            now_monotonic=now_monotonic,
        )

    def should_log_problem_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return self._voice_gate.should_log_problem_diag(
            now_monotonic=now_monotonic,
            min_interval_s=min_interval_s,
        )

    def should_log_success_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return self._voice_gate.should_log_success_diag(
            now_monotonic=now_monotonic,
            min_interval_s=min_interval_s,
        )

    def note_diag_logged(self, *, success: bool, now_monotonic: float) -> None:
        self._voice_gate.note_diag_logged(
            success=success,
            now_monotonic=now_monotonic,
        )

    def current_mode_contract(self):
        return self.aec_engine.product_mode_contract_for(
            selected_backend=self.telemetry.selected_backend,
            requested_backend=self.telemetry.requested_backend,
            audio_mode=self.device_graph.audio_mode or self._audio_mode_supplier() or "notebook_builtin",
            degradation_cause=self.telemetry.degradation_cause,
        )

    def aec_requires_reference(self) -> bool:
        return bool(self.current_mode_contract().requires_reference)

    def evaluate_capture_gate(self, **kwargs):
        kwargs.setdefault("capture_gate_policy", self.telemetry.capture_gate_policy or self.current_mode_contract().capture_gate_policy)
        return self._voice_gate.evaluate_capture_gate(**kwargs)

    def enqueue_render_pcm(self, pcm: bytes) -> None:
        self._ensure_player()
        player = self._runtime_resources.player
        if player is not None:
            player.enqueue_pcm16(pcm)

    def stop_render_output(self) -> None:
        player = self._runtime_resources.player
        if player is not None:
            player.stop()

    def runtime_pending_snapshot(self) -> dict[str, int]:
        rt = self._runtime_resources.rt
        mic = self._runtime_resources.mic
        player = self._runtime_resources.player
        return {
            "pending_events": int(rt.pending_event_count if rt else 0),
            "pending_mic": int(mic.pending_chunk_count if mic else 0),
            "pending_player_bytes": int(player.buffered_bytes if player else 0),
        }

    def note_reference_health(self, *, ready: bool, available_samples: int, callback_age_ms: int) -> None:
        self.recovery.observe_reference_health(
            ready=ready,
            available_samples=available_samples,
            callback_age_ms=callback_age_ms,
            session_state=self.session_state.value,
        )

    def note_aec_observation(
        self,
        *,
        backend: str,
        reference_miss: bool,
        aec_quality: float,
        improvement_ratio: float,
        delay_samples: int,
        calibration_latency: int,
        similarity: float,
        webrtc_success: bool,
    ) -> None:
        self.recovery.observe_aec_health(
            backend=backend,
            reference_miss=reference_miss,
            aec_quality=aec_quality,
            improvement_ratio=improvement_ratio,
            delay_samples=delay_samples,
            calibration_latency=calibration_latency,
            similarity=similarity,
            webrtc_success=webrtc_success,
            session_state=self.session_state.value,
        )

    def start_handsfree(self) -> None:
        self._start_session(mode="handsfree", turn_mode="server_vad", caption="Hands-free: Realtime aktivní (server VAD).", reset_transport_input=False)

    def ptt_pressed(self) -> None:
        if self._mode_supplier() == "handsfree":
            return
        self._start_session(mode="ptt", turn_mode="ptt", caption="PTT: poslouchám…", reset_transport_input=True)

    def ptt_released(self) -> None:
        if self._mode_supplier() != "ptt":
            return
        rt = self._runtime_resources.rt or self.transport.realtime
        if rt is None:
            return
        self._voice_gate.close()
        try:
            mic = self._runtime_resources.mic
            if mic:
                mic.stop()
        except Exception:
            pass
        self._runtime_resources.mic = None
        self.device_graph.mic = None
        self._voice_gate.awaiting_transcript = True
        rt.commit_input_audio()
        rt.request_response()
        self._set_presentation_state(SessionPresentationState.TRANSCRIBING, reason="ptt_commit")
        self._log_sink("audio_push_to_realtime", self._build_log_payload(turn_mode="ptt", phase="commit"))
        self._caption_sink("PTT: čekám na odpověď…")

    def request_stop(self) -> None:
        self._set_session_state(SessionState.STOPPING, reason=FailureReason.USER_STOP.value)
        self.awaiting_transcript = False
        self.shutdown_runtime_resources()
        self._mode_setter("idle")
        self.reset_runtime_tracking()
        self.reset_voice_gate_runtime()
        self.telemetry.clear_reconnect()
        self.telemetry.mark_session_stopped()
        self._set_session_state(SessionState.IDLE, reason=FailureReason.USER_STOP.value)

    def _start_session(self, *, mode: str, turn_mode: str, caption: str, reset_transport_input: bool) -> None:
        self._mode_setter(mode)
        self._set_session_state(SessionState.STARTING, reason=f"{mode}_requested")
        self._resolve_devices()
        self._start_session_if_needed()
        self.reset_runtime_tracking()
        self._require_devices_ready()
        self.device_graph.input_device = self._input_device_getter()
        self.device_graph.output_device = self._output_device_getter()
        self.device_graph.audio_mode = self._audio_mode_supplier() or "notebook_builtin"
        existing_duplex = self._runtime_resources.duplex
        if existing_duplex is not None:
            try:
                existing_duplex.stop()
            except Exception:
                pass
        duplex = DuplexAudioSession(
            samplerate=24000,
            input_device=self.device_graph.input_device,
            output_device=self.device_graph.output_device,
            blocksize=self._preferred_frame_size(),
        )
        self._runtime_resources.duplex = duplex
        self._runtime_resources.player = duplex.player
        self._runtime_resources.mic = duplex.mic
        self.device_graph.duplex = duplex
        self.device_graph.player = duplex.player
        self.device_graph.mic = duplex.mic
        requested_backend = self.aec_engine.requested_backend_for_audio_mode(self.device_graph.audio_mode)
        self.telemetry.mark_session_started(
            requested_backend=requested_backend,
            device_fingerprint=self._device_fingerprint_supplier(),
            audio_mode=self.device_graph.audio_mode,
        )
        self._set_session_state(SessionState.PROBING, reason="backend_selection")
        self.telemetry.mark_probe_started()
        decision = self.aec_engine.select_backend(
            audio_mode=self.device_graph.audio_mode,
            windows_healthcheck=self._probe_windows_system_aec,
            webrtc_healthcheck=self._probe_webrtc_apm,
        )
        self.telemetry.mark_probe_completed()
        self.recovery.record_probe_outcome(
            requested_backend=decision.requested_backend,
            selected_backend=decision.selected_backend,
            fallback_reason=decision.fallback_reason,
            degraded=decision.degraded,
        )
        self._apply_backend_selection(decision, reason="session_start")
        rt = self.transport.ensure_connected(turn_mode, self.telemetry.reconnect_attempts)
        self._runtime_resources.rt = self.transport.realtime
        self._start_rt_loop()
        if reset_transport_input:
            rt.clear_input_audio()
            self._log_sink("audio_push_to_realtime", self._build_log_payload(turn_mode=turn_mode, phase="clear_input"))
        duplex.start_mic()
        self._voice_gate.open()
        self.telemetry.mark_session_activated()
        final_state = SessionState.DEGRADED if decision.degraded else SessionState.ACTIVE
        self._set_session_state(final_state, reason="session_started")
        if getattr(duplex.mic, "using_resampler", False):
            self._caption_sink(f"Mikrofon jede na {duplex.mic.input_samplerate} Hz, resampluji na 24000 Hz.")
        self._caption_sink(caption)

    def _apply_backend_selection(self, decision: BackendSelectionDecision, *, reason: str) -> None:
        normalized_backend = normalize_audio_aec_mode(decision.selected_backend)
        self._aec_mode_setter(normalized_backend)
        contract = decision.mode_contract or self.aec_engine.product_mode_contract_for(
            selected_backend=normalized_backend,
            requested_backend=decision.requested_backend,
            audio_mode=self.device_graph.audio_mode or self._audio_mode_supplier() or "notebook_builtin",
            degradation_cause=decision.degradation_cause,
        )
        self.telemetry.note_backend_selected(
            selected_backend=normalized_backend,
            fallback_reason=decision.fallback_reason,
            degradation_cause=decision.degradation_cause,
            mode_contract=contract,
        )
        if decision.degraded:
            self._caption_sink(f"{contract.ui_status} Důvod: {decision.degradation_cause or decision.fallback_reason or 'nezjištěn'}. Recovery: {contract.recovery_policy}.")
        elif decision.selected_backend != decision.requested_backend or contract.selected_backend == "headset_clean":
            self._caption_sink(contract.ui_status)
        self._log_sink(
            "audio_backend_selected",
            self._build_log_payload(
                reason=reason,
                probe_details=decision.probe_details,
                product_mode_key=contract.key,
                product_status=contract.session_status,
                capture_gate_policy=contract.capture_gate_policy,
                recovery_policy=contract.recovery_policy,
            ),
        )

    def _attempt_backend_fallback(self, reason: str) -> bool:
        current_backend = normalize_audio_aec_mode(self.telemetry.selected_backend)
        next_backend = self.aec_engine.next_backend_after(
            current_backend,
            requested_backend=self.telemetry.requested_backend,
        )
        if not next_backend or next_backend == current_backend:
            return False
        degradation_cause = reason if next_backend == "degraded_no_aec" else ""
        contract = self.aec_engine.product_mode_contract_for(
            selected_backend=next_backend,
            requested_backend=self.telemetry.requested_backend,
            audio_mode=self.device_graph.audio_mode or self._audio_mode_supplier() or "notebook_builtin",
            degradation_cause=degradation_cause,
        )
        self._aec_mode_setter(next_backend)
        self.telemetry.note_backend_selected(
            selected_backend=next_backend,
            fallback_reason=reason,
            degradation_cause=degradation_cause,
            mode_contract=contract,
        )
        if next_backend == "degraded_no_aec":
            self._caption_sink(f"{contract.ui_status} Důvod: {reason}. Recovery: {contract.recovery_policy}.")
        else:
            self._caption_sink(f"Audio: fallback {current_backend} -> {next_backend} ({reason}).")
        self._log_sink(
            "audio_backend_fallback",
            self._build_log_payload(
                reason=reason,
                from_backend=current_backend,
                to_backend=next_backend,
            ),
        )
        self._set_session_state(SessionState.RECOVERING, reason=reason)
        self._restore_session_state(reason)
        return True

    def _restore_session_state(self, reason: str) -> None:
        state = SessionState.DEGRADED if self.telemetry.selected_backend == "degraded_no_aec" else SessionState.ACTIVE
        self._set_session_state(state, reason=reason)

    def _probe_windows_system_aec(self) -> tuple[bool, str]:
        health = windows_system_aec_healthcheck()
        if hasattr(health, "as_tuple"):
            return health.as_tuple()
        if isinstance(health, tuple) and len(health) == 2:
            return bool(health[0]), str(health[1])
        return bool(getattr(health, "ok", False)), str(getattr(health, "reason", "Windows System AEC backend nevrátil detail."))

    def _probe_webrtc_apm(self) -> tuple[bool, str]:
        if _WebRTCAudioProcessor is None:
            return False, "WebRTC APM backend není dostupný."
        return True, "WebRTC APM backend je dostupný."

    def _sync_ui_state(self) -> None:
        self._state_sink(session_state_to_ui_state(self.session_state, self.presentation_state))

    def _set_presentation_state(self, presentation: SessionPresentationState | None, *, reason: str) -> None:
        if self.session_state not in {SessionState.ACTIVE, SessionState.DEGRADED} and presentation is not None:
            return
        self.presentation_state = presentation
        self._sync_ui_state()
        self._log_sink("audio_ui_state", self._build_log_payload(reason=reason, ui_state=session_state_to_ui_state(self.session_state, self.presentation_state)))

    def _set_session_state(self, state: SessionState, *, reason: str) -> None:
        validate_session_transition(self.session_state, state)
        self.session_state = state
        if state not in {SessionState.ACTIVE, SessionState.DEGRADED}:
            self.presentation_state = None
        self._sync_ui_state()
        self._log_sink("audio_session_state", self._build_log_payload(reason=reason, session_state=state.value))

    def set_presentation_state(self, presentation: SessionPresentationState | None, *, reason: str = "external_ui") -> None:
        self._set_presentation_state(presentation, reason=reason)

    def enter_recovering(self, reason: str) -> None:
        if self._mode_supplier() == "idle":
            return
        if self.session_state not in {SessionState.RECOVERING, SessionState.FAILED}:
            self._set_session_state(SessionState.RECOVERING, reason=reason)

    def fail(self, message: str, *, reason: str) -> None:
        self.shutdown_runtime_resources()
        self._mode_setter("idle")
        self.awaiting_transcript = False
        self.reset_runtime_tracking()
        self.reset_voice_gate_runtime()
        self.telemetry.mark_session_stopped()
        self._set_session_state(SessionState.FAILED, reason=reason)
        self._error_sink(message)

    def handle_transport_error(self, message: str) -> None:
        self.recovery.handle_transport_error(message, session_state=self.session_state.value)

    def note_assistant_output_started(self) -> None:
        self.note_tts_rendering(rendering_active=True, now_monotonic=__import__("time").monotonic())
        if self.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}:
            self._set_presentation_state(SessionPresentationState.SPEAKING, reason="assistant_audio")

    def note_speech_started(self, *, during_assistant_output: bool) -> None:
        if self.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}:
            reason = "speech_started_during_render" if during_assistant_output else "speech_started"
            self._set_presentation_state(None, reason=reason)

    def note_user_turn_committed(self) -> None:
        if self.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}:
            self._set_presentation_state(SessionPresentationState.TRANSCRIBING, reason="barge_in")

    def note_response_done(self) -> None:
        self.note_tts_rendering(rendering_active=False, now_monotonic=__import__("time").monotonic())
        target = SessionState.DEGRADED if self.telemetry.selected_backend == "degraded_no_aec" else SessionState.ACTIVE
        if self.session_state == SessionState.RECOVERING:
            self._set_session_state(target, reason="response_done")
            return
        if self.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}:
            self._set_presentation_state(None, reason="response_done")

    def note_xrun(self, *, source: str = "runtime") -> None:
        self.recovery.note_xrun(source=source, session_state=self.session_state.value)

    def note_device_reset(self, *, source: str = "runtime") -> None:
        self.recovery.note_device_reset(source=source, session_state=self.session_state.value)

    def note_barge_in_result(self, *, success: bool, reason: str = "") -> None:
        self.recovery.note_barge_in_result(success=success, reason=reason)

    def handle_user_transcript(self, text: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        self.telemetry.note_server_activity()
        self.telemetry.note_response_started()
        self._caption_sink(f"Ty: {normalized}")
        self._log_sink("user", {"chars": len(normalized)})
        self.awaiting_transcript = False
        if self.presentation_state != SessionPresentationState.SPEAKING and self.session_state not in {SessionState.FAILED, SessionState.STOPPING}:
            self._set_presentation_state(SessionPresentationState.THINKING, reason="user_transcript")

    def handle_assistant_done(self, text: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        self.telemetry.note_server_activity()
        self._caption_sink(f"AI: {normalized}")
        self._log_sink("assistant", {"chars": len(normalized)})

    def handle_assistant_audio(self, pcm: bytes) -> None:
        self.telemetry.note_server_activity()
        self.note_assistant_output_started()
        latency_ms = self.telemetry.note_response_first_audio()
        if latency_ms is not None:
            self._log_sink("assistant_audio_first_delta", {"latency_ms": latency_ms, "bytes": len(pcm)})
        try:
            self.enqueue_render_pcm(pcm)
        except Exception as exc:
            self._log_sink("error", {"message": str(exc)})
            self.fail(f"{exc}\n\n{format_device_help()}", reason="assistant_audio_enqueue_failed")

    def handle_speech_started(self) -> None:
        try:
            self.stop_render_output()
        except Exception:
            pass
        self.awaiting_transcript = False
        self.note_speech_started(during_assistant_output=bool(self.telemetry.response_first_audio_at > 0.0))
        self.telemetry.clear_current_turn()
        self.telemetry.note_server_activity()
        self._log_sink("speech_started", self._build_log_payload())

    def handle_speech_stopped(self) -> None:
        self.awaiting_transcript = True
        self.note_user_turn_committed()
        self.telemetry.note_turn_committed()
        self.telemetry.note_server_activity()
        self._log_sink("speech_stopped", self._build_log_payload())

    def handle_response_done(self) -> None:
        total_latency_ms = self.telemetry.note_response_done()
        self._log_sink("response_done", self._build_log_payload(total_latency_ms=total_latency_ms))
        self.note_response_done()
        if self._mode_supplier() != "handsfree" and self.session_state in {SessionState.ACTIVE, SessionState.DEGRADED}:
            self._set_presentation_state(SessionPresentationState.QUIESCENT, reason="ptt_response_complete")
        self.awaiting_transcript = False
        self.telemetry.note_server_activity()

    def check_runtime_health(self) -> None:
        pending_snapshot = self.runtime_pending_snapshot()
        pending_events = int(pending_snapshot["pending_events"])
        duplex = self._runtime_resources.duplex
        if duplex is not None:
            duplex_state = duplex.get_runtime_state()
            pending_mic = int(duplex_state.get("pending_chunk_count", 0) or 0)
            pending_player_bytes = int(duplex_state.get("buffered_bytes", 0) or 0)
        else:
            pending_mic = int(pending_snapshot["pending_mic"])
            pending_player_bytes = int(pending_snapshot["pending_player_bytes"])

        gate_runtime = self.voice_gate_runtime
        now = time.monotonic()
        if (
            now - self._last_echo_stat_log_at >= 5.0
            and (gate_runtime.echo_drop_count != gate_runtime.last_echo_drop_reported or gate_runtime.barge_in_chunk_count != gate_runtime.last_barge_in_reported)
        ):
            self._log_sink("echo_guard", {"dropped_echo_chunks": gate_runtime.echo_drop_count, "barge_in_chunks": gate_runtime.barge_in_chunk_count})
            self._last_echo_stat_log_at = now
            gate_runtime.last_echo_drop_reported = gate_runtime.echo_drop_count
            gate_runtime.last_barge_in_reported = gate_runtime.barge_in_chunk_count

        self.recovery.observe_runtime_health(
            session_state=self.session_state.value,
            presentation_state=(self.presentation_state.value if self.presentation_state is not None else ""),
            pending_events=pending_events,
            pending_mic=pending_mic,
            pending_player_bytes=pending_player_bytes,
            transport_connected=bool(self.transport.realtime is not None and self.transport.realtime.is_connected),
        )

    def _build_log_payload(self, **extra: object) -> dict[str, object]:
        payload = self.telemetry.snapshot(session_state=self.session_state.value).to_log_payload()
        contract = self.current_mode_contract()
        payload.update(
            {
                "session_state": self.session_state.value,
                "configured_backend": self.aec_engine.requested_backend,
                "requested_backend": self.telemetry.requested_backend,
                "requested_backend_effective": self.telemetry.requested_backend,
                "selected_backend": self.telemetry.selected_backend,
                "product_mode_key": self.telemetry.product_mode_key,
                "product_status": self.telemetry.product_status,
                "capture_gate_policy": self.telemetry.capture_gate_policy,
                "recovery_policy": self.telemetry.recovery_policy,
                "aec_requires_reference": contract.requires_reference,
                "backend_chain": list(self.aec_engine.backend_chain_for(self.telemetry.requested_backend)),
                "turn_mode": self.transport.turn_mode,
                "session_telemetry_snapshot": self.telemetry.serializable_snapshot(session_state=self.session_state.value).to_log_payload(),
                "transport_health": (
                    self.transport.connection_health_snapshot()
                    if hasattr(self.transport, "connection_health_snapshot")
                    else {
                        "turn_mode": getattr(self.transport, "turn_mode", "server_vad"),
                        "is_connected": bool(getattr(getattr(self.transport, "realtime", None), "is_connected", False)),
                        "has_realtime": getattr(self.transport, "realtime", None) is not None,
                    }
                ),
            }
        )
        payload.update(extra)
        return payload

    def _require_devices_ready(self) -> None:
        if self._input_device_getter() is None or self._output_device_getter() is None:
            raise RuntimeError("Nenalezen mikrofon nebo výstupní zařízení.\n\n" + format_device_help())

    def _classify_failure_reason(self, message: str) -> str:
        return self.recovery.classify_failure_reason(message)

    def _handle_transport_error(self, message: str) -> None:
        self.handle_transport_error(message)
