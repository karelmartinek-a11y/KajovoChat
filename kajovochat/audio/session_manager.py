from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .aec_engine import AecEngine, BackendSelectionDecision
from .device_graph import DuplexDeviceGraph
from .recovery import FailureReason, RecoverySupervisor
from .runtime_resources import AudioRuntimeResources
from .session_state import SessionState, session_state_to_ui_state, validate_session_transition
from .telemetry import AudioTelemetry
from .transport_bridge import RealtimeTransportBridge
from .voice_gate import (
    VoiceGate,
    VoiceGateRuntimeState,
    is_guard_active as voice_gate_is_guard_active,
    note_diag_logged,
    note_reference_cache,
    note_tts_window,
    record_gate_outcome,
    should_log_problem_diag as voice_gate_should_log_problem_diag,
    should_log_success_diag as voice_gate_should_log_success_diag,
    update_playback_reference_state,
)
from ..settings import AppSettings, normalize_audio_aec_mode
from ..services.audio_service import AudioPlayer, DuplexAudioSession, RealtimeMicStream, format_device_help
from ..services.realtime_service import RealtimeService
from ..services.windows_native_aec import probe_windows_native_aec

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
        stop_realtime_session: Callable[[], None],
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
        self._stop_realtime_session = stop_realtime_session
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
        self.aec_engine = AecEngine(settings.audio_aec_mode)
        self.transport = RealtimeTransportBridge(
            settings=settings,
            guard_profile_supplier=guard_profile_supplier,
            state_sink=state_sink,
            caption_sink=caption_sink,
            error_sink=self._handle_transport_error,
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
            stop_session=stop_realtime_session,
            error_sink=error_sink,
            selected_backend_supplier=lambda: self.telemetry.selected_backend,
            fallback_handler=self._attempt_backend_fallback,
            restore_session_state=self._restore_session_state,
        )
        self.device_graph = DuplexDeviceGraph()

    @property
    def mic_enabled(self) -> threading.Event:
        return self._voice_gate.mic_enabled

    @property
    def voice_gate_runtime(self) -> VoiceGateRuntimeState:
        return self._voice_gate.runtime

    @property
    def awaiting_transcript(self) -> bool:
        return self._voice_gate.awaiting_transcript

    @awaiting_transcript.setter
    def awaiting_transcript(self, value: bool) -> None:
        self._voice_gate.awaiting_transcript = bool(value)

    def tick(self) -> None:
        self.recovery.tick()

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
        update_playback_reference_state(
            self._voice_gate.runtime,
            is_playing_out=is_playing_out,
            now_monotonic=now_monotonic,
            trailing_hold_s=trailing_hold_s,
        )

    def is_guard_active(self, *, is_playing_out: bool) -> bool:
        return voice_gate_is_guard_active(
            self._voice_gate.runtime,
            mode=self._mode_supplier(),
            is_playing_out=is_playing_out,
        )

    def cache_reference(self, reference_pcm16: bytes, *, now_monotonic: float) -> None:
        note_reference_cache(self._voice_gate.runtime, reference_pcm16, now_monotonic=now_monotonic)

    def note_tts_rendering(self, *, rendering_active: bool, now_monotonic: float) -> None:
        note_tts_window(
            self._voice_gate.runtime,
            rendering_active=rendering_active,
            now_monotonic=now_monotonic,
        )

    def record_gate_outcome(self, *, drop_chunk: bool, barge_in_candidate: bool):
        return record_gate_outcome(
            self._voice_gate.runtime,
            drop_chunk=drop_chunk,
            barge_in_candidate=barge_in_candidate,
        )

    def should_log_problem_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return voice_gate_should_log_problem_diag(
            self._voice_gate.runtime,
            now_monotonic=now_monotonic,
            min_interval_s=min_interval_s,
        )

    def should_log_success_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return voice_gate_should_log_success_diag(
            self._voice_gate.runtime,
            now_monotonic=now_monotonic,
            min_interval_s=min_interval_s,
        )

    def note_diag_logged(self, *, success: bool, now_monotonic: float) -> None:
        note_diag_logged(
            self._voice_gate.runtime,
            success=success,
            now_monotonic=now_monotonic,
        )

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
        changed = self.telemetry.note_reference_health(
            ready=ready,
            available_samples=available_samples,
            callback_age_ms=callback_age_ms,
        )
        if changed:
            self._log_sink(
                "audio_reference_health",
                self._build_log_payload(
                    ready=ready,
                    available_samples=available_samples,
                    callback_age_ms=callback_age_ms,
                ),
            )
        selected_backend = self.telemetry.selected_backend
        unhealthy = (
            selected_backend in {"windows_system_aec", "webrtc_apm"}
            and not ready
            and self.telemetry.reference_consecutive_misses >= 12
            and callback_age_ms >= 0
            and callback_age_ms >= 120
        )
        if unhealthy:
            self.recovery.request_fallback(FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value)

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
        selected_backend = self.telemetry.selected_backend
        if reference_miss or selected_backend != "windows_system_aec":
            self.telemetry.poor_aec_consecutive = 0
            return
        if backend in {"webrtc", "windows_system_capture"} or webrtc_success:
            self.telemetry.poor_aec_consecutive = 0
            return
        delay_error = abs(int(delay_samples or 0) - int(calibration_latency or 0))
        poor_native_block = bool(
            aec_quality < 0.025
            and improvement_ratio < 0.14
            and similarity < 0.5
            and delay_error >= 180
        )
        if poor_native_block:
            self.telemetry.poor_aec_events += 1
            self.telemetry.poor_aec_consecutive += 1
        else:
            self.telemetry.poor_aec_consecutive = 0
        if self.telemetry.poor_aec_consecutive >= 6:
            self.recovery.request_fallback(FailureReason.WINDOWS_SYSTEM_AEC_UNHEALTHY.value)

    def start_handsfree(self) -> None:
        self._start_session(mode="handsfree", turn_mode="server_vad", caption="Hands-free: Realtime aktivní (server VAD).", reset_transport_input=False)
        self._state_sink("listening")

    def ptt_pressed(self) -> None:
        if self._mode_supplier() == "handsfree":
            return
        self._start_session(mode="ptt", turn_mode="ptt", caption="PTT: poslouchám…", reset_transport_input=True)
        self._state_sink("listening")

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
        self._state_sink("transcribing")
        self._log_sink("audio_push_to_realtime", self._build_log_payload(turn_mode="ptt", phase="commit"))
        self._caption_sink("PTT: čekám na odpověď…")

    def request_stop(self) -> None:
        self._set_session_state(SessionState.STOPPING, reason=FailureReason.USER_STOP.value)
        self._voice_gate.close()
        self._voice_gate.awaiting_transcript = False
        self.transport.close()
        self._runtime_resources.rt = None
        self.device_graph.stop_io()
        self._stop_realtime_session()
        self.telemetry.clear_reconnect()
        self.telemetry.mark_session_stopped()
        self._set_session_state(SessionState.IDLE, reason=FailureReason.USER_STOP.value)
        self._state_sink("idle")

    def _start_session(self, *, mode: str, turn_mode: str, caption: str, reset_transport_input: bool) -> None:
        self._mode_setter(mode)
        self._set_session_state(SessionState.INITIALIZING, reason=f"{mode}_requested")
        self._resolve_devices()
        self._start_session_if_needed()
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
        self._set_session_state(SessionState.CALIBRATING, reason="backend_selection")
        decision = self.aec_engine.select_backend(
            audio_mode=self.device_graph.audio_mode,
            windows_healthcheck=self._probe_windows_system_aec,
            webrtc_healthcheck=self._probe_webrtc_apm,
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
        final_state = SessionState.DEGRADED if decision.degraded else SessionState.READY
        self._set_session_state(final_state, reason="session_started")
        if getattr(duplex.mic, "using_resampler", False):
            self._caption_sink(f"Mikrofon jede na {duplex.mic.input_samplerate} Hz, resampluji na 24000 Hz.")
        self._caption_sink(caption)

    def _apply_backend_selection(self, decision: BackendSelectionDecision, *, reason: str) -> None:
        normalized_backend = normalize_audio_aec_mode(decision.selected_backend)
        self._aec_mode_setter(normalized_backend)
        self.telemetry.note_backend_selected(
            selected_backend=normalized_backend,
            fallback_reason=decision.fallback_reason,
            degradation_cause=decision.degradation_cause,
        )
        if decision.degraded:
            self._caption_sink("Audio: session běží v nouzovém režimu degraded_no_aec.")
        elif decision.selected_backend != decision.requested_backend:
            self._caption_sink(
                f"Audio: backend fallback {decision.requested_backend} -> {decision.selected_backend}."
            )
        self._log_sink(
            "audio_backend_selected",
            self._build_log_payload(
                reason=reason,
                probe_details=decision.probe_details,
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
        self._aec_mode_setter(next_backend)
        self.telemetry.note_backend_selected(
            selected_backend=next_backend,
            fallback_reason=reason,
            degradation_cause=degradation_cause,
        )
        self._caption_sink(f"Audio: fallback {current_backend} -> {next_backend} ({reason}).")
        self._log_sink(
            "audio_backend_fallback",
            self._build_log_payload(
                reason=reason,
                from_backend=current_backend,
                to_backend=next_backend,
            ),
        )
        next_state = SessionState.DEGRADED if next_backend == "degraded_no_aec" else SessionState.RECOVERING
        self._set_session_state(next_state, reason=reason)
        return True

    def _restore_session_state(self, reason: str) -> None:
        state = SessionState.DEGRADED if self.telemetry.selected_backend == "degraded_no_aec" else SessionState.READY
        self._set_session_state(state, reason=reason)

    def _probe_windows_system_aec(self) -> tuple[bool, str]:
        probe = probe_windows_native_aec()
        return bool(probe.available), str(probe.reason or "Windows System AEC probe nevrátil detail.")

    def _probe_webrtc_apm(self) -> tuple[bool, str]:
        if _WebRTCAudioProcessor is None:
            return False, "WebRTC APM backend není dostupný."
        return True, "WebRTC APM backend je dostupný."

    def _set_session_state(self, state: SessionState, *, reason: str) -> None:
        validate_session_transition(self.session_state, state)
        self.session_state = state
        self._state_sink(session_state_to_ui_state(state))
        self._log_sink("audio_session_state", self._build_log_payload(reason=reason, session_state=state.value))

    def note_assistant_rendering(self) -> None:
        self.note_tts_rendering(rendering_active=True, now_monotonic=__import__("time").monotonic())
        if self.session_state in {SessionState.READY, SessionState.DEGRADED, SessionState.BARGE_IN_TRANSITION}:
            self._set_session_state(SessionState.ASSISTANT_RENDERING, reason="assistant_audio")

    def note_speech_started(self, *, assistant_rendering: bool) -> None:
        if assistant_rendering and self.session_state in {SessionState.ASSISTANT_RENDERING, SessionState.READY, SessionState.DEGRADED}:
            self._set_session_state(SessionState.DOUBLE_TALK, reason="speech_started_during_render")
        elif self.session_state in {SessionState.READY, SessionState.DEGRADED}:
            self._set_session_state(self.session_state, reason="speech_started")

    def note_barge_in_transition(self) -> None:
        if self.session_state in {
            SessionState.ASSISTANT_RENDERING,
            SessionState.DOUBLE_TALK,
            SessionState.READY,
            SessionState.DEGRADED,
        }:
            self._set_session_state(SessionState.BARGE_IN_TRANSITION, reason="barge_in")

    def note_response_done(self) -> None:
        self.note_tts_rendering(rendering_active=False, now_monotonic=__import__("time").monotonic())
        target = SessionState.DEGRADED if self.telemetry.selected_backend == "degraded_no_aec" else SessionState.READY
        if self.session_state in {
            SessionState.ASSISTANT_RENDERING,
            SessionState.DOUBLE_TALK,
            SessionState.BARGE_IN_TRANSITION,
            SessionState.RECOVERING,
            SessionState.READY,
            SessionState.DEGRADED,
        }:
            self._set_session_state(target, reason="response_done")

    def note_xrun(self, *, source: str = "runtime") -> None:
        self.recovery.note_xrun(source=source)

    def note_device_reset(self, *, source: str = "runtime") -> None:
        self.recovery.note_device_reset(source=source)

    def note_barge_in_result(self, *, success: bool, reason: str = "") -> None:
        self.recovery.note_barge_in_result(success=success, reason=reason)

    def _build_log_payload(self, **extra: object) -> dict[str, object]:
        payload = self.telemetry.snapshot(session_state=self.session_state).to_log_payload()
        payload.update(
            {
                "session_state": self.session_state.value,
                "configured_backend": self.aec_engine.requested_backend,
                "requested_backend": self.telemetry.requested_backend,
                "requested_backend_effective": self.telemetry.requested_backend,
                "selected_backend": self.telemetry.selected_backend,
                "backend_chain": list(self.aec_engine.backend_chain_for(self.telemetry.requested_backend)),
                "turn_mode": self.transport.turn_mode,
            }
        )
        payload.update(extra)
        return payload

    def _require_devices_ready(self) -> None:
        if self._input_device_getter() is None or self._output_device_getter() is None:
            raise RuntimeError("Nenalezen mikrofon nebo výstupní zařízení.\n\n" + format_device_help())

    def _classify_failure_reason(self, message: str) -> str:
        lowered = (message or "").lower()
        if any(marker in lowered for marker in ("timed out", "timeout")):
            return FailureReason.TRANSPORT_TIMEOUT.value
        if any(marker in lowered for marker in ("connection", "socket", "reset", "closed", "disconnect", "broken pipe")):
            return FailureReason.TRANSPORT_DISCONNECT.value
        if any(marker in lowered for marker in ("microphone", "reproduktor", "device", "zařízení", "nenalezen mikrofon")):
            return FailureReason.DEVICE_UNAVAILABLE.value
        if "reference" in lowered:
            return FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value
        if "windows" in lowered and "aec" in lowered:
            return FailureReason.WINDOWS_SYSTEM_AEC_UNHEALTHY.value
        return FailureReason.UNKNOWN.value

    def _handle_transport_error(self, message: str) -> None:
        failure_reason = self._classify_failure_reason(message)
        self.telemetry.last_failure_reason = failure_reason
        self._log_sink(
            "audio_session_error",
            self._build_log_payload(
                message=message,
                failure_reason=failure_reason,
                recoverable=failure_reason in {
                    FailureReason.TRANSPORT_DISCONNECT.value,
                    FailureReason.TRANSPORT_TIMEOUT.value,
                },
            ),
        )
        recoverable = failure_reason in {
            FailureReason.TRANSPORT_DISCONNECT.value,
            FailureReason.TRANSPORT_TIMEOUT.value,
        }
        if self._mode_supplier() != "idle" and recoverable:
            self._set_session_state(SessionState.RECOVERING, reason=failure_reason)
            self.recovery.schedule(message, failure_reason)
            return
        if failure_reason in {
            FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value,
            FailureReason.WINDOWS_SYSTEM_AEC_UNHEALTHY.value,
        } and self.recovery.request_fallback(failure_reason):
            return
        self._set_session_state(SessionState.FAILED, reason=failure_reason)
        self._stop_realtime_session()
        self._state_sink("error")
        self._error_sink(message)
