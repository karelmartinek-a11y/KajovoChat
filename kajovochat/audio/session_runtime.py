from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from .runtime_bindings import ConversationAudioRuntimeBindings


class ConversationAudioRuntimeController:
    """Vlastník životního cyklu realtime audio smyčky."""

    def __init__(
        self,
        owner: object,
        *,
        estimate_voice_likelihood_from_pcm16,
        closed_pose_factory,
        backend_aware_aec_metrics,
        state_speaking: str,
        echo_trailing_hold_s: float,
        on_fatal=None,
    ) -> None:
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._runtime: Optional[ConversationAudioRuntimeLoop] = None
        self._bindings = ConversationAudioRuntimeBindings(
            owner,
            closed_pose_factory=closed_pose_factory,
            state_speaking=state_speaking,
            echo_trailing_hold_s=echo_trailing_hold_s,
        )
        self._on_fatal = on_fatal
        self._kwargs = {
            "estimate_voice_likelihood_from_pcm16": estimate_voice_likelihood_from_pcm16,
            "backend_aware_aec_metrics": backend_aware_aec_metrics,
            "on_fatal": self._handle_runtime_fatal,
        }

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def runtime(self) -> Optional["ConversationAudioRuntimeLoop"]:
        return self._runtime

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _handle_runtime_fatal(self, exc: Exception, *, stage: str) -> None:
        self._stop_event.set()
        if self._on_fatal is None:
            return
        try:
            self._on_fatal(exc, stage=stage)
        except Exception:
            return

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._runtime = ConversationAudioRuntimeLoop(
            self._bindings,
            stop_event=self._stop_event,
            **self._kwargs,
        )
        self._thread = threading.Thread(target=self._runtime.run, daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        self._thread = None
        self._runtime = None

class ConversationAudioRuntimeLoop:
    """Provozní realtime audio smyčka oddělená od GUI workeru."""

    def __init__(
        self,
        bindings: ConversationAudioRuntimeBindings,
        *,
        stop_event,
        estimate_voice_likelihood_from_pcm16,
        backend_aware_aec_metrics,
        on_fatal,
    ) -> None:
        self._bindings = bindings
        self.owner = bindings.owner
        self._stop_event = stop_event
        self._estimate_voice_likelihood_from_pcm16 = estimate_voice_likelihood_from_pcm16
        self._backend_aware_aec_metrics = backend_aware_aec_metrics
        self._on_fatal = on_fatal

    def run(self) -> None:
        owner = self.owner
        try:
            while not self._stop_event.is_set():
                self._bindings.tick_realtime()
                self._bindings.adapt_guard_if_needed()

                duplex, current_out_level, is_playing_out = self._bindings.resolve_playback_state()
                guard_active = self._bindings.guard_active(is_playing_out=is_playing_out)

                if self._bindings.has_active_capture(duplex):
                    capture_queue = self._bindings.capture_queue_for(duplex)
                    for _ in range(6):
                        try:
                            chunk_item = capture_queue.get_nowait()
                        except queue.Empty:
                            break
                        mic_captured_at_mono_ns = 0
                        if hasattr(chunk_item, "pcm_bytes"):
                            chunk = bytes(chunk_item.pcm_bytes)
                            mic_captured_at_mono_ns = int(getattr(chunk_item, "captured_at_mono_ns", 0) or 0)
                        else:
                            chunk = chunk_item
                        if chunk:
                            processed_chunk = chunk
                            aec_result: dict[str, object] = {}
                            similarity = 0.0
                            residual_level = owner._pcm16_level(chunk)
                            aec_quality = 0.0
                            double_talk = False
                            raw_voice_likelihood = 0.0
                            predicted_level = 0.0
                            improvement_ratio = 0.0
                            effective_similarity = 0.0
                            effective_aec_quality = 0.0
                            aec_backend = "custom"
                            webrtc_success = False
                            reference_miss = False
                            if guard_active and duplex is not None:
                                try:
                                    previous_calibration_latency = int(owner._guard_calibration.get("latency_samples", 0) or 0)
                                    calibration_latency = previous_calibration_latency
                                    reference = duplex.get_echo_reference_for_capture(
                                        max_samples=max(8192, len(chunk) // 2 + 1920),
                                        captured_at_mono_ns=mic_captured_at_mono_ns or None,
                                    )
                                    reference_stats = duplex.get_echo_reference_stats()
                                except Exception:
                                    previous_calibration_latency = 0
                                    calibration_latency = 0
                                    reference = b""
                                    reference_stats = {"available_samples": 0, "total_samples": 0, "played_samples": 0, "callback_age_ms": -1}
                                aec_requires_reference = owner._session_manager.aec_requires_reference()
                                reference_needed = owner._reference_needed_samples(len(chunk))
                                callback_age_ms = int(reference_stats.get("callback_age_ms", -1) or -1)
                                played_samples = int(reference_stats.get("played_samples", 0) or 0)
                                available_samples = int(reference_stats.get("available_samples", 0) or 0)
                                reference_choice = owner._session_manager.select_reference_source(
                                    aec_requires_reference=aec_requires_reference,
                                    now_monotonic=time.monotonic(),
                                    reference_needed=reference_needed,
                                    live_reference_pcm16=np.asarray(reference, dtype=np.int16).astype(np.int16, copy=False).tobytes() if len(reference) else b"",
                                    available_samples=available_samples,
                                    played_samples=played_samples,
                                    callback_age_ms=callback_age_ms,
                                )
                                reference_ready = bool(reference_choice.ready)
                                reference_miss = bool(reference_choice.miss)
                                if reference_choice.source.startswith("cached"):
                                    reference = np.frombuffer(reference_choice.reference_pcm16, dtype=np.int16).copy()
                                    reference_stats = dict(reference_stats)
                                    reference_stats["available_samples"] = reference_choice.available_samples
                                    reference_stats["callback_age_ms"] = reference_choice.callback_age_ms
                                elif not aec_requires_reference:
                                    reference = np.empty((0,), dtype=np.int16)
                                owner._session_manager.note_reference_health(
                                    ready=reference_ready,
                                    available_samples=int(reference_choice.available_samples),
                                    callback_age_ms=int(reference_choice.callback_age_ms),
                                )
                                if reference_ready:
                                    try:
                                        owner._session_manager.cache_reference(
                                            np.asarray(reference, dtype=np.int16).astype(np.int16, copy=False).tobytes(),
                                            now_monotonic=time.monotonic(),
                                        )
                                    except Exception:
                                        pass
                                    try:
                                        aec_result = owner._aec.process(
                                            chunk,
                                            reference,
                                            max_shift_samples=max(960, calibration_latency + max(240, owner._aec.filter_length // 2)),
                                            expected_shift=calibration_latency or None,
                                            aec_mode=owner._aec_mode,
                                        )
                                        processed_chunk = bytes(aec_result.get("pcm", chunk))
                                        similarity = float(aec_result.get("similarity", 0.0) or 0.0)
                                        residual_level = float(aec_result.get("residual_level", owner._pcm16_level(processed_chunk)) or 0.0)
                                        aec_quality = float(aec_result.get("aec_quality", 0.0) or 0.0)
                                        double_talk = bool(aec_result.get("double_talk", False))
                                        raw_voice_likelihood = float(aec_result.get("voice_likelihood", 0.0) or 0.0)
                                        predicted_level = float(aec_result.get("predicted_level", 0.0) or 0.0)
                                        improvement_ratio = float(aec_result.get("improvement_ratio", 0.0) or 0.0)
                                        aec_backend = str(aec_result.get("backend", "custom") or "custom")
                                        webrtc_success = bool(aec_result.get("webrtc_success", False))
                                        native_attempted = bool(aec_result.get("native_attempted", False))
                                        native_selected = bool(aec_result.get("native_selected", False))
                                        selection_reason = str(aec_result.get("selection_reason", "custom_fallback") or "custom_fallback")
                                        delay_samples = int(aec_result.get("delay_samples", 0) or 0)
                                        effective_similarity, effective_aec_quality = self._backend_aware_aec_metrics(
                                            backend=aec_backend,
                                            similarity=similarity,
                                            aec_quality=aec_quality,
                                            improvement_ratio=improvement_ratio,
                                            residual_level=residual_level,
                                            output_level=current_out_level,
                                            webrtc_success=webrtc_success,
                                            native_selected=native_selected,
                                        )
                                        owner._session_manager.note_aec_observation(
                                            backend=aec_backend,
                                            reference_miss=False,
                                            aec_quality=effective_aec_quality,
                                            improvement_ratio=improvement_ratio,
                                            delay_samples=delay_samples,
                                            calibration_latency=calibration_latency,
                                            similarity=effective_similarity,
                                            webrtc_success=webrtc_success,
                                        )
                                        owner._record_aec_diag_sample(
                                            residual_level=residual_level,
                                            aec_quality=effective_aec_quality,
                                            double_talk=double_talk,
                                            delay_samples=delay_samples,
                                            similarity=effective_similarity,
                                            reference_miss=False,
                                        )
                                        can_refresh_latency = bool(
                                            delay_samples > 0
                                            and (
                                                (effective_similarity >= 0.55 and effective_aec_quality >= 0.08)
                                                or (aec_backend == "webrtc" and effective_similarity >= 0.4 and improvement_ratio >= 0.05)
                                            )
                                            and not double_talk
                                            and reference_stats.get("available_samples", 0) >= max(len(chunk) // 2 + 256, 1024)
                                        )
                                        if can_refresh_latency:
                                            owner._consider_runtime_latency_update(
                                                delay_samples=delay_samples,
                                                similarity=effective_similarity,
                                                aec_quality=effective_aec_quality,
                                                improvement_ratio=improvement_ratio,
                                                backend=aec_backend,
                                                double_talk=double_talk,
                                                prefer_webrtc=owner._aec_mode == "webrtc_apm",
                                            )
                                        elif webrtc_success and delay_samples > 0 and not double_talk:
                                            owner._consider_runtime_latency_update(
                                                delay_samples=delay_samples,
                                                similarity=max(0.4, effective_similarity),
                                                aec_quality=max(0.08, effective_aec_quality),
                                                improvement_ratio=improvement_ratio,
                                                backend=aec_backend,
                                                double_talk=False,
                                                prefer_webrtc=owner._aec_mode == "webrtc_apm",
                                            )
                                    except Exception:
                                        processed_chunk = chunk
                                        similarity = 0.0
                                        residual_level = owner._pcm16_level(chunk)
                                        aec_quality = 0.0
                                        double_talk = False
                                        raw_voice_likelihood = 0.0
                                        predicted_level = 0.0
                                        improvement_ratio = 0.0
                                        aec_backend = "custom"
                                        webrtc_success = False
                                        native_attempted = False
                                        native_selected = False
                                        selection_reason = "custom_fallback"
                                        effective_similarity = 0.0
                                        effective_aec_quality = 0.0
                                        previous_calibration_latency = calibration_latency
                                        reference_stats = {"available_samples": 0, "total_samples": 0, "callback_age_ms": -1}
                                        reference_miss = True
                                else:
                                    predicted_level = 0.0
                                    improvement_ratio = 0.0
                                    aec_backend = "custom"
                                    webrtc_success = False
                                    native_attempted = False
                                    native_selected = False
                                    selection_reason = "custom_fallback"
                                    effective_similarity = 0.0
                                    effective_aec_quality = 0.0
                                    owner._record_aec_diag_sample(
                                        residual_level=residual_level,
                                        aec_quality=0.0,
                                        double_talk=False,
                                        delay_samples=0,
                                        similarity=0.0,
                                        reference_miss=True,
                                    )
                            in_level = owner._pcm16_level(processed_chunk)
                            processed_voice_likelihood = self._estimate_voice_likelihood_from_pcm16(processed_chunk)
                            if is_playing_out:
                                playback_safe_raw_voice = raw_voice_likelihood * (0.45 if (double_talk or aec_quality > 0.22) else 0.2)
                                voice_likelihood = max(processed_voice_likelihood, playback_safe_raw_voice)
                            else:
                                voice_likelihood = max(raw_voice_likelihood, processed_voice_likelihood)
                            if not effective_similarity and not effective_aec_quality and (similarity > 0.0 or aec_quality > 0.0):
                                effective_similarity, effective_aec_quality = self._backend_aware_aec_metrics(
                                    backend=aec_backend,
                                    similarity=similarity,
                                    aec_quality=aec_quality,
                                    improvement_ratio=improvement_ratio,
                                    residual_level=residual_level,
                                    output_level=current_out_level,
                                    webrtc_success=webrtc_success,
                                    native_selected=native_selected,
                                )
                            effective_aec_quality = max(effective_aec_quality, 0.1 if webrtc_success else 0.0)
                            now_for_diag = time.monotonic()
                            owner._last_raw_in_level = float(in_level)
                            gate_decision = owner._session_manager.evaluate_capture_gate(
                                mode=owner._mode,
                                guard_active=guard_active,
                                playback_active=is_playing_out,
                                similarity=effective_similarity,
                                input_level=in_level,
                                output_level=current_out_level,
                                default_profile=owner.settings.normalized_audio_guard_profile(),
                                profile=owner._guard_profile,
                                residual_level=residual_level,
                                voice_likelihood=voice_likelihood,
                                double_talk=double_talk,
                                aec_quality=effective_aec_quality,
                                effective_similarity=effective_similarity,
                                effective_aec_quality=effective_aec_quality,
                                now_monotonic=now_for_diag,
                            )
                            drop_chunk = bool(gate_decision.drop_chunk)
                            drop_reason = str(gate_decision.drop_reason)
                            delay_drift = abs(int(owner._guard_calibration.get("latency_samples", 0) or 0) - int(aec_result.get("delay_samples", 0) or 0))
                            diag_interval_s = 0.8 if (reference_miss or effective_similarity >= 0.4 or aec_backend == "webrtc" or aec_backend == "windows_system_aec" or double_talk) else 5.0
                            should_log_problem_diag = bool(
                                guard_active
                                and owner._session_manager.should_log_problem_diag(
                                    now_monotonic=now_for_diag,
                                    min_interval_s=diag_interval_s,
                                )
                                and (
                                    reference_miss
                                    or effective_similarity >= 0.45
                                    or webrtc_success
                                    or effective_aec_quality < 0.18
                                    or double_talk
                                    or delay_drift > 96
                                )
                            )
                            should_log_success_diag = bool(
                                guard_active
                                and not should_log_problem_diag
                                and owner._session_manager.should_log_success_diag(
                                    now_monotonic=now_for_diag,
                                    min_interval_s=2.0,
                                )
                                and (
                                    aec_backend == "webrtc"
                                    or aec_backend == "windows_system_aec"
                                    or webrtc_success
                                    or (not reference_miss and effective_similarity >= 0.2)
                                    or (not reference_miss and predicted_level > 0.0 and improvement_ratio > 0.0)
                                )
                            )
                            if should_log_problem_diag or should_log_success_diag:
                                owner._log_event(
                                    "aec_diag",
                                    similarity=round(effective_similarity, 3),
                                    residual_level=round(residual_level, 4),
                                    aec_quality=round(effective_aec_quality, 3),
                                    predicted_level=round(predicted_level, 4),
                                    improvement_ratio=round(improvement_ratio, 3),
                                    backend=aec_backend,
                                    webrtc_success=bool(webrtc_success),
                                    native_attempted=bool(native_attempted),
                                    native_selected=bool(native_selected),
                                    selection_reason=selection_reason,
                                    double_talk=bool(double_talk),
                                    voice_likelihood=round(voice_likelihood, 3),
                                    delay_samples=int(aec_result.get("delay_samples", 0) or 0) if guard_active and duplex is not None else 0,
                                    calibration_latency=int(previous_calibration_latency if guard_active and duplex is not None else owner._guard_calibration.get("latency_samples", 0) or 0),
                                    reference_available=int(reference_stats.get("available_samples", 0) or 0),
                                    reference_callback_age_ms=int(reference_stats.get("callback_age_ms", -1) or -1),
                                    reference_miss=bool(reference_miss),
                                )
                                if should_log_problem_diag:
                                    owner._session_manager.note_diag_logged(success=False, now_monotonic=now_for_diag)
                                else:
                                    owner._session_manager.note_diag_logged(success=True, now_monotonic=now_for_diag)
                            barge_in_candidate = bool(gate_decision.barge_in_candidate)
                            barge_in_confirmed = bool(gate_decision.barge_in_confirmed)
                            owner._guard_telemetry.add_sample(
                                input_level=float(gate_decision.effective_input_level),
                                output_level=current_out_level,
                                similarity=effective_similarity,
                                voice_likelihood=voice_likelihood,
                                dropped=drop_chunk,
                                playback_active=is_playing_out,
                                reason=drop_reason,
                                barge_in_candidate=barge_in_candidate,
                                residual_level=residual_level,
                                aec_quality=effective_aec_quality,
                                double_talk=double_talk,
                            )
                            gate_outcome = gate_decision.side_effects
                            owner._session_manager.maybe_force_server_vad_turn_commit(
                                input_level=float(gate_decision.effective_input_level),
                                voice_likelihood=float(voice_likelihood),
                                now_monotonic=now_for_diag,
                                drop_chunk=bool(drop_chunk),
                                playback_active=bool(is_playing_out),
                                barge_in_confirmed=bool(barge_in_confirmed),
                            )
                            if barge_in_confirmed and is_playing_out:
                                owner._session_manager.note_user_turn_committed()
                            if drop_chunk:
                                owner._last_post_gate_in_level = 0.0
                                owner._last_in_level = 0.0
                                if gate_outcome.should_log_echo_drop:
                                    owner._log_event(
                                        "echo_drop",
                                        reason=drop_reason,
                                        similarity=round(effective_similarity, 3),
                                        input_level=round(float(gate_decision.effective_input_level), 3),
                                        output_level=round(current_out_level, 3),
                                        voice_likelihood=round(voice_likelihood, 3),
                                    )
                                continue
                            owner._last_post_gate_in_level = float(in_level)
                            owner._last_in_level = in_level
                            rt = owner._session_manager.transport.realtime
                            if rt is None:
                                continue
                            try:
                                if rt.append_audio_pcm16(processed_chunk):
                                    owner._session_manager.note_input_audio_appended(len(processed_chunk))
                            except AttributeError:
                                continue

                if duplex is not None:
                    try:
                        owner._last_out_level = current_out_level
                        out_pose = duplex.get_lipsync_snapshot()
                    except Exception:
                        owner._last_out_level = 0.0
                        out_pose = self._bindings.closed_pose()
                else:
                    owner._last_out_level = 0.0
                    out_pose = self._bindings.closed_pose()

                try:
                    self._bindings.emit_levels(out_pose=out_pose)
                except RuntimeError:
                    self._stop_event.set()
                    break
                time.sleep(0.005)
        except Exception as exc:
            self._stop_event.set()
            if self._on_fatal is not None:
                self._on_fatal(exc, stage="runtime_loop")
