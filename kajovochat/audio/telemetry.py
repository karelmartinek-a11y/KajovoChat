from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .contracts import BackendHealthSnapshot, SessionHealth

if TYPE_CHECKING:
    from .aec_engine import AecProductMode


@dataclass
class AecDiagnosticsTelemetry:
    samples: float = 0.0
    double_talk: float = 0.0
    low_quality: float = 0.0
    reference_miss: float = 0.0
    reference_ready: float = 0.0
    aligned: float = 0.0
    strong_aligned: float = 0.0
    residual_sum: float = 0.0
    quality_sum: float = 0.0
    aligned_residual_sum: float = 0.0
    aligned_quality_sum: float = 0.0
    delay_error_sum: float = 0.0
    max_delay_error: float = 0.0

    def reset(self) -> None:
        self.samples = 0.0
        self.double_talk = 0.0
        self.low_quality = 0.0
        self.reference_miss = 0.0
        self.reference_ready = 0.0
        self.aligned = 0.0
        self.strong_aligned = 0.0
        self.residual_sum = 0.0
        self.quality_sum = 0.0
        self.aligned_residual_sum = 0.0
        self.aligned_quality_sum = 0.0
        self.delay_error_sum = 0.0
        self.max_delay_error = 0.0

    def record_sample(
        self,
        *,
        residual_level: float,
        aec_quality: float,
        double_talk: bool,
        delay_samples: int,
        similarity: float,
        reference_miss: bool,
        calibration_latency: int,
    ) -> None:
        self.samples += 1.0
        self.residual_sum += float(residual_level)
        self.quality_sum += float(aec_quality)
        if double_talk:
            self.double_talk += 1.0
        if aec_quality < 0.18:
            self.low_quality += 1.0
        if reference_miss:
            self.reference_miss += 1.0
        else:
            self.reference_ready += 1.0
        if similarity >= 0.2:
            self.aligned += 1.0
            self.aligned_residual_sum += float(residual_level)
            self.aligned_quality_sum += float(aec_quality)
        if similarity >= 0.5:
            self.strong_aligned += 1.0
        delay_error = abs(int(delay_samples) - int(calibration_latency)) if calibration_latency > 0 and delay_samples > 0 else 0
        self.delay_error_sum += float(delay_error)
        self.max_delay_error = max(float(self.max_delay_error), float(delay_error))

    def build_summary(self) -> dict[str, float]:
        samples = max(1.0, float(self.samples or 0.0))
        aligned = max(1.0, float(self.aligned or 0.0))
        return {
            "samples": int(self.samples or 0.0),
            "avg_residual": float(self.residual_sum or 0.0) / samples,
            "avg_quality": float(self.quality_sum or 0.0) / samples,
            "double_talk_ratio": float(self.double_talk or 0.0) / samples,
            "low_quality_ratio": float(self.low_quality or 0.0) / samples,
            "reference_miss_ratio": float(self.reference_miss or 0.0) / samples,
            "reference_ready_ratio": float(self.reference_ready or 0.0) / samples,
            "aligned_ratio": float(self.aligned or 0.0) / samples,
            "strong_alignment_ratio": float(self.strong_aligned or 0.0) / samples,
            "avg_quality_when_aligned": float(self.aligned_quality_sum or 0.0) / aligned,
            "avg_residual_when_aligned": float(self.aligned_residual_sum or 0.0) / aligned,
            "avg_delay_error": float(self.delay_error_sum or 0.0) / samples,
            "max_delay_error": float(self.max_delay_error or 0.0),
        }


@dataclass(frozen=True)
class ReferenceHealthEvent:
    mono_ts: float
    ready: bool
    available_samples: int
    callback_age_ms: int
    state: str

    def to_log_payload(self) -> dict[str, object]:
        return {
            "mono_ts": round(float(self.mono_ts), 6),
            "ready": bool(self.ready),
            "available_samples": int(self.available_samples),
            "callback_age_ms": int(self.callback_age_ms),
            "state": self.state,
        }


@dataclass(frozen=True)
class RecoveryStoryEvent:
    mono_ts: float
    category: str
    reason: str
    action: str
    session_state: str
    selected_backend: str
    target_backend: str = ""
    cooldown_s: float = 0.0
    detail: dict[str, object] = field(default_factory=dict)

    def to_log_payload(self) -> dict[str, object]:
        payload = {
            "mono_ts": round(float(self.mono_ts), 6),
            "category": self.category,
            "reason": self.reason,
            "action": self.action,
            "session_state": self.session_state,
            "selected_backend": self.selected_backend,
            "target_backend": self.target_backend,
            "cooldown_s": round(float(self.cooldown_s), 3),
        }
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class SessionTelemetrySnapshot:
    session_state: str
    requested_backend: str
    selected_backend: str
    product_mode_key: str
    product_status: str
    capture_gate_policy: str
    recovery_policy: str
    fallback_chain_step: int
    fallback_reason: str
    degradation_cause: str
    reconnect_attempts: int
    recovery_attempts_total: int
    recovery_successes_total: int
    recovery_failures_total: int
    xrun_events_total: int
    device_resets_total: int
    reference_health_timeline: list[dict[str, object]]
    recovery_story: list[dict[str, object]]
    backend_health_score: float
    timings: dict[str, object]
    turn_latency: dict[str, object]
    health: dict[str, object]

    def to_log_payload(self) -> dict[str, object]:
        return {
            "session_state": self.session_state,
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "product_mode_key": self.product_mode_key,
            "product_status": self.product_status,
            "capture_gate_policy": self.capture_gate_policy,
            "recovery_policy": self.recovery_policy,
            "fallback_chain_step": int(self.fallback_chain_step),
            "fallback_reason": self.fallback_reason,
            "degradation_cause": self.degradation_cause,
            "reconnect_attempts": int(self.reconnect_attempts),
            "recovery_attempts_total": int(self.recovery_attempts_total),
            "recovery_successes_total": int(self.recovery_successes_total),
            "recovery_failures_total": int(self.recovery_failures_total),
            "xrun_events_total": int(self.xrun_events_total),
            "device_resets_total": int(self.device_resets_total),
            "reference_health_timeline": list(self.reference_health_timeline),
            "recovery_story": list(self.recovery_story),
            "backend_health_score": float(self.backend_health_score),
            "timings": dict(self.timings),
            "turn_latency": dict(self.turn_latency),
            "health": dict(self.health),
        }

    def to_log_dict(self) -> dict[str, object]:
        return self.to_log_payload()


@dataclass
class AudioTelemetry:
    reconnect_attempts: int = 0
    scheduled_reconnect_at: float = 0.0
    last_server_activity_at: float = field(default_factory=time.monotonic)
    session_started_at: float = 0.0
    session_probe_started_at: float = 0.0
    session_probe_completed_at: float = 0.0
    session_activated_at: float = 0.0
    session_stopped_at: float = 0.0
    requested_backend: str = "windows_system_aec"
    selected_backend: str = "windows_system_aec"
    product_mode_key: str = "notebook_builtin_windows_system_aec"
    product_status: str = "Notebook builtin + Windows System AEC"
    capture_gate_policy: str = "windows_system_aec"
    recovery_policy: str = "prefer_current_until_failure"
    fallback_chain_step: int = 0
    fallback_reason: str = ""
    degradation_cause: str = ""
    device_fingerprint: str = "unknown"
    audio_mode: str = "notebook_builtin"
    last_failure_reason: str = ""
    recovery_attempts_total: int = 0
    recovery_successes_total: int = 0
    recovery_failures_total: int = 0
    reference_ready: bool = False
    reference_available_samples: int = 0
    reference_callback_age_ms: int = -1
    reference_health_state: str = "unknown"
    reference_ready_events: int = 0
    reference_miss_events: int = 0
    reference_consecutive_misses: int = 0
    reference_health_timeline: list[ReferenceHealthEvent] = field(default_factory=list)
    recovery_story: list[RecoveryStoryEvent] = field(default_factory=list)
    last_backend_switch_at: float = 0.0
    poor_aec_events: int = 0
    poor_aec_consecutive: int = 0
    degraded_transitions_total: int = 0
    backend_switches_total: int = 0
    xrun_events_total: int = 0
    device_resets_total: int = 0
    barge_in_attempts_total: int = 0
    barge_in_successes_total: int = 0
    pending_events: int = 0
    pending_mic_chunks: int = 0
    pending_player_bytes: int = 0
    last_backlog_log_at: float = 0.0
    last_player_progress_at: float = 0.0
    last_player_buffer_bytes: int = 0
    turn_committed_at: float = 0.0
    response_started_at: float = 0.0
    response_first_audio_at: float = 0.0
    turns_total: int = 0
    responses_completed_total: int = 0
    response_first_audio_latency_total_ms: int = 0
    response_first_audio_latency_max_ms: int = 0
    response_total_latency_total_ms: int = 0
    response_total_latency_max_ms: int = 0

    def mark_session_started(self, *, requested_backend: str, device_fingerprint: str, audio_mode: str) -> None:
        now = time.monotonic()
        self.session_started_at = now
        self.session_probe_started_at = 0.0
        self.session_probe_completed_at = 0.0
        self.session_activated_at = 0.0
        self.session_stopped_at = 0.0
        self.last_server_activity_at = now
        self.reconnect_attempts = 0
        self.scheduled_reconnect_at = 0.0
        self.requested_backend = requested_backend
        self.selected_backend = requested_backend
        self.audio_mode = audio_mode or "notebook_builtin"
        self.product_mode_key = f"{self.audio_mode}_{requested_backend}" if requested_backend != "headset_clean" else "headset_clean"
        self.product_status = requested_backend
        self.capture_gate_policy = requested_backend
        self.recovery_policy = "steady"
        self.fallback_chain_step = 0
        self.fallback_reason = ""
        self.degradation_cause = ""
        self.device_fingerprint = device_fingerprint or "unknown"
        self.last_failure_reason = ""
        self.recovery_attempts_total = 0
        self.recovery_successes_total = 0
        self.recovery_failures_total = 0
        self.reference_ready = False
        self.reference_available_samples = 0
        self.reference_callback_age_ms = -1
        self.reference_health_state = "starting"
        self.reference_ready_events = 0
        self.reference_miss_events = 0
        self.reference_consecutive_misses = 0
        self.reference_health_timeline = []
        self.recovery_story = []
        self.last_backend_switch_at = now
        self.poor_aec_events = 0
        self.poor_aec_consecutive = 0
        self.degraded_transitions_total = 0
        self.backend_switches_total = 0
        self.xrun_events_total = 0
        self.device_resets_total = 0
        self.barge_in_attempts_total = 0
        self.barge_in_successes_total = 0
        self.reset_runtime_watchdog(now_monotonic=now)
        self.reset_turn_timing()
        self.record_recovery_story(
            category="session",
            reason="session_started",
            action="start",
            session_state="starting",
            selected_backend=requested_backend,
            target_backend=requested_backend,
        )

    def mark_probe_started(self) -> None:
        now = time.monotonic()
        self.session_probe_started_at = now
        self.record_recovery_story(
            category="session",
            reason="backend_probe_started",
            action="probe",
            session_state="probing",
            selected_backend=self.selected_backend,
        )

    def mark_probe_completed(self) -> None:
        self.session_probe_completed_at = time.monotonic()

    def mark_session_activated(self) -> None:
        self.session_activated_at = time.monotonic()

    def mark_session_stopped(self) -> None:
        self.session_stopped_at = time.monotonic()
        self.record_recovery_story(
            category="session",
            reason="session_stopped",
            action="stop",
            session_state="stopping",
            selected_backend=self.selected_backend,
        )

    def reset_runtime_watchdog(self, *, now_monotonic: float | None = None) -> None:
        now = float(now_monotonic if now_monotonic is not None else time.monotonic())
        self.last_server_activity_at = now
        self.pending_events = 0
        self.pending_mic_chunks = 0
        self.pending_player_bytes = 0
        self.last_backlog_log_at = 0.0
        self.last_player_progress_at = now
        self.last_player_buffer_bytes = 0

    def clear_current_turn(self) -> None:
        self.turn_committed_at = 0.0
        self.response_started_at = 0.0
        self.response_first_audio_at = 0.0

    def reset_turn_timing(self) -> None:
        self.clear_current_turn()
        self.turns_total = 0
        self.responses_completed_total = 0
        self.response_first_audio_latency_total_ms = 0
        self.response_first_audio_latency_max_ms = 0
        self.response_total_latency_total_ms = 0
        self.response_total_latency_max_ms = 0

    def note_server_activity(self) -> None:
        self.last_server_activity_at = time.monotonic()

    def note_transport_activity(self) -> None:
        self.note_server_activity()

    def set_last_failure_reason(self, reason: str) -> None:
        self.last_failure_reason = reason or ""

    def schedule_reconnect(self, *, delay_s: float, failure_reason: str) -> None:
        self.reconnect_attempts += 1
        self.recovery_attempts_total += 1
        self.last_failure_reason = failure_reason
        self.scheduled_reconnect_at = time.monotonic() + max(0.0, float(delay_s))

    def note_reconnect_failure(self, failure_reason: str) -> None:
        self.recovery_failures_total += 1
        self.last_failure_reason = failure_reason

    def clear_reconnect(self) -> None:
        self.reconnect_attempts = 0
        self.scheduled_reconnect_at = 0.0

    def note_recovery_success(self) -> None:
        self.recovery_successes_total += 1

    def note_xrun(self) -> None:
        self.xrun_events_total += 1

    def note_device_reset(self) -> None:
        self.device_resets_total += 1

    def note_barge_in_result(self, *, success: bool) -> None:
        self.barge_in_attempts_total += 1
        if success:
            self.barge_in_successes_total += 1

    def note_backend_selected(
        self,
        *,
        selected_backend: str,
        fallback_reason: str = "",
        degradation_cause: str = "",
        mode_contract: "AecProductMode | None" = None,
    ) -> None:
        now = time.monotonic()
        if self.selected_backend and self.selected_backend != selected_backend:
            self.backend_switches_total += 1
            self.fallback_chain_step += 1
        self.selected_backend = selected_backend
        self.fallback_reason = fallback_reason
        self.degradation_cause = degradation_cause
        if mode_contract is not None:
            self.product_mode_key = mode_contract.key
            self.product_status = mode_contract.session_status
            self.capture_gate_policy = mode_contract.capture_gate_policy
            self.recovery_policy = mode_contract.recovery_policy
        if selected_backend == "degraded_no_aec":
            self.degraded_transitions_total += 1
        self.last_backend_switch_at = now

    def record_recovery_story(
        self,
        *,
        category: str,
        reason: str,
        action: str,
        session_state: str,
        selected_backend: str | None = None,
        target_backend: str = "",
        cooldown_s: float = 0.0,
        detail: dict[str, object] | None = None,
    ) -> None:
        event = RecoveryStoryEvent(
            mono_ts=time.monotonic(),
            category=str(category),
            reason=str(reason),
            action=str(action),
            session_state=str(session_state),
            selected_backend=str(selected_backend or self.selected_backend),
            target_backend=str(target_backend),
            cooldown_s=float(cooldown_s),
            detail=dict(detail or {}),
        )
        self.recovery_story.append(event)
        if len(self.recovery_story) > 64:
            self.recovery_story = self.recovery_story[-64:]

    def compute_health_score(self) -> float:
        reference_penalty = min(0.35, float(self.reference_consecutive_misses) * 0.03)
        poor_aec_penalty = min(0.35, float(self.poor_aec_consecutive) * 0.04)
        reconnect_penalty = min(0.2, float(self.reconnect_attempts) * 0.04)
        degraded_penalty = 0.2 if self.selected_backend == "degraded_no_aec" else 0.0
        xrun_penalty = min(0.12, float(self.xrun_events_total) * 0.02)
        device_reset_penalty = min(0.18, float(self.device_resets_total) * 0.06)
        score = 1.0 - reference_penalty - poor_aec_penalty - reconnect_penalty - degraded_penalty - xrun_penalty - device_reset_penalty
        return float(max(0.0, min(1.0, score)))

    def note_reference_health(
        self,
        *,
        ready: bool,
        available_samples: int,
        callback_age_ms: int,
    ) -> bool:
        self.reference_ready = bool(ready)
        self.reference_available_samples = max(0, int(available_samples or 0))
        self.reference_callback_age_ms = int(callback_age_ms if callback_age_ms is not None else -1)
        if ready:
            self.reference_ready_events += 1
            self.reference_consecutive_misses = 0
            next_state = "ready"
        else:
            self.reference_miss_events += 1
            self.reference_consecutive_misses += 1
            next_state = "stale" if self.reference_available_samples > 0 else "missing"
        changed = next_state != self.reference_health_state
        self.reference_health_state = next_state
        event = ReferenceHealthEvent(
            mono_ts=time.monotonic(),
            ready=self.reference_ready,
            available_samples=self.reference_available_samples,
            callback_age_ms=self.reference_callback_age_ms,
            state=self.reference_health_state,
        )
        self.reference_health_timeline.append(event)
        if len(self.reference_health_timeline) > 48:
            self.reference_health_timeline = self.reference_health_timeline[-48:]
        return changed

    def note_aec_health(
        self,
        *,
        poor_block: bool,
        reference_miss: bool,
        accepted_backend: bool,
    ) -> None:
        if reference_miss or not accepted_backend:
            self.poor_aec_consecutive = 0
            return
        if poor_block:
            self.poor_aec_events += 1
            self.poor_aec_consecutive += 1
        else:
            self.poor_aec_consecutive = 0

    def note_runtime_backlog(
        self,
        *,
        pending_events: int,
        pending_mic: int,
        pending_player_bytes: int,
        now_monotonic: float | None = None,
    ) -> dict[str, object]:
        now = float(now_monotonic if now_monotonic is not None else time.monotonic())
        self.pending_events = max(0, int(pending_events))
        self.pending_mic_chunks = max(0, int(pending_mic))
        self.pending_player_bytes = max(0, int(pending_player_bytes))
        backlog_log_due = bool(
            now - self.last_backlog_log_at >= 5.0
            and (self.pending_events > 0 or self.pending_mic_chunks > 0 or self.pending_player_bytes > 0)
        )
        if backlog_log_due:
            self.last_backlog_log_at = now
        if self.pending_player_bytes != self.last_player_buffer_bytes:
            self.last_player_progress_at = now
            self.last_player_buffer_bytes = self.pending_player_bytes
        playback_stagnating = bool(
            self.pending_player_bytes > 0 and now - self.last_player_progress_at > 8.0
        )
        if playback_stagnating:
            self.last_player_progress_at = now
            self.last_player_buffer_bytes = 0
            self.pending_player_bytes = 0
        return {
            "backlog_log_due": backlog_log_due,
            "playback_stagnating": playback_stagnating,
        }

    def note_turn_committed(self) -> None:
        self.turns_total += 1
        self.turn_committed_at = time.monotonic()
        self.response_started_at = 0.0
        self.response_first_audio_at = 0.0

    def note_response_started(self) -> None:
        if self.response_started_at <= 0.0:
            self.response_started_at = time.monotonic()

    def note_response_first_audio(self) -> int | None:
        if self.response_first_audio_at > 0.0:
            return None
        self.response_first_audio_at = time.monotonic()
        if self.response_started_at <= 0.0:
            return None
        latency_ms = int((self.response_first_audio_at - self.response_started_at) * 1000)
        self.response_first_audio_latency_total_ms += latency_ms
        self.response_first_audio_latency_max_ms = max(self.response_first_audio_latency_max_ms, latency_ms)
        return latency_ms

    def note_response_done(self) -> int | None:
        self.responses_completed_total += 1
        latency_ms = None
        if self.turn_committed_at > 0.0:
            latency_ms = int((time.monotonic() - self.turn_committed_at) * 1000)
            self.response_total_latency_total_ms += latency_ms
            self.response_total_latency_max_ms = max(self.response_total_latency_max_ms, latency_ms)
        self.turn_committed_at = 0.0
        self.response_started_at = 0.0
        self.response_first_audio_at = 0.0
        return latency_ms

    def serializable_snapshot(self, *, session_state: str) -> SessionTelemetrySnapshot:
        now = time.monotonic()
        reference_timeline = [event.to_log_payload() for event in self.reference_health_timeline]
        recovery_story = [event.to_log_payload() for event in self.recovery_story]
        timings = {
            "session_start_mono": round(float(self.session_started_at or 0.0), 6),
            "session_probe_start_mono": round(float(self.session_probe_started_at or 0.0), 6),
            "session_probe_done_mono": round(float(self.session_probe_completed_at or 0.0), 6),
            "session_active_mono": round(float(self.session_activated_at or 0.0), 6),
            "session_stop_mono": round(float(self.session_stopped_at or 0.0), 6),
            "uptime_s": round(max(0.0, now - self.session_started_at) if self.session_started_at else 0.0, 3),
            "active_for_s": round(max(0.0, now - self.session_activated_at) if self.session_activated_at else 0.0, 3),
            "probe_duration_s": round(max(0.0, self.session_probe_completed_at - self.session_probe_started_at) if self.session_probe_started_at and self.session_probe_completed_at else 0.0, 3),
            "last_server_activity_age_s": round(max(0.0, now - self.last_server_activity_at) if self.last_server_activity_at else 0.0, 3),
        }
        turn_latency = {
            "turns_total": int(self.turns_total),
            "responses_completed_total": int(self.responses_completed_total),
            "avg_first_audio_latency_ms": round(
                float(self.response_first_audio_latency_total_ms) / float(self.responses_completed_total)
                if self.responses_completed_total > 0
                else 0.0,
                3,
            ),
            "max_first_audio_latency_ms": int(self.response_first_audio_latency_max_ms),
            "avg_response_latency_ms": round(
                float(self.response_total_latency_total_ms) / float(self.responses_completed_total)
                if self.responses_completed_total > 0
                else 0.0,
                3,
            ),
            "max_response_latency_ms": int(self.response_total_latency_max_ms),
        }
        health = {
            "reference_ready": bool(self.reference_ready),
            "reference_health_state": self.reference_health_state,
            "reference_available_samples": int(self.reference_available_samples),
            "reference_callback_age_ms": int(self.reference_callback_age_ms),
            "reference_ready_events": int(self.reference_ready_events),
            "reference_miss_events": int(self.reference_miss_events),
            "reference_consecutive_misses": int(self.reference_consecutive_misses),
            "poor_aec_events": int(self.poor_aec_events),
            "poor_aec_consecutive": int(self.poor_aec_consecutive),
            "last_failure_reason": self.last_failure_reason,
            "pending_events": int(self.pending_events),
            "pending_mic_chunks": int(self.pending_mic_chunks),
            "pending_player_bytes": int(self.pending_player_bytes),
        }
        return SessionTelemetrySnapshot(
            session_state=session_state,
            requested_backend=self.requested_backend,
            selected_backend=self.selected_backend,
            product_mode_key=self.product_mode_key,
            product_status=self.product_status,
            capture_gate_policy=self.capture_gate_policy,
            recovery_policy=self.recovery_policy,
            fallback_chain_step=self.fallback_chain_step,
            fallback_reason=self.fallback_reason,
            degradation_cause=self.degradation_cause,
            reconnect_attempts=self.reconnect_attempts,
            recovery_attempts_total=self.recovery_attempts_total,
            recovery_successes_total=self.recovery_successes_total,
            recovery_failures_total=self.recovery_failures_total,
            xrun_events_total=self.xrun_events_total,
            device_resets_total=self.device_resets_total,
            reference_health_timeline=reference_timeline,
            recovery_story=recovery_story,
            backend_health_score=self.compute_health_score(),
            timings=timings,
            turn_latency=turn_latency,
            health=health,
        )

    def snapshot(self, *, session_state: str) -> SessionHealth:
        now = time.monotonic()
        started_age = max(0.0, now - self.session_started_at) if self.session_started_at else 0.0
        active_age = max(0.0, now - self.session_activated_at) if self.session_activated_at else 0.0
        server_idle = max(0.0, now - self.last_server_activity_at) if self.last_server_activity_at else 0.0
        backend_health = BackendHealthSnapshot(
            backend=self.selected_backend,
            health_score=self.compute_health_score(),
            requested_backend=self.requested_backend,
            audio_mode=self.audio_mode,
            product_mode_key=self.product_mode_key,
            product_status=self.product_status,
            capture_gate_policy=self.capture_gate_policy,
            recovery_policy=self.recovery_policy,
            reference_ready=self.reference_ready,
            reference_available_samples=self.reference_available_samples,
            reference_callback_age_ms=self.reference_callback_age_ms,
            reference_health_state=self.reference_health_state,
            poor_aec_events=self.poor_aec_events,
            poor_aec_consecutive=self.poor_aec_consecutive,
            fallback_reason=self.fallback_reason,
            degradation_cause=self.degradation_cause,
            last_failure_reason=self.last_failure_reason,
            barge_in_success_ratio=(
                float(self.barge_in_successes_total) / float(self.barge_in_attempts_total)
                if self.barge_in_attempts_total > 0
                else 0.0
            ),
            recoveries=self.recovery_successes_total,
            xruns=self.xrun_events_total,
            device_resets=self.device_resets_total,
        )
        return SessionHealth(
            requested_backend=self.requested_backend,
            selected_backend=self.selected_backend,
            fallback_reason=self.fallback_reason,
            degradation_cause=self.degradation_cause,
            device_fingerprint=self.device_fingerprint,
            audio_mode=self.audio_mode,
            product_mode_key=self.product_mode_key,
            product_status=self.product_status,
            capture_gate_policy=self.capture_gate_policy,
            recovery_policy=self.recovery_policy,
            session_state=session_state,
            session_started_at_mono=self.session_started_at,
            session_activated_at_mono=self.session_activated_at,
            uptime_s=started_age,
            active_for_s=active_age,
            last_server_activity_age_s=server_idle,
            reference_ready=self.reference_ready,
            reference_health=self.reference_health_state,
            reference_available_samples=self.reference_available_samples,
            reference_callback_age_ms=self.reference_callback_age_ms,
            reference_ready_events=self.reference_ready_events,
            reference_miss_events=self.reference_miss_events,
            reference_consecutive_misses=self.reference_consecutive_misses,
            poor_aec_events=self.poor_aec_events,
            poor_aec_consecutive=self.poor_aec_consecutive,
            recovery_attempts_scheduled=self.reconnect_attempts,
            recovery_attempts_total=self.recovery_attempts_total,
            recovery_successes_total=self.recovery_successes_total,
            degraded_transitions_total=self.degraded_transitions_total,
            backend_switches_total=self.backend_switches_total,
            xrun_events_total=self.xrun_events_total,
            device_resets_total=self.device_resets_total,
            barge_in_attempts_total=self.barge_in_attempts_total,
            barge_in_successes_total=self.barge_in_successes_total,
            health_score=self.compute_health_score(),
            next_reconnect_at_mono=self.scheduled_reconnect_at,
            last_failure_reason=self.last_failure_reason,
            backend_health=backend_health,
        )
