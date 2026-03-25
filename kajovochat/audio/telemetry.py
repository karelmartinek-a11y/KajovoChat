from __future__ import annotations

import time
from dataclasses import dataclass, field

from .contracts import BackendHealthSnapshot, SessionHealth


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


@dataclass
class AudioTelemetry:
    reconnect_attempts: int = 0
    scheduled_reconnect_at: float = 0.0
    last_server_activity_at: float = field(default_factory=time.monotonic)
    session_started_at: float = 0.0
    session_activated_at: float = 0.0
    session_stopped_at: float = 0.0
    requested_backend: str = "windows_system_aec"
    selected_backend: str = "windows_system_aec"
    fallback_reason: str = ""
    degradation_cause: str = ""
    device_fingerprint: str = "unknown"
    audio_mode: str = "notebook_builtin"
    last_failure_reason: str = ""
    recovery_attempts_total: int = 0
    reference_ready: bool = False
    reference_available_samples: int = 0
    reference_callback_age_ms: int = -1
    reference_health_state: str = "unknown"
    reference_ready_events: int = 0
    reference_miss_events: int = 0
    reference_consecutive_misses: int = 0
    last_backend_switch_at: float = 0.0
    poor_aec_events: int = 0
    poor_aec_consecutive: int = 0
    recovery_successes_total: int = 0
    degraded_transitions_total: int = 0
    backend_switches_total: int = 0
    xrun_events_total: int = 0
    device_resets_total: int = 0
    barge_in_attempts_total: int = 0
    barge_in_successes_total: int = 0

    def mark_session_started(self, *, requested_backend: str, device_fingerprint: str, audio_mode: str) -> None:
        now = time.monotonic()
        self.session_started_at = now
        self.session_activated_at = 0.0
        self.session_stopped_at = 0.0
        self.last_server_activity_at = now
        self.reconnect_attempts = 0
        self.scheduled_reconnect_at = 0.0
        self.requested_backend = requested_backend
        self.selected_backend = requested_backend
        self.fallback_reason = ""
        self.degradation_cause = ""
        self.device_fingerprint = device_fingerprint or "unknown"
        self.audio_mode = audio_mode or "notebook_builtin"
        self.last_failure_reason = ""
        self.reference_ready = False
        self.reference_available_samples = 0
        self.reference_callback_age_ms = -1
        self.reference_health_state = "starting"
        self.reference_ready_events = 0
        self.reference_miss_events = 0
        self.reference_consecutive_misses = 0
        self.last_backend_switch_at = now
        self.poor_aec_events = 0
        self.poor_aec_consecutive = 0
        self.recovery_successes_total = 0
        self.degraded_transitions_total = 0
        self.backend_switches_total = 0
        self.xrun_events_total = 0
        self.device_resets_total = 0
        self.barge_in_attempts_total = 0
        self.barge_in_successes_total = 0

    def mark_session_activated(self) -> None:
        self.session_activated_at = time.monotonic()

    def mark_session_stopped(self) -> None:
        self.session_stopped_at = time.monotonic()

    def note_server_activity(self) -> None:
        self.last_server_activity_at = time.monotonic()

    def schedule_reconnect(self, *, delay_s: float, failure_reason: str) -> None:
        self.reconnect_attempts += 1
        self.recovery_attempts_total += 1
        self.last_failure_reason = failure_reason
        self.scheduled_reconnect_at = time.monotonic() + max(0.0, float(delay_s))

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
    ) -> None:
        if self.selected_backend and self.selected_backend != selected_backend:
            self.backend_switches_total += 1
        self.selected_backend = selected_backend
        self.fallback_reason = fallback_reason
        self.degradation_cause = degradation_cause
        if selected_backend == "degraded_no_aec":
            self.degraded_transitions_total += 1
        self.last_backend_switch_at = time.monotonic()

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
        self.reference_callback_age_ms = int(callback_age_ms or -1)
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
        return changed

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
