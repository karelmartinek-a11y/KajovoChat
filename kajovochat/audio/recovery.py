from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .telemetry import AudioTelemetry


class FailureReason(str, Enum):
    TRANSPORT_DISCONNECT = "transport_disconnect"
    TRANSPORT_TIMEOUT = "transport_timeout"
    PLAYBACK_STAGNATION = "playback_stagnation"
    DEVICE_UNAVAILABLE = "device_unavailable"
    WINDOWS_SYSTEM_AEC_UNAVAILABLE = "windows_system_aec_unavailable"
    WINDOWS_SYSTEM_AEC_UNHEALTHY = "windows_system_aec_unhealthy"
    WEBRTC_APM_UNAVAILABLE = "webrtc_apm_unavailable"
    REFERENCE_PIPELINE_UNHEALTHY = "reference_pipeline_unhealthy"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    SESSION_START_FAILED = "session_start_failed"
    USER_STOP = "user_stop"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RecoveryPolicy:
    action: str
    cooldown_s: float
    target_state: str


@dataclass
class RecoverySupervisor:
    telemetry: AudioTelemetry
    transport: object
    mode_supplier: Callable[[], str]
    state_sink: Callable[[str], None]
    caption_sink: Callable[[str], None]
    log_sink: Callable[[str, object], None]
    enter_recovering: Callable[[str], None]
    stop_session: Callable[[], None]
    fail_session: Callable[[str, str], None]
    error_sink: Callable[[str], None]
    selected_backend_supplier: Callable[[], str]
    fallback_handler: Callable[[str], bool]
    restore_session_state: Callable[[str], None]
    stop_playback: Callable[[], None]
    _last_fallback_at: float = field(init=False, default=0.0)
    _last_fallback_reason: str = field(init=False, default="")
    _last_reconnect_at: float = field(init=False, default=0.0)
    _last_reconnect_reason: str = field(init=False, default="")

    REFERENCE_MISS_THRESHOLD = 12
    REFERENCE_STALE_AGE_MS = 120
    POOR_NATIVE_AEC_THRESHOLD = 6
    XRUN_FALLBACK_THRESHOLD = 3
    DEVICE_RESET_FALLBACK_THRESHOLD = 2
    TRANSPORT_IDLE_TIMEOUT_S = 25.0
    BACKLOG_LOG_PERIOD_S = 5.0
    PLAYBACK_STAGNATION_S = 8.0
    FALLBACK_COOLDOWN_S = 8.0
    RECONNECT_GUARD_S = 1.0
    MAX_RECONNECT_ATTEMPTS = 5

    POLICIES: dict[str, RecoveryPolicy] = field(
        init=False,
        default_factory=lambda: {
            FailureReason.TRANSPORT_DISCONNECT.value: RecoveryPolicy("transport_reconnect", 0.8, "recovering"),
            FailureReason.TRANSPORT_TIMEOUT.value: RecoveryPolicy("transport_reconnect", 0.8, "recovering"),
            FailureReason.PLAYBACK_STAGNATION.value: RecoveryPolicy("transport_reconnect", 1.5, "recovering"),
            FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value: RecoveryPolicy("backend_fallback", 8.0, "recovering"),
            FailureReason.WINDOWS_SYSTEM_AEC_UNHEALTHY.value: RecoveryPolicy("backend_fallback", 8.0, "recovering"),
            FailureReason.WINDOWS_SYSTEM_AEC_UNAVAILABLE.value: RecoveryPolicy("backend_fallback", 8.0, "recovering"),
            FailureReason.WEBRTC_APM_UNAVAILABLE.value: RecoveryPolicy("backend_fallback", 8.0, "recovering"),
            FailureReason.DEVICE_UNAVAILABLE.value: RecoveryPolicy("backend_fallback", 8.0, "recovering"),
            FailureReason.SESSION_START_FAILED.value: RecoveryPolicy("fail_session", 0.0, "failed"),
            FailureReason.RECOVERY_EXHAUSTED.value: RecoveryPolicy("fail_session", 0.0, "failed"),
            FailureReason.UNKNOWN.value: RecoveryPolicy("fail_session", 0.0, "failed"),
        },
    )

    def classify_failure_reason(self, message: str) -> str:
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

    def record_probe_outcome(self, *, requested_backend: str, selected_backend: str, fallback_reason: str, degraded: bool) -> None:
        action = "probe_selected"
        if selected_backend != requested_backend:
            action = "probe_fallback"
        if degraded:
            action = "probe_degraded"
        self.telemetry.record_recovery_story(
            category="backend_probe",
            reason=fallback_reason or "backend_probe",
            action=action,
            session_state="probing",
            selected_backend=requested_backend,
            target_backend=selected_backend,
            cooldown_s=0.0,
            detail={"degraded": bool(degraded)},
        )

    def observe_reference_health(
        self,
        *,
        ready: bool,
        available_samples: int,
        callback_age_ms: int,
        session_state: str,
    ) -> None:
        changed = self.telemetry.note_reference_health(
            ready=ready,
            available_samples=available_samples,
            callback_age_ms=callback_age_ms,
        )
        if changed:
            self.log_sink(
                "audio_reference_health",
                {
                    "ready": bool(ready),
                    "available_samples": int(available_samples),
                    "callback_age_ms": int(callback_age_ms),
                    "reference_health_state": self.telemetry.reference_health_state,
                    "selected_backend": self.selected_backend_supplier(),
                },
            )
        unhealthy = (
            self.telemetry.selected_backend in {"windows_system_aec", "webrtc_apm"}
            and not ready
            and self.telemetry.reference_consecutive_misses >= self.REFERENCE_MISS_THRESHOLD
            and callback_age_ms >= self.REFERENCE_STALE_AGE_MS
        )
        if unhealthy:
            self.handle_failure(
                FailureReason.REFERENCE_PIPELINE_UNHEALTHY.value,
                session_state=session_state,
                message="Reference pipeline není zdravá.",
                detail={
                    "reference_consecutive_misses": self.telemetry.reference_consecutive_misses,
                    "callback_age_ms": int(callback_age_ms),
                },
            )

    def observe_aec_health(
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
        session_state: str,
    ) -> None:
        selected_backend = self.telemetry.selected_backend
        accepted_backend = selected_backend == "windows_system_aec" and backend == "windows_system_aec" and not webrtc_success
        delay_error = abs(int(delay_samples or 0) - int(calibration_latency or 0))
        has_reported_native_metrics = any(
            float(value or 0.0) > 0.0
            for value in (aec_quality, improvement_ratio, similarity)
        ) or int(delay_samples or 0) > 0
        poor_native_block = bool(
            accepted_backend
            and has_reported_native_metrics
            and aec_quality < 0.025
            and improvement_ratio < 0.14
            and similarity < 0.5
            and delay_error >= 180
        )
        self.telemetry.note_aec_health(
            poor_block=poor_native_block,
            reference_miss=reference_miss,
            accepted_backend=accepted_backend,
        )
        if self.telemetry.poor_aec_consecutive >= self.POOR_NATIVE_AEC_THRESHOLD:
            self.handle_failure(
                FailureReason.WINDOWS_SYSTEM_AEC_UNHEALTHY.value,
                session_state=session_state,
                message="Windows System AEC zůstává nestabilní.",
                detail={
                    "poor_aec_consecutive": self.telemetry.poor_aec_consecutive,
                    "delay_error": int(delay_error),
                },
            )

    def observe_runtime_health(
        self,
        *,
        session_state: str,
        presentation_state: str,
        pending_events: int,
        pending_mic: int,
        pending_player_bytes: int,
        transport_connected: bool,
    ) -> None:
        now = time.monotonic()
        runtime_flags = self.telemetry.note_runtime_backlog(
            pending_events=pending_events,
            pending_mic=pending_mic,
            pending_player_bytes=pending_player_bytes,
            now_monotonic=now,
        )
        if runtime_flags["backlog_log_due"]:
            self.log_sink(
                "backlog",
                {
                    "rt_events": int(self.telemetry.pending_events),
                    "mic_chunks": int(self.telemetry.pending_mic_chunks),
                    "player_bytes": int(self.telemetry.pending_player_bytes),
                },
            )
        if runtime_flags["playback_stagnating"]:
            self.log_sink("watchdog", {"message": "audio playback stagnuje", "buffered_bytes": int(pending_player_bytes)})
            try:
                self.stop_playback()
            except Exception:
                pass
            self.handle_failure(
                FailureReason.PLAYBACK_STAGNATION.value,
                session_state=session_state,
                message="Audio playback stagnuje.",
                detail={"buffered_bytes": int(pending_player_bytes)},
            )
            return
        if (
            self.mode_supplier() != "idle"
            and transport_connected
            and session_state in {"active", "degraded", "recovering"}
            and presentation_state in {"transcribing", "thinking"}
            and time.monotonic() - self.telemetry.last_server_activity_at > self.TRANSPORT_IDLE_TIMEOUT_S
        ):
            self.log_sink("watchdog", {"message": "realtime bez aktivity", "state": presentation_state})
            self.handle_failure(
                FailureReason.TRANSPORT_TIMEOUT.value,
                session_state=session_state,
                message="Watchdog: realtime bez aktivity.",
                detail={"presentation_state": presentation_state},
            )

    def note_xrun(self, *, source: str = "runtime", session_state: str) -> None:
        self.telemetry.note_xrun()
        self.log_sink(
            "audio_xrun",
            {
                "source": source,
                "selected_backend": self.selected_backend_supplier(),
                "xrun_events_total": self.telemetry.xrun_events_total,
            },
        )
        if self.telemetry.xrun_events_total >= self.XRUN_FALLBACK_THRESHOLD:
            self.handle_failure(
                FailureReason.DEVICE_UNAVAILABLE.value,
                session_state=session_state,
                message="XRUN eskaloval do fallbacku.",
                detail={"source": source, "xrun_events_total": self.telemetry.xrun_events_total},
            )

    def note_device_reset(self, *, source: str = "runtime", session_state: str) -> None:
        self.telemetry.note_device_reset()
        self.telemetry.set_last_failure_reason(FailureReason.DEVICE_UNAVAILABLE.value)
        self.log_sink(
            "audio_device_reset",
            {
                "source": source,
                "selected_backend": self.selected_backend_supplier(),
                "device_resets_total": self.telemetry.device_resets_total,
                "failure_reason": FailureReason.DEVICE_UNAVAILABLE.value,
            },
        )
        if self.telemetry.device_resets_total >= self.DEVICE_RESET_FALLBACK_THRESHOLD:
            self.handle_failure(
                FailureReason.DEVICE_UNAVAILABLE.value,
                session_state=session_state,
                message="Device reset eskaloval do fallbacku.",
                detail={"source": source, "device_resets_total": self.telemetry.device_resets_total},
            )

    def note_barge_in_result(self, *, success: bool, reason: str = "") -> None:
        self.telemetry.note_barge_in_result(success=success)
        self.log_sink(
            "audio_barge_in_result",
            {
                "success": bool(success),
                "reason": reason,
                "selected_backend": self.selected_backend_supplier(),
                "barge_in_attempts_total": self.telemetry.barge_in_attempts_total,
                "barge_in_successes_total": self.telemetry.barge_in_successes_total,
            },
        )

    def handle_transport_error(self, message: str, *, session_state: str) -> None:
        self.handle_failure(
            self.classify_failure_reason(message),
            session_state=session_state,
            message=message,
        )

    def handle_failure(
        self,
        reason: str,
        *,
        session_state: str,
        message: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        failure_reason = str(reason or FailureReason.UNKNOWN.value)
        self.telemetry.set_last_failure_reason(failure_reason)
        policy = self.POLICIES.get(failure_reason, self.POLICIES[FailureReason.UNKNOWN.value])
        payload = {
            "message": message,
            "failure_reason": failure_reason,
            "recoverable": policy.action in {"transport_reconnect", "backend_fallback"},
            "recovery_action": policy.action,
            "recovery_target_state": policy.target_state,
            "cooldown_s": float(policy.cooldown_s),
        }
        if detail:
            payload.update(detail)
        self.log_sink("audio_session_error", payload)
        if policy.action == "transport_reconnect":
            self._schedule_transport_reconnect(
                reason=failure_reason,
                message=message,
                session_state=session_state,
                cooldown_s=policy.cooldown_s,
                detail=detail or {},
            )
            return
        if policy.action == "backend_fallback":
            self._request_backend_fallback(
                reason=failure_reason,
                message=message,
                session_state=session_state,
                cooldown_s=policy.cooldown_s,
                detail=detail or {},
            )
            return
        self.telemetry.record_recovery_story(
            category="failure",
            reason=failure_reason,
            action="fail_session",
            session_state=policy.target_state,
            selected_backend=self.selected_backend_supplier(),
            cooldown_s=policy.cooldown_s,
            detail=detail or {"message": message},
        )
        self.fail_session(message, failure_reason)

    def _request_backend_fallback(
        self,
        *,
        reason: str,
        message: str,
        session_state: str,
        cooldown_s: float,
        detail: dict[str, object],
    ) -> None:
        now = time.monotonic()
        if reason == self._last_fallback_reason and now - self._last_fallback_at < cooldown_s:
            self.log_sink(
                "audio_backend_fallback_suppressed",
                {
                    "reason": reason,
                    "selected_backend": self.selected_backend_supplier(),
                    "cooldown_s": round(cooldown_s - (now - self._last_fallback_at), 3),
                },
            )
            self.telemetry.record_recovery_story(
                category="fallback",
                reason=reason,
                action="suppressed",
                session_state=session_state,
                selected_backend=self.selected_backend_supplier(),
                cooldown_s=cooldown_s,
                detail=detail,
            )
            return
        self.enter_recovering(reason)
        before_backend = self.selected_backend_supplier()
        changed = self.fallback_handler(reason)
        if changed:
            self._last_fallback_at = now
            self._last_fallback_reason = reason
            self.telemetry.record_recovery_story(
                category="fallback",
                reason=reason,
                action="backend_fallback",
                session_state="recovering",
                selected_backend=before_backend,
                target_backend=self.selected_backend_supplier(),
                cooldown_s=cooldown_s,
                detail=detail or {"message": message},
            )
            return
        self.telemetry.record_recovery_story(
            category="fallback",
            reason=reason,
            action="fallback_unavailable",
            session_state="failed",
            selected_backend=before_backend,
            cooldown_s=cooldown_s,
            detail=detail or {"message": message},
        )
        self.fail_session(message, reason)

    def _schedule_transport_reconnect(
        self,
        *,
        reason: str,
        message: str,
        session_state: str,
        cooldown_s: float,
        detail: dict[str, object],
    ) -> None:
        now = time.monotonic()
        if self.telemetry.scheduled_reconnect_at and now < self.telemetry.scheduled_reconnect_at:
            self.log_sink(
                "reconnect_suppressed",
                {
                    "reason": reason,
                    "selected_backend": self.selected_backend_supplier(),
                    "scheduled_reconnect_at": self.telemetry.scheduled_reconnect_at,
                },
            )
            return
        if reason == self._last_reconnect_reason and now - self._last_reconnect_at < max(self.RECONNECT_GUARD_S, cooldown_s):
            self.log_sink(
                "reconnect_suppressed",
                {
                    "reason": reason,
                    "selected_backend": self.selected_backend_supplier(),
                    "cooldown_s": round(max(self.RECONNECT_GUARD_S, cooldown_s) - (now - self._last_reconnect_at), 3),
                },
            )
            self.telemetry.record_recovery_story(
                category="transport_reconnect",
                reason=reason,
                action="suppressed",
                session_state=session_state,
                selected_backend=self.selected_backend_supplier(),
                cooldown_s=max(self.RECONNECT_GUARD_S, cooldown_s),
                detail=detail,
            )
            return
        delay = min(8.0, max(cooldown_s, 0.8 * (2 ** max(0, self.telemetry.reconnect_attempts))))
        self.enter_recovering(reason)
        self.telemetry.schedule_reconnect(delay_s=delay, failure_reason=reason)
        self.caption_sink(f"Realtime: plánuji reconnect za {delay:.1f} s")
        self.log_sink(
            "reconnect_scheduled",
            {
                "reason": message,
                "failure_reason": reason,
                "attempt": self.telemetry.reconnect_attempts,
                "delay_s": delay,
                "selected_backend": self.selected_backend_supplier(),
            },
        )
        self.telemetry.record_recovery_story(
            category="transport_reconnect",
            reason=reason,
            action="scheduled",
            session_state="recovering",
            selected_backend=self.selected_backend_supplier(),
            cooldown_s=delay,
            detail=detail or {"message": message},
        )
        self._last_reconnect_at = now
        self._last_reconnect_reason = reason
        self.transport.close()

    def tick(self) -> None:
        if self.mode_supplier() == "idle":
            return
        realtime = getattr(self.transport, "realtime", None)
        if realtime is not None and getattr(realtime, "is_connected", False):
            return
        if self.telemetry.scheduled_reconnect_at and time.monotonic() < self.telemetry.scheduled_reconnect_at:
            return
        try:
            self.caption_sink("Realtime: obnovuji spojení…")
            self.transport.ensure_connected(getattr(self.transport, "turn_mode", "semantic_vad"), self.telemetry.reconnect_attempts)
            self.log_sink(
                "reconnect_ok",
                {
                    "attempt": self.telemetry.reconnect_attempts,
                    "selected_backend": self.selected_backend_supplier(),
                    "failure_reason": self.telemetry.last_failure_reason,
                },
            )
            self.telemetry.note_recovery_success()
            self.telemetry.record_recovery_story(
                category="transport_reconnect",
                reason=self.telemetry.last_failure_reason or FailureReason.TRANSPORT_DISCONNECT.value,
                action="success",
                session_state="recovering",
                selected_backend=self.selected_backend_supplier(),
            )
            self.telemetry.clear_reconnect()
            self.restore_session_state("reconnect_ok")
        except Exception as exc:
            failure_reason = FailureReason.TRANSPORT_DISCONNECT.value
            self.telemetry.note_reconnect_failure(failure_reason)
            self.log_sink(
                "reconnect_failed",
                {
                    "message": str(exc),
                    "attempt": self.telemetry.reconnect_attempts,
                    "selected_backend": self.selected_backend_supplier(),
                },
            )
            self.telemetry.record_recovery_story(
                category="transport_reconnect",
                reason=failure_reason,
                action="failed",
                session_state="recovering",
                selected_backend=self.selected_backend_supplier(),
                detail={"message": str(exc), "attempt": self.telemetry.reconnect_attempts},
            )
            if self.telemetry.reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
                self.log_sink(
                    "recovery_exhausted",
                    {
                        "message": str(exc),
                        "attempt": self.telemetry.reconnect_attempts,
                        "selected_backend": self.selected_backend_supplier(),
                        "failure_reason": FailureReason.RECOVERY_EXHAUSTED.value,
                    },
                )
                self.telemetry.set_last_failure_reason(FailureReason.RECOVERY_EXHAUSTED.value)
                self.telemetry.record_recovery_story(
                    category="transport_reconnect",
                    reason=FailureReason.RECOVERY_EXHAUSTED.value,
                    action="exhausted",
                    session_state="failed",
                    selected_backend=self.selected_backend_supplier(),
                    detail={"message": str(exc)},
                )
                self.stop_session()
                self.error_sink(f"Realtime se nepodařilo obnovit: {exc}")
                return
            self._schedule_transport_reconnect(
                reason=failure_reason,
                message=str(exc),
                session_state="recovering",
                cooldown_s=self.POLICIES[failure_reason].cooldown_s,
                detail={"attempt": self.telemetry.reconnect_attempts},
            )
