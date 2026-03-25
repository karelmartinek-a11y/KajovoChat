from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .telemetry import AudioTelemetry


class FailureReason(str, Enum):
    TRANSPORT_DISCONNECT = "transport_disconnect"
    TRANSPORT_TIMEOUT = "transport_timeout"
    DEVICE_UNAVAILABLE = "device_unavailable"
    WINDOWS_SYSTEM_AEC_UNAVAILABLE = "windows_system_aec_unavailable"
    WINDOWS_SYSTEM_AEC_UNHEALTHY = "windows_system_aec_unhealthy"
    WEBRTC_APM_UNAVAILABLE = "webrtc_apm_unavailable"
    REFERENCE_PIPELINE_UNHEALTHY = "reference_pipeline_unhealthy"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    SESSION_START_FAILED = "session_start_failed"
    USER_STOP = "user_stop"
    UNKNOWN = "unknown"


@dataclass
class RecoverySupervisor:
    telemetry: AudioTelemetry
    transport: object
    mode_supplier: Callable[[], str]
    state_sink: Callable[[str], None]
    caption_sink: Callable[[str], None]
    log_sink: Callable[[str, object], None]
    stop_session: Callable[[], None]
    error_sink: Callable[[str], None]
    selected_backend_supplier: Callable[[], str]
    fallback_handler: Callable[[str], bool]
    restore_session_state: Callable[[str], None]

    def __post_init__(self) -> None:
        self._last_fallback_at = 0.0
        self._last_fallback_reason = ""

    def schedule(self, reason: str, failure_reason: str) -> None:
        if self.telemetry.scheduled_reconnect_at and time.monotonic() < self.telemetry.scheduled_reconnect_at:
            return
        delay = min(8.0, 0.8 * (2 ** max(0, self.telemetry.reconnect_attempts)))
        self.telemetry.schedule_reconnect(delay_s=delay, failure_reason=failure_reason)
        self.caption_sink(f"Realtime: plánuji reconnect za {delay:.1f} s")
        self.log_sink(
            "reconnect_scheduled",
            {
                "reason": reason,
                "failure_reason": failure_reason,
                "attempt": self.telemetry.reconnect_attempts,
                "delay_s": delay,
                "selected_backend": self.selected_backend_supplier(),
            },
        )
        self.transport.close()
        self.state_sink("reconnecting")

    def request_fallback(self, reason: str) -> bool:
        now = time.monotonic()
        if reason == self._last_fallback_reason and now - self._last_fallback_at < 8.0:
            self.log_sink(
                "audio_backend_fallback_suppressed",
                {"reason": reason, "selected_backend": self.selected_backend_supplier(), "cooldown_s": round(8.0 - (now - self._last_fallback_at), 3)},
            )
            return False
        changed = self.fallback_handler(reason)
        if changed:
            self._last_fallback_at = now
            self._last_fallback_reason = reason
        return changed

    def note_xrun(self, *, source: str = "runtime") -> None:
        self.telemetry.note_xrun()
        self.log_sink(
            "audio_xrun",
            {
                "source": source,
                "selected_backend": self.selected_backend_supplier(),
                "xrun_events_total": self.telemetry.xrun_events_total,
            },
        )
        if self.telemetry.xrun_events_total >= 3:
            self.request_fallback(FailureReason.DEVICE_UNAVAILABLE.value)

    def note_device_reset(self, *, source: str = "runtime") -> None:
        self.telemetry.note_device_reset()
        self.telemetry.last_failure_reason = FailureReason.DEVICE_UNAVAILABLE.value
        self.log_sink(
            "audio_device_reset",
            {
                "source": source,
                "selected_backend": self.selected_backend_supplier(),
                "device_resets_total": self.telemetry.device_resets_total,
                "failure_reason": FailureReason.DEVICE_UNAVAILABLE.value,
            },
        )
        if self.telemetry.device_resets_total >= 2:
            self.request_fallback(FailureReason.DEVICE_UNAVAILABLE.value)

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

    def tick(self) -> None:
        if self.mode_supplier() == "idle":
            return
        rt = self.transport.realtime
        if rt is not None and rt.is_connected:
            return
        if self.telemetry.scheduled_reconnect_at and time.monotonic() < self.telemetry.scheduled_reconnect_at:
            return
        try:
            self.caption_sink("Realtime: obnovuji spojení…")
            self.transport.ensure_connected(self.transport.turn_mode, self.telemetry.reconnect_attempts)
            self.log_sink(
                "reconnect_ok",
                {
                    "attempt": self.telemetry.reconnect_attempts,
                    "selected_backend": self.selected_backend_supplier(),
                    "failure_reason": self.telemetry.last_failure_reason,
                },
            )
            self.telemetry.note_recovery_success()
            self.telemetry.clear_reconnect()
            self.restore_session_state("reconnect_ok")
            if self.mode_supplier() == "handsfree":
                self.state_sink("listening")
        except Exception as exc:
            self.log_sink(
                "reconnect_failed",
                {
                    "message": str(exc),
                    "attempt": self.telemetry.reconnect_attempts,
                    "selected_backend": self.selected_backend_supplier(),
                },
            )
            if self.telemetry.reconnect_attempts >= 5:
                self.log_sink(
                    "recovery_exhausted",
                    {
                        "message": str(exc),
                        "attempt": self.telemetry.reconnect_attempts,
                        "selected_backend": self.selected_backend_supplier(),
                        "failure_reason": FailureReason.RECOVERY_EXHAUSTED.value,
                    },
                )
                self.telemetry.last_failure_reason = FailureReason.RECOVERY_EXHAUSTED.value
                self.stop_session()
                self.error_sink(f"Realtime se nepodařilo obnovit: {exc}")
                return
            self.schedule(str(exc), FailureReason.TRANSPORT_DISCONNECT.value)
