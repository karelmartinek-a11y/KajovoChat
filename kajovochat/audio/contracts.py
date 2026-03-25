from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CaptureFrame:
    frame_index: int
    mono_ns: int
    raw_mic_pcm16: bytes
    processed_mic_pcm16: bytes
    render_ref_pcm16: Optional[bytes]
    sample_rate: int
    channels: int
    aec_backend: str
    aec_quality: float
    residual_level: float
    vad_probability: float
    double_talk: bool
    stream_delay_ms: int
    device_clock_locked: bool

    def to_log_payload(self) -> dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "mono_ns": int(self.mono_ns),
            "raw_mic_bytes": len(self.raw_mic_pcm16),
            "processed_mic_bytes": len(self.processed_mic_pcm16),
            "render_ref_bytes": len(self.render_ref_pcm16 or b""),
            "sample_rate": int(self.sample_rate),
            "channels": int(self.channels),
            "aec_backend": self.aec_backend,
            "aec_quality": float(self.aec_quality),
            "residual_level": float(self.residual_level),
            "vad_probability": float(self.vad_probability),
            "double_talk": bool(self.double_talk),
            "stream_delay_ms": int(self.stream_delay_ms),
            "device_clock_locked": bool(self.device_clock_locked),
        }


@dataclass(frozen=True)
class RenderFrame:
    frame_index: int
    mono_ns: int
    pcm16: bytes
    tts_active: bool
    prompted_by_assistant_turn: Optional[str] = None

    def to_log_payload(self) -> dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "mono_ns": int(self.mono_ns),
            "pcm_bytes": len(self.pcm16),
            "tts_active": bool(self.tts_active),
            "prompted_by_assistant_turn": self.prompted_by_assistant_turn,
        }


@dataclass(frozen=True)
class BackendHealthSnapshot:
    backend: str
    health_score: float = 0.0
    requested_backend: str = ""
    audio_mode: str = ""
    reference_ready: bool = False
    reference_available_samples: int = 0
    reference_callback_age_ms: int = -1
    reference_health_state: str = "unknown"
    poor_aec_events: int = 0
    poor_aec_consecutive: int = 0
    fallback_reason: str = ""
    degradation_cause: str = ""
    last_failure_reason: str = ""
    reference_loss_ratio: float = 0.0
    aec_effective_ratio: float = 0.0
    double_talk_ratio: float = 0.0
    barge_in_success_ratio: float = 0.0
    recoveries: int = 0
    xruns: int = 0
    device_resets: int = 0

    def to_log_payload(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "health_score": float(self.health_score),
            "requested_backend": self.requested_backend,
            "audio_mode": self.audio_mode,
            "reference_ready": bool(self.reference_ready),
            "reference_available_samples": int(self.reference_available_samples),
            "reference_callback_age_ms": int(self.reference_callback_age_ms),
            "reference_health_state": self.reference_health_state,
            "poor_aec_events": int(self.poor_aec_events),
            "poor_aec_consecutive": int(self.poor_aec_consecutive),
            "fallback_reason": self.fallback_reason,
            "degradation_cause": self.degradation_cause,
            "last_failure_reason": self.last_failure_reason,
            "reference_loss_ratio": float(self.reference_loss_ratio),
            "aec_effective_ratio": float(self.aec_effective_ratio),
            "double_talk_ratio": float(self.double_talk_ratio),
            "barge_in_success_ratio": float(self.barge_in_success_ratio),
            "recoveries": int(self.recoveries),
            "xruns": int(self.xruns),
            "device_resets": int(self.device_resets),
        }

    def to_log_dict(self) -> dict[str, object]:
        return self.to_log_payload()


@dataclass(frozen=True)
class SessionHealth:
    requested_backend: str
    selected_backend: str
    fallback_reason: str
    degradation_cause: str
    device_fingerprint: str
    audio_mode: str
    session_state: str
    session_started_at_mono: float
    session_activated_at_mono: float
    uptime_s: float
    active_for_s: float
    last_server_activity_age_s: float
    reference_ready: bool
    reference_health: str
    reference_available_samples: int
    reference_callback_age_ms: int
    reference_ready_events: int
    reference_miss_events: int
    reference_consecutive_misses: int
    poor_aec_events: int
    poor_aec_consecutive: int
    recovery_attempts_scheduled: int
    recovery_attempts_total: int
    next_reconnect_at_mono: float
    last_failure_reason: str
    recovery_successes_total: int = 0
    degraded_transitions_total: int = 0
    backend_switches_total: int = 0
    xrun_events_total: int = 0
    device_resets_total: int = 0
    barge_in_attempts_total: int = 0
    barge_in_successes_total: int = 0
    health_score: float = 0.0
    backend_health: BackendHealthSnapshot = field(default_factory=lambda: BackendHealthSnapshot(backend="unknown", health_score=0.0))

    def to_log_payload(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "fallback_reason": self.fallback_reason,
            "degradation_cause": self.degradation_cause,
            "device_fingerprint": self.device_fingerprint,
            "audio_mode": self.audio_mode,
            "session_state": self.session_state,
            "session_timing": {
                "started_at_mono": round(self.started_at_mono, 6),
                "activated_at_mono": round(self.activated_at_mono, 6),
                "uptime_s": round(self.uptime_s, 3),
                "active_for_s": round(self.active_for_s, 3),
                "last_server_activity_age_s": round(self.last_server_activity_age_s, 3),
            },
            "reference": {
                "ready": bool(self.reference_ready),
                "health": self.reference_health,
                "available_samples": int(self.reference_available_samples),
                "callback_age_ms": int(self.reference_callback_age_ms),
                "ready_events": int(self.reference_ready_events),
                "miss_events": int(self.reference_miss_events),
                "consecutive_misses": int(self.reference_consecutive_misses),
            },
            "aec_health": {
                "poor_events": int(self.poor_aec_events),
                "poor_consecutive": int(self.poor_aec_consecutive),
            },
            "backend_health": self.backend_health.to_log_payload(),
            "recovery_attempts": {
                "scheduled": int(self.recovery_attempts_scheduled),
                "total": int(self.recovery_attempts_total),
                "successes_total": int(self.recovery_successes_total),
                "degraded_transitions_total": int(self.degraded_transitions_total),
                "backend_switches_total": int(self.backend_switches_total),
                "xrun_events_total": int(self.xrun_events_total),
                "device_resets_total": int(self.device_resets_total),
                "barge_in_attempts_total": int(self.barge_in_attempts_total),
                "barge_in_successes_total": int(self.barge_in_successes_total),
                "next_reconnect_at_mono": round(self.next_reconnect_at_mono, 6),
            },
            "health_score": float(self.health_score),
            "last_failure_reason": self.last_failure_reason,
        }

    @property
    def started_at_mono(self) -> float:
        return self.session_started_at_mono

    @property
    def activated_at_mono(self) -> float:
        return self.session_activated_at_mono

    @property
    def scheduled_reconnects(self) -> int:
        return self.recovery_attempts_scheduled

    def to_log_dict(self) -> dict[str, object]:
        return self.to_log_payload()


@dataclass(frozen=True)
class AudioSessionEvent:
    name: str
    mono_ns: int
    session_state: str
    detail: dict[str, object] = field(default_factory=dict)
