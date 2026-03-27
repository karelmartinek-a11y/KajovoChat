from __future__ import annotations

import time
from typing import Any

from .log_formatters import format_audio_log_payload
from .telemetry import AecDiagnosticsTelemetry


def _is_echo_guard_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    return normalized in {"playback_voice_echo", "playback_voice_lock"}


class ConversationAudioObserver:
    """Obsluhuje audio debug payloady, UI stav a caption/log výstupy."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._aec_diag_telemetry = AecDiagnosticsTelemetry()

    def emit_guard_debug(self) -> None:
        owner = self._owner
        snapshot = owner._guard_telemetry.snapshot(window_s=15.0)
        now = time.monotonic()
        learning_remaining_s = max(0.0, float(owner._guard_learning_until) - now)
        monitor_state = "stable"
        recommend_preflight = False
        top_reason = str(snapshot.get("top_reason", "") or "")
        samples = int(snapshot.get("samples", 0) or 0)
        latency_missing = int(owner._guard_calibration.get("latency_samples", 0) or 0) <= 0
        profile_saturated = bool(
            float(owner._guard_profile.get("echo_similarity_drop", 0.0) or 0.0) >= 0.96
            and float(owner._guard_profile.get("echo_similarity_soft", 0.0) or 0.0) >= 0.82
            and float(owner._guard_profile.get("server_vad_threshold", 0.0) or 0.0) >= 0.88
        )
        playback_heavy = bool(
            float(snapshot.get("playback_ratio", 0.0) or 0.0) >= 0.78
            and float(snapshot.get("avg_output", 0.0) or 0.0) >= 0.2
        )
        echo_signature_persistent = bool(
            float(snapshot.get("avg_similarity", 0.0) or 0.0) >= 0.3
            and float(snapshot.get("avg_voice_likelihood", 0.0) or 0.0) >= 0.5
            and float(snapshot.get("drop_rate", 0.0) or 0.0) >= 0.12
        )
        severe_echo = bool(
            owner._guard_adaptor.state == "echo_heavy"
            and _is_echo_guard_reason(top_reason)
            and samples >= 120
            and float(snapshot.get("drop_rate", 0.0) or 0.0) >= 0.22
            and float(snapshot.get("playback_ratio", 0.0) or 0.0) >= 0.7
            and float(owner._guard_profile.get("echo_similarity_drop", 0.0) or 0.0) >= 0.96
            and float(owner._guard_profile.get("echo_similarity_soft", 0.0) or 0.0) >= 0.82
        )
        persistent_saturated_echo = bool(
            owner._guard_adaptor.state == "echo_heavy"
            and _is_echo_guard_reason(top_reason)
            and samples >= 160
            and latency_missing
            and profile_saturated
            and playback_heavy
            and echo_signature_persistent
        )
        if getattr(owner, "_audio_policy", None) is not None:
            try:
                owner._audio_policy.consider_live_tuning(
                    now_monotonic=now,
                    raw_input_level=float(getattr(owner, "_last_raw_in_level", 0.0) or 0.0),
                    post_gate_input_level=float(getattr(owner, "_last_post_gate_in_level", getattr(owner, "_last_in_level", 0.0)) or 0.0),
                    output_level=float(getattr(owner, "_last_out_level", 0.0) or 0.0),
                    top_reason=top_reason,
                    monitor_state=monitor_state,
                    samples=samples,
                    playback_ratio=float(snapshot.get("playback_ratio", 0.0) or 0.0),
                    avg_voice_likelihood=float(snapshot.get("avg_voice_likelihood", 0.0) or 0.0),
                )
            except Exception:
                pass
        if owner._guard_adaptor.state == "echo_heavy" and _is_echo_guard_reason(top_reason) and samples >= 12:
            if severe_echo or persistent_saturated_echo:
                monitor_state = "needs_preflight"
                recommend_preflight = True
            elif learning_remaining_s > 0.0:
                monitor_state = "relearning"
            else:
                monitor_state = "needs_preflight"
                recommend_preflight = True
        elif learning_remaining_s > 0.0 and owner._guard_adaptor.state != "normal":
            monitor_state = "learning"
        payload = {
            "state": owner._guard_adaptor.state,
            "profile": dict(owner._guard_profile),
            "telemetry": snapshot,
            "audio_mode": owner._audio_mode,
            "aec_aware": owner._guard_aec_aware,
            "learning_mode": learning_remaining_s > 0.0,
            "learning_remaining_s": round(learning_remaining_s, 1),
            "monitor_state": monitor_state,
            "recommend_preflight": recommend_preflight,
            "raw_input_level": round(float(getattr(owner, "_last_raw_in_level", 0.0) or 0.0), 5),
            "post_gate_input_level": round(float(getattr(owner, "_last_post_gate_in_level", getattr(owner, "_last_in_level", 0.0)) or 0.0), 5),
            "output_level": round(float(getattr(owner, "_last_out_level", 0.0) or 0.0), 5),
            "echo_trailing_hold_s": round(float(getattr(owner, "_echo_trailing_hold_s", 0.18) or 0.18), 3),
            "native_aec_available": owner._native_aec_probe.available,
            "native_aec_reason": owner._native_aec_probe.reason,
            "calibration": dict(owner._guard_calibration),
            "input_device_name": owner._input_device_name,
            "output_device_name": owner._output_device_name,
        }
        if getattr(owner, "_logger", None) is not None:
            self.log_event(
                "guard_monitor_snapshot",
                monitor_state=monitor_state,
                recommend_preflight=recommend_preflight,
                learning_mode=payload["learning_mode"],
                learning_remaining_s=payload["learning_remaining_s"],
                guard_state=str(owner._guard_adaptor.state),
                top_reason=top_reason,
                raw_input_level=round(float(getattr(owner, "_last_raw_in_level", 0.0) or 0.0), 5),
                post_gate_input_level=round(float(getattr(owner, "_last_post_gate_in_level", getattr(owner, "_last_in_level", 0.0)) or 0.0), 5),
                output_level=round(float(getattr(owner, "_last_out_level", 0.0) or 0.0), 5),
                echo_trailing_hold_s=round(float(getattr(owner, "_echo_trailing_hold_s", 0.18) or 0.18), 3),
                telemetry={
                    "samples": int(snapshot.get("samples", 0) or 0),
                    "drop_rate": round(float(snapshot.get("drop_rate", 0.0) or 0.0), 5),
                    "avg_similarity": round(float(snapshot.get("avg_similarity", 0.0) or 0.0), 5),
                    "avg_voice_likelihood": round(float(snapshot.get("avg_voice_likelihood", 0.0) or 0.0), 5),
                    "avg_output": round(float(snapshot.get("avg_output", 0.0) or 0.0), 5),
                    "avg_aec_quality": round(float(snapshot.get("avg_aec_quality", 0.0) or 0.0), 5),
                },
                profile={key: round(float(value), 5) for key, value in owner._guard_profile.items()},
                calibration={
                    "latency_samples": int(owner._guard_calibration.get("latency_samples", 0) or 0),
                    "device_fingerprint": str(owner._guard_calibration.get("device_fingerprint", "") or ""),
                    "last_monitor_recommendation": str(owner._guard_calibration.get("last_monitor_recommendation", "") or ""),
                },
            )
        owner.guard_debug_updated.emit(payload)

    def set_ui_state(self, value: str, *, valid_states: set[str], idle_state: str) -> None:
        owner = self._owner
        normalized = (value or idle_state).strip().lower()
        if normalized not in valid_states:
            normalized = idle_state
        owner._ui_state = normalized
        owner.state_changed.emit(normalized)

    def record_aec_diag_sample(
        self,
        *,
        residual_level: float,
        aec_quality: float,
        double_talk: bool,
        delay_samples: int,
        similarity: float,
        reference_miss: bool,
    ) -> None:
        owner = self._owner
        calibration_latency = int(owner._guard_calibration.get("latency_samples", 0) or 0)
        self._aec_diag_telemetry.record_sample(
            residual_level=residual_level,
            aec_quality=aec_quality,
            double_talk=double_talk,
            delay_samples=delay_samples,
            similarity=similarity,
            reference_miss=reference_miss,
            calibration_latency=calibration_latency,
        )

    def build_aec_summary(self) -> dict[str, float]:
        return self._aec_diag_telemetry.build_summary()

    def reset_aec_diag(self) -> None:
        self._aec_diag_telemetry.reset()

    def append_caption(self, line: str) -> None:
        owner = self._owner
        owner._captions = (owner._captions + "\n" + line).strip()
        owner._captions = "\n".join(owner._captions.splitlines()[-12:])
        owner.captions_updated.emit(owner._captions)

    def set_caption_preview(self, prefix: str, text: str) -> None:
        owner = self._owner
        base = owner._captions.splitlines()[-11:]
        preview = (text or "").replace("\n", " ").strip()
        owner.captions_updated.emit("\n".join(base + [f"{prefix}: {preview}"]))

    def log_event(self, record_type: str, **extra: object) -> None:
        owner = self._owner
        if not owner._logger:
            return
        payload = {"type": record_type}
        payload.update(extra)
        payload = format_audio_log_payload(record_type, payload)
        owner._logger.append(payload)
        if owner._logger.last_error:
            self.append_caption(f"Logování: {owner._logger.last_error}")

    def log_conversation_text(self, record_type: str, text: str) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return
        self.log_event(record_type, chars=len(normalized))
