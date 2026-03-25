from __future__ import annotations

import time
from typing import Any

from .log_formatters import format_audio_log_payload
from .telemetry import AecDiagnosticsTelemetry


class ConversationAudioObserver:
    """Obsluhuje audio debug payloady, UI stav a caption/log výstupy."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._aec_diag_telemetry = AecDiagnosticsTelemetry()

    def emit_guard_debug(self) -> None:
        owner = self._owner
        snapshot = owner._guard_telemetry.snapshot(window_s=15.0)
        payload = {
            "state": owner._guard_adaptor.state,
            "profile": dict(owner._guard_profile),
            "telemetry": snapshot,
            "audio_mode": owner._audio_mode,
            "aec_aware": owner._guard_aec_aware,
            "learning_mode": time.monotonic() < owner._guard_learning_until,
            "native_aec_available": owner._native_aec_probe.available,
            "native_aec_reason": owner._native_aec_probe.reason,
            "calibration": dict(owner._guard_calibration),
            "input_device_name": owner._input_device_name,
            "output_device_name": owner._output_device_name,
        }
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
