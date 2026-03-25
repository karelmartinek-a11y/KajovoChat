from __future__ import annotations

from datetime import datetime
from typing import Any

from ..settings import DEFAULT_AUDIO_GUARD_PROFILE
from ..services.guard_replay import append_guard_replay_metrics
from ..services.guard_telemetry import GuardTelemetry
from ..services.log_service import RealtimeLogWriter

_REALTIME_MODEL = "gpt-realtime"
_TTS_VOICE = "alloy"
_TTS_SPEED = 1.0


class ConversationAudioLifecycle:
    """Obsluhuje start a ukončení audio relace včetně session-level telemetrie."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def start_session_if_needed(self) -> None:
        owner = self._owner
        if owner._logger:
            return
        owner._captions = ""
        owner.captions_updated.emit(owner._captions)
        owner._session_manager.reset_runtime_tracking()
        owner._audio_observer.reset_aec_diag()
        owner._session_manager.reset_voice_gate_runtime()
        owner._guard_telemetry = GuardTelemetry()
        owner._aec.reset()
        owner._guard_learning_until = __import__("time").monotonic() + 30.0
        log_dir = owner.settings.validate_log_dir()
        session_name = datetime.now().strftime("kajovochat_%Y%m%d_%H%M%S")
        owner._logger = RealtimeLogWriter(log_dir=log_dir, session_name=session_name)
        owner._session_log_dir = log_dir
        owner._session_name = session_name
        owner._append_caption(f"Log: {str(owner._logger.jsonl_path)}")

        owner._log_event(
            "session_start",
            settings={
                "openai_base_url": "wss://api.openai.com/v1/realtime",
                "realtime_model": _REALTIME_MODEL,
                "answer_language_mode": owner.settings.answer_language_mode,
                "fixed_answer_language": owner.settings.fixed_answer_language,
                "response_style": owner.settings.response_style,
                "audio_aec_mode": owner._aec_mode,
                "audio_device_mode": getattr(owner.settings, "audio_device_mode", "auto"),
                "audio_session_profile": getattr(owner.settings, "audio_session_profile", "production"),
                "audio_diagnostics_enabled": bool(getattr(owner.settings, "audio_diagnostics_enabled", False)),
                "tts_voice": _TTS_VOICE,
                "tts_speed": _TTS_SPEED,
                "audio": {
                    "input_device": owner._resolved_input_device,
                    "output_device": owner._resolved_output_device,
                    "input_device_name": owner._input_device_name,
                    "output_device_name": owner._output_device_name,
                    "audio_mode": owner._audio_mode,
                    "calibration": dict(owner._guard_calibration),
                },
            },
        )
        owner._append_caption(
            f"Relace: model={_REALTIME_MODEL}, hlas={_TTS_VOICE}, jazyk={owner.settings.answer_language_mode}, styl={owner.settings.response_style}"
        )

    def end_session(self) -> None:
        owner = self._owner
        if not owner._logger:
            return
        telemetry = owner._guard_telemetry.snapshot(window_s=60.0)
        aec_summary = owner._build_aec_summary()
        owner._log_event(
            "session_end_guard",
            profile=owner._guard_profile,
            telemetry=telemetry,
            guard_state=owner._guard_adaptor.state,
            aec_aware=owner._guard_aec_aware,
            aec_summary=aec_summary,
        )
        owner._log_event("session_end", dropped_records=owner._logger.dropped_records, last_error=owner._logger.last_error)
        if int(aec_summary.get("samples", 0) or 0) > 0:
            owner._log_event(
                "aec_summary",
                **{
                    key: (round(float(value), 4) if isinstance(value, float) else value)
                    for key, value in aec_summary.items()
                },
            )
            owner._append_caption(
                "AEC summary: "
                f"samples={int(aec_summary['samples'])} "
                f"q={float(aec_summary['avg_quality']):.3f} "
                f"residual={float(aec_summary['avg_residual']):.4f} "
                f"dt={float(aec_summary['double_talk_ratio']):.3f} "
                f"delay_err={float(aec_summary['avg_delay_error']):.1f}"
            )
        try:
            owner.settings.audio_guard_profile = owner.settings.normalized_audio_guard_profile() | {
                key: float(value) for key, value in owner._guard_profile.items() if key in DEFAULT_AUDIO_GUARD_PROFILE
            }
            owner.settings.audio_guard_calibration = dict(owner._guard_calibration)
            owner.settings.save()
        except Exception:
            pass
        try:
            if owner._session_log_dir is not None:
                append_guard_replay_metrics(
                    owner._session_log_dir,
                    {
                        "session": owner._session_name,
                        "audio_mode": owner._audio_mode,
                        "guard_state": owner._guard_adaptor.state,
                        "aec_aware": owner._guard_aec_aware,
                        "profile": {key: round(float(value), 5) for key, value in owner._guard_profile.items()},
                        "telemetry": telemetry,
                    },
                )
        except Exception:
            pass
        try:
            owner._logger.close()
        except Exception:
            pass
        owner._logger = None
        owner._session_log_dir = None
        owner._session_name = ""
