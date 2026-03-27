from __future__ import annotations

import time
from typing import Any, Callable

_GUARD_ADAPT_INTERVAL_S = 1.2
_LIVE_RELEARN_EXTENSION_S = 90.0
_LIVE_RELEARN_MAX_HORIZON_S = 240.0


class ConversationAudioRuntimeBindings:
    """Explicitní kontrakt mezi realtime smyčkou a session vrstvou."""

    def __init__(
        self,
        owner: Any,
        *,
        closed_pose_factory: Callable[[], dict[str, object]],
        state_speaking: str,
        echo_trailing_hold_s: float,
    ) -> None:
        self._owner = owner
        self._closed_pose_factory = closed_pose_factory
        self._state_speaking = state_speaking
        self._echo_trailing_hold_s = float(echo_trailing_hold_s)

    @property
    def owner(self) -> Any:
        return self._owner

    def tick_realtime(self) -> None:
        owner = self._owner
        owner._session_manager.tick()
        rt = owner._session_manager.transport.realtime
        if rt is not None:
            rt.pump_events()
        owner._session_manager.check_runtime_health()

    def adapt_guard_if_needed(self) -> None:
        owner = self._owner
        now = time.monotonic()
        if now - owner._guard_last_adapt_at < _GUARD_ADAPT_INTERVAL_S:
            return
        telemetry_snapshot = owner._guard_telemetry.snapshot(window_s=15.0)
        learning_mode = now < owner._guard_learning_until
        previous_state = str(owner._guard_adaptor.state)
        previous_profile = dict(owner._guard_profile)
        owner._guard_aec_aware = (
            float(telemetry_snapshot.get("playback_ratio", 0.0) or 0.0) > 0.22
            and float(telemetry_snapshot.get("avg_output", 0.0) or 0.0) > 0.028
            and float(telemetry_snapshot.get("avg_similarity", 0.0) or 0.0) < 0.08
        )
        adaptation = owner._guard_adaptor.adapt(
            owner._guard_profile,
            telemetry_snapshot,
            learning_mode=learning_mode,
            aec_aware=owner._guard_aec_aware,
        )
        owner._guard_profile = adaptation.profile
        owner._guard_last_adapt_at = now
        learning_extended = self._extend_learning_window_if_needed(now=now, telemetry_snapshot=telemetry_snapshot, guard_state=adaptation.state)
        self._log_guard_adaptation(
            now=now,
            telemetry_snapshot=telemetry_snapshot,
            previous_state=previous_state,
            learning_mode=learning_mode,
            previous_profile=previous_profile,
            current_state=adaptation.state,
            learning_extended=learning_extended,
        )
        owner._emit_guard_debug()

    def _extend_learning_window_if_needed(
        self,
        *,
        now: float,
        telemetry_snapshot: dict[str, float | str],
        guard_state: str,
    ) -> bool:
        owner = self._owner
        top_reason = str(telemetry_snapshot.get("top_reason", "") or "")
        samples = int(telemetry_snapshot.get("samples", 0) or 0)
        should_extend = samples >= 12 and (
            guard_state == "echo_heavy" or top_reason == "playback_voice_echo"
        )
        if not should_extend:
            return False
        extended_until = min(
            max(float(owner._guard_learning_until), now + _LIVE_RELEARN_EXTENSION_S),
            now + _LIVE_RELEARN_MAX_HORIZON_S,
        )
        if extended_until <= float(owner._guard_learning_until):
            return False
        owner._guard_learning_until = extended_until
        if hasattr(owner, "_log_event"):
            owner._log_event(
                "guard_learning_extended",
                reason="live_echo_retraining",
                guard_state=guard_state,
                top_reason=top_reason,
                learning_until_mono=round(float(extended_until), 3),
            )
        return True

    def _log_guard_adaptation(
        self,
        *,
        now: float,
        telemetry_snapshot: dict[str, float | str],
        previous_state: str,
        learning_mode: bool,
        previous_profile: dict[str, float],
        current_state: str,
        learning_extended: bool,
    ) -> None:
        owner = self._owner
        if not hasattr(owner, "_log_event"):
            return
        changed_profile = {
            key: {
                "before": round(float(previous_profile.get(key, 0.0) or 0.0), 5),
                "after": round(float(owner._guard_profile.get(key, 0.0) or 0.0), 5),
            }
            for key in sorted(owner._guard_profile)
            if abs(float(owner._guard_profile.get(key, 0.0) or 0.0) - float(previous_profile.get(key, 0.0) or 0.0)) >= 0.0005
        }
        owner._log_event(
            "guard_adaptation",
            previous_state=previous_state,
            current_state=current_state,
            learning_mode=learning_mode,
            learning_remaining_s=round(max(0.0, float(owner._guard_learning_until) - now), 3),
            learning_extended=learning_extended,
            aec_aware=bool(owner._guard_aec_aware),
            telemetry={
                "samples": int(telemetry_snapshot.get("samples", 0) or 0),
                "drop_rate": round(float(telemetry_snapshot.get("drop_rate", 0.0) or 0.0), 5),
                "avg_similarity": round(float(telemetry_snapshot.get("avg_similarity", 0.0) or 0.0), 5),
                "avg_voice_likelihood": round(float(telemetry_snapshot.get("avg_voice_likelihood", 0.0) or 0.0), 5),
                "playback_ratio": round(float(telemetry_snapshot.get("playback_ratio", 0.0) or 0.0), 5),
                "barge_in_ratio": round(float(telemetry_snapshot.get("barge_in_ratio", 0.0) or 0.0), 5),
                "avg_output": round(float(telemetry_snapshot.get("avg_output", 0.0) or 0.0), 5),
                "avg_residual": round(float(telemetry_snapshot.get("avg_residual", 0.0) or 0.0), 5),
                "avg_aec_quality": round(float(telemetry_snapshot.get("avg_aec_quality", 0.0) or 0.0), 5),
                "top_reason": str(telemetry_snapshot.get("top_reason", "") or ""),
            },
            profile={key: round(float(value), 5) for key, value in owner._guard_profile.items()},
            changed_profile=changed_profile,
        )

    def resolve_playback_state(self) -> tuple[object | None, float, bool]:
        owner = self._owner
        current_out_level = 0.0
        is_playing_out = False
        duplex = owner._active_duplex()
        if duplex is not None:
            try:
                current_out_level = float(duplex.get_level())
                _buffered = duplex.buffered_bytes
            except Exception:
                current_out_level = 0.0
                _buffered = 0
            trailing_hold_s = float(getattr(owner, "_echo_trailing_hold_s", self._echo_trailing_hold_s) or self._echo_trailing_hold_s)
            # Samotná fronta v bufferu ještě neznamená, že je výstup opravdu slyšet.
            # Pro ochranu mikrofonu bereme jako aktivní jen skutečně slyšitelný playback.
            is_playing_out = (
                current_out_level > float(owner._guard_profile["playback_activity_level"])
                or owner.ui_state == self._state_speaking
            )
            owner._session_manager.note_playback_activity(
                is_playing_out=is_playing_out,
                now_monotonic=time.monotonic(),
                trailing_hold_s=trailing_hold_s,
            )
        return duplex, float(current_out_level), bool(is_playing_out)

    def guard_active(self, *, is_playing_out: bool) -> bool:
        return bool(self._owner._session_manager.is_guard_active(is_playing_out=is_playing_out))

    def has_active_capture(self, duplex: object | None) -> bool:
        owner = self._owner
        runtime = owner._runtime_resources
        return bool(owner._session_manager.mic_enabled.is_set() and (duplex is not None or runtime.mic is not None) and owner._session_manager.transport.realtime is not None)

    def capture_queue_for(self, duplex: object | None):
        owner = self._owner
        runtime = owner._runtime_resources
        return duplex.queue if duplex is not None else runtime.mic.queue

    def emit_levels(self, *, out_pose: dict[str, object]) -> None:
        owner = self._owner
        now = time.time()
        if now - owner._last_level_emit_t < 0.016:
            return
        in_lvl = owner._last_in_level if owner._session_manager.mic_enabled.is_set() else 0.0
        out_lvl = owner._last_out_level
        owner.input_level.emit(float(in_lvl))
        owner.output_level.emit(float(out_lvl))
        owner.output_pose.emit(out_pose)
        owner._last_level_emit_t = now

    def closed_pose(self) -> dict[str, object]:
        return self._closed_pose_factory()
