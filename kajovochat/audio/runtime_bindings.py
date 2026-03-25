from __future__ import annotations

import time
from typing import Any, Callable


class ConversationAudioRuntimeBindings:
    """Explicitní kontrakt mezi realtime smyčkou a worker/session vrstvou."""

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
        owner._attempt_reconnect_if_needed()
        if owner._rt:
            owner._rt.pump_events()
        owner._check_runtime_health()

    def adapt_guard_if_needed(self) -> None:
        owner = self._owner
        if time.monotonic() - owner._guard_last_adapt_at < 1.2:
            return
        telemetry_snapshot = owner._guard_telemetry.snapshot(window_s=15.0)
        owner._guard_aec_aware = (
            float(telemetry_snapshot.get("playback_ratio", 0.0) or 0.0) > 0.22
            and float(telemetry_snapshot.get("avg_output", 0.0) or 0.0) > 0.028
            and float(telemetry_snapshot.get("avg_similarity", 0.0) or 0.0) < 0.08
        )
        adaptation = owner._guard_adaptor.adapt(
            owner._guard_profile,
            telemetry_snapshot,
            learning_mode=time.monotonic() < owner._guard_learning_until,
            aec_aware=owner._guard_aec_aware,
        )
        owner._guard_profile = adaptation.profile
        owner._guard_last_adapt_at = time.monotonic()
        owner._emit_guard_debug()

    def resolve_playback_state(self) -> tuple[object | None, float, bool]:
        owner = self._owner
        current_out_level = 0.0
        is_playing_out = False
        duplex = owner._active_duplex()
        if duplex is not None:
            try:
                current_out_level = float(duplex.get_level())
                buffered = duplex.buffered_bytes
            except Exception:
                current_out_level = 0.0
                buffered = 0
            is_playing_out = (
                buffered > 0
                or current_out_level > float(owner._guard_profile["playback_activity_level"])
                or owner._ui_state == self._state_speaking
            )
            owner._session_manager.note_playback_activity(
                is_playing_out=is_playing_out,
                now_monotonic=time.monotonic(),
                trailing_hold_s=self._echo_trailing_hold_s,
            )
        return duplex, float(current_out_level), bool(is_playing_out)

    def guard_active(self, *, is_playing_out: bool) -> bool:
        return bool(self._owner._session_manager.is_guard_active(is_playing_out=is_playing_out))

    def has_active_capture(self, duplex: object | None) -> bool:
        owner = self._owner
        return bool(owner._mic_enabled.is_set() and (duplex is not None or owner._mic is not None) and owner._rt is not None)

    def capture_queue_for(self, duplex: object | None):
        owner = self._owner
        return duplex.queue if duplex is not None else owner._mic.queue

    def emit_levels(self, *, out_pose: dict[str, object]) -> None:
        owner = self._owner
        now = time.time()
        if now - owner._last_level_emit_t < 0.016:
            return
        in_lvl = owner._last_in_level if owner._mic_enabled.is_set() else 0.0
        out_lvl = owner._last_out_level
        owner.input_level.emit(float(in_lvl))
        owner.output_level.emit(float(out_lvl))
        owner.output_pose.emit(out_pose)
        owner._last_level_emit_t = now

    def closed_pose(self) -> dict[str, object]:
        return self._closed_pose_factory()
