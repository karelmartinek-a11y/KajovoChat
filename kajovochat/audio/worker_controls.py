from __future__ import annotations

from typing import Any, Callable


class ConversationAudioWorkerControls:
    """Obsluhuje veřejné audio akce workeru mimo GUI třídu."""

    def __init__(
        self,
        owner: Any,
        *,
        sanitize_text_fn: Callable[[str], str],
        state_idle: str,
        state_error: str,
        closed_pose_factory: Callable[[], dict[str, object]],
    ) -> None:
        self._owner = owner
        self._sanitize_text_fn = sanitize_text_fn
        self._state_idle = state_idle
        self._state_error = state_error
        self._closed_pose_factory = closed_pose_factory

    def request_stop(self) -> None:
        owner = self._owner
        owner._stop_all.set()
        owner._mode = "idle"
        owner._session_manager.shutdown_runtime_resources()
        owner._transport_runtime.reset()
        owner._session_manager.reset_voice_gate_runtime()
        owner.input_level.emit(0.0)
        owner.output_level.emit(0.0)
        owner.output_pose.emit(self._closed_pose_factory())
        owner._set_state(self._state_idle)
        owner._emit_guard_debug()
        owner._end_session()

    def start_handsfree(self) -> None:
        owner = self._owner
        try:
            owner._session_manager.start_handsfree()
        except Exception as exc:
            owner._set_state(self._state_error)
            owner.error.emit(self._sanitize_text_fn(str(exc)))

    def ptt_pressed(self) -> None:
        owner = self._owner
        try:
            owner._session_manager.ptt_pressed()
        except Exception as exc:
            owner._set_state(self._state_error)
            owner.error.emit(self._sanitize_text_fn(str(exc)))

    def ptt_released(self) -> None:
        owner = self._owner
        owner._awaiting_transcript = True
        owner._session_manager.ptt_released()
