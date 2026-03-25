from __future__ import annotations

from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    READY = "ready"
    ASSISTANT_RENDERING = "assistant_rendering"
    DOUBLE_TALK = "double_talk"
    BARGE_IN_TRANSITION = "barge_in_transition"
    RECOVERING = "recovering"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    FAILED = "failed"


class SessionTransitionError(RuntimeError):
    """Signalizuje neplatný přechod stavového automatu audio relace."""


_ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {SessionState.INITIALIZING, SessionState.STOPPING, SessionState.FAILED},
    SessionState.INITIALIZING: {SessionState.CALIBRATING, SessionState.IDLE, SessionState.STOPPING, SessionState.FAILED},
    SessionState.CALIBRATING: {
        SessionState.READY,
        SessionState.DEGRADED,
        SessionState.RECOVERING,
        SessionState.IDLE,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.READY: {
        SessionState.ASSISTANT_RENDERING,
        SessionState.DOUBLE_TALK,
        SessionState.BARGE_IN_TRANSITION,
        SessionState.RECOVERING,
        SessionState.DEGRADED,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.ASSISTANT_RENDERING: {
        SessionState.READY,
        SessionState.DOUBLE_TALK,
        SessionState.BARGE_IN_TRANSITION,
        SessionState.RECOVERING,
        SessionState.DEGRADED,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.DOUBLE_TALK: {
        SessionState.READY,
        SessionState.ASSISTANT_RENDERING,
        SessionState.BARGE_IN_TRANSITION,
        SessionState.RECOVERING,
        SessionState.DEGRADED,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.BARGE_IN_TRANSITION: {
        SessionState.READY,
        SessionState.ASSISTANT_RENDERING,
        SessionState.RECOVERING,
        SessionState.DEGRADED,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.RECOVERING: {SessionState.READY, SessionState.DEGRADED, SessionState.STOPPING, SessionState.FAILED},
    SessionState.DEGRADED: {
        SessionState.RECOVERING,
        SessionState.READY,
        SessionState.ASSISTANT_RENDERING,
        SessionState.BARGE_IN_TRANSITION,
        SessionState.STOPPING,
        SessionState.FAILED,
    },
    SessionState.STOPPING: {SessionState.IDLE, SessionState.FAILED},
    SessionState.FAILED: {SessionState.IDLE, SessionState.INITIALIZING},
}

_SESSION_UI_STATE_MAP: dict[SessionState, str] = {
    SessionState.IDLE: "idle",
    SessionState.INITIALIZING: "connecting",
    SessionState.CALIBRATING: "connecting",
    SessionState.READY: "listening",
    SessionState.ASSISTANT_RENDERING: "speaking",
    SessionState.DOUBLE_TALK: "listening",
    SessionState.BARGE_IN_TRANSITION: "transcribing",
    SessionState.RECOVERING: "reconnecting",
    SessionState.DEGRADED: "listening",
    SessionState.STOPPING: "idle",
    SessionState.FAILED: "error",
}


def is_valid_session_transition(current: SessionState, target: SessionState) -> bool:
    if current == target:
        return True
    return bool(target in _ALLOWED_TRANSITIONS.get(current, set()))


def require_session_transition(current: SessionState, target: SessionState) -> None:
    if not is_valid_session_transition(current, target):
        raise SessionTransitionError(f"Neplatný přechod audio relace: {current.value} -> {target.value}")


def validate_session_transition(current: SessionState, target: SessionState) -> None:
    require_session_transition(current, target)


def session_state_to_ui_state(state: SessionState) -> str:
    return _SESSION_UI_STATE_MAP.get(state, "idle")
