from __future__ import annotations

from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    PROBING = "probing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    FAILED = "failed"


class SessionPresentationState(str, Enum):
    THINKING = "thinking"
    SPEAKING = "speaking"
    TRANSCRIBING = "transcribing"
    QUIESCENT = "quiescent"


class SessionTransitionError(RuntimeError):
    """Signalizuje neplatný přechod stavového automatu audio relace."""


_ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE: {SessionState.STARTING, SessionState.FAILED},
    SessionState.STARTING: {SessionState.PROBING, SessionState.FAILED},
    SessionState.PROBING: {SessionState.ACTIVE, SessionState.DEGRADED, SessionState.FAILED},
    SessionState.ACTIVE: {SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED},
    SessionState.DEGRADED: {SessionState.RECOVERING, SessionState.STOPPING, SessionState.FAILED},
    SessionState.RECOVERING: {SessionState.ACTIVE, SessionState.DEGRADED, SessionState.STOPPING, SessionState.FAILED},
    SessionState.STOPPING: {SessionState.IDLE, SessionState.FAILED},
    SessionState.FAILED: {SessionState.IDLE},
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


def session_state_to_ui_state(state: SessionState, presentation: SessionPresentationState | None = None) -> str:
    if state in {SessionState.IDLE, SessionState.STOPPING}:
        return "idle"
    if state in {SessionState.STARTING, SessionState.PROBING}:
        return "connecting"
    if state == SessionState.RECOVERING:
        return "reconnecting"
    if state == SessionState.FAILED:
        return "error"
    if presentation == SessionPresentationState.SPEAKING:
        return "speaking"
    if presentation == SessionPresentationState.THINKING:
        return "thinking"
    if presentation == SessionPresentationState.TRANSCRIBING:
        return "transcribing"
    if presentation == SessionPresentationState.QUIESCENT:
        return "idle"
    return "listening"
