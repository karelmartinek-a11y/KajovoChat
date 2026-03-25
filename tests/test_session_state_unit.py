from __future__ import annotations

import pytest

from kajovochat.audio.session_state import (
    SessionPresentationState,
    SessionState,
    session_state_to_ui_state,
    validate_session_transition,
)


def test_session_state_transition_model_matches_target_architecture() -> None:
    validate_session_transition(SessionState.IDLE, SessionState.STARTING)
    validate_session_transition(SessionState.STARTING, SessionState.PROBING)
    validate_session_transition(SessionState.PROBING, SessionState.ACTIVE)
    validate_session_transition(SessionState.ACTIVE, SessionState.RECOVERING)
    validate_session_transition(SessionState.RECOVERING, SessionState.DEGRADED)
    validate_session_transition(SessionState.DEGRADED, SessionState.STOPPING)
    validate_session_transition(SessionState.STOPPING, SessionState.IDLE)

    with pytest.raises(Exception):
        validate_session_transition(SessionState.IDLE, SessionState.ACTIVE)


def test_session_state_to_ui_state_is_single_official_mapping() -> None:
    assert session_state_to_ui_state(SessionState.STARTING) == "connecting"
    assert session_state_to_ui_state(SessionState.RECOVERING) == "reconnecting"
    assert session_state_to_ui_state(SessionState.ACTIVE, SessionPresentationState.SPEAKING) == "speaking"
    assert session_state_to_ui_state(SessionState.ACTIVE, SessionPresentationState.TRANSCRIBING) == "transcribing"
    assert session_state_to_ui_state(SessionState.DEGRADED) == "listening"
    assert session_state_to_ui_state(SessionState.FAILED) == "error"
