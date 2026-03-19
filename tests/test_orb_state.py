from __future__ import annotations

from kajovochat.orb.config import create_default_config
from kajovochat.orb.state import StateController


def test_state_controller_blends_towards_target() -> None:
    controller = StateController(create_default_config())
    controller.set_state("speaking")

    profile_a, blend_a = controller.update(0.05)
    profile_b, blend_b = controller.update(0.35)

    assert 0.0 < blend_a < 1.0
    assert blend_b > blend_a
    assert profile_b.speaking_mix >= profile_a.speaking_mix


def test_state_controller_rejects_invalid_state() -> None:
    controller = StateController(create_default_config())
    try:
        controller.set_state("bad-state")
    except ValueError:
        pass
    else:
        raise AssertionError("Neplatný stav musí vyhodit ValueError.")
