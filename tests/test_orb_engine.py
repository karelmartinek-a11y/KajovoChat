from __future__ import annotations

import numpy as np

from kajovochat.orb.config import LivingOrbConfig, create_default_config
from kajovochat.orb.engine import OrbEngine


def test_orb_engine_explicit_features_override_pcm_for_one_update() -> None:
    engine = OrbEngine()
    engine.push_audio_frame(np.ones((512,), dtype=np.float32) * 0.8, sample_rate=24000)
    engine.set_audio_features({"loudness": 0.05, "rms": 0.05, "speaking_gate": 0.0})
    engine.update(0.016)
    first_radius = engine.current_frame.radius
    engine.update(0.016)
    second_radius = engine.current_frame.radius

    assert second_radius > first_radius


def test_orb_config_validation_requires_profiles() -> None:
    cfg = LivingOrbConfig(state_profiles={})
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Konfigurace bez profilů musí selhat.")


def test_orb_engine_state_change_affects_frame() -> None:
    engine = OrbEngine(create_default_config())
    engine.set_state("thinking")
    engine.update(0.1)

    assert engine.current_frame.thinking_activity > 0.2
