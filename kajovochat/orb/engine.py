from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .audio import AudioAnalyzer, AudioFeatureFrame
from .config import LivingOrbConfig, create_default_config
from .controller import OrbAnimationController, OrbFrameParameters
from .renderer import LivingOrbRenderer
from .state import StateController


class OrbEngine:
    """Veřejné integrační API pro living orb renderer."""

    def __init__(self, config: LivingOrbConfig | None = None, renderer: LivingOrbRenderer | None = None) -> None:
        self.config = config or create_default_config()
        self.config.validate()
        self.renderer = renderer
        self.audio = AudioAnalyzer(self.config)
        self.state = StateController(self.config)
        self.controller = OrbAnimationController(self.config)
        self._time = 0.0
        self._last_audio_frame = np.zeros((0,), dtype=np.float32)
        self._last_audio_rate: int | None = None
        self._explicit_features: Mapping[str, float] | None = None
        self._frame: OrbFrameParameters = self.controller.build_frame(
            time_value=0.0,
            state_profile=self.config.state_profiles["idle"],
            state_blend=1.0,
            audio=AudioFeatureFrame(),
        )

    @property
    def current_frame(self) -> OrbFrameParameters:
        return self._frame

    def set_state(self, state: str) -> None:
        self.state.set_state(state)

    def push_audio_frame(self, samples: np.ndarray, sample_rate: int | None = None) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self._last_audio_frame = x
        self._last_audio_rate = int(sample_rate) if sample_rate else None

    def set_audio_features(self, features: Mapping[str, float]) -> None:
        self._explicit_features = dict(features)

    def update(self, dt: float) -> None:
        dt = max(0.0, min(0.1, float(dt)))
        self._time += dt
        profile, blend = self.state.update(dt)
        if self._explicit_features is not None:
            smoothed = self.audio.update_from_features(self._explicit_features, dt)
            # Explicitní feature override má prioritu pouze pro nejbližší update.
            self._explicit_features = None
        else:
            smoothed = self.audio.update_from_pcm(self._last_audio_frame, dt, sample_rate=self._last_audio_rate)
        self._frame = self.controller.build_frame(
            time_value=self._time,
            state_profile=profile,
            state_blend=blend,
            audio=smoothed,
        )

    def render(self) -> None:
        if self.renderer is None:
            raise RuntimeError("OrbEngine nemá připojený renderer.")
        self.renderer.render(self._frame)

    def resize(self, width: int, height: int) -> None:
        if self.renderer is not None:
            self.renderer.resize(width, height)

    def shutdown(self) -> None:
        if self.renderer is not None:
            self.renderer.shutdown()
