from __future__ import annotations

from dataclasses import dataclass

from .config import LivingOrbConfig, OrbStateProfile


def _ease_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


@dataclass
class StateController:
    config: LivingOrbConfig
    current_state: str = "idle"
    target_state: str = "idle"
    transition_progress: float = 1.0
    _source_state: str = "idle"

    def set_state(self, state: str) -> None:
        normalized = (state or "idle").strip().lower()
        if normalized not in self.config.state_profiles:
            raise ValueError(f"Neplatný orb stav: {state}")
        if normalized == self.target_state and self.transition_progress < 1.0:
            return
        if normalized != self.target_state:
            self._source_state = self.current_state if self.transition_progress >= 1.0 else self._source_state
            self.target_state = normalized
            self.transition_progress = 0.0

    def update(self, dt: float) -> tuple[OrbStateProfile, float]:
        if self.transition_progress < 1.0:
            self.transition_progress = min(
                1.0,
                self.transition_progress + max(0.0, dt) / self.config.state_transition_seconds,
            )
            if self.transition_progress >= 1.0:
                self.current_state = self.target_state
                self._source_state = self.current_state
        source = self._source_state if self.transition_progress < 1.0 else self.current_state
        target = self.target_state
        blend = _ease_cubic(self.transition_progress)
        return self._blend_profiles(self.config.state_profiles[source], self.config.state_profiles[target], blend), blend

    @staticmethod
    def _blend_profiles(a: OrbStateProfile, b: OrbStateProfile, t: float) -> OrbStateProfile:
        def _mix(x: float, y: float) -> float:
            return float(x + (y - x) * t)

        return OrbStateProfile(
            core_intensity=_mix(a.core_intensity, b.core_intensity),
            glow_intensity=_mix(a.glow_intensity, b.glow_intensity),
            aura_intensity=_mix(a.aura_intensity, b.aura_intensity),
            shell_deformation=_mix(a.shell_deformation, b.shell_deformation),
            warp_strength=_mix(a.warp_strength, b.warp_strength),
            turbulence=_mix(a.turbulence, b.turbulence),
            shimmer=_mix(a.shimmer, b.shimmer),
            interference=_mix(a.interference, b.interference),
            focus=_mix(a.focus, b.focus),
            thinking_activity=_mix(a.thinking_activity, b.thinking_activity),
            speaking_mix=_mix(a.speaking_mix, b.speaking_mix),
            listening_tension=_mix(a.listening_tension, b.listening_tension),
            radius_bias=_mix(a.radius_bias, b.radius_bias),
        )
