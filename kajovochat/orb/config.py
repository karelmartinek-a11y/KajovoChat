from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OrbStateProfile:
    core_intensity: float
    glow_intensity: float
    aura_intensity: float
    shell_deformation: float
    warp_strength: float
    turbulence: float
    shimmer: float
    interference: float
    focus: float
    thinking_activity: float
    speaking_mix: float
    listening_tension: float
    radius_bias: float


@dataclass(slots=True)
class LivingOrbConfig:
    core_radius: float = 0.23
    glow_intensity: float = 1.05
    aura_radius: float = 0.58
    shell_deformation_strength: float = 0.12
    noise_scale_primary: float = 2.1
    noise_scale_secondary: float = 4.8
    domain_warp_primary: float = 0.19
    domain_warp_secondary: float = 0.09
    background_intensity: float = 0.18
    micro_shimmer_strength: float = 0.17
    interference_strength: float = 0.13
    attack_seconds: float = 0.045
    release_seconds: float = 0.22
    peak_attack_seconds: float = 0.018
    peak_release_seconds: float = 0.28
    speaking_hold_seconds: float = 0.22
    state_transition_seconds: float = 0.42
    transient_decay_seconds: float = 0.18
    low_band_weight: float = 1.15
    mid_band_weight: float = 1.0
    high_band_weight: float = 0.9
    centroid_weight: float = 0.55
    flux_weight: float = 0.8
    transient_weight: float = 0.95
    background_color: tuple[float, float, float] = (0.020, 0.034, 0.060)
    haze_color: tuple[float, float, float] = (0.060, 0.120, 0.180)
    core_color: tuple[float, float, float] = (0.88, 0.95, 1.00)
    glow_color: tuple[float, float, float] = (0.42, 0.72, 1.00)
    aura_color: tuple[float, float, float] = (0.16, 0.48, 0.92)
    edge_color: tuple[float, float, float] = (0.90, 0.98, 1.00)
    state_profiles: dict[str, OrbStateProfile] = field(default_factory=dict)

    def validate(self) -> None:
        required = {"idle", "listening", "thinking", "speaking"}
        missing = required.difference(self.state_profiles)
        if missing:
            raise ValueError(f"Chybí orb state profily: {', '.join(sorted(missing))}")
        if self.core_radius <= 0.0 or self.aura_radius <= self.core_radius:
            raise ValueError("Neplatná konfigurace poloměrů orb vrstvy.")
        if self.state_transition_seconds <= 0.0:
            raise ValueError("Přechod stavů musí mít kladnou délku.")


def create_default_config() -> LivingOrbConfig:
    cfg = LivingOrbConfig(
        state_profiles={
            "idle": OrbStateProfile(
                core_intensity=0.88,
                glow_intensity=0.82,
                aura_intensity=0.74,
                shell_deformation=0.72,
                warp_strength=0.82,
                turbulence=0.54,
                shimmer=0.42,
                interference=0.36,
                focus=0.20,
                thinking_activity=0.14,
                speaking_mix=0.08,
                listening_tension=0.12,
                radius_bias=0.00,
            ),
            "listening": OrbStateProfile(
                core_intensity=0.94,
                glow_intensity=0.96,
                aura_intensity=0.82,
                shell_deformation=0.58,
                warp_strength=0.70,
                turbulence=0.36,
                shimmer=0.52,
                interference=0.52,
                focus=0.82,
                thinking_activity=0.16,
                speaking_mix=0.10,
                listening_tension=0.88,
                radius_bias=-0.02,
            ),
            "thinking": OrbStateProfile(
                core_intensity=1.00,
                glow_intensity=1.08,
                aura_intensity=0.92,
                shell_deformation=0.94,
                warp_strength=1.00,
                turbulence=0.86,
                shimmer=0.46,
                interference=0.28,
                focus=0.44,
                thinking_activity=1.00,
                speaking_mix=0.16,
                listening_tension=0.22,
                radius_bias=0.03,
            ),
            "speaking": OrbStateProfile(
                core_intensity=1.12,
                glow_intensity=1.18,
                aura_intensity=1.05,
                shell_deformation=1.02,
                warp_strength=1.08,
                turbulence=0.92,
                shimmer=0.84,
                interference=0.48,
                focus=0.56,
                thinking_activity=0.28,
                speaking_mix=1.00,
                listening_tension=0.24,
                radius_bias=0.06,
            ),
        }
    )
    cfg.validate()
    return cfg
