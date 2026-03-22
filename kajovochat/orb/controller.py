from __future__ import annotations

import math
from dataclasses import dataclass

from .audio import AudioFeatureFrame
from .config import LivingOrbConfig, OrbStateProfile


@dataclass
class OrbFrameParameters:
    time: float
    radius: float
    core_radius: float
    core_intensity: float
    glow_intensity: float
    aura_radius: float
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
    speaking_boost: float
    transient_spike: float
    detail_sharpness: float
    low_pulse: float
    mid_motion: float
    high_shimmer: float
    background_intensity: float
    state_blend: float


class OrbAnimationController:
    def __init__(self, config: LivingOrbConfig) -> None:
        self.config = config

    def build_frame(
        self,
        *,
        time_value: float,
        state_profile: OrbStateProfile,
        state_blend: float,
        audio: AudioFeatureFrame,
    ) -> OrbFrameParameters:
        breathe = 0.018 * math.sin(time_value * 0.78) + 0.011 * math.sin(time_value * 1.61 + 0.7)
        drift = 0.014 * math.sin(time_value * 0.23 + 1.8)
        speaking_boost = audio.speaking_gate * (0.14 + audio.loudness * 0.09) + audio.transient_activity * 0.04
        radius = 0.33 + state_profile.radius_bias + breathe + drift + audio.loudness * 0.035 + audio.low_band * 0.022 + speaking_boost
        detail_sharpness = min(1.0, 0.28 + audio.spectral_centroid * self.config.centroid_weight + audio.high_band * 0.22)
        transient_spike = min(1.0, audio.spectral_flux * self.config.flux_weight + audio.transient_activity * self.config.transient_weight)
        return OrbFrameParameters(
            time=time_value,
            radius=radius,
            core_radius=self.config.core_radius + audio.loudness * 0.025 + speaking_boost * 0.18,
            core_intensity=self.config.glow_intensity * state_profile.core_intensity + audio.loudness * 0.45 + transient_spike * 0.18,
            glow_intensity=self.config.glow_intensity * state_profile.glow_intensity + audio.rms * 0.42 + audio.low_band * 0.18,
            aura_radius=self.config.aura_radius + state_profile.aura_intensity * 0.06 + audio.low_band * 0.04,
            aura_intensity=state_profile.aura_intensity + audio.low_band * 0.20 + audio.speaking_gate * 0.12,
            shell_deformation=self.config.shell_deformation_strength * (0.72 + state_profile.shell_deformation * 0.9 + audio.mid_band * 0.42),
            warp_strength=self.config.domain_warp_primary * (0.66 + state_profile.warp_strength) + audio.mid_band * 0.05,
            turbulence=0.10 + state_profile.turbulence * 0.28 + audio.mid_band * 0.14,
            shimmer=self.config.micro_shimmer_strength * (0.58 + state_profile.shimmer) + audio.high_band * 0.15,
            interference=self.config.interference_strength * (0.55 + state_profile.interference) + audio.spectral_centroid * 0.05,
            focus=state_profile.focus + audio.peak_envelope * 0.10,
            thinking_activity=state_profile.thinking_activity,
            speaking_mix=max(state_profile.speaking_mix, audio.speaking_gate * 0.92),
            listening_tension=state_profile.listening_tension,
            speaking_boost=speaking_boost,
            transient_spike=transient_spike,
            detail_sharpness=detail_sharpness,
            low_pulse=audio.low_band * self.config.low_band_weight,
            mid_motion=audio.mid_band * self.config.mid_band_weight,
            high_shimmer=audio.high_band * self.config.high_band_weight,
            background_intensity=self.config.background_intensity + audio.loudness * 0.04,
            state_blend=state_blend,
        )
