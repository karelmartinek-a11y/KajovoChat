from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import AecBackendContext, AecBackendResult


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


class WebRtcApmBackendRunner:
    """Spouští produkční duplex fallback nad WebRTC APM backendem."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def run(
        self,
        *,
        context: AecBackendContext,
    ) -> Optional[AecBackendResult]:
        owner = self._owner
        external_backend = getattr(owner, "_external_backend")
        if external_backend is None:
            return None
        prep = context.prep
        try:
            external_pcm = external_backend.process(
                mic_pcm=context.mic_pcm,
                reference_pcm=prep.segment.astype(np.int16, copy=False),
                delay_ms=int(round(prep.best_shift * (1000.0 / float(owner.samplerate)))),
            )
            external_cleaned = np.frombuffer(external_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            external_residual_level = _rms(external_cleaned)
            external_improvement = max(0.0, 1.0 - min(1.0, external_residual_level / max(prep.mic_level, 1e-4)))
            external_predicted_level = max(0.0, prep.mic_level - external_residual_level)
            strong_external = bool(
                external_improvement >= 0.18
                and external_residual_level <= max(0.0035, prep.mic_level * 0.12)
            )
            prefer_external = bool(
                external_improvement >= context.improvement_ratio + 0.02
                or (
                    prep.similarity >= 0.35
                    and external_residual_level <= context.residual_level * 1.08
                    and external_improvement >= max(0.02, context.improvement_ratio - 0.01)
                )
                or (
                    prep.similarity >= 0.5 and external_residual_level <= context.residual_level * 1.02
                )
                or (
                    prep.stable_delay_lock
                    and external_improvement >= max(0.08, context.improvement_ratio + 0.015)
                    and external_residual_level <= context.residual_level * 0.96
                )
                or (
                    prep.similarity >= 0.18
                    and context.improvement_ratio < 0.02
                    and external_improvement >= 0.05
                    and external_residual_level <= context.residual_level * 0.96
                )
            )
            if prep.anchor_shift > 0 and prep.shift_error > max(320, int(getattr(owner, "_filter_length")) // 2) and prep.similarity < 0.68:
                prefer_external = False
            if prep.anchor_shift > 0 and prep.shift_error > max(240, int(getattr(owner, "_filter_length")) // 3) and prep.voice_likelihood >= 0.55:
                prefer_external = False
            if context.prefer_native_mode and context.backend_used == "windows_system_aec":
                prefer_external = bool(
                    external_improvement >= context.improvement_ratio + 0.35
                    or (
                        strong_external
                        and external_improvement >= context.improvement_ratio + 0.18
                        and external_residual_level <= context.residual_level * 0.75
                        and prep.similarity >= 0.6
                    )
                )
            if not strong_external:
                if external_improvement < context.improvement_ratio + 0.12:
                    prefer_external = False
                if prep.voice_likelihood >= 0.55 and external_improvement < 0.45:
                    prefer_external = False
            if not prefer_external:
                return None
            if (
                prep.similarity >= 0.4
                and external_improvement >= 0.05
                and (
                    int(getattr(owner, "_delay_lock_shift")) <= 0
                    or abs(int(prep.best_shift) - int(getattr(owner, "_delay_lock_shift"))) <= max(128, int(getattr(owner, "_filter_length")) // 4)
                )
            ):
                owner._delay_lock_shift = int(prep.best_shift)
                owner._delay_lock_votes = max(int(getattr(owner, "_delay_lock_votes")), 3)
            return AecBackendResult(
                pcm=np.frombuffer(external_pcm, dtype=np.int16).copy(),
                similarity=float(prep.similarity),
                residual_level=float(external_residual_level),
                improvement_ratio=float(external_improvement),
                predicted_level=max(float(context.predicted_level), float(external_predicted_level)),
                delay_samples=int(prep.best_shift),
                backend="webrtc",
                webrtc_success=bool(strong_external),
                native_selected=False,
                selection_reason="webrtc_apm",
            )
        except Exception:
            return None
