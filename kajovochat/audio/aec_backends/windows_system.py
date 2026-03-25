from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from .base import AecBackendContext, AecBackendResult


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


class WindowsSystemAecBackendRunner:
    """Spouští produkční systémový AEC backend nad native helperem."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def run(
        self,
        *,
        context: AecBackendContext,
    ) -> Optional[AecBackendResult | dict[str, object]]:
        owner = self._owner
        if not getattr(owner, "_windows_native_probe").available:
            return None
        prep = context.prep
        try:
            native_session = owner._ensure_windows_native_session()
            if native_session is None:
                return None
            stream_delay_ms = int(round(prep.best_shift * (1000.0 / float(owner.samplerate))))
            native_session.submit_capture_frame(
                raw_mic_pcm16=context.mic_pcm,
                mono_ns=time.monotonic_ns(),
                stream_delay_ms=stream_delay_ms,
                render_ref_pcm16=prep.segment.astype(np.int16, copy=False).tobytes(),
            )
            native_frame = native_session.read_capture_frame(timeout_ms=0)
            if native_frame is None:
                return None
            native_pcm = native_frame.processed_mic_pcm16
            native_cleaned = np.frombuffer(native_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            native_residual_level = _rms(native_cleaned)
            native_improvement = max(0.0, 1.0 - min(1.0, native_residual_level / max(prep.mic_level, 1e-4)))
            native_predicted_level = max(0.0, prep.mic_level - native_residual_level)
            native_strong = bool(
                native_improvement >= 0.2
                and native_residual_level <= max(0.0035, prep.mic_level * 0.12)
            )
            prefer_native = bool(
                native_improvement >= context.improvement_ratio + 0.02
                or (
                    prep.similarity >= 0.3
                    and native_residual_level <= context.residual_level * 1.05
                    and native_improvement >= max(0.02, context.improvement_ratio - 0.01)
                )
                or (
                    prep.stable_delay_lock
                    and native_improvement >= max(0.08, context.improvement_ratio + 0.015)
                    and native_residual_level <= context.residual_level * 0.96
                )
                or (
                    context.backend_used == "custom"
                    and native_improvement >= 0.12
                    and native_residual_level <= context.residual_level * 0.98
                )
            )
            if prep.anchor_shift > 0 and prep.shift_error > max(240, int(getattr(owner, "_filter_length")) // 3) and prep.voice_likelihood >= 0.55:
                prefer_native = False
            if not prefer_native:
                return {
                    "native_attempted": True,
                    "native_selected": False,
                }
            if (
                prep.similarity >= 0.35
                and native_improvement >= 0.05
                and (
                    int(getattr(owner, "_delay_lock_shift")) <= 0
                    or abs(int(prep.best_shift) - int(getattr(owner, "_delay_lock_shift"))) <= max(128, int(getattr(owner, "_filter_length")) // 4)
                )
            ):
                owner._delay_lock_shift = int(prep.best_shift)
                owner._delay_lock_votes = max(int(getattr(owner, "_delay_lock_votes")), 3)
            return AecBackendResult(
                pcm=np.frombuffer(native_pcm, dtype=np.int16).copy(),
                similarity=float(prep.similarity),
                residual_level=float(native_frame.residual_level or native_residual_level),
                improvement_ratio=float(native_improvement),
                predicted_level=max(float(context.predicted_level), float(native_predicted_level)),
                delay_samples=int(prep.best_shift),
                backend="windows_system_aec",
                webrtc_success=bool(native_strong),
                native_attempted=True,
                native_selected=True,
                selection_reason="windows_system_aec_session",
            )
        except Exception:
            return None
