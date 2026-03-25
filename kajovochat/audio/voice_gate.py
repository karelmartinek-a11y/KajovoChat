from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class VoiceGateThresholds:
    echo_similarity_drop: float
    echo_similarity_soft: float
    barge_in_min_input_level: float
    barge_in_output_ratio: float

    @classmethod
    def from_profiles(
        cls,
        *,
        default_profile: dict[str, float],
        profile: Optional[dict[str, float]] = None,
    ) -> "VoiceGateThresholds":
        active_profile = dict(default_profile)
        if profile:
            active_profile.update(profile)
        return cls(
            echo_similarity_drop=float(active_profile["echo_similarity_drop"]),
            echo_similarity_soft=float(active_profile["echo_similarity_soft"]),
            barge_in_min_input_level=float(active_profile["barge_in_min_input_level"]),
            barge_in_output_ratio=float(active_profile["barge_in_output_ratio"]),
        )


@dataclass
class VoiceGateRuntimeState:
    mic_suppressed_until: float = 0.0
    echo_drop_count: int = 0
    barge_in_chunk_count: int = 0
    barge_in_streak: int = 0
    last_echo_drop_reported: int = 0
    last_barge_in_reported: int = 0
    last_aec_diag_log_at: float = 0.0
    last_aec_success_log_at: float = 0.0
    playback_reference_armed: bool = False
    reference_warmup_until: float = 0.0
    cached_echo_reference: bytes = b""
    cached_reference_at: float = 0.0
    tts_start_hold_until: float = 0.0
    tts_tail_hold_until: float = 0.0


@dataclass(frozen=True)
class VoiceGateSnapshot:
    mic_suppressed_until: float
    echo_drop_count: int
    barge_in_chunk_count: int
    barge_in_streak: int
    playback_reference_armed: bool
    reference_warmup_until: float
    cached_reference_samples: int
    cached_reference_age_s: float
    tts_start_hold_until: float
    tts_tail_hold_until: float
    in_tts_hold: bool
    mic_capture_window_active: bool


@dataclass(frozen=True)
class ReferenceGateDecision:
    ready: bool
    miss: bool
    source: str
    available_samples: int
    callback_age_ms: int


@dataclass(frozen=True)
class ReferenceSelectionDecision:
    ready: bool
    miss: bool
    source: str
    available_samples: int
    callback_age_ms: int
    reference_pcm16: bytes


@dataclass(frozen=True)
class GateSideEffects:
    echo_drop_count: int
    barge_in_chunk_count: int
    should_log_echo_drop: bool


@dataclass(frozen=True)
class GateDecision:
    drop_chunk: bool
    drop_reason: str
    barge_in_candidate: bool
    barge_in_confirmed: bool
    effective_input_level: float
    effective_similarity: float
    effective_aec_quality: float
    in_tts_hold: bool
    thresholds: VoiceGateThresholds
    side_effects: GateSideEffects
    capture_gate_policy: str




@dataclass(frozen=True)
class VoiceGatePolicy:
    name: str
    echo_similarity_drop_bias: float = 0.0
    echo_similarity_soft_bias: float = 0.0
    barge_in_min_input_multiplier: float = 1.0
    barge_in_output_ratio_multiplier: float = 1.0
    base_barge_in_streak: int = 2
    tts_hold_barge_in_streak: int = 3
    disable_echo_drop: bool = False


def _policy_for_capture_gate(name: str) -> VoiceGatePolicy:
    normalized = (name or 'standard').strip().lower()
    if normalized == 'degraded_no_aec':
        return VoiceGatePolicy(
            name='degraded_no_aec',
            echo_similarity_drop_bias=-0.08,
            echo_similarity_soft_bias=-0.06,
            barge_in_min_input_multiplier=1.6,
            barge_in_output_ratio_multiplier=1.3,
            base_barge_in_streak=3,
            tts_hold_barge_in_streak=4,
        )
    if normalized == 'headset_clean':
        return VoiceGatePolicy(
            name='headset_clean',
            barge_in_min_input_multiplier=0.92,
            barge_in_output_ratio_multiplier=0.9,
            base_barge_in_streak=1,
            tts_hold_barge_in_streak=2,
            disable_echo_drop=True,
        )
    return VoiceGatePolicy(name=normalized or 'standard')

@dataclass
class VoiceGate:
    """Jediný source of truth pro hlasovou UX politiku session vrstvy."""

    mic_enabled: threading.Event = field(default_factory=threading.Event)
    awaiting_transcript: bool = False
    runtime: VoiceGateRuntimeState = field(default_factory=VoiceGateRuntimeState)

    def open(self) -> None:
        self.mic_enabled.set()

    def close(self) -> None:
        self.mic_enabled.clear()

    def reset_runtime(self) -> None:
        self.runtime = VoiceGateRuntimeState()

    def snapshot(self, *, now_monotonic: Optional[float] = None, mode: str = "idle") -> VoiceGateSnapshot:
        now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        runtime = self.runtime
        in_tts_hold = bool(now_monotonic < runtime.tts_start_hold_until or now_monotonic < runtime.tts_tail_hold_until)
        return VoiceGateSnapshot(
            mic_suppressed_until=float(runtime.mic_suppressed_until),
            echo_drop_count=int(runtime.echo_drop_count),
            barge_in_chunk_count=int(runtime.barge_in_chunk_count),
            barge_in_streak=int(runtime.barge_in_streak),
            playback_reference_armed=bool(runtime.playback_reference_armed),
            reference_warmup_until=float(runtime.reference_warmup_until),
            cached_reference_samples=len(runtime.cached_echo_reference) // 2,
            cached_reference_age_s=max(0.0, now_monotonic - float(runtime.cached_reference_at)) if runtime.cached_reference_at > 0.0 else float("inf"),
            tts_start_hold_until=float(runtime.tts_start_hold_until),
            tts_tail_hold_until=float(runtime.tts_tail_hold_until),
            in_tts_hold=in_tts_hold,
            mic_capture_window_active=bool(mode == "handsfree" and now_monotonic < runtime.mic_suppressed_until),
        )

    def update_playback_reference_state(self, *, is_playing_out: bool, now_monotonic: float, trailing_hold_s: float) -> None:
        runtime = self.runtime
        if is_playing_out:
            if not runtime.playback_reference_armed:
                runtime.playback_reference_armed = True
                runtime.reference_warmup_until = now_monotonic + 0.03
            runtime.mic_suppressed_until = max(runtime.mic_suppressed_until, now_monotonic + trailing_hold_s)
        else:
            runtime.playback_reference_armed = False
            runtime.reference_warmup_until = 0.0

    def is_guard_active(self, *, mode: str, is_playing_out: bool, now_monotonic: Optional[float] = None) -> bool:
        now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return bool(mode == "handsfree" and (is_playing_out or now_monotonic < self.runtime.mic_suppressed_until))

    def resolve_reference_gate(self, *, aec_requires_reference: bool, now_monotonic: float, reference_needed: int, available_samples: int, played_samples: int, callback_age_ms: int) -> ReferenceGateDecision:
        runtime = self.runtime
        if not aec_requires_reference:
            return ReferenceGateDecision(True, False, "system", int(available_samples), int(callback_age_ms))
        if not runtime.playback_reference_armed:
            return ReferenceGateDecision(False, True, "none", int(available_samples), int(callback_age_ms))
        enough_headroom = available_samples >= reference_needed + max(240, reference_needed // 4)
        warmup_done = bool(now_monotonic >= runtime.reference_warmup_until or played_samples >= max(240, reference_needed // 4) or enough_headroom)
        ready = bool(warmup_done and available_samples >= reference_needed and (callback_age_ms >= 0 or played_samples >= max(240, reference_needed // 4)))
        return ReferenceGateDecision(ready, not ready, "live" if ready else "none", int(available_samples), int(callback_age_ms))

    def can_use_cached_reference(self, *, now_monotonic: float, reference_needed: int, cached_samples: int | None = None, max_cache_age_s: float = 0.5) -> bool:
        runtime = self.runtime
        cached_samples = len(runtime.cached_echo_reference) // 2 if cached_samples is None else int(cached_samples)
        if cached_samples < max(448, reference_needed - 448):
            return False
        return bool((now_monotonic - runtime.cached_reference_at) <= max_cache_age_s)

    def select_reference_source(self, *, aec_requires_reference: bool, now_monotonic: float, reference_needed: int, live_reference_pcm16: bytes, available_samples: int, played_samples: int, callback_age_ms: int) -> ReferenceSelectionDecision:
        gate = self.resolve_reference_gate(aec_requires_reference=aec_requires_reference, now_monotonic=now_monotonic, reference_needed=reference_needed, available_samples=available_samples, played_samples=played_samples, callback_age_ms=callback_age_ms)
        if gate.ready:
            selected_pcm16 = bytes(live_reference_pcm16)
            if selected_pcm16:
                self.note_reference_cache(selected_pcm16, now_monotonic=now_monotonic)
            return ReferenceSelectionDecision(True, False, gate.source, int(available_samples), int(callback_age_ms), selected_pcm16)
        if not aec_requires_reference:
            return ReferenceSelectionDecision(True, False, "system", int(available_samples), int(callback_age_ms), b"")
        cached_samples = len(self.runtime.cached_echo_reference) // 2
        if self.can_use_cached_reference(now_monotonic=now_monotonic, reference_needed=reference_needed, cached_samples=cached_samples):
            return ReferenceSelectionDecision(True, False, "cached", int(cached_samples), max(0, int(callback_age_ms)), bytes(self.runtime.cached_echo_reference))
        if available_samples <= 0 and self.runtime.cached_echo_reference and played_samples > 0 and (now_monotonic - self.runtime.cached_reference_at) <= 0.35 and cached_samples >= max(640, reference_needed):
            return ReferenceSelectionDecision(True, False, "cached_tail", int(cached_samples), max(0, int(callback_age_ms)), bytes(self.runtime.cached_echo_reference))
        return ReferenceSelectionDecision(False, True, "none", int(available_samples), int(callback_age_ms), b"")

    def note_reference_cache(self, reference_pcm16: bytes, *, now_monotonic: float) -> None:
        self.runtime.cached_echo_reference = bytes(reference_pcm16)
        self.runtime.cached_reference_at = float(now_monotonic)

    def note_tts_window(self, *, rendering_active: bool, now_monotonic: float, start_hold_s: float = 0.18, tail_hold_s: float = 0.24) -> None:
        runtime = self.runtime
        if rendering_active:
            runtime.tts_start_hold_until = max(runtime.tts_start_hold_until, now_monotonic + start_hold_s)
            runtime.tts_tail_hold_until = 0.0
        else:
            runtime.tts_tail_hold_until = max(runtime.tts_tail_hold_until, now_monotonic + tail_hold_s)
            runtime.barge_in_streak = 0

    def record_gate_outcome(self, *, drop_chunk: bool, barge_in_candidate: bool) -> GateSideEffects:
        runtime = self.runtime
        should_log_echo_drop = False
        if drop_chunk:
            runtime.echo_drop_count += 1
            should_log_echo_drop = runtime.echo_drop_count <= 3
        if barge_in_candidate:
            runtime.barge_in_chunk_count += 1
        return GateSideEffects(int(runtime.echo_drop_count), int(runtime.barge_in_chunk_count), bool(should_log_echo_drop))

    def should_log_problem_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return bool(now_monotonic - self.runtime.last_aec_diag_log_at >= min_interval_s)

    def should_log_success_diag(self, *, now_monotonic: float, min_interval_s: float) -> bool:
        return bool(now_monotonic - self.runtime.last_aec_success_log_at >= min_interval_s)

    def note_diag_logged(self, *, success: bool, now_monotonic: float) -> None:
        if success:
            self.runtime.last_aec_success_log_at = float(now_monotonic)
        else:
            self.runtime.last_aec_diag_log_at = float(now_monotonic)

    def evaluate_capture_gate(self, *, mode: str, guard_active: bool, playback_active: bool, similarity: float, input_level: float, output_level: float, default_profile: dict[str, float], profile: Optional[dict[str, float]] = None, residual_level: Optional[float] = None, voice_likelihood: float = 0.0, double_talk: bool = False, aec_quality: float = 0.0, effective_similarity: Optional[float] = None, effective_aec_quality: Optional[float] = None, now_monotonic: Optional[float] = None, capture_gate_policy: str = "standard") -> GateDecision:
        now_monotonic = time.monotonic() if now_monotonic is None else float(now_monotonic)
        policy = _policy_for_capture_gate(capture_gate_policy)
        thresholds = VoiceGateThresholds.from_profiles(default_profile=default_profile, profile=profile)
        thresholds = VoiceGateThresholds(
            echo_similarity_drop=max(0.18, thresholds.echo_similarity_drop + policy.echo_similarity_drop_bias),
            echo_similarity_soft=max(0.12, thresholds.echo_similarity_soft + policy.echo_similarity_soft_bias),
            barge_in_min_input_level=max(0.01, thresholds.barge_in_min_input_level * policy.barge_in_min_input_multiplier),
            barge_in_output_ratio=max(0.45, thresholds.barge_in_output_ratio * policy.barge_in_output_ratio_multiplier),
        )
        runtime = self.runtime
        in_tts_hold = bool(now_monotonic < runtime.tts_start_hold_until or now_monotonic < runtime.tts_tail_hold_until)
        effective_input_level = max(float(input_level), float(voice_likelihood) * 0.45)
        drop_chunk, drop_reason = should_drop_mic_chunk(mode=mode, guard_active=guard_active, playback_active=playback_active, similarity=similarity, input_level=effective_input_level, output_level=output_level, default_profile=default_profile, profile={
            'echo_similarity_drop': thresholds.echo_similarity_drop,
            'echo_similarity_soft': thresholds.echo_similarity_soft,
            'barge_in_min_input_level': thresholds.barge_in_min_input_level,
            'barge_in_output_ratio': thresholds.barge_in_output_ratio,
        }, residual_level=residual_level, voice_likelihood=voice_likelihood, double_talk=double_talk, aec_quality=aec_quality, capture_gate_policy=policy.name)
        min_voice_likelihood = 0.42
        if policy.name == "degraded_no_aec":
            min_voice_likelihood = 0.68
        elif policy.name == "headset_clean":
            min_voice_likelihood = 0.34
        barge_in_candidate = bool(playback_active and voice_likelihood >= min_voice_likelihood and not drop_chunk and effective_input_level >= max(thresholds.barge_in_min_input_level * (0.95 if in_tts_hold else 0.8), float(output_level) * (thresholds.barge_in_output_ratio * (0.88 if in_tts_hold else 0.72))))
        runtime.barge_in_streak = runtime.barge_in_streak + 1 if barge_in_candidate else 0
        required_streak = policy.tts_hold_barge_in_streak if in_tts_hold else policy.base_barge_in_streak
        barge_in_confirmed = bool(barge_in_candidate and runtime.barge_in_streak >= required_streak)
        side_effects = self.record_gate_outcome(drop_chunk=drop_chunk, barge_in_candidate=barge_in_candidate)
        return GateDecision(bool(drop_chunk), str(drop_reason), barge_in_candidate, barge_in_confirmed, float(effective_input_level), float(similarity if effective_similarity is None else effective_similarity), float(aec_quality if effective_aec_quality is None else effective_aec_quality), in_tts_hold, thresholds, side_effects, policy.name)


def update_playback_reference_state(runtime: VoiceGateRuntimeState, *, is_playing_out: bool, now_monotonic: float, trailing_hold_s: float) -> None:
    VoiceGate(runtime=runtime).update_playback_reference_state(is_playing_out=is_playing_out, now_monotonic=now_monotonic, trailing_hold_s=trailing_hold_s)


def is_guard_active(runtime: VoiceGateRuntimeState, *, mode: str, is_playing_out: bool) -> bool:
    return VoiceGate(runtime=runtime).is_guard_active(mode=mode, is_playing_out=is_playing_out)


def resolve_reference_gate(runtime: VoiceGateRuntimeState, *, aec_requires_reference: bool, now_monotonic: float, reference_needed: int, available_samples: int, played_samples: int, callback_age_ms: int) -> ReferenceGateDecision:
    return VoiceGate(runtime=runtime).resolve_reference_gate(aec_requires_reference=aec_requires_reference, now_monotonic=now_monotonic, reference_needed=reference_needed, available_samples=available_samples, played_samples=played_samples, callback_age_ms=callback_age_ms)


def can_use_cached_reference(runtime: VoiceGateRuntimeState, *, now_monotonic: float, reference_needed: int, cached_samples: int | None = None, max_cache_age_s: float = 0.5) -> bool:
    return VoiceGate(runtime=runtime).can_use_cached_reference(now_monotonic=now_monotonic, reference_needed=reference_needed, cached_samples=cached_samples, max_cache_age_s=max_cache_age_s)


def note_reference_cache(runtime: VoiceGateRuntimeState, reference_pcm16: bytes, *, now_monotonic: float) -> None:
    VoiceGate(runtime=runtime).note_reference_cache(reference_pcm16, now_monotonic=now_monotonic)


def note_tts_window(runtime: VoiceGateRuntimeState, *, rendering_active: bool, now_monotonic: float, start_hold_s: float = 0.18, tail_hold_s: float = 0.24) -> None:
    VoiceGate(runtime=runtime).note_tts_window(rendering_active=rendering_active, now_monotonic=now_monotonic, start_hold_s=start_hold_s, tail_hold_s=tail_hold_s)


def record_gate_outcome(runtime: VoiceGateRuntimeState, *, drop_chunk: bool, barge_in_candidate: bool) -> GateSideEffects:
    return VoiceGate(runtime=runtime).record_gate_outcome(drop_chunk=drop_chunk, barge_in_candidate=barge_in_candidate)


def should_log_problem_diag(runtime: VoiceGateRuntimeState, *, now_monotonic: float, min_interval_s: float) -> bool:
    return VoiceGate(runtime=runtime).should_log_problem_diag(now_monotonic=now_monotonic, min_interval_s=min_interval_s)


def should_log_success_diag(runtime: VoiceGateRuntimeState, *, now_monotonic: float, min_interval_s: float) -> bool:
    return VoiceGate(runtime=runtime).should_log_success_diag(now_monotonic=now_monotonic, min_interval_s=min_interval_s)


def note_diag_logged(runtime: VoiceGateRuntimeState, *, success: bool, now_monotonic: float) -> None:
    VoiceGate(runtime=runtime).note_diag_logged(success=success, now_monotonic=now_monotonic)


def should_drop_mic_chunk(*, mode: str, guard_active: bool, playback_active: bool, similarity: float, input_level: float, output_level: float, default_profile: dict[str, float], profile: Optional[dict[str, float]] = None, residual_level: Optional[float] = None, voice_likelihood: float = 0.0, double_talk: bool = False, aec_quality: float = 0.0, capture_gate_policy: str = "standard") -> tuple[bool, str]:
    policy = _policy_for_capture_gate(capture_gate_policy)
    thresholds = VoiceGateThresholds.from_profiles(default_profile=default_profile, profile=profile)
    residual = float(input_level if residual_level is None else residual_level)
    if mode != "handsfree" or not guard_active:
        return False, ""
    if policy.disable_echo_drop:
        return False, ""
    strong_user = bool(input_level >= thresholds.barge_in_min_input_level and input_level >= max(thresholds.barge_in_min_input_level, output_level * thresholds.barge_in_output_ratio))
    if double_talk and (voice_likelihood >= 0.42 or strong_user):
        return False, ""
    if similarity >= thresholds.echo_similarity_drop and not strong_user and residual <= max(0.08, output_level * 1.05):
        return True, "echo_similarity"
    if playback_active and aec_quality < 0.04 and similarity >= max(0.66, thresholds.echo_similarity_soft) and not strong_user and voice_likelihood < 0.5:
        return True, "echo_similarity_fallback"
    if playback_active and similarity >= thresholds.echo_similarity_soft and residual <= max(0.045, output_level * (0.98 if aec_quality > 0.2 else 1.08)):
        return True, "echo_residual"
    if playback_active and aec_quality < 0.05 and output_level >= 0.06 and residual <= 0.022 and voice_likelihood < 0.26:
        return True, "quiet_bleed"
    return False, ""


def backend_aware_aec_metrics(*, backend: str, similarity: float, aec_quality: float, improvement_ratio: float, residual_level: float, output_level: float, webrtc_success: bool, native_selected: bool) -> tuple[float, float]:
    effective_similarity = float(similarity)
    effective_quality = float(aec_quality)
    if backend == "webrtc":
        effective_similarity = max(effective_similarity, 0.42 if webrtc_success else 0.0, min(0.72, float(improvement_ratio) * 0.9))
        effective_quality = max(effective_quality, 0.12 if webrtc_success else 0.0, min(0.35, float(improvement_ratio) * 0.45))
    elif backend == "windows_system_aec":
        effective_similarity = max(effective_similarity, 0.38 if native_selected else 0.22, min(0.68, float(improvement_ratio) * 0.82))
        effective_quality = max(effective_quality, 0.18 if native_selected else 0.08, min(0.32, float(improvement_ratio) * 0.4))
    if residual_level <= max(0.0012, output_level * 0.02):
        effective_quality = max(effective_quality, 0.1)
    return float(effective_similarity), float(effective_quality)


def evaluate_capture_gate(*, mode: str, guard_active: bool, playback_active: bool, similarity: float, input_level: float, output_level: float, default_profile: dict[str, float], profile: Optional[dict[str, float]] = None, runtime: Optional[VoiceGateRuntimeState] = None, residual_level: Optional[float] = None, voice_likelihood: float = 0.0, double_talk: bool = False, aec_quality: float = 0.0, capture_gate_policy: str = "standard") -> GateDecision:
    gate = VoiceGate(runtime=runtime or VoiceGateRuntimeState())
    return gate.evaluate_capture_gate(mode=mode, guard_active=guard_active, playback_active=playback_active, similarity=similarity, input_level=input_level, output_level=output_level, default_profile=default_profile, profile=profile, residual_level=residual_level, voice_likelihood=voice_likelihood, double_talk=double_talk, aec_quality=aec_quality, capture_gate_policy=capture_gate_policy)
