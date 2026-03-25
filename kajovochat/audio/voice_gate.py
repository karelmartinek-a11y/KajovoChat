from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VoiceGate:
    """Jednoduchý stav gate pro zachytávání mikrofonu a čekání na transcript."""

    mic_enabled: threading.Event = field(default_factory=threading.Event)
    awaiting_transcript: bool = False
    runtime: "VoiceGateRuntimeState" = field(default_factory=lambda: VoiceGateRuntimeState())

    def open(self) -> None:
        self.mic_enabled.set()

    def close(self) -> None:
        self.mic_enabled.clear()

    def reset_runtime(self) -> None:
        self.runtime = VoiceGateRuntimeState()


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
class ReferenceGateDecision:
    ready: bool
    miss: bool
    source: str
    available_samples: int
    callback_age_ms: int


@dataclass(frozen=True)
class GateDecision:
    drop_chunk: bool
    drop_reason: str
    barge_in_candidate: bool
    barge_in_confirmed: bool
    effective_input_level: float


@dataclass(frozen=True)
class GateSideEffects:
    echo_drop_count: int
    barge_in_chunk_count: int
    should_log_echo_drop: bool


def update_playback_reference_state(
    runtime: VoiceGateRuntimeState,
    *,
    is_playing_out: bool,
    now_monotonic: float,
    trailing_hold_s: float,
) -> None:
    """Aktualizuje reference arm/warmup a hold-off kolem playbacku."""

    if is_playing_out:
        if not runtime.playback_reference_armed:
            runtime.playback_reference_armed = True
            runtime.reference_warmup_until = now_monotonic + 0.03
        runtime.mic_suppressed_until = max(runtime.mic_suppressed_until, now_monotonic + trailing_hold_s)
    else:
        runtime.playback_reference_armed = False
        runtime.reference_warmup_until = 0.0


def is_guard_active(
    runtime: VoiceGateRuntimeState,
    *,
    mode: str,
    is_playing_out: bool,
) -> bool:
    """Vrátí, zda má být VoiceGate pro aktuální frame aktivní."""

    return bool(
        mode == "handsfree"
        and (is_playing_out or time.monotonic() < runtime.mic_suppressed_until)
    )


def resolve_reference_gate(
    runtime: VoiceGateRuntimeState,
    *,
    aec_requires_reference: bool,
    now_monotonic: float,
    reference_needed: int,
    available_samples: int,
    played_samples: int,
    callback_age_ms: int,
) -> ReferenceGateDecision:
    """Rozhodne, zda je reference připravená bez zásahu workeru."""

    if not aec_requires_reference:
        return ReferenceGateDecision(
            ready=True,
            miss=False,
            source="system",
            available_samples=int(available_samples),
            callback_age_ms=int(callback_age_ms),
        )
    if not runtime.playback_reference_armed:
        return ReferenceGateDecision(
            ready=False,
            miss=True,
            source="none",
            available_samples=int(available_samples),
            callback_age_ms=int(callback_age_ms),
        )
    enough_headroom = available_samples >= reference_needed + max(240, reference_needed // 4)
    warmup_done = bool(
        now_monotonic >= runtime.reference_warmup_until
        or played_samples >= max(240, reference_needed // 4)
        or enough_headroom
    )
    ready = bool(
        warmup_done
        and available_samples >= reference_needed
        and (callback_age_ms >= 0 or played_samples >= max(240, reference_needed // 4))
    )
    return ReferenceGateDecision(
        ready=ready,
        miss=not ready,
        source="live" if ready else "none",
        available_samples=int(available_samples),
        callback_age_ms=int(callback_age_ms),
    )


def can_use_cached_reference(
    runtime: VoiceGateRuntimeState,
    *,
    now_monotonic: float,
    reference_needed: int,
    cached_samples: int | None = None,
    max_cache_age_s: float = 0.5,
) -> bool:
    """Určí, zda je poslední uložená reference ještě provozně použitelná."""

    cached_samples = len(runtime.cached_echo_reference) // 2 if cached_samples is None else int(cached_samples)
    if cached_samples < max(448, reference_needed - 448):
        return False
    return bool((now_monotonic - runtime.cached_reference_at) <= max_cache_age_s)


def note_reference_cache(runtime: VoiceGateRuntimeState, reference_pcm16: bytes, *, now_monotonic: float) -> None:
    """Uloží poslední použitelnou playback referenci do runtime stavu gate."""

    runtime.cached_echo_reference = bytes(reference_pcm16)
    runtime.cached_reference_at = float(now_monotonic)


def note_tts_window(
    runtime: VoiceGateRuntimeState,
    *,
    rendering_active: bool,
    now_monotonic: float,
    start_hold_s: float = 0.18,
    tail_hold_s: float = 0.24,
) -> None:
    """Aktualizuje hold-off okna kolem začátku a konce TTS renderu."""

    if rendering_active:
        runtime.tts_start_hold_until = max(runtime.tts_start_hold_until, now_monotonic + start_hold_s)
        runtime.tts_tail_hold_until = 0.0
    else:
        runtime.tts_tail_hold_until = max(runtime.tts_tail_hold_until, now_monotonic + tail_hold_s)
        runtime.barge_in_streak = 0


def record_gate_outcome(
    runtime: VoiceGateRuntimeState,
    *,
    drop_chunk: bool,
    barge_in_candidate: bool,
) -> GateSideEffects:
    """Zapíše provozní side-effecty gate do session-owned runtime stavu."""

    should_log_echo_drop = False
    if drop_chunk:
        runtime.echo_drop_count += 1
        should_log_echo_drop = runtime.echo_drop_count <= 3
    if barge_in_candidate:
        runtime.barge_in_chunk_count += 1
    return GateSideEffects(
        echo_drop_count=int(runtime.echo_drop_count),
        barge_in_chunk_count=int(runtime.barge_in_chunk_count),
        should_log_echo_drop=bool(should_log_echo_drop),
    )


def should_log_problem_diag(runtime: VoiceGateRuntimeState, *, now_monotonic: float, min_interval_s: float) -> bool:
    """Vrátí, zda je možné znovu emitovat problémovou AEC diagnostiku."""

    return bool(now_monotonic - runtime.last_aec_diag_log_at >= min_interval_s)


def should_log_success_diag(runtime: VoiceGateRuntimeState, *, now_monotonic: float, min_interval_s: float) -> bool:
    """Vrátí, zda je možné znovu emitovat úspěšnou AEC diagnostiku."""

    return bool(now_monotonic - runtime.last_aec_success_log_at >= min_interval_s)


def note_diag_logged(runtime: VoiceGateRuntimeState, *, success: bool, now_monotonic: float) -> None:
    """Zapíše čas poslední emitované AEC diagnostiky."""

    if success:
        runtime.last_aec_success_log_at = float(now_monotonic)
    else:
        runtime.last_aec_diag_log_at = float(now_monotonic)


def should_drop_mic_chunk(
    *,
    mode: str,
    guard_active: bool,
    playback_active: bool,
    similarity: float,
    input_level: float,
    output_level: float,
    default_profile: dict[str, float],
    profile: Optional[dict[str, float]] = None,
    residual_level: Optional[float] = None,
    voice_likelihood: float = 0.0,
    double_talk: bool = False,
    aec_quality: float = 0.0,
) -> tuple[bool, str]:
    """Vyhodnotí, zda má být capture chunk zahožen jako echo nebo bleed."""

    active_profile = dict(default_profile)
    if profile:
        active_profile.update(profile)
    echo_similarity_drop = float(active_profile["echo_similarity_drop"])
    echo_similarity_soft = float(active_profile["echo_similarity_soft"])
    barge_in_min_input_level = float(active_profile["barge_in_min_input_level"])
    barge_in_output_ratio = float(active_profile["barge_in_output_ratio"])
    residual = float(input_level if residual_level is None else residual_level)

    if mode != "handsfree" or not guard_active:
        return False, ""

    strong_user = (
        input_level >= barge_in_min_input_level
        and input_level >= max(barge_in_min_input_level, output_level * barge_in_output_ratio)
    )
    if double_talk and (voice_likelihood >= 0.42 or strong_user):
        return False, ""
    if similarity >= echo_similarity_drop and not strong_user and residual <= max(0.08, output_level * 1.05):
        return True, "echo_similarity"
    if playback_active and aec_quality < 0.04 and similarity >= max(0.66, echo_similarity_soft) and not strong_user and voice_likelihood < 0.5:
        return True, "echo_similarity_fallback"
    if playback_active and similarity >= echo_similarity_soft and residual <= max(0.045, output_level * (0.98 if aec_quality > 0.2 else 1.08)):
        return True, "echo_residual"
    if playback_active and aec_quality < 0.05 and output_level >= 0.06 and residual <= 0.022 and voice_likelihood < 0.26:
        return True, "quiet_bleed"
    return False, ""


def backend_aware_aec_metrics(
    *,
    backend: str,
    similarity: float,
    aec_quality: float,
    improvement_ratio: float,
    residual_level: float,
    output_level: float,
    webrtc_success: bool,
    native_selected: bool,
) -> tuple[float, float]:
    """Vrátí efektivní similarity a kvalitu pro downstream guard a runtime logiku."""

    effective_similarity = float(similarity)
    effective_quality = float(aec_quality)
    if backend == "webrtc":
        effective_similarity = max(
            effective_similarity,
            0.42 if webrtc_success else 0.0,
            min(0.72, float(improvement_ratio) * 0.9),
        )
        effective_quality = max(
            effective_quality,
            0.12 if webrtc_success else 0.0,
            min(0.35, float(improvement_ratio) * 0.45),
        )
    elif backend == "windows_native":
        effective_similarity = max(
            effective_similarity,
            0.38 if native_selected else 0.0,
            min(0.68, float(improvement_ratio) * 0.82),
        )
        effective_quality = max(
            effective_quality,
            0.1 if native_selected else 0.0,
            min(0.32, float(improvement_ratio) * 0.4),
        )
    elif backend == "windows_system_capture":
        effective_similarity = max(effective_similarity, 0.35 if native_selected else 0.22)
        effective_quality = max(effective_quality, 0.18 if native_selected else 0.08)
    if residual_level <= max(0.0012, output_level * 0.02):
        effective_quality = max(effective_quality, 0.1)
    return float(effective_similarity), float(effective_quality)


def evaluate_capture_gate(
    *,
    mode: str,
    guard_active: bool,
    playback_active: bool,
    similarity: float,
    input_level: float,
    output_level: float,
    default_profile: dict[str, float],
    profile: Optional[dict[str, float]] = None,
    runtime: Optional[VoiceGateRuntimeState] = None,
    residual_level: Optional[float] = None,
    voice_likelihood: float = 0.0,
    double_talk: bool = False,
    aec_quality: float = 0.0,
) -> GateDecision:
    """Vrátí jednotné rozhodnutí VoiceGate pro drop i barge-in."""

    active_profile = dict(default_profile)
    if profile:
        active_profile.update(profile)
    now_monotonic = time.monotonic()
    in_tts_hold = bool(
        runtime is not None
        and (
            now_monotonic < runtime.tts_start_hold_until
            or now_monotonic < runtime.tts_tail_hold_until
        )
    )
    effective_input_level = max(
        float(input_level),
        float(voice_likelihood) * 0.45,
    )
    drop_chunk, drop_reason = should_drop_mic_chunk(
        mode=mode,
        guard_active=guard_active,
        playback_active=playback_active,
        similarity=similarity,
        input_level=effective_input_level,
        output_level=output_level,
        default_profile=default_profile,
        profile=profile,
        residual_level=residual_level,
        voice_likelihood=voice_likelihood,
        double_talk=double_talk,
        aec_quality=aec_quality,
    )
    barge_in_candidate = bool(
        playback_active
        and voice_likelihood >= 0.42
        and not drop_chunk
        and effective_input_level >= max(
            float(active_profile["barge_in_min_input_level"]) * (0.95 if in_tts_hold else 0.8),
            float(output_level) * (float(active_profile["barge_in_output_ratio"]) * (0.88 if in_tts_hold else 0.72)),
        )
    )
    if runtime is not None:
        runtime.barge_in_streak = runtime.barge_in_streak + 1 if barge_in_candidate else 0
        required_streak = 3 if in_tts_hold else 2
        barge_in_confirmed = bool(barge_in_candidate and runtime.barge_in_streak >= required_streak)
    else:
        barge_in_confirmed = bool(barge_in_candidate)
    return GateDecision(
        drop_chunk=bool(drop_chunk),
        drop_reason=str(drop_reason),
        barge_in_candidate=barge_in_candidate,
        barge_in_confirmed=barge_in_confirmed,
        effective_input_level=float(effective_input_level),
    )
