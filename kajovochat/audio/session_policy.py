from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import sounddevice as sd

from ..settings import DEFAULT_AUDIO_GUARD_PROFILE, normalize_audio_device_mode
from ..services.audio_service import AudioPlayer, DuplexAudioSession, build_device_fingerprint, pick_audio_device
from .voice_gate import can_use_cached_reference as voice_gate_can_use_cached_reference
from .voice_gate import resolve_reference_gate


@dataclass
class ConversationAudioLatencyState:
    candidate_samples: int = 0
    candidate_hits: int = 0
    last_committed: int = 0


class ConversationAudioPolicy:
    """Drží provozní audio policy, aby worker nebyl plný frame-level rozhodování."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._latency = ConversationAudioLatencyState(
            last_committed=int(owner._guard_calibration.get("latency_samples", 0) or 0)
        )

    def current_device_fingerprint(self) -> str:
        owner = self._owner
        return build_device_fingerprint(owner._resolved_input_device, owner._resolved_output_device, 24000)

    def configure_aec_from_calibration(self) -> None:
        owner = self._owner
        calibration = owner._guard_calibration or {}
        filter_length = int(calibration.get("filter_length", owner._aec.filter_length) or owner._aec.filter_length)
        latency = int(calibration.get("latency_samples", owner._aec.last_shift) or owner._aec.last_shift)
        owner._aec.configure(
            filter_length=filter_length,
            max_shift_samples=max(960, latency + max(240, filter_length // 2)),
        )
        if latency > 0:
            owner._aec._last_shift = latency
            self._latency.last_committed = latency

    def consider_runtime_latency_update(
        self,
        *,
        delay_samples: int,
        similarity: float,
        aec_quality: float,
        improvement_ratio: float,
        backend: str,
        double_talk: bool,
        prefer_webrtc: bool = False,
    ) -> None:
        owner = self._owner
        if delay_samples <= 0 or double_talk:
            self._latency.candidate_hits = 0
            return
        committed = int(owner._guard_calibration.get("latency_samples", self._latency.last_committed) or self._latency.last_committed or 0)
        if committed <= 0:
            committed = int(delay_samples)
        webrtc_close_window = max(96, owner._aec.filter_length // (4 if prefer_webrtc else 5))
        stable_candidate = bool(
            (
                backend == "webrtc"
                and similarity >= (0.42 if prefer_webrtc else 0.5)
                and improvement_ratio >= (0.08 if prefer_webrtc else 0.12)
                and aec_quality >= (0.04 if prefer_webrtc else 0.05)
                and abs(int(delay_samples) - committed) <= webrtc_close_window
            )
            or (similarity >= 0.55 and aec_quality >= 0.08)
        )
        if not stable_candidate:
            self._latency.candidate_hits = max(0, self._latency.candidate_hits - 1)
            return

        max_step = max(160, owner._aec.filter_length // 2)
        if backend == "webrtc" and committed > 0 and abs(int(delay_samples) - committed) > max_step + 96:
            self._latency.candidate_hits = max(0, self._latency.candidate_hits - 1)
            return
        if abs(int(delay_samples) - committed) > max_step:
            target = committed + max_step if delay_samples > committed else committed - max_step
        else:
            target = int(delay_samples)

        if self._latency.candidate_samples <= 0 or abs(int(target) - int(self._latency.candidate_samples)) > 48:
            self._latency.candidate_samples = int(target)
            self._latency.candidate_hits = 1
            return

        self._latency.candidate_samples = int(round((self._latency.candidate_samples * 0.6) + (int(target) * 0.4)))
        self._latency.candidate_hits += 1
        required_hits = 3 if (backend == "webrtc" and prefer_webrtc) else (4 if backend == "webrtc" else 3)
        if self._latency.candidate_hits < required_hits:
            return

        new_latency = int(self._latency.candidate_samples)
        owner._guard_calibration["latency_samples"] = new_latency
        self._latency.last_committed = new_latency
        self._latency.candidate_hits = 0

    def reference_needed_samples(self, chunk_bytes: int) -> int:
        owner = self._owner
        return max(int(chunk_bytes) // 2 + max(96, owner._aec.filter_length // 6), 640)

    def is_reference_ready(
        self,
        *,
        now_monotonic: float,
        reference_needed: int,
        available_samples: int,
        played_samples: int,
        callback_age_ms: int,
    ) -> bool:
        owner = self._owner
        decision = resolve_reference_gate(
            owner._session_manager.voice_gate_runtime,
            aec_requires_reference=True,
            now_monotonic=now_monotonic,
            reference_needed=reference_needed,
            available_samples=available_samples,
            played_samples=played_samples,
            callback_age_ms=callback_age_ms,
        )
        return bool(decision.ready)

    def can_use_cached_reference(self, *, now_monotonic: float, reference_needed: int, cached_samples: int) -> bool:
        owner = self._owner
        runtime = owner._session_manager.voice_gate_runtime
        if cached_samples < max(448, reference_needed - 448):
            return False
        return voice_gate_can_use_cached_reference(
            runtime,
            now_monotonic=now_monotonic,
            reference_needed=reference_needed,
            cached_samples=cached_samples,
        )

    def ensure_player(self) -> None:
        owner = self._owner
        if owner._duplex is not None:
            owner._player = owner._duplex.player
            owner._mic = owner._duplex.mic
            return
        if owner._player is not None:
            return
        owner._player = AudioPlayer(
            samplerate=24000,
            device=owner._resolved_output_device,
            blocksize=owner._preferred_frame_size(),
        )

    def active_duplex(self) -> Optional[DuplexAudioSession]:
        return self._owner._duplex

    def device_calibration_matches(self) -> bool:
        owner = self._owner
        calibration = owner._guard_calibration or {}
        fingerprint = str(calibration.get("device_fingerprint", "")).strip().lower()
        if fingerprint and fingerprint != "unknown":
            return fingerprint == self.current_device_fingerprint().strip().lower()
        input_name = str(calibration.get("input_device_name", "")).strip().lower()
        output_name = str(calibration.get("output_device_name", "")).strip().lower()
        if not input_name or not output_name:
            return False
        return input_name == owner._input_device_name.strip().lower() and output_name == owner._output_device_name.strip().lower()

    def apply_device_class_policy(self) -> None:
        owner = self._owner
        profile = dict(owner._guard_profile)
        if owner._audio_mode in {"wired_headset", "bluetooth_headset", "external_headphones"}:
            profile["echo_similarity_soft"] = max(0.54, profile["echo_similarity_soft"] - 0.05)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] - 0.06)
            profile["playback_activity_level"] = max(0.02, profile["playback_activity_level"] * 0.82)
        elif owner._audio_mode in {"external_speakers", "notebook_builtin"}:
            profile["server_vad_threshold"] = min(0.9, profile["server_vad_threshold"] + 0.01)
            profile["barge_in_output_ratio"] = min(1.9, profile["barge_in_output_ratio"] + 0.06)
        owner._guard_profile = profile

    def apply_aec_mode_policy(self) -> None:
        owner = self._owner
        profile = dict(owner._guard_profile)
        if owner._aec_mode == "webrtc_apm":
            profile["echo_similarity_soft"] = max(0.58, profile["echo_similarity_soft"] - 0.03)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] - 0.025)
            profile["playback_activity_level"] = max(0.025, profile["playback_activity_level"] * 0.92)
            profile["barge_in_output_ratio"] = max(1.18, profile["barge_in_output_ratio"] * 0.97)
        elif owner._aec_mode == "headset_clean":
            profile["echo_similarity_soft"] = max(0.52, profile["echo_similarity_soft"] - 0.08)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] - 0.08)
            profile["playback_activity_level"] = max(0.018, profile["playback_activity_level"] * 0.76)
            profile["barge_in_output_ratio"] = max(1.1, profile["barge_in_output_ratio"] * 0.9)
        elif owner._aec_mode == "custom_lab":
            profile["echo_similarity_soft"] = min(0.76, profile["echo_similarity_soft"] + 0.02)
            profile["echo_similarity_drop"] = max(profile["echo_similarity_soft"] + 0.04, profile["echo_similarity_drop"] + 0.01)
            profile["playback_activity_level"] = min(0.08, profile["playback_activity_level"] * 1.03)
        owner._guard_profile = profile

    def apply_saved_calibration(self) -> None:
        owner = self._owner
        if not self.device_calibration_matches():
            self.apply_device_class_policy()
            self.apply_aec_mode_policy()
            return
        saved_profile = owner._guard_calibration.get("profile")
        if isinstance(saved_profile, dict) and saved_profile:
            merged = dict(owner.settings.normalized_audio_guard_profile())
            merged.update({key: float(value) for key, value in saved_profile.items() if key in DEFAULT_AUDIO_GUARD_PROFILE})
            owner._guard_profile = merged
        self.configure_aec_from_calibration()
        self.apply_aec_mode_policy()

    def resolve_audio_devices(self) -> None:
        """Vybere stabilní vstup/výstup a odvodí audio topologii relace."""
        owner = self._owner
        in_dev, in_note = pick_audio_device("input", None)
        out_dev, out_note = pick_audio_device("output", None)
        owner._resolved_input_device = in_dev
        owner._resolved_output_device = out_dev

        try:
            in_name = sd.query_devices(in_dev, "input")["name"] if in_dev is not None else "default"
        except Exception:
            in_name = "(neznámé)"
        try:
            out_name = sd.query_devices(out_dev, "output")["name"] if out_dev is not None else "default"
        except Exception:
            out_name = "(neznámé)"
        owner._input_device_name = str(in_name)
        owner._output_device_name = str(out_name)
        combined_names = f"{owner._input_device_name} {owner._output_device_name}".lower()
        owner._audio_mode = "notebook_builtin"
        if any(token in combined_names for token in ("bluetooth", "airpods", "buds", "hands-free")):
            owner._audio_mode = "bluetooth_headset"
        elif any(token in combined_names for token in ("headphone", "headphones", "headset", "earbuds")):
            owner._audio_mode = "wired_headset"
        elif any(token in combined_names for token in ("speaker", "speakers", "monitor", "hdmi", "display audio", "dock")) and not any(
            token in combined_names for token in ("built-in", "builtin", "internal", "laptop", "notebook")
        ):
            owner._audio_mode = "external_speakers"
        configured_audio_mode = normalize_audio_device_mode(getattr(owner.settings, "audio_device_mode", "auto"))
        if configured_audio_mode != "auto":
            owner._audio_mode = configured_audio_mode
        self.apply_saved_calibration()
        owner._append_caption(f"Mic: {in_dev if in_dev is not None else 'Default'} – {in_name}")
        owner._append_caption(f"Spk: {out_dev if out_dev is not None else 'Default'} – {out_name}")

        notes = []
        if in_note != "selected:settings":
            notes.append(f"mic:{in_note}")
        if out_note != "selected:settings":
            notes.append(f"spk:{out_note}")
        if notes:
            owner._append_caption("Audio: " + ", ".join(notes))
