from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import sounddevice as sd

from ..settings import DEFAULT_AUDIO_GUARD_PROFILE, normalize_audio_device_mode
from .devices import build_device_fingerprint, pick_audio_device
from .io import AudioPlayer, DuplexAudioSession


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
        self._live_miss_streak = 0
        self._live_echo_streak = 0
        self._last_live_tune_at = 0.0

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

    def ensure_player(self) -> None:
        owner = self._owner
        runtime = owner._runtime_resources
        if runtime.duplex is not None:
            runtime.player = runtime.duplex.player
            runtime.mic = runtime.duplex.mic
            return
        if runtime.player is not None:
            return
        runtime.player = AudioPlayer(
            samplerate=24000,
            device=owner._resolved_output_device,
            blocksize=owner._preferred_frame_size(),
        )

    def active_duplex(self) -> Optional[DuplexAudioSession]:
        return self._owner._runtime_resources.duplex

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

    def consider_live_tuning(
        self,
        *,
        now_monotonic: float,
        raw_input_level: float,
        post_gate_input_level: float,
        output_level: float,
        top_reason: str,
        monitor_state: str,
        samples: int,
        playback_ratio: float,
        avg_voice_likelihood: float,
    ) -> bool:
        """Jemně upraví guard za běhu podle toho, co skutečně prošlo přes gate."""
        owner = self._owner
        if now_monotonic - self._last_live_tune_at < 0.95:
            return False

        raw_input_level = float(raw_input_level)
        post_gate_input_level = float(post_gate_input_level)
        output_level = float(output_level)
        playback_ratio = float(playback_ratio)
        avg_voice_likelihood = float(avg_voice_likelihood)
        top_reason = str(top_reason or "").strip().lower()
        monitor_state = str(monitor_state or "").strip().lower()

        user_like_signal = bool(raw_input_level >= 0.02 and raw_input_level >= post_gate_input_level + 0.006)
        stale_playback_lock = bool(
            top_reason in {"playback_voice_echo", "playback_voice_lock"}
            and output_level <= 0.015
            and raw_input_level >= 0.02
            and avg_voice_likelihood >= 0.42
        )
        blocked_user_signal = bool(
            user_like_signal
            and post_gate_input_level <= 0.004
            and output_level <= 0.02
        )
        sustained_echo = bool(
            top_reason in {"playback_voice_echo", "playback_voice_lock"}
            and playback_ratio >= 0.52
            and avg_voice_likelihood >= 0.42
            and output_level >= 0.03
            and raw_input_level < 0.024
        )

        if blocked_user_signal or stale_playback_lock:
            self._live_miss_streak = min(12, self._live_miss_streak + 1)
        else:
            self._live_miss_streak = max(0, self._live_miss_streak - 1)

        if sustained_echo:
            self._live_echo_streak = min(12, self._live_echo_streak + 1)
        else:
            self._live_echo_streak = max(0, self._live_echo_streak - 1)

        should_relax = self._live_miss_streak >= 2
        should_tighten = self._live_echo_streak >= 6 and self._live_miss_streak == 0
        if not should_relax and not should_tighten:
            return False

        profile = dict(owner._guard_profile)
        before = dict(profile)
        reason = "missing_user_capture" if should_relax else "sustained_echo"

        if should_relax:
            profile["playback_activity_level"] = max(0.02, float(profile["playback_activity_level"]) - 0.0025)
            profile["echo_similarity_soft"] = min(0.84, float(profile["echo_similarity_soft"]) + 0.01)
            profile["echo_similarity_drop"] = min(0.98, max(float(profile["echo_similarity_soft"]) + 0.04, float(profile["echo_similarity_drop"]) + 0.012))
            profile["barge_in_min_input_level"] = max(0.035, float(profile["barge_in_min_input_level"]) - 0.002)
            profile["barge_in_output_ratio"] = max(1.08, float(profile["barge_in_output_ratio"]) - 0.01)
            profile["server_vad_threshold"] = max(0.66, float(profile["server_vad_threshold"]) - 0.004)
            owner._echo_trailing_hold_s = max(0.08, float(getattr(owner, "_echo_trailing_hold_s", 0.18)) - 0.02)
            owner._guard_learning_until = max(float(owner._guard_learning_until), now_monotonic + 30.0)
        else:
            profile["playback_activity_level"] = min(0.16, float(profile["playback_activity_level"]) + 0.0015)
            profile["echo_similarity_soft"] = max(0.54, float(profile["echo_similarity_soft"]) - 0.006)
            profile["echo_similarity_drop"] = max(float(profile["echo_similarity_soft"]) + 0.04, float(profile["echo_similarity_drop"]) - 0.008)
            profile["barge_in_min_input_level"] = min(0.22, float(profile["barge_in_min_input_level"]) + 0.001)
            profile["barge_in_output_ratio"] = min(1.9, float(profile["barge_in_output_ratio"]) + 0.006)
            owner._echo_trailing_hold_s = min(0.28, float(getattr(owner, "_echo_trailing_hold_s", 0.18)) + 0.01)

        owner._guard_profile = profile
        self._last_live_tune_at = float(now_monotonic)
        if hasattr(owner, "_log_event"):
            changed_profile = {
                key: {
                    "before": round(float(before.get(key, 0.0) or 0.0), 5),
                    "after": round(float(profile.get(key, 0.0) or 0.0), 5),
                }
                for key in sorted(profile)
                if abs(float(profile.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)) >= 0.0005
            }
            owner._log_event(
                "guard_live_tuned",
                reason=reason,
                monitor_state=monitor_state,
                top_reason=top_reason,
                raw_input_level=round(raw_input_level, 5),
                post_gate_input_level=round(post_gate_input_level, 5),
                output_level=round(output_level, 5),
                playback_ratio=round(playback_ratio, 5),
                voice_likelihood=round(avg_voice_likelihood, 5),
                samples=int(samples),
                miss_streak=int(self._live_miss_streak),
                echo_streak=int(self._live_echo_streak),
                echo_trailing_hold_s=round(float(getattr(owner, "_echo_trailing_hold_s", 0.18)), 3),
                profile=profile,
                changed_profile=changed_profile,
            )
        return True

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
