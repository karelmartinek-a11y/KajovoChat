from __future__ import annotations

from dataclasses import dataclass

from ..settings import DEFAULT_AUDIO_GUARD_PROFILE


@dataclass
class GuardAdaptationResult:
    profile: dict[str, float]
    state: str


class GuardAdaptor:
    """Pomalá online adaptace audio guardu s hysteresi."""

    def __init__(self) -> None:
        self._state = "normal"

    @property
    def state(self) -> str:
        return self._state

    def adapt(
        self,
        profile: dict[str, float],
        telemetry: dict[str, float | str],
        *,
        learning_mode: bool = False,
        aec_aware: bool = False,
    ) -> GuardAdaptationResult:
        updated = dict(DEFAULT_AUDIO_GUARD_PROFILE)
        updated.update(profile)

        drop_rate = float(telemetry.get("drop_rate", 0.0) or 0.0)
        similarity = float(telemetry.get("avg_similarity", 0.0) or 0.0)
        voice_likelihood = float(telemetry.get("avg_voice_likelihood", 0.0) or 0.0)
        playback_ratio = float(telemetry.get("playback_ratio", 0.0) or 0.0)
        barge_in_ratio = float(telemetry.get("barge_in_ratio", 0.0) or 0.0)
        avg_output = float(telemetry.get("avg_output", 0.0) or 0.0)
        step = 1.8 if learning_mode else 1.0

        if aec_aware and playback_ratio > 0.22 and avg_output > 0.03 and similarity < 0.08:
            self._state = "aec_aware"
        elif similarity > 0.28 and playback_ratio > 0.18:
            self._state = "echo_heavy"
        elif barge_in_ratio > 0.08 and voice_likelihood > 0.46:
            self._state = "barge_ready"
        elif drop_rate < 0.08 and similarity < 0.18:
            self._state = "normal"

        if self._state == "echo_heavy":
            updated["echo_similarity_soft"] = min(0.9, updated["echo_similarity_soft"] + 0.01 * step)
            updated["echo_similarity_drop"] = min(0.97, updated["echo_similarity_drop"] + 0.012 * step)
            updated["playback_activity_level"] = min(0.16, updated["playback_activity_level"] + 0.002 * step)
            updated["server_vad_threshold"] = min(0.9, updated["server_vad_threshold"] + 0.004 * step)
        elif self._state == "aec_aware":
            updated["playback_activity_level"] = min(0.16, updated["playback_activity_level"] + 0.003 * step)
            updated["barge_in_output_ratio"] = min(1.9, updated["barge_in_output_ratio"] + 0.018 * step)
            updated["barge_in_min_input_level"] = min(0.22, updated["barge_in_min_input_level"] + 0.004 * step)
            updated["server_vad_threshold"] = min(0.9, updated["server_vad_threshold"] + 0.003 * step)
            updated["echo_similarity_soft"] = max(0.54, updated["echo_similarity_soft"] - 0.002)
        elif self._state == "barge_ready":
            updated["barge_in_min_input_level"] = max(0.04, updated["barge_in_min_input_level"] - 0.003 * step)
            updated["barge_in_output_ratio"] = max(1.12, updated["barge_in_output_ratio"] - 0.01 * step)
            updated["echo_similarity_soft"] = max(0.56, updated["echo_similarity_soft"] - 0.003 * step)
        else:
            updated["echo_similarity_soft"] += (DEFAULT_AUDIO_GUARD_PROFILE["echo_similarity_soft"] - updated["echo_similarity_soft"]) * 0.08
            updated["echo_similarity_drop"] += (DEFAULT_AUDIO_GUARD_PROFILE["echo_similarity_drop"] - updated["echo_similarity_drop"]) * 0.08
            updated["playback_activity_level"] += (
                DEFAULT_AUDIO_GUARD_PROFILE["playback_activity_level"] - updated["playback_activity_level"]
            ) * 0.08
            updated["barge_in_min_input_level"] += (
                DEFAULT_AUDIO_GUARD_PROFILE["barge_in_min_input_level"] - updated["barge_in_min_input_level"]
            ) * 0.08
            updated["barge_in_output_ratio"] += (
                DEFAULT_AUDIO_GUARD_PROFILE["barge_in_output_ratio"] - updated["barge_in_output_ratio"]
            ) * 0.08

        updated["echo_similarity_drop"] = max(updated["echo_similarity_soft"] + 0.04, updated["echo_similarity_drop"])
        updated["server_vad_threshold"] = max(0.66, min(0.9, updated["server_vad_threshold"]))
        updated["playback_activity_level"] = max(0.02, min(0.16, updated["playback_activity_level"]))
        updated["barge_in_min_input_level"] = max(0.04, min(0.22, updated["barge_in_min_input_level"]))
        updated["barge_in_output_ratio"] = max(1.1, min(1.9, updated["barge_in_output_ratio"]))
        updated["echo_similarity_soft"] = max(0.54, min(0.9, updated["echo_similarity_soft"]))
        updated["echo_similarity_drop"] = max(0.62, min(0.98, updated["echo_similarity_drop"]))

        return GuardAdaptationResult(profile={key: float(value) for key, value in updated.items()}, state=self._state)
