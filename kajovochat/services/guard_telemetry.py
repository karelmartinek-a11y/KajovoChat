from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class GuardSample:
    timestamp: float
    input_level: float
    output_level: float
    similarity: float
    voice_likelihood: float
    residual_level: float = 0.0
    aec_quality: float = 0.0
    double_talk: bool = False
    dropped: bool = False
    playback_active: bool = False
    reason: str = ""
    barge_in_candidate: bool = False


class GuardTelemetry:
    """Klouzavá telemetrie guardu pro adaptaci a diagnostiku."""

    def __init__(self, retention_s: float = 60.0) -> None:
        self._retention_s = float(retention_s)
        self._samples: Deque[GuardSample] = deque()

    def add_sample(
        self,
        *,
        input_level: float,
        output_level: float,
        similarity: float,
        voice_likelihood: float,
        dropped: bool,
        playback_active: bool,
        reason: str,
        barge_in_candidate: bool,
        residual_level: float = 0.0,
        aec_quality: float = 0.0,
        double_talk: bool = False,
    ) -> None:
        now = time.monotonic()
        self._samples.append(
            GuardSample(
                timestamp=now,
                input_level=float(input_level),
                output_level=float(output_level),
                similarity=float(similarity),
                voice_likelihood=float(voice_likelihood),
                residual_level=float(residual_level),
                aec_quality=float(aec_quality),
                double_talk=bool(double_talk),
                dropped=bool(dropped),
                playback_active=bool(playback_active),
                reason=(reason or "").strip(),
                barge_in_candidate=bool(barge_in_candidate),
            )
        )
        self._trim(now)

    def snapshot(self, window_s: float = 15.0) -> dict[str, float | str]:
        now = time.monotonic()
        self._trim(now)
        window_start = now - float(window_s)
        samples = [sample for sample in self._samples if sample.timestamp >= window_start]
        if not samples:
            return {
                "samples": 0,
                "avg_input": 0.0,
                "avg_output": 0.0,
                "avg_similarity": 0.0,
                "avg_voice_likelihood": 0.0,
                "drop_rate": 0.0,
                "playback_ratio": 0.0,
                "barge_in_ratio": 0.0,
                "double_talk_ratio": 0.0,
                "avg_residual": 0.0,
                "avg_aec_quality": 0.0,
                "max_input": 0.0,
                "max_output": 0.0,
                "top_reason": "-",
            }

        reason_counter = Counter(sample.reason for sample in samples if sample.reason)
        count = len(samples)
        return {
            "samples": count,
            "avg_input": sum(sample.input_level for sample in samples) / count,
            "avg_output": sum(sample.output_level for sample in samples) / count,
            "avg_similarity": sum(sample.similarity for sample in samples) / count,
            "avg_voice_likelihood": sum(sample.voice_likelihood for sample in samples) / count,
            "drop_rate": sum(1 for sample in samples if sample.dropped) / count,
            "playback_ratio": sum(1 for sample in samples if sample.playback_active) / count,
            "barge_in_ratio": sum(1 for sample in samples if sample.barge_in_candidate) / count,
            "double_talk_ratio": sum(1 for sample in samples if sample.double_talk) / count,
            "avg_residual": sum(sample.residual_level for sample in samples) / count,
            "avg_aec_quality": sum(sample.aec_quality for sample in samples) / count,
            "max_input": max(sample.input_level for sample in samples),
            "max_output": max(sample.output_level for sample in samples),
            "top_reason": reason_counter.most_common(1)[0][0] if reason_counter else "-",
        }

    def _trim(self, now: float) -> None:
        min_time = now - self._retention_s
        while self._samples and self._samples[0].timestamp < min_time:
            self._samples.popleft()
