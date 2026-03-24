from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kajovochat.tools.analyze_aec_log import summarize_session_jsonl


def test_aec_log_analyzer_prefers_embedded_summary() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = Path(temp_dir) / "session.jsonl"
        lines = [
            {"type": "aec_diag", "aec_quality": 0.12, "residual_level": 0.06, "double_talk": False, "delay_samples": 520, "calibration_latency": 480},
            {
                "type": "session_end_guard",
                "aec_summary": {
                    "samples": 8,
                    "avg_quality": 0.44,
                    "avg_residual": 0.021,
                    "double_talk_ratio": 0.25,
                    "low_quality_ratio": 0.125,
                    "reference_miss_ratio": 0.375,
                    "reference_ready_ratio": 0.625,
                    "aligned_ratio": 0.5,
                    "strong_alignment_ratio": 0.25,
                    "avg_quality_when_aligned": 0.61,
                    "avg_residual_when_aligned": 0.014,
                    "avg_delay_error": 18.0,
                    "max_delay_error": 64.0,
                },
            },
        ]
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines), encoding="utf-8")

        result = summarize_session_jsonl(path)

        assert result["samples"] == 8
        assert result["avg_quality"] == 0.44
        assert result["aligned_ratio"] == 0.5
        assert result["avg_quality_when_aligned"] == 0.61
        assert result["diag_count"] == 1


def test_aec_log_analyzer_falls_back_to_diag_records() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = Path(temp_dir) / "session.jsonl"
        lines = [
            {"type": "aec_diag", "aec_quality": 0.12, "residual_level": 0.06, "double_talk": False, "delay_samples": 520, "calibration_latency": 480, "similarity": 0.12, "reference_miss": True},
            {"type": "aec_diag", "aec_quality": 0.32, "residual_level": 0.03, "double_talk": True, "delay_samples": 500, "calibration_latency": 480, "similarity": 0.56, "reference_miss": False},
        ]
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines), encoding="utf-8")

        result = summarize_session_jsonl(path)

        assert result["samples"] == 2
        assert result["diag_count"] == 2
        assert result["low_quality_ratio"] == 0.5
        assert result["double_talk_ratio"] == 0.5
        assert result["reference_miss_ratio"] == 0.5
        assert result["reference_ready_ratio"] == 0.5
        assert result["aligned_ratio"] == 0.5
        assert result["strong_alignment_ratio"] == 0.5
        assert result["avg_quality_when_aligned"] == 0.32
