from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_session_jsonl(path: Path) -> dict[str, Any]:
    samples = 0
    diag_count = 0
    low_quality = 0
    double_talk = 0
    reference_miss = 0
    reference_ready = 0
    aligned = 0
    strong_aligned = 0
    avg_quality = 0.0
    avg_residual = 0.0
    avg_quality_when_aligned = 0.0
    avg_residual_when_aligned = 0.0
    avg_delay_error = 0.0
    max_delay_error = 0.0
    summary_from_log: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        record_type = str(item.get("type", "")).strip().lower()
        if record_type == "aec_diag":
            diag_count += 1
            quality = float(item.get("aec_quality", 0.0) or 0.0)
            residual = float(item.get("residual_level", 0.0) or 0.0)
            delay = int(item.get("delay_samples", 0) or 0)
            calib = int(item.get("calibration_latency", 0) or 0)
            similarity = float(item.get("similarity", 0.0) or 0.0)
            delay_error = abs(delay - calib) if delay > 0 and calib > 0 else 0
            avg_quality += quality
            avg_residual += residual
            avg_delay_error += float(delay_error)
            max_delay_error = max(max_delay_error, float(delay_error))
            if quality < 0.18:
                low_quality += 1
            if bool(item.get("double_talk")):
                double_talk += 1
            if bool(item.get("reference_miss")):
                reference_miss += 1
            else:
                reference_ready += 1
            if similarity >= 0.2:
                aligned += 1
                avg_quality_when_aligned += quality
                avg_residual_when_aligned += residual
            if similarity >= 0.5:
                strong_aligned += 1
        elif record_type == "aec_summary":
            summary_from_log = dict(item)
        elif record_type == "session_end_guard":
            embedded = item.get("aec_summary")
            if isinstance(embedded, dict):
                summary_from_log = dict(embedded)

    if summary_from_log:
        samples = int(summary_from_log.get("samples", 0) or 0)
        aligned_samples = max(1, aligned)
        return {
            "path": str(path),
            "samples": samples,
            "avg_quality": float(summary_from_log.get("avg_quality", 0.0) or 0.0),
            "avg_residual": float(summary_from_log.get("avg_residual", 0.0) or 0.0),
            "double_talk_ratio": float(summary_from_log.get("double_talk_ratio", 0.0) or 0.0),
            "low_quality_ratio": float(summary_from_log.get("low_quality_ratio", 0.0) or 0.0),
            "reference_miss_ratio": float(summary_from_log.get("reference_miss_ratio", reference_miss / max(1, diag_count)) or 0.0),
            "reference_ready_ratio": float(summary_from_log.get("reference_ready_ratio", reference_ready / max(1, diag_count)) or 0.0),
            "aligned_ratio": float(summary_from_log.get("aligned_ratio", aligned / max(1, diag_count)) or 0.0),
            "strong_alignment_ratio": float(summary_from_log.get("strong_alignment_ratio", strong_aligned / max(1, diag_count)) or 0.0),
            "avg_quality_when_aligned": float(summary_from_log.get("avg_quality_when_aligned", avg_quality_when_aligned / aligned_samples) or 0.0),
            "avg_residual_when_aligned": float(summary_from_log.get("avg_residual_when_aligned", avg_residual_when_aligned / aligned_samples) or 0.0),
            "avg_delay_error": float(summary_from_log.get("avg_delay_error", 0.0) or 0.0),
            "max_delay_error": float(summary_from_log.get("max_delay_error", 0.0) or 0.0),
            "diag_count": diag_count,
        }

    samples = max(1, diag_count)
    aligned_samples = max(1, aligned)
    return {
        "path": str(path),
        "samples": diag_count,
        "avg_quality": avg_quality / samples,
        "avg_residual": avg_residual / samples,
        "double_talk_ratio": double_talk / samples,
        "low_quality_ratio": low_quality / samples,
        "reference_miss_ratio": reference_miss / samples,
        "reference_ready_ratio": reference_ready / samples,
        "aligned_ratio": aligned / samples,
        "strong_alignment_ratio": strong_aligned / samples,
        "avg_quality_when_aligned": avg_quality_when_aligned / aligned_samples,
        "avg_residual_when_aligned": avg_residual_when_aligned / aligned_samples,
        "avg_delay_error": avg_delay_error / samples,
        "max_delay_error": max_delay_error,
        "diag_count": diag_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shrne AEC diagnostiku ze session JSONL logu.")
    parser.add_argument("path", help="Cesta k session .jsonl logu")
    args = parser.parse_args(argv)

    summary = summarize_session_jsonl(Path(args.path).expanduser().resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
