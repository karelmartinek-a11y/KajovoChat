from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_guard_replay_metrics(log_dir: Path, payload: dict[str, Any]) -> Path:
    """Uloží anonymizované metriky guardu pro pozdější replay ladění."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "guard_replay_metrics.jsonl"
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path
