from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audio_architecture_harness import run_all_scenarios, write_evidence

EVIDENCE_ROOT = ROOT / "docs" / "audio_acceptance_evidence"
MATRIX_PATH = ROOT / "FINAL_ACCEPTANCE_MATRIX.md"
FINAL_DOC_PATH = ROOT / "docs" / "final_audio_architecture.md"


def _evidence_dir_for(kind: str) -> Path:
    return EVIDENCE_ROOT / kind


def _snapshot_ref(kind: str, scenario: str) -> str:
    return f"docs/audio_acceptance_evidence/{kind}/{scenario}_snapshot.json"


def _build_matrix(results: list[object]) -> str:
    lines = [
        "# FINAL_ACCEPTANCE_MATRIX",
        "",
        "| scénář | způsob ověření | backend chain | expected final state | telemetry evidence | pass/fail |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        verification = {
            "integration": "pytest + scripted harness",
            "acceptance": "scripted harness + telemetry snapshot + session log",
            "soak": "scripted harness + fault injection + telemetry snapshot",
        }.get(result.kind, "scripted harness")
        lines.append(
            f"| {result.scenario} | {verification} | `{' -> '.join(result.backend_chain)}` | `{result.final_state}` | `{_snapshot_ref(result.kind, result.scenario)}` | **{result.verdict}** |"
        )
    lines.extend(
        [
            "",
            "## Poznámky",
            "",
            "- Každý scénář má vedle snapshotu i odpovídající `*.jsonl` session log a `*_verdict.json` soubor ve stejné složce.",
            "- Acceptance scénáře jsou navržené jako deterministické scripted runs bez nutnosti reálného HW v CI.",
            "- Hardwarově závislé vlastnosti notebookového a headset prostředí jsou oddělené od produkčního rozhodování a pokryté simulací/fault injection harnessy.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_final_doc(results: list[object]) -> str:
    acceptance = [item for item in results if item.kind == "acceptance"]
    integration = [item for item in results if item.kind == "integration"]
    soak = [item for item in results if item.kind == "soak"]
    return "\n".join(
        [
            "# Finální stav audio architektury",
            "",
            "Repo po etapě 8 drží finální cílovou audio architekturu bez druhého source of truth v produkční větvi.",
            "",
            "## Jediné produkční autority",
            "",
            "- `AudioSessionManager` je jediný session entry point a aplikační vrstva pro audio relaci.",
            "- `VoiceGate` je jediný source of truth pro hlasovou UX politiku: capture gate, reference gating, TTS hold-off a barge-in potvrzení.",
            "- `RecoverySupervisor` je jediný source of truth pro reconnect a backend fallback policy.",
            "- `AudioTelemetry` je jediný source of truth pro session health, fallback story, recovery story a serializovatelný snapshot.",
            "- `windows_system_aec` je finální helper-backed produkční backend detail schovaný za session-level kontraktem `kajovochat.audio.windows_system_aec`.",
            "",
            "## Produkční backend chain",
            "",
            "1. `windows_system_aec`",
            "2. `webrtc_apm`",
            "3. `degraded_no_aec`",
            "",
            "Pro headset topologii se chain neřeší přes AEC fallback, ale přepíná se do explicitního first-class režimu `headset_clean`.",
            "",
            "## Důkazy",
            "",
            f"- acceptance scénáře: {len(acceptance)}",
            f"- integration scénáře: {len(integration)}",
            f"- soak/fault-injection scénáře: {len(soak)}",
            "- tabulka verdictů: `FINAL_ACCEPTANCE_MATRIX.md`",
            "- evidence soubory: `docs/audio_acceptance_evidence/*`",
            "- scripted harness: `tools/audio_architecture_harness.py`",
            "- generátor evidence: `tools/generate_audio_acceptance_evidence.py`",
            "",
            "## Poznámka k HW závislosti",
            "",
            "Plně reálné notebook/headset chování vyžaduje fyzický hardware a konkrétní Windows audio topologii. CI proto používá deterministické scripted runs a fault injection, ale produkční rozhodovací logika zůstává stejná jako v aplikaci.",
        ]
    ) + "\n"


def main() -> int:
    results = write_evidence(EVIDENCE_ROOT)
    MATRIX_PATH.write_text(_build_matrix(results), encoding="utf-8")
    FINAL_DOC_PATH.write_text(_build_final_doc(results), encoding="utf-8")
    print(json.dumps([
        {
            "scenario": item.scenario,
            "kind": item.kind,
            "backend_chain": item.backend_chain,
            "final_state": item.final_state,
            "verdict": item.verdict,
        }
        for item in results
    ], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
