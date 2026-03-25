from __future__ import annotations

from tests.audio_acceptance_harness import SOAK_SCENARIOS, run_scenarios


def test_soak_and_fault_injection_scenarios_are_reproducible() -> None:
    results = run_scenarios(SOAK_SCENARIOS)

    assert len(results) == 5
    assert all(result.pass_fail == "PASS" for result in results)
    assert any(result.session_state in {"recovering", "degraded"} for result in results)
    assert all(result.telemetry_snapshot["recovery_story"] for result in results)
