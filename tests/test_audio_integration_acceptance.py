from __future__ import annotations

from tests.audio_acceptance_harness import ACCEPTANCE_SCENARIOS, INTEGRATION_SCENARIOS, run_scenarios


def test_acceptance_scenarios_have_pass_verdict_and_snapshot() -> None:
    results = run_scenarios(ACCEPTANCE_SCENARIOS)

    assert len(results) == 5
    assert all(result.pass_fail == "PASS" for result in results)
    assert all(result.telemetry_snapshot["selected_backend"] for result in results)
    assert all(result.telemetry_snapshot["recovery_story"] for result in results)


def test_integration_scenarios_cover_required_backend_and_transport_paths() -> None:
    results = run_scenarios(INTEGRATION_SCENARIOS)

    assert len(results) == 7
    assert all(result.pass_fail == "PASS" for result in results)
    selected_backends = {result.selected_backend for result in results}
    assert {"windows_system_aec", "webrtc_apm", "degraded_no_aec", "headset_clean"}.issubset(selected_backends)
