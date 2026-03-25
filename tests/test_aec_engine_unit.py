from __future__ import annotations

from kajovochat.audio.aec_engine import AecEngine


def test_aec_engine_selects_windows_then_webrtc_then_degraded() -> None:
    engine = AecEngine("windows_system_aec")

    decision = engine.select_backend(
        audio_mode="notebook_builtin",
        windows_healthcheck=lambda: (False, "windows unavailable"),
        webrtc_healthcheck=lambda: (False, "webrtc unavailable"),
    )

    assert decision.requested_backend == "windows_system_aec"
    assert decision.selected_backend == "degraded_no_aec"
    assert decision.degradation_cause == "webrtc_apm_unavailable"
    assert decision.mode_contract is not None
    assert decision.mode_contract.capture_gate_policy == "degraded_no_aec"
    assert decision.mode_contract.recovery_policy == "probe_richer_backend_again"


def test_aec_engine_marks_headset_clean_as_first_class_mode() -> None:
    engine = AecEngine("windows_system_aec")

    contract = engine.product_mode_contract_for(
        selected_backend="headset_clean",
        requested_backend="headset_clean",
        audio_mode="wired_headset",
    )

    assert contract.key == "headset_clean"
    assert contract.requires_reference is False
    assert contract.capture_gate_policy == "headset_clean"
    assert contract.recovery_policy == "topology_locked"
