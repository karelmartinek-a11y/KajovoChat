from __future__ import annotations

from kajovochat.audio.windows_system_aec import (
    FINAL_WINDOWS_SYSTEM_AEC_VARIANT,
    WindowsSystemAecProbe,
    probe_windows_system_aec,
    windows_system_aec_healthcheck,
)
from kajovochat.services.windows_native_aec import WindowsNativeAECProbe


def test_windows_system_aec_probe_wraps_native_detail(monkeypatch) -> None:
    monkeypatch.setattr(
        "kajovochat.audio.windows_system_aec._probe_windows_native_aec_impl",
        lambda: WindowsNativeAECProbe(
            available=True,
            reason="helper detail",
            helper_path="C:/fake/backend.dll",
            installed_driver=True,
            published_name="oem42.inf",
            original_name="kajovochat_windows_apo.inf",
            provider_name="KajovoChat",
        ),
    )

    probe = probe_windows_system_aec()

    assert probe.available is True
    assert probe.implementation_variant == FINAL_WINDOWS_SYSTEM_AEC_VARIANT
    assert probe.system_capture_contract is True
    assert "systémovém capture kontraktu" in probe.reason


def test_windows_system_aec_healthcheck_exposes_session_level_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "kajovochat.audio.windows_system_aec.probe_windows_system_aec",
        lambda: WindowsSystemAecProbe(False, "Windows System AEC backend není připraven."),
    )

    health = windows_system_aec_healthcheck()

    assert health.ok is False
    assert health.reason == "Windows System AEC backend není připraven."
    assert health.implementation_variant == FINAL_WINDOWS_SYSTEM_AEC_VARIANT
    assert health.as_tuple() == (False, "Windows System AEC backend není připraven.")
