from __future__ import annotations

from dataclasses import dataclass

from ..services.windows_native_aec import (
    WindowsNativeAECProbe,
    probe_windows_native_aec as _probe_windows_native_aec_impl,
)

FINAL_WINDOWS_SYSTEM_AEC_VARIANT = "helper_backed_production_backend"


@dataclass(frozen=True)
class WindowsSystemAecProbe:
    available: bool
    reason: str
    implementation_variant: str = FINAL_WINDOWS_SYSTEM_AEC_VARIANT
    installed_driver: bool = False
    system_capture_contract: bool = False

    def to_log_payload(self) -> dict[str, object]:
        return {
            "available": bool(self.available),
            "reason": self.reason,
            "implementation_variant": self.implementation_variant,
            "installed_driver": bool(self.installed_driver),
            "system_capture_contract": bool(self.system_capture_contract),
        }


@dataclass(frozen=True)
class WindowsSystemAecHealthcheck:
    ok: bool
    reason: str
    implementation_variant: str = FINAL_WINDOWS_SYSTEM_AEC_VARIANT

    def as_tuple(self) -> tuple[bool, str]:
        return bool(self.ok), self.reason



def _public_reason_from_native_probe(probe: WindowsNativeAECProbe) -> str:
    if probe.available:
        if probe.installed_driver:
            return "Windows System AEC backend je připraven v systémovém capture kontraktu."
        return "Windows System AEC backend je připraven v render-reference kontraktu."
    if probe.installed_driver:
        return "Windows System AEC backend není připraven: systémový capture kontrakt je přítomen, ale produkční backend není kompletní."
    return "Windows System AEC backend není připraven."



def probe_windows_system_aec() -> WindowsSystemAecProbe:
    native_probe = _probe_windows_native_aec_impl()
    return WindowsSystemAecProbe(
        available=bool(native_probe.available),
        reason=_public_reason_from_native_probe(native_probe),
        installed_driver=bool(native_probe.installed_driver),
        system_capture_contract=bool(native_probe.available and native_probe.installed_driver),
    )



def windows_system_aec_healthcheck() -> WindowsSystemAecHealthcheck:
    probe = probe_windows_system_aec()
    return WindowsSystemAecHealthcheck(
        ok=bool(probe.available),
        reason=probe.reason,
    )


__all__ = [
    "FINAL_WINDOWS_SYSTEM_AEC_VARIANT",
    "WindowsSystemAecHealthcheck",
    "WindowsSystemAecProbe",
    "probe_windows_system_aec",
    "windows_system_aec_healthcheck",
]
