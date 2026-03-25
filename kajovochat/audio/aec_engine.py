from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..settings import normalize_audio_aec_mode, normalize_audio_device_mode
from .recovery import FailureReason


@dataclass(frozen=True)
class BackendSelectionDecision:
    requested_backend: str
    selected_backend: str
    fallback_reason: str = ""
    degradation_cause: str = ""
    probe_details: dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return self.selected_backend == "degraded_no_aec"


@dataclass(frozen=True)
class AecEngine:
    configured_mode: str

    def requested_backend_for_audio_mode(self, audio_mode: str) -> str:
        requested = normalize_audio_aec_mode(self.configured_mode)
        if requested in {"windows_system_aec", "webrtc_apm"} and normalize_audio_device_mode(audio_mode) in {
            "wired_headset",
            "bluetooth_headset",
            "external_headphones",
        }:
            return "headset_clean"
        return requested

    @property
    def requested_backend(self) -> str:
        return self.requested_backend_for_audio_mode("notebook_builtin")

    def backend_chain_for(self, requested_backend: str) -> tuple[str, ...]:
        mode = normalize_audio_aec_mode(requested_backend)
        if mode == "windows_system_aec":
            return ("windows_system_aec", "webrtc_apm", "degraded_no_aec")
        if mode == "webrtc_apm":
            return ("webrtc_apm", "degraded_no_aec")
        if mode == "headset_clean":
            return ("headset_clean",)
        if mode == "degraded_no_aec":
            return ("degraded_no_aec",)
        return ("custom_lab",)

    @property
    def backend_chain(self) -> tuple[str, ...]:
        return self.backend_chain_for(self.requested_backend)

    def next_backend_after(self, current_backend: str, *, requested_backend: Optional[str] = None) -> Optional[str]:
        chain = self.backend_chain_for(requested_backend or self.requested_backend)
        try:
            index = chain.index(normalize_audio_aec_mode(current_backend))
        except ValueError:
            return chain[0] if chain else None
        next_index = index + 1
        if next_index >= len(chain):
            return None
        return chain[next_index]

    def select_backend(
        self,
        *,
        audio_mode: str = "notebook_builtin",
        windows_healthcheck: Callable[[], tuple[bool, str]],
        webrtc_healthcheck: Callable[[], tuple[bool, str]],
    ) -> BackendSelectionDecision:
        requested = self.requested_backend_for_audio_mode(audio_mode)
        if requested in {"custom_lab", "headset_clean"}:
            return BackendSelectionDecision(requested_backend=requested, selected_backend=requested)

        probe_details: dict[str, str] = {}
        fallback_reason = ""
        for backend in self.backend_chain_for(requested):
            if backend == "windows_system_aec":
                ok, reason = windows_healthcheck()
                probe_details[backend] = reason
                if ok:
                    return BackendSelectionDecision(
                        requested_backend=requested,
                        selected_backend=backend,
                        fallback_reason=fallback_reason,
                        probe_details=probe_details,
                    )
                fallback_reason = FailureReason.WINDOWS_SYSTEM_AEC_UNAVAILABLE.value
                continue
            if backend == "webrtc_apm":
                ok, reason = webrtc_healthcheck()
                probe_details[backend] = reason
                if ok:
                    return BackendSelectionDecision(
                        requested_backend=requested,
                        selected_backend=backend,
                        fallback_reason=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                        probe_details=probe_details,
                    )
                fallback_reason = FailureReason.WEBRTC_APM_UNAVAILABLE.value
                continue
            if backend == "degraded_no_aec":
                probe_details[backend] = "Nouzový průchozí režim bez AEC je vždy dostupný."
                return BackendSelectionDecision(
                    requested_backend=requested,
                    selected_backend=backend,
                    fallback_reason=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                    degradation_cause=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                    probe_details=probe_details,
                )
        return BackendSelectionDecision(
            requested_backend=requested,
            selected_backend=requested,
            probe_details=probe_details,
        )
