from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..settings import normalize_audio_aec_mode, normalize_audio_device_mode
from .recovery import FailureReason


@dataclass(frozen=True)
class AecProductMode:
    key: str
    requested_backend: str
    selected_backend: str
    audio_mode: str
    session_status: str
    ui_status: str
    telemetry_path: str
    capture_gate_policy: str
    requires_reference: bool
    degraded: bool = False
    degradation_reason: str = ""
    recovery_policy: str = "steady"
    recovery_retry_budget: int = 0


@dataclass(frozen=True)
class BackendSelectionDecision:
    requested_backend: str
    selected_backend: str
    fallback_reason: str = ""
    degradation_cause: str = ""
    probe_details: dict[str, str] = field(default_factory=dict)
    mode_contract: AecProductMode | None = None

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

    def product_mode_contract_for(
        self,
        *,
        selected_backend: str,
        audio_mode: str,
        requested_backend: Optional[str] = None,
        degradation_cause: str = "",
    ) -> AecProductMode:
        normalized_backend = normalize_audio_aec_mode(selected_backend)
        normalized_audio_mode = normalize_audio_device_mode(audio_mode)
        requested = normalize_audio_aec_mode(requested_backend or self.requested_backend_for_audio_mode(normalized_audio_mode))
        if normalized_backend == "headset_clean":
            return AecProductMode(
                key="headset_clean",
                requested_backend=requested,
                selected_backend=normalized_backend,
                audio_mode=normalized_audio_mode,
                session_status="Headset clean",
                ui_status="Audio: headset clean režim bez AEC a bez očekávání playback reference.",
                telemetry_path="headset_clean",
                capture_gate_policy="headset_clean",
                requires_reference=False,
                degraded=False,
                recovery_policy="topology_locked",
                recovery_retry_budget=0,
            )
        if normalized_backend == "degraded_no_aec":
            auto_recover = requested in {"windows_system_aec", "webrtc_apm"} and normalized_audio_mode == "notebook_builtin"
            return AecProductMode(
                key=f"{normalized_audio_mode}_degraded_no_aec",
                requested_backend=requested,
                selected_backend=normalized_backend,
                audio_mode=normalized_audio_mode,
                session_status="Nouzový režim bez AEC",
                ui_status="Audio: nouzový režim degraded_no_aec bez AEC, s konzervativním capture gate.",
                telemetry_path=f"{normalized_audio_mode}+degraded_no_aec",
                capture_gate_policy="degraded_no_aec",
                requires_reference=False,
                degraded=True,
                degradation_reason=degradation_cause,
                recovery_policy="probe_richer_backend_again" if auto_recover else "stay_degraded",
                recovery_retry_budget=2 if auto_recover else 0,
            )
        if normalized_backend == "webrtc_apm":
            return AecProductMode(
                key=f"{normalized_audio_mode}_webrtc_apm",
                requested_backend=requested,
                selected_backend=normalized_backend,
                audio_mode=normalized_audio_mode,
                session_status="Notebook builtin + WebRTC APM",
                ui_status="Audio: notebook builtin režim přes webrtc_apm.",
                telemetry_path=f"{normalized_audio_mode}+webrtc_apm",
                capture_gate_policy="webrtc_apm",
                requires_reference=True,
                degraded=False,
                recovery_policy="prefer_current_until_failure",
                recovery_retry_budget=0,
            )
        if normalized_backend == "windows_system_aec":
            return AecProductMode(
                key=f"{normalized_audio_mode}_windows_system_aec",
                requested_backend=requested,
                selected_backend=normalized_backend,
                audio_mode=normalized_audio_mode,
                session_status="Notebook builtin + Windows System AEC",
                ui_status="Audio: notebook builtin režim přes windows_system_aec.",
                telemetry_path=f"{normalized_audio_mode}+windows_system_aec",
                capture_gate_policy="windows_system_aec",
                requires_reference=False,
                degraded=False,
                recovery_policy="prefer_current_until_failure",
                recovery_retry_budget=0,
            )
        return AecProductMode(
            key=normalized_backend,
            requested_backend=requested,
            selected_backend=normalized_backend,
            audio_mode=normalized_audio_mode,
            session_status=normalized_backend,
            ui_status=f"Audio: režim {normalized_backend}.",
            telemetry_path=f"{normalized_audio_mode}+{normalized_backend}",
            capture_gate_policy=normalized_backend,
            requires_reference=normalized_backend not in {"degraded_no_aec", "headset_clean", "windows_system_aec"},
            degraded=False,
            recovery_policy="steady",
            recovery_retry_budget=0,
        )

    def select_backend(
        self,
        *,
        audio_mode: str = "notebook_builtin",
        windows_healthcheck: Callable[[], tuple[bool, str]],
        webrtc_healthcheck: Callable[[], tuple[bool, str]],
    ) -> BackendSelectionDecision:
        requested = self.requested_backend_for_audio_mode(audio_mode)
        if requested in {"custom_lab", "headset_clean"}:
            contract = self.product_mode_contract_for(
                selected_backend=requested,
                requested_backend=requested,
                audio_mode=audio_mode,
            )
            return BackendSelectionDecision(requested_backend=requested, selected_backend=requested, mode_contract=contract)

        probe_details: dict[str, str] = {}
        fallback_reason = ""
        for backend in self.backend_chain_for(requested):
            if backend == "windows_system_aec":
                ok, reason = windows_healthcheck()
                probe_details[backend] = reason
                if ok:
                    contract = self.product_mode_contract_for(
                        selected_backend=backend,
                        requested_backend=requested,
                        audio_mode=audio_mode,
                    )
                    return BackendSelectionDecision(
                        requested_backend=requested,
                        selected_backend=backend,
                        fallback_reason=fallback_reason,
                        probe_details=probe_details,
                        mode_contract=contract,
                    )
                fallback_reason = FailureReason.WINDOWS_SYSTEM_AEC_UNAVAILABLE.value
                continue
            if backend == "webrtc_apm":
                ok, reason = webrtc_healthcheck()
                probe_details[backend] = reason
                if ok:
                    contract = self.product_mode_contract_for(
                        selected_backend=backend,
                        requested_backend=requested,
                        audio_mode=audio_mode,
                    )
                    return BackendSelectionDecision(
                        requested_backend=requested,
                        selected_backend=backend,
                        fallback_reason=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                        probe_details=probe_details,
                        mode_contract=contract,
                    )
                fallback_reason = FailureReason.WEBRTC_APM_UNAVAILABLE.value
                continue
            if backend == "degraded_no_aec":
                probe_details[backend] = "Nouzový průchozí režim bez AEC je vždy dostupný."
                contract = self.product_mode_contract_for(
                    selected_backend=backend,
                    requested_backend=requested,
                    audio_mode=audio_mode,
                    degradation_cause=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                )
                return BackendSelectionDecision(
                    requested_backend=requested,
                    selected_backend=backend,
                    fallback_reason=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                    degradation_cause=fallback_reason or FailureReason.WEBRTC_APM_UNAVAILABLE.value,
                    probe_details=probe_details,
                    mode_contract=contract,
                )
        contract = self.product_mode_contract_for(
            selected_backend=requested,
            requested_backend=requested,
            audio_mode=audio_mode,
        )
        return BackendSelectionDecision(
            requested_backend=requested,
            selected_backend=requested,
            probe_details=probe_details,
            mode_contract=contract,
        )
