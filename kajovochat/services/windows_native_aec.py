from __future__ import annotations

import ctypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class WindowsNativeAECProbe:
    available: bool
    reason: str
    helper_path: str = ""
    installed_driver: bool = False
    published_name: str = ""
    original_name: str = ""
    provider_name: str = ""


def _candidate_helper_paths(*, prefer_apo: bool = False) -> list[Path]:
    paths: list[Path] = []
    apo_paths: list[Path] = []
    aec_paths: list[Path] = []
    env_path = os.environ.get("KAJOVOCHAT_WINDOWS_AEC_DLL", "").strip()
    if env_path:
        aec_paths.append(Path(env_path).expanduser())
    apo_env_path = os.environ.get("KAJOVOCHAT_WINDOWS_APO_DLL", "").strip()
    if apo_env_path:
        apo_paths.append(Path(apo_env_path).expanduser())

    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent.parent
    native_root = repo_root / "native" / "windows_aec_helper"
    apo_root = repo_root / "native" / "windows_apo_helper"

    for config in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel"):
        aec_paths.append(native_root / "build" / "bin" / config / "kajovochat_windows_aec.dll")
        aec_paths.append(native_root / "build" / config / "kajovochat_windows_aec.dll")
        aec_paths.append(native_root / "out" / config / "kajovochat_windows_aec.dll")
        apo_paths.append(apo_root / "build" / "bin" / config / "kajovochat_windows_apo.dll")
        apo_paths.append(apo_root / "build" / config / "kajovochat_windows_apo.dll")
        apo_paths.append(apo_root / "out" / config / "kajovochat_windows_apo.dll")

    aec_paths.append(native_root / "kajovochat_windows_aec.dll")
    apo_paths.append(apo_root / "kajovochat_windows_apo.dll")
    aec_paths.append(module_dir / "native" / "kajovochat_windows_aec.dll")
    aec_paths.append(module_dir / "kajovochat_windows_aec.dll")
    apo_paths.append(module_dir / "native" / "kajovochat_windows_apo.dll")
    apo_paths.append(module_dir / "kajovochat_windows_apo.dll")

    cwd = Path.cwd()
    aec_paths.append(cwd / "kajovochat_windows_aec.dll")
    apo_paths.append(cwd / "kajovochat_windows_apo.dll")
    paths.extend(apo_paths if prefer_apo else aec_paths)
    paths.extend(aec_paths if prefer_apo else apo_paths)
    return paths


def _probe_installed_audio_processing_object() -> tuple[bool, str, str, str]:
    if os.name != "nt":
        return False, "", "", ""

    try:
        completed = subprocess.run(
            ["pnputil", "/enum-drivers"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return False, "", "", ""

    if completed.returncode != 0 or not completed.stdout:
        return False, "", "", ""

    published_name = ""
    original_name = ""
    provider_name = ""
    class_name = ""

    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            if (
                original_name.lower() == "kajovochat_windows_apo.inf"
                and provider_name == "KajovoChat"
                and class_name == "AudioProcessingObject"
            ):
                return True, published_name, original_name, provider_name
            published_name = ""
            original_name = ""
            provider_name = ""
            class_name = ""
            continue

        if line.startswith("Published Name:"):
            published_name = line.split(":", 1)[1].strip()
        elif line.startswith("Original Name:"):
            original_name = line.split(":", 1)[1].strip()
        elif line.startswith("Provider Name:"):
            provider_name = line.split(":", 1)[1].strip()
        elif line.startswith("Class Name:"):
            class_name = line.split(":", 1)[1].strip()

    if (
        original_name.lower() == "kajovochat_windows_apo.inf"
        and provider_name == "KajovoChat"
        and class_name == "AudioProcessingObject"
    ):
        return True, published_name, original_name, provider_name

    return False, "", "", ""


def probe_windows_native_aec() -> WindowsNativeAECProbe:
    if os.name != "nt":
        return WindowsNativeAECProbe(False, "Windows-native AEC neni dostupny mimo Windows.")

    installed_driver, published_name, original_name, provider_name = _probe_installed_audio_processing_object()

    for candidate in _candidate_helper_paths(prefer_apo=installed_driver):
        try:
            if candidate.exists():
                reason = "Windows-native AEC helper nalezen."
                if installed_driver:
                    reason = f"{reason} APO driver je nainstalovany jako {published_name}."
                return WindowsNativeAECProbe(
                    True,
                    reason,
                    str(candidate),
                    installed_driver=installed_driver,
                    published_name=published_name,
                    original_name=original_name,
                    provider_name=provider_name,
                )
        except Exception:
            continue

    if installed_driver:
        return WindowsNativeAECProbe(
            False,
            f"APO driver je nainstalovany jako {published_name}, ale helper DLL nebyla nalezena.",
            installed_driver=True,
            published_name=published_name,
            original_name=original_name,
            provider_name=provider_name,
        )

    return WindowsNativeAECProbe(
        False,
        "Windows-native AEC helper nebyl nalezen. Nastav KAJOVOCHAT_WINDOWS_AEC_DLL nebo KAJOVOCHAT_WINDOWS_APO_DLL.",
    )


class WindowsNativeAECBackend:
    """Tenká adapter vrstva pro nativni Windows AEC helper DLL."""

    def __init__(self, *, input_samplerate: int, filter_length: int, max_shift_samples: int) -> None:
        probe = probe_windows_native_aec()
        if not probe.available:
            raise RuntimeError(probe.reason)
        self.input_samplerate = int(input_samplerate)
        self.filter_length = int(filter_length)
        self.max_shift_samples = int(max_shift_samples)
        self.helper_path = probe.helper_path
        self._dll = ctypes.WinDLL(self.helper_path)
        self._handle: Optional[int] = None
        self._configure_symbols()
        self._handle = int(self._create(self.input_samplerate, self.filter_length, self.max_shift_samples))
        if not self._handle:
            raise RuntimeError("Windows-native AEC helper vytvoril prazdny handle.")

    def _configure_symbols(self) -> None:
        self._create = self._dll.kajovochat_aec_create
        self._create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._create.restype = ctypes.c_void_p

        self._destroy = self._dll.kajovochat_aec_destroy
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = None

        self._process = self._dll.kajovochat_aec_process
        self._process.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._process.restype = ctypes.c_int

    def close(self) -> None:
        if self._handle:
            try:
                self._destroy(ctypes.c_void_p(self._handle))
            except Exception:
                pass
            self._handle = None

    def __del__(self) -> None:
        self.close()

    @property
    def is_ready(self) -> bool:
        return bool(self._handle)

    def process(
        self,
        *,
        mic_pcm: bytes,
        reference_pcm: np.ndarray,
        delay_ms: int,
    ) -> bytes:
        if not self._handle:
            return mic_pcm

        mic = np.frombuffer(mic_pcm, dtype=np.int16).reshape(-1)
        ref = np.asarray(reference_pcm, dtype=np.int16).reshape(-1)
        if mic.size == 0 or ref.size == 0:
            return mic_pcm

        output = np.zeros_like(mic, dtype=np.int16)
        quality = ctypes.c_double(0.0)
        improvement = ctypes.c_double(0.0)
        residual = ctypes.c_double(0.0)
        strong = ctypes.c_int(0)
        rc = self._process(
            ctypes.c_void_p(self._handle),
            mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            int(mic.size),
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            int(ref.size),
            int(delay_ms),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            int(output.size),
            ctypes.byref(quality),
            ctypes.byref(improvement),
            ctypes.byref(residual),
            ctypes.byref(strong),
        )
        if rc != 0:
            raise RuntimeError(f"Windows-native AEC helper vratil chybu rc={rc}.")
        return output.tobytes()
