from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio.contracts import BackendHealthSnapshot, CaptureFrame, RenderFrame


@dataclass(frozen=True)
class WindowsNativeAECProbe:
    available: bool
    reason: str
    helper_path: str = ""
    installed_driver: bool = False
    published_name: str = ""
    original_name: str = ""
    provider_name: str = ""


@dataclass(frozen=True)
class WindowsNativeAECSessionConfig:
    samplerate: int
    channels: int = 1
    frame_samples: int = 240
    filter_length: int = 256
    max_shift_samples: int = 960
    device_clock_locked: bool = True


@dataclass(frozen=True)
class WindowsNativeAECSessionHealth:
    frame_index: int
    processed_frames: int
    xruns: int
    device_resets: int
    last_error: str
    backend_snapshot: BackendHealthSnapshot


def _candidate_helper_paths(*, prefer_apo: bool = False) -> list[Path]:
    paths: list[Path] = []
    apo_paths: list[Path] = []
    aec_paths: list[Path] = []
    env_path = os.environ.get("KAJOVOCHAT_WINDOWS_AEC_DLL", "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())
    apo_env_path = os.environ.get("KAJOVOCHAT_WINDOWS_APO_DLL", "").strip()
    if apo_env_path:
        paths.append(Path(apo_env_path).expanduser())

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
        self._capture_handle: Optional[int] = None
        self._using_system_capture_contract = False
        self._last_quality = 0.0
        self._last_voice_probability = 0.0
        self._last_improvement = 0.0
        self._last_residual = 0.0
        self._last_flags = 0
        self._configure_symbols()
        if probe.installed_driver and self._apo_capture_available:
            self._capture_handle = int(self._apo_capture_create(self.input_samplerate))
            if self._capture_handle:
                self._using_system_capture_contract = True
        if not self._using_system_capture_contract:
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
        self._apo_capture_available = all(
            hasattr(self._dll, symbol)
            for symbol in (
                "kajovochat_apo_capture_create",
                "kajovochat_apo_capture_destroy",
                "kajovochat_apo_capture_process",
            )
        )
        if self._apo_capture_available:
            self._apo_capture_create = self._dll.kajovochat_apo_capture_create
            self._apo_capture_create.argtypes = [ctypes.c_int]
            self._apo_capture_create.restype = ctypes.c_void_p

            self._apo_capture_destroy = self._dll.kajovochat_apo_capture_destroy
            self._apo_capture_destroy.argtypes = [ctypes.c_void_p]
            self._apo_capture_destroy.restype = None

            self._apo_capture_process = self._dll.kajovochat_apo_capture_process
            self._apo_capture_process.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
            ]
            self._apo_capture_process.restype = ctypes.c_int

    def close(self) -> None:
        if self._capture_handle:
            try:
                self._apo_capture_destroy(ctypes.c_void_p(self._capture_handle))
            except Exception:
                pass
            self._capture_handle = None
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
        return bool(self._handle or self._capture_handle)

    @property
    def using_system_capture_contract(self) -> bool:
        return bool(self._using_system_capture_contract)

    @property
    def last_quality(self) -> float:
        return float(self._last_quality)

    @property
    def last_voice_probability(self) -> float:
        return float(self._last_voice_probability)

    @property
    def last_improvement(self) -> float:
        return float(self._last_improvement)

    @property
    def last_residual(self) -> float:
        return float(self._last_residual)

    @property
    def last_flags(self) -> int:
        return int(self._last_flags)

    def process(
        self,
        *,
        mic_pcm: bytes,
        reference_pcm: np.ndarray,
        delay_ms: int,
    ) -> bytes:
        if self._capture_handle:
            mic = np.frombuffer(mic_pcm, dtype=np.int16).reshape(-1)
            if mic.size == 0:
                return mic_pcm
            output = np.zeros_like(mic, dtype=np.int16)
            quality = ctypes.c_double(0.0)
            voice = ctypes.c_double(0.0)
            flags = ctypes.c_int(0)
            rc = self._apo_capture_process(
                ctypes.c_void_p(self._capture_handle),
                mic.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                int(mic.size),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                int(output.size),
                ctypes.byref(quality),
                ctypes.byref(voice),
                ctypes.byref(flags),
            )
            if rc != 0:
                raise RuntimeError(f"Windows APO capture helper vratil chybu rc={rc}.")
            self._last_quality = float(quality.value)
            self._last_voice_probability = float(voice.value)
            self._last_improvement = 0.0
            self._last_residual = 0.0
            self._last_flags = int(flags.value)
            return output.tobytes()
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
        self._last_quality = float(quality.value)
        self._last_voice_probability = 0.0
        self._last_improvement = float(improvement.value)
        self._last_residual = float(residual.value)
        self._last_flags = int(strong.value)
        return output.tobytes()


class WindowsNativeAECSession:
    """Session-oriented obalka nad Windows native AEC backendem."""

    def __init__(
        self,
        config: WindowsNativeAECSessionConfig,
        *,
        probe: Optional[WindowsNativeAECProbe] = None,
        backend: Optional[WindowsNativeAECBackend] = None,
    ) -> None:
        self.config = config
        self.probe = probe or probe_windows_native_aec()
        if not self.probe.available:
            raise RuntimeError(self.probe.reason)
        self._backend = backend or WindowsNativeAECBackend(
            input_samplerate=int(config.samplerate),
            filter_length=int(config.filter_length),
            max_shift_samples=int(config.max_shift_samples),
        )
        self._pending_frames: queue.Queue[CaptureFrame] = queue.Queue()
        self._started = False
        self._frame_index = 0
        self._processed_frames = 0
        self._xruns = 0
        self._device_resets = 0
        self._last_error = ""
        self._last_render_frame: Optional[RenderFrame] = None

    @property
    def is_started(self) -> bool:
        return bool(self._started)

    @property
    def using_system_capture_contract(self) -> bool:
        return bool(getattr(self._backend, "using_system_capture_contract", False))

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False
        while True:
            try:
                self._pending_frames.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self.stop()
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def write_render_frame(self, frame: RenderFrame) -> None:
        self._last_render_frame = frame

    def submit_capture_frame(
        self,
        *,
        raw_mic_pcm16: bytes,
        mono_ns: Optional[int] = None,
        stream_delay_ms: int = 0,
        render_ref_pcm16: Optional[bytes] = None,
    ) -> None:
        if not self._started:
            raise RuntimeError("Windows native AEC session neni spustena.")
        capture_mono_ns = int(mono_ns if mono_ns is not None else time.monotonic_ns())
        render_ref = render_ref_pcm16
        if render_ref is None and self._last_render_frame is not None:
            render_ref = self._last_render_frame.pcm16
        try:
            if self.using_system_capture_contract:
                processed_pcm16 = self._backend.process(
                    mic_pcm=raw_mic_pcm16,
                    reference_pcm=np.empty((0,), dtype=np.int16),
                    delay_ms=int(stream_delay_ms),
                )
            else:
                reference_array = np.frombuffer(render_ref or b"", dtype=np.int16).copy()
                processed_pcm16 = self._backend.process(
                    mic_pcm=raw_mic_pcm16,
                    reference_pcm=reference_array,
                    delay_ms=int(stream_delay_ms),
                )
            frame = CaptureFrame(
                frame_index=int(self._frame_index),
                mono_ns=capture_mono_ns,
                raw_mic_pcm16=raw_mic_pcm16,
                processed_mic_pcm16=processed_pcm16,
                render_ref_pcm16=render_ref,
                sample_rate=int(self.config.samplerate),
                channels=int(self.config.channels),
                aec_backend="windows_system_capture" if self.using_system_capture_contract else "windows_native",
                aec_quality=float(getattr(self._backend, "last_quality", 0.0)),
                residual_level=float(getattr(self._backend, "last_residual", 0.0)),
                vad_probability=float(getattr(self._backend, "last_voice_probability", 0.0)),
                double_talk=bool(int(getattr(self._backend, "last_flags", 0)) & 0x1) if self.using_system_capture_contract else False,
                stream_delay_ms=int(stream_delay_ms),
                device_clock_locked=bool(self.config.device_clock_locked),
            )
            self._pending_frames.put(frame)
            self._processed_frames += 1
            self._frame_index += 1
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            raise

    def read_capture_frame(self, timeout_ms: int = 0) -> Optional[CaptureFrame]:
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)
        try:
            if timeout_s <= 0.0:
                return self._pending_frames.get_nowait()
            return self._pending_frames.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def get_health_snapshot(self) -> WindowsNativeAECSessionHealth:
        backend_name = "windows_system_capture" if self.using_system_capture_contract else "windows_native"
        return WindowsNativeAECSessionHealth(
            frame_index=int(self._frame_index),
            processed_frames=int(self._processed_frames),
            xruns=int(self._xruns),
            device_resets=int(self._device_resets),
            last_error=self._last_error,
            backend_snapshot=BackendHealthSnapshot(
                backend=backend_name,
                health_score=1.0 if self._started and not self._last_error else 0.0,
                requested_backend="windows_system_aec",
                audio_mode="notebook_builtin",
                reference_ready=bool(self.using_system_capture_contract or self._last_render_frame is not None),
                reference_available_samples=0 if self.using_system_capture_contract else len((self._last_render_frame.pcm16 if self._last_render_frame else b"")) // 2,
                reference_callback_age_ms=0,
                reference_health_state="system_capture" if self.using_system_capture_contract else "render_feed",
                poor_aec_events=0,
                poor_aec_consecutive=0,
                fallback_reason="",
                degradation_cause="",
                last_failure_reason=self._last_error,
                reference_loss_ratio=0.0,
                aec_effective_ratio=max(0.0, min(1.0, float(getattr(self._backend, "last_quality", 0.0)))),
                double_talk_ratio=1.0 if bool(int(getattr(self._backend, "last_flags", 0)) & 0x1) else 0.0,
                barge_in_success_ratio=0.0,
                recoveries=0,
                xruns=int(self._xruns),
                device_resets=int(self._device_resets),
            ),
        )


def open_windows_native_aec_session(config: WindowsNativeAECSessionConfig) -> WindowsNativeAECSession:
    session = WindowsNativeAECSession(config)
    session.start()
    return session
