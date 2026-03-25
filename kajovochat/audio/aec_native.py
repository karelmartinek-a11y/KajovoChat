from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..services.windows_native_aec import (
    WindowsNativeAECBackend,
    WindowsNativeAECProbe,
    WindowsNativeAECSession,
    WindowsNativeAECSessionConfig,
    probe_windows_native_aec,
)


@dataclass
class WindowsNativeAecResources:
    """Vlastník probe/backend/session lifecycle pro native Windows AEC."""

    samplerate: int
    filter_length: int
    max_shift_samples: int

    def __post_init__(self) -> None:
        self._probe = probe_windows_native_aec()
        self._backend: Optional[WindowsNativeAECBackend] = None
        self._session: Optional[WindowsNativeAECSession] = None
        self._backend_attempted = False

    @property
    def probe(self) -> WindowsNativeAECProbe:
        return self._probe

    @property
    def backend(self) -> Optional[WindowsNativeAECBackend]:
        return self._backend

    @property
    def session(self) -> Optional[WindowsNativeAECSession]:
        return self._session

    def reconfigure(self, *, filter_length: int, max_shift_samples: int) -> None:
        self.filter_length = int(filter_length)
        self.max_shift_samples = int(max_shift_samples)
        self.reset()

    def reset(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
        self._session = None
        self._backend = None
        self._backend_attempted = False

    def ensure_backend(self) -> Optional[WindowsNativeAECBackend]:
        if self._backend is not None:
            return self._backend
        if self._backend_attempted:
            return None
        self._backend_attempted = True
        if self._probe.available:
            try:
                self._backend = WindowsNativeAECBackend(
                    input_samplerate=int(self.samplerate),
                    filter_length=int(self.filter_length),
                    max_shift_samples=int(self.max_shift_samples),
                )
            except Exception:
                self._backend = None
        return self._backend

    def ensure_session(self) -> Optional[WindowsNativeAECSession]:
        if self._session is not None:
            return self._session
        backend = self.ensure_backend()
        if backend is None:
            return None
        try:
            self._session = WindowsNativeAECSession(
                WindowsNativeAECSessionConfig(
                    samplerate=int(self.samplerate),
                    channels=1,
                    frame_samples=240,
                    filter_length=int(self.filter_length),
                    max_shift_samples=int(self.max_shift_samples),
                    device_clock_locked=True,
                ),
                probe=self._probe,
                backend=backend,
            )
            self._session.start()
        except Exception:
            self._session = None
        return self._session
