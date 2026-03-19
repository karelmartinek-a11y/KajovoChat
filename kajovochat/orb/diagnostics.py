from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat


@dataclass(slots=True)
class OpenGLProbeResult:
    available: bool
    backend: str
    message: str
    vendor: str = ""
    renderer: str = ""
    version: str = ""

    def summary(self) -> str:
        parts = [self.backend, self.message]
        if self.vendor:
            parts.append(f"vendor={self.vendor}")
        if self.renderer:
            parts.append(f"renderer={self.renderer}")
        if self.version:
            parts.append(f"version={self.version}")
        return " | ".join(parts)


def probe_opengl_support() -> OpenGLProbeResult:
    surface = QOffscreenSurface()
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setMajorVersion(3)
    fmt.setMinorVersion(3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    surface.setFormat(fmt)
    surface.create()

    if not surface.isValid():
        return OpenGLProbeResult(
            available=False,
            backend="fallback-2d",
            message="Qt nevytvořilo validní offscreen surface pro OpenGL.",
        )

    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create():
        return OpenGLProbeResult(
            available=False,
            backend="fallback-2d",
            message="Qt nevytvořilo OpenGL context.",
        )

    if not context.makeCurrent(surface):
        return OpenGLProbeResult(
            available=False,
            backend="fallback-2d",
            message="OpenGL context se nepodařilo aktivovat.",
        )

    actual = context.format()
    version = f"{actual.majorVersion()}.{actual.minorVersion()}"
    context.doneCurrent()
    return OpenGLProbeResult(
        available=True,
        backend="gpu-opengl",
        message="OpenGL context je dostupný.",
        version=version,
    )
