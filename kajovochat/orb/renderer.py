from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .config import LivingOrbConfig
from .controller import OrbFrameParameters
from .shaders import FRAGMENT_SHADER, VERTEX_SHADER


class LivingOrbRenderer:
    def __init__(self, config: LivingOrbConfig) -> None:
        self.config = config
        self._ctx: moderngl.Context | None = None
        self._program: moderngl.Program | None = None
        self._vao: moderngl.VertexArray | None = None
        self._vbo: moderngl.Buffer | None = None
        self._width = 1
        self._height = 1

    def initialize(self) -> None:
        try:
            import moderngl
        except Exception as exc:
            raise RuntimeError("Chybí dependency moderngl pro GPU living orb renderer.") from exc
        try:
            self._ctx = moderngl.create_context()
        except Exception as exc:
            raise RuntimeError(f"Nepodařilo se vytvořit OpenGL context pro living orb: {exc}") from exc
        self._program = self._ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        quad = np.asarray(
            [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            dtype="f4",
        )
        self._vbo = self._ctx.buffer(quad.tobytes())
        self._vao = self._ctx.simple_vertex_array(self._program, self._vbo, "in_pos")
        self.resize(self._width, self._height)

    @property
    def is_ready(self) -> bool:
        return self._ctx is not None and self._program is not None and self._vao is not None

    def resize(self, width: int, height: int) -> None:
        self._width = max(1, int(width))
        self._height = max(1, int(height))
        if self._ctx is not None:
            self._ctx.viewport = (0, 0, self._width, self._height)

    def render(self, frame: OrbFrameParameters) -> None:
        if not self.is_ready:
            raise RuntimeError("Renderer living orb není inicializovaný.")
        import moderngl
        assert self._ctx is not None
        assert self._program is not None
        assert self._vao is not None
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._ctx.clear(0.0, 0.0, 0.0, 0.0)
        self._program["u_resolution"].value = (float(self._width), float(self._height))
        self._program["u_background_color"].value = self.config.background_color
        self._program["u_haze_color"].value = self.config.haze_color
        self._program["u_core_color"].value = self.config.core_color
        self._program["u_glow_color"].value = self.config.glow_color
        self._program["u_aura_color"].value = self.config.aura_color
        self._program["u_edge_color"].value = self.config.edge_color
        for key, value in asdict(frame).items():
            uniform_name = f"u_{key}"
            try:
                self._program[uniform_name].value = float(value)
            except KeyError:
                continue
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def shutdown(self) -> None:
        for obj in (self._vao, self._vbo, self._program):
            if obj is None:
                continue
            try:
                obj.release()
            except Exception:
                pass
        self._vao = None
        self._vbo = None
        self._program = None
        if self._ctx is not None:
            try:
                self._ctx.release()
            except Exception:
                pass
        self._ctx = None
