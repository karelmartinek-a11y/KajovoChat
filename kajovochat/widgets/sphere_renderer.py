from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from PySide6.QtGui import QImage


def _to_rgba8888(img: QImage) -> QImage:
    if img.isNull():
        return QImage()
    if img.format() != QImage.Format_RGBA8888:
        return img.convertToFormat(QImage.Format_RGBA8888)
    return img


def qimage_to_rgba_numpy(img: QImage) -> np.ndarray:
    """QImage -> numpy uint8 array (H, W, 4) in RGBA order."""
    img = _to_rgba8888(img)
    if img.isNull():
        return np.zeros((0, 0, 4), dtype=np.uint8)

    w, h = img.width(), img.height()
    # PySide6 vrací memoryview bez setsize(); použijeme bezpečný převod.
    mv = memoryview(img.bits())
    # tobytes() in PySide6 does not accept a length; slice manually to the expected size.
    buf = mv.tobytes()
    if len(buf) < img.sizeInBytes():
        buf = mv.tobytes()  # fallback; shouldn't happen
    buf = buf[: img.sizeInBytes()]
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, img.bytesPerLine()))
    arr = arr[:, : w * 4].reshape((h, w, 4)).copy()  # detach from Qt buffer
    return arr


def rgba_numpy_to_qimage(arr: np.ndarray) -> QImage:
    """numpy uint8 array (H, W, 4) RGBA -> QImage (deep copy)."""
    if arr.dtype != np.uint8:
        raise TypeError("Expected uint8")
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("Expected (H, W, 4)")

    h, w = int(arr.shape[0]), int(arr.shape[1])
    img = QImage(w, h, QImage.Format_RGBA8888)
    mv = memoryview(img.bits())
    out = np.ndarray(shape=(h, img.bytesPerLine()), dtype=np.uint8, buffer=mv)
    out[:, : w * 4] = arr.reshape((h, w * 4))
    return img


@dataclass(frozen=True)
class SphereGrid:
    size: int
    idx_flat: np.ndarray  # (N,) flat indices into size*size
    x: np.ndarray  # (N,) normalized coords [-1..1]
    y: np.ndarray
    z: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    alpha: np.ndarray  # (N,) 0..255


@lru_cache(maxsize=16)
def get_sphere_grid(size: int) -> SphereGrid:
    size = int(size)
    if size < 8:
        size = 8

    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    # Centered NDC coordinates. +0.5 aligns sampling to pixel centers.
    cx = size / 2.0
    cy = size / 2.0
    x = (xs + 0.5 - cx) / cx
    y = -((ys + 0.5 - cy) / cy)
    d2 = x * x + y * y
    mask = d2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(np.maximum(0.0, 1.0 - d2[mask]))

    # Anti-aliased edge alpha.
    d = np.sqrt(np.maximum(0.0, d2))
    edge = max(2.0 / size, 0.002)
    a = np.clip((1.0 - d) / edge, 0.0, 1.0)

    idx_flat = np.flatnonzero(mask.ravel())
    x_f = x.ravel()[idx_flat]
    y_f = y.ravel()[idx_flat]
    z_f = z.ravel()[idx_flat]

    # On a unit sphere, position vector equals normal.
    nx, ny, nz = x_f, y_f, z_f
    alpha = (a.ravel()[idx_flat] * 255.0).astype(np.uint8)

    return SphereGrid(
        size=size,
        idx_flat=idx_flat,
        x=x_f,
        y=y_f,
        z=z_f,
        nx=nx,
        ny=ny,
        nz=nz,
        alpha=alpha,
    )


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return (x / n, y / n, z / n)


def _bilinear_sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray, wrap_x: bool = True) -> np.ndarray:
    """Sample tex (H,W,C) float32 in [0..1] at u,v (N,) -> (N,C)."""
    th, tw, c = tex.shape
    # Map to texture pixel coordinates.
    x = u * (tw - 1)
    y = v * (th - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    if wrap_x:
        x0 = np.mod(x0, tw)
        x1 = np.mod(x1, tw)
    else:
        x0 = np.clip(x0, 0, tw - 1)
        x1 = np.clip(x1, 0, tw - 1)

    y0 = np.clip(y0, 0, th - 1)
    y1 = np.clip(y1, 0, th - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]

    wx = wx[:, None]
    wy = wy[:, None]
    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    return c0 * (1.0 - wy) + c1 * wy


class SphereRenderer:
    """CPU sphere shading using numpy (fast enough for UI-scale orbs)."""

    def __init__(
        self,
        base_texture: QImage,
        clouds_texture: Optional[QImage] = None,
    ) -> None:
        base = qimage_to_rgba_numpy(base_texture)
        if base.size == 0:
            self._base_rgb = np.zeros((1, 1, 3), dtype=np.float32)
        else:
            self._base_rgb = (base[..., :3].astype(np.float32) / 255.0)

        self._clouds_rgba: Optional[np.ndarray] = None
        if clouds_texture is not None and not clouds_texture.isNull():
            clouds = qimage_to_rgba_numpy(clouds_texture)
            if clouds.size:
                self._clouds_rgba = clouds.astype(np.float32) / 255.0

    def render_moon(
        self,
        size: int,
        angle_deg: float,
        light_dir: Tuple[float, float, float] = (-0.45, 0.25, 0.85),
    ) -> QImage:
        grid = get_sphere_grid(size)
        lx, ly, lz = _normalize(light_dir)

        rot = math.radians(angle_deg)
        lon = np.arctan2(grid.x, grid.z) + rot
        u = (lon / (2.0 * math.pi) + 0.5) % 1.0
        v = 0.5 - (np.arcsin(np.clip(grid.y, -1.0, 1.0)) / math.pi)
        v = np.clip(v, 0.0, 1.0)

        tex = _bilinear_sample(self._base_rgb, u, v, wrap_x=True)

        ndl = np.clip(grid.nx * lx + grid.ny * ly + grid.nz * lz, 0.0, 1.0)
        # Stronger terminator + subtle limb darkening.
        shade = (0.18 + 0.92 * (ndl ** 0.85))
        shade *= (0.60 + 0.40 * grid.nz)
        rgb = tex * shade[:, None]

        # Slight contrast boost in linear space.
        rgb = np.clip(rgb * 1.08, 0.0, 1.0)

        return self._compose_rgba_image(grid, rgb, atmosphere=False)

    def render_earth(
        self,
        size: int,
        angle_deg: float,
        clouds_angle_deg: Optional[float] = None,
        light_dir: Tuple[float, float, float] = (-0.55, 0.18, 0.82),
    ) -> QImage:
        grid = get_sphere_grid(size)
        lx, ly, lz = _normalize(light_dir)
        rot = math.radians(angle_deg)

        lon = np.arctan2(grid.x, grid.z) + rot
        u = (lon / (2.0 * math.pi) + 0.5) % 1.0
        v = 0.5 - (np.arcsin(np.clip(grid.y, -1.0, 1.0)) / math.pi)
        v = np.clip(v, 0.0, 1.0)

        tex = _bilinear_sample(self._base_rgb, u, v, wrap_x=True)

        ndl = np.clip(grid.nx * lx + grid.ny * ly + grid.nz * lz, 0.0, 1.0)
        shade = 0.22 + 0.95 * (ndl ** 1.0)
        shade *= (0.58 + 0.42 * grid.nz)  # limb darkening
        rgb = tex * shade[:, None]

        # Ocean-ish pixels get specular highlight.
        r, g, b = tex[:, 0], tex[:, 1], tex[:, 2]
        ocean = (b > (g + 0.05)) & (b > (r + 0.05)) & (b > 0.20)
        ocean_f = ocean.astype(np.float32)

        vx, vy, vz = (0.0, 0.0, 1.0)
        hx, hy, hz = _normalize((lx + vx, ly + vy, lz + vz))
        ndh = np.clip(grid.nx * hx + grid.ny * hy + grid.nz * hz, 0.0, 1.0)
        spec = (ndh ** 60.0) * 0.65 * ocean_f * (ndl ** 0.85)
        rgb = np.clip(rgb + spec[:, None], 0.0, 1.0)

        # Clouds: rotate slightly faster than surface.
        if self._clouds_rgba is not None:
            ca = clouds_angle_deg
            if ca is None:
                ca = angle_deg * 1.35
            c_rot = math.radians(ca)
            c_lon = np.arctan2(grid.x, grid.z) + c_rot
            cu = (c_lon / (2.0 * math.pi) + 0.5) % 1.0
            ctex = _bilinear_sample(self._clouds_rgba, cu, v, wrap_x=True)
            cloud_alpha = np.clip(ctex[:, 3] * 0.40, 0.0, 0.40)  # already 0..1
            cloud_lit = (0.80 + 0.35 * ndl)  # brighten on day side
            clouds_rgb = np.clip(ctex[:, :3] * cloud_lit[:, None], 0.0, 1.0)
            rgb = rgb * (1.0 - cloud_alpha[:, None]) + clouds_rgb * cloud_alpha[:, None]

        # Saturation and gamma for display.
        rgb = np.clip(rgb, 0.0, 1.0)

        return self._compose_rgba_image(grid, rgb, atmosphere=True)

    def _compose_rgba_image(self, grid: SphereGrid, rgb_linear: np.ndarray, atmosphere: bool) -> QImage:
        # Apply gamma for display.
        rgb = np.power(np.clip(rgb_linear, 0.0, 1.0), 1.0 / 2.2)

        out = np.zeros((grid.size * grid.size, 4), dtype=np.uint8)
        out[grid.idx_flat, :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
        out[grid.idx_flat, 3] = grid.alpha

        if atmosphere:
            # Blue rim based on view-angle (nz).
            rim = np.power(np.clip(1.0 - grid.nz, 0.0, 1.0), 3.0)
            rim = rim * (grid.alpha.astype(np.float32) / 255.0)
            # Additive atmosphere in sRGB-ish space.
            add = np.zeros((grid.size * grid.size, 3), dtype=np.float32)
            add[grid.idx_flat, 0] = 0.18 * rim
            add[grid.idx_flat, 1] = 0.35 * rim
            add[grid.idx_flat, 2] = 0.75 * rim
            rgb_u8 = out[:, :3].astype(np.float32) / 255.0
            rgb_u8 = np.clip(rgb_u8 + add, 0.0, 1.0)
            out[:, :3] = (rgb_u8 * 255.0 + 0.5).astype(np.uint8)

        out = out.reshape((grid.size, grid.size, 4))
        return rgba_numpy_to_qimage(out)
