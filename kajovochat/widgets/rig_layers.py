from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QTransform

from .sphere_renderer import qimage_to_rgba_numpy


@dataclass(slots=True)
class RigLayer:
    name: str
    role: str
    path: str
    absolute_path: str
    anchor_x: float
    anchor_y: float
    pivot_x: float
    pivot_y: float
    opacity: float
    enabled: bool
    exists: bool
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class RigDefinition:
    meta: dict[str, Any]
    canvas: dict[str, Any]
    fallback: dict[str, Any]
    pivots: dict[str, Any]
    masks: dict[str, Any]
    deformation_ranges: dict[str, Any]
    z_order: dict[str, Any]
    state_presets: dict[str, Any]
    fallback_layers: list[RigLayer] = field(default_factory=list)
    production_layers: list[RigLayer] = field(default_factory=list)
    fallback_mode: bool = True
    production_ready: bool = False
    issues: list[str] = field(default_factory=list)
    assets_root: str = ""

    def active_layers(self) -> list[RigLayer]:
        return self.fallback_layers if self.fallback_mode else self.production_layers

    def find_layer(self, name: str) -> RigLayer | None:
        for layer in self.fallback_layers + self.production_layers:
            if layer.name == name:
                return layer
        return None


@lru_cache(maxsize=32)
def load_image_cached(path: str) -> QImage:
    if not path:
        return QImage()
    return QImage(path)


def layer_image(layer: RigLayer) -> QImage:
    return load_image_cached(layer.absolute_path) if layer.exists else QImage()


@lru_cache(maxsize=32)
def load_content_bbox_cached(path: str) -> tuple[float, float, float, float]:
    image = load_image_cached(path)
    if image.isNull():
        return (0.0, 0.0, 1.0, 1.0)

    rgba = qimage_to_rgba_numpy(image)
    if rgba.size == 0:
        return (0.0, 0.0, float(image.width()), float(image.height()))

    rgb = rgba[..., :3].astype("int16")
    samples = [
        rgb[0, 0],
        rgb[0, -1],
        rgb[-1, 0],
        rgb[-1, -1],
        rgb[min(10, rgb.shape[0] - 1), min(10, rgb.shape[1] - 1)],
    ]
    bg = sum(samples) / float(len(samples))
    distance = ((rgb - bg) ** 2).sum(axis=2) ** 0.5
    bright = rgb.mean(axis=2)
    alpha = rgba[..., 3].astype("uint8").copy()
    mask = (distance < 22.0) | ((distance < 34.0) & (bright > 220.0))
    alpha[mask] = 0
    edge = 12
    alpha[:edge, :] = 0
    alpha[-edge:, :] = 0
    alpha[:, :edge] = 0
    alpha[:, -edge:] = 0

    ys, xs = (alpha > 8).nonzero()
    if xs.size == 0 or ys.size == 0:
        return (0.0, 0.0, float(image.width()), float(image.height()))

    left = float(xs.min())
    top = float(ys.min())
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return (left, top, width, height)


def layer_content_bbox(layer: RigLayer) -> QRectF:
    if not layer.exists:
        return QRectF(0.0, 0.0, 1.0, 1.0)
    left, top, width, height = load_content_bbox_cached(layer.absolute_path)
    return QRectF(left, top, width, height)


def layer_target_rect(frame_rect: QRectF, layer: RigLayer) -> QRectF:
    width = frame_rect.width()
    height = frame_rect.height()
    left = frame_rect.left() + width * (layer.anchor_x - layer.pivot_x)
    top = frame_rect.top() + height * (layer.anchor_y - layer.pivot_y)
    return QRectF(left, top, width, height)


def build_pivot_transform(target_rect: QRectF, *, tx: float = 0.0, ty: float = 0.0, rot_deg: float = 0.0, scale: float = 1.0) -> QTransform:
    pivot = QPointF(target_rect.center().x(), target_rect.center().y())
    transform = QTransform()
    transform.translate(pivot.x() + tx, pivot.y() + ty)
    transform.rotate(rot_deg)
    transform.scale(scale, scale)
    transform.translate(-pivot.x(), -pivot.y())
    return transform


def apply_mask(image: QImage, mask: QImage) -> QImage:
    if image.isNull() or mask.isNull():
        return image
    result = QImage(image.size(), QImage.Format_ARGB32_Premultiplied)
    result.fill(0)
    painter = QPainter(result)
    painter.drawImage(0, 0, image)
    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    painter.drawImage(result.rect(), mask)
    painter.end()
    return result


def rig_definition_from_manifest(manifest: dict[str, Any]) -> RigDefinition:
    runtime = dict(manifest.get("runtime", {}))

    def _build_layers(items: list[dict[str, Any]]) -> list[RigLayer]:
        layers: list[RigLayer] = []
        for item in items:
            layers.append(
                RigLayer(
                    name=str(item.get("name", "")),
                    role=str(item.get("role", "")),
                    path=str(item.get("path", "")),
                    absolute_path=str(item.get("absolute_path", "")),
                    anchor_x=float(item.get("anchor_x", 0.5)),
                    anchor_y=float(item.get("anchor_y", 0.5)),
                    pivot_x=float(item.get("pivot_x", 0.5)),
                    pivot_y=float(item.get("pivot_y", 0.5)),
                    opacity=float(item.get("opacity", 1.0)),
                    enabled=bool(item.get("enabled", True)),
                    exists=bool(item.get("exists", False)),
                    width=int(item["width"]) if item.get("width") else None,
                    height=int(item["height"]) if item.get("height") else None,
                )
            )
        return layers

    resolved_layers = runtime.get("resolved_layers", {})
    return RigDefinition(
        meta=dict(manifest.get("meta", {})),
        canvas=dict(manifest.get("canvas", {})),
        fallback=dict(manifest.get("fallback", {})),
        pivots=dict(manifest.get("pivots", {})),
        masks=dict(manifest.get("masks", {})),
        deformation_ranges=dict(manifest.get("deformation_ranges", {})),
        z_order=dict(manifest.get("z_order", {})),
        state_presets=dict(manifest.get("state_presets", {})),
        fallback_layers=_build_layers(list(resolved_layers.get("fallback", []))),
        production_layers=_build_layers(list(resolved_layers.get("production", []))),
        fallback_mode=bool(runtime.get("fallback_mode", True)),
        production_ready=bool(runtime.get("production_ready", False)),
        issues=[str(item) for item in runtime.get("issues", [])],
        assets_root=str(runtime.get("assets_root", "")),
    )
