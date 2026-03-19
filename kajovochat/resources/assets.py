from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PySide6.QtGui import QImage


_REQUIRED_TALKING_HEAD_KEYS = (
    "meta",
    "canvas",
    "fallback",
    "layers",
    "pivots",
    "masks",
    "deformation_ranges",
    "z_order",
    "state_presets",
)
_REQUIRED_LAYER_KEYS = ("path", "anchor_x", "anchor_y", "pivot_x", "pivot_y", "opacity", "enabled")


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def asset_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "assets_manifest.json"


def talking_head_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "talking_head_manifest.json"


def load_asset_manifest() -> dict[str, Any]:
    return json.loads(asset_manifest_path().read_text(encoding="utf-8"))


def _asset_file_entries(manifest: dict[str, Any]) -> dict[str, dict[str, object]]:
    files = manifest.get("files")
    if isinstance(files, dict):
        return {str(name): value for name, value in files.items() if isinstance(value, dict)}
    return {str(name): value for name, value in manifest.items() if isinstance(value, dict) and "sha256" in value}


def verify_asset_manifest(*, max_asset_bytes: int = 5_000_000) -> list[str]:
    issues: list[str] = []
    manifest = load_asset_manifest()
    root = assets_dir()
    files = _asset_file_entries(manifest)

    for name, expected in files.items():
        path = root / name
        if not path.exists():
            issues.append(f"Chybí asset: {name}")
            continue
        data = path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        actual_size = len(data)
        if actual_hash != str(expected.get("sha256", "")).lower():
            issues.append(f"Nesedí hash assetu: {name}")
        if actual_size != int(expected.get("bytes", -1)):
            issues.append(f"Nesedí velikost assetu: {name}")
        if actual_size > max_asset_bytes:
            issues.append(f"Asset překračuje limit {max_asset_bytes} B: {name}")

    for path in root.glob("*"):
        if path.is_file() and path.name not in files:
            issues.append(f"Asset chybí v manifestu: {path.name}")

    return issues


def _ensure_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Sekce '{label}' musí být objekt.")
    return value


def _ensure_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Sekce '{label}' musí být pole.")
    return value


def _resolve_asset_path(assets_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else assets_root / relative_path


def _image_size(path: Path) -> tuple[int, int] | None:
    image = QImage(str(path))
    if image.isNull():
        return None
    return image.width(), image.height()


def _validate_layer_block(
    layers: list[Any],
    *,
    label: str,
    assets_root: Path,
    canvas: dict[str, Any],
    missing_fatal: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    resolved_layers: list[dict[str, Any]] = []
    canvas_width = int(canvas.get("width", 0) or 0)
    canvas_height = int(canvas.get("height", 0) or 0)

    for index, raw_layer in enumerate(layers):
        layer = _ensure_mapping(raw_layer, label=f"{label}[{index}]")
        missing_keys = [key for key in _REQUIRED_LAYER_KEYS if key not in layer]
        if missing_keys:
            issues.append(f"Vrstva '{label}[{index}]' postrádá klíče: {', '.join(missing_keys)}")
            continue

        relative_path = str(layer.get("path", "")).strip()
        if not relative_path:
            issues.append(f"Vrstva '{label}[{index}]' má prázdnou cestu assetu.")
            continue

        absolute_path = _resolve_asset_path(assets_root, relative_path)
        exists = absolute_path.exists()
        if not exists:
            message = f"Chybí asset vrstvy '{layer.get('name', label + '[' + str(index) + ']')}': {relative_path}"
            if missing_fatal:
                issues.append(message)
            else:
                issues.append(f"{message} (aktivuji fallback rig)")

        width = height = None
        if exists:
            size = _image_size(absolute_path)
            if size is None:
                issues.append(f"Asset vrstvy '{layer.get('name', label + '[' + str(index) + ']')}' nejde načíst jako obrázek: {relative_path}")
            else:
                width, height = size
                if canvas_width and width and abs(width - canvas_width) > max(2, int(canvas_width * 0.30)):
                    issues.append(f"Vrstva '{layer.get('name', label + '[' + str(index) + ']')}' má podezřelou šířku {width}px vůči canvasu {canvas_width}px.")
                if canvas_height and height and abs(height - canvas_height) > max(2, int(canvas_height * 0.30)):
                    issues.append(f"Vrstva '{layer.get('name', label + '[' + str(index) + ']')}' má podezřelou výšku {height}px vůči canvasu {canvas_height}px.")

        resolved_layers.append(
            {
                **layer,
                "path": relative_path,
                "absolute_path": str(absolute_path),
                "exists": exists,
                "width": width,
                "height": height,
            }
        )

    return resolved_layers, issues


def validate_talking_head_manifest_data(
    data: dict[str, Any],
    *,
    assets_root: Path | None = None,
    strict: bool = False,
) -> list[str]:
    issues: list[str] = []
    assets_root = assets_dir() if assets_root is None else Path(assets_root)

    for key in _REQUIRED_TALKING_HEAD_KEYS:
        if key not in data:
            issues.append(f"Manifest talking head postrádá povinnou sekci '{key}'.")

    if issues:
        return issues

    canvas = _ensure_mapping(data["canvas"], label="canvas")
    if int(canvas.get("width", 0) or 0) <= 0 or int(canvas.get("height", 0) or 0) <= 0:
        issues.append("Sekce 'canvas' musí obsahovat kladné rozměry width a height.")

    layers = _ensure_mapping(data["layers"], label="layers")
    fallback_layers = _ensure_list(layers.get("fallback", []), label="layers.fallback")
    production_layers = _ensure_list(layers.get("production", []), label="layers.production")

    _, fallback_issues = _validate_layer_block(
        fallback_layers,
        label="layers.fallback",
        assets_root=assets_root,
        canvas=canvas,
        missing_fatal=True,
    )
    issues.extend(fallback_issues)

    _, production_issues = _validate_layer_block(
        production_layers,
        label="layers.production",
        assets_root=assets_root,
        canvas=canvas,
        missing_fatal=strict,
    )
    issues.extend(production_issues if strict else [issue for issue in production_issues if "nejde načíst" in issue or "podezřelou" in issue])

    return issues


def load_talking_head_manifest(
    *,
    manifest_path: Path | None = None,
    assets_root: Path | None = None,
    strict: bool = False,
    fallback_image_override: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = talking_head_manifest_path() if manifest_path is None else Path(manifest_path)
    assets_root = assets_dir() if assets_root is None else Path(assets_root)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    issues = validate_talking_head_manifest_data(data, assets_root=assets_root, strict=strict)
    if issues and strict:
        raise ValueError("; ".join(issues))

    canvas = _ensure_mapping(data["canvas"], label="canvas")
    layers = _ensure_mapping(data["layers"], label="layers")
    fallback_layers, fallback_runtime_issues = _validate_layer_block(
        _ensure_list(layers.get("fallback", []), label="layers.fallback"),
        label="layers.fallback",
        assets_root=assets_root,
        canvas=canvas,
        missing_fatal=True,
    )
    production_layers, production_runtime_issues = _validate_layer_block(
        _ensure_list(layers.get("production", []), label="layers.production"),
        label="layers.production",
        assets_root=assets_root,
        canvas=canvas,
        missing_fatal=False,
    )

    if fallback_image_override:
        override_path = Path(fallback_image_override)
        absolute_override = override_path if override_path.is_absolute() else assets_root / override_path
        if not absolute_override.exists():
            raise FileNotFoundError(f"Fallback asset override neexistuje: {absolute_override}")
        for layer in fallback_layers:
            if str(layer.get("role", "")) == "head_base":
                layer["path"] = str(override_path)
                layer["absolute_path"] = str(absolute_override)
                layer["exists"] = True
                size = _image_size(absolute_override)
                layer["width"] = size[0] if size else None
                layer["height"] = size[1] if size else None

    production_ready = bool(production_layers) and all(bool(layer.get("exists")) for layer in production_layers if layer.get("enabled", True))
    fallback_ready = bool(fallback_layers) and all(bool(layer.get("exists")) for layer in fallback_layers if layer.get("enabled", True))

    all_issues = issues + fallback_runtime_issues + production_runtime_issues
    if not fallback_ready:
        fatal = [issue for issue in all_issues if "layers.fallback" in issue or "fallback" in issue.lower()]
        raise FileNotFoundError("; ".join(fatal or ["Fallback talking head rig není dostupný."]))

    runtime = {
        "assets_root": str(assets_root),
        "fallback_mode": not production_ready,
        "production_ready": production_ready,
        "issues": list(dict.fromkeys(all_issues)),
        "resolved_layers": {
            "fallback": fallback_layers,
            "production": production_layers,
        },
    }

    return {**data, "runtime": runtime}
