from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def asset_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "assets_manifest.json"


def talking_head_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "talking_head_manifest.json"


def load_asset_manifest() -> dict[str, dict[str, object]]:
    return json.loads(asset_manifest_path().read_text(encoding="utf-8"))


def _load_json_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Manifest nenalezen: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Neplatný JSON manifest: {manifest_path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Manifest musí být objekt: {manifest_path}")
    return data


def _manifest_issue(message: str, *, issues: list[str]) -> None:
    issues.append(message)


def _ensure_required_keys(manifest: dict[str, Any], required: tuple[str, ...], *, strict: bool) -> None:
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"V manifestu chybí klíč/klíče: {', '.join(missing)}")
    if strict:
        for key in required:
            if manifest.get(key) is None:
                raise ValueError(f"V manifestu chybí klíč: {key}")


def _resolve_layer(item: dict[str, Any], *, assets_root: Path) -> dict[str, Any]:
    layer = dict(item)
    rel_path = str(layer.get("path", ""))
    absolute_path = (assets_root / rel_path).resolve()
    exists = absolute_path.exists()
    layer["absolute_path"] = str(absolute_path)
    layer["exists"] = exists
    if exists:
        try:
            from PySide6.QtGui import QImage

            image = QImage(str(absolute_path))
            if not image.isNull():
                layer["width"] = image.width()
                layer["height"] = image.height()
        except Exception:
            pass
    return layer


def load_talking_head_manifest(
    *,
    manifest_path: Path | None = None,
    assets_root: Path | None = None,
    fallback_image_override: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path) if manifest_path is not None else talking_head_manifest_path()
    root = Path(assets_root) if assets_root is not None else assets_dir()
    manifest = _load_json_manifest(path)
    _ensure_required_keys(
        manifest,
        ("meta", "canvas", "fallback", "layers", "pivots", "masks", "deformation_ranges", "z_order", "state_presets"),
        strict=strict,
    )

    fallback = dict(manifest.get("fallback", {}))
    layers = dict(manifest.get("layers", {}))
    issues: list[str] = []

    resolved_layers = {"fallback": [], "production": []}
    production_ready = True

    for group_name in ("fallback", "production"):
        group_layers = layers.get(group_name, [])
        if not isinstance(group_layers, list):
            raise ValueError(f"Skupina vrstev musí být seznam: {group_name}")
        for item in group_layers:
            if not isinstance(item, dict):
                raise ValueError(f"Vrstva musí být objekt: {group_name}")
            resolved = _resolve_layer(item, assets_root=root)
            if fallback_image_override and group_name == "fallback" and resolved.get("role") == "head_base":
                override_path = Path(fallback_image_override).resolve()
                resolved["absolute_path"] = str(override_path)
                resolved["exists"] = override_path.exists()
            if strict and not resolved.get("exists", False):
                raise ValueError(f"Chybí asset pro vrstvu: {resolved.get('path', '')}")
            if group_name == "production" and not resolved.get("exists", False):
                production_ready = False
            resolved_layers[group_name].append(resolved)

    if not production_ready:
        _manifest_issue("aktivuji fallback rig, protože production assets nejsou kompletní", issues=issues)
    if fallback.get("force_when_production_incomplete", False) and not production_ready:
        fallback_mode = True
    else:
        fallback_mode = bool(fallback.get("enabled", True)) and not production_ready

    runtime = {
        "assets_root": str(root.resolve()),
        "resolved_layers": resolved_layers,
        "production_ready": production_ready,
        "fallback_mode": fallback_mode,
        "issues": issues,
    }
    manifest["runtime"] = runtime
    return manifest


def verify_asset_manifest(*, max_asset_bytes: int = 5_000_000) -> list[str]:
    issues: list[str] = []
    manifest = load_asset_manifest()
    root = assets_dir()

    for name, expected in manifest.items():
        path = root / name
        if not path.exists():
            issues.append(f"chybí asset: {name}")
            continue
        data = path.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        actual_size = len(data)
        if actual_hash != expected.get("sha256"):
            issues.append(f"nesedí hash assetu: {name}")
        if actual_size != int(expected.get("bytes", -1)):
            issues.append(f"nesedí velikost assetu: {name}")
        if actual_size > max_asset_bytes:
            issues.append(f"asset překračuje limit {max_asset_bytes} B: {name}")

    for path in root.glob("*"):
        if path.is_file() and path.name not in manifest:
            issues.append(f"asset chybí v manifestu: {path.name}")

    return issues
