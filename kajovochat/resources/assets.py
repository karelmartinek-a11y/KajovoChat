from __future__ import annotations

import hashlib
import json
from pathlib import Path


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def asset_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "assets_manifest.json"


def load_asset_manifest() -> dict[str, dict[str, object]]:
    return json.loads(asset_manifest_path().read_text(encoding="utf-8"))


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
