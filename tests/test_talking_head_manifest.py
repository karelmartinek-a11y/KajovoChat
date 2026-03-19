from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

from kajovochat.resources.assets import assets_dir, load_talking_head_manifest


def test_load_talking_head_manifest_valid() -> None:
    manifest = load_talking_head_manifest()
    assert manifest["meta"]["name"] == "kajovo_talking_head"
    assert manifest["runtime"]["fallback_mode"] is True
    assert manifest["runtime"]["production_ready"] is False


def test_talking_head_manifest_missing_key() -> None:
    manifest = load_talking_head_manifest()
    broken = deepcopy(manifest)
    broken.pop("canvas", None)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "broken_manifest.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_talking_head_manifest(manifest_path=path, strict=True)


def test_talking_head_manifest_missing_asset_path() -> None:
    manifest = load_talking_head_manifest()
    broken = deepcopy(manifest)
    broken["layers"]["fallback"][0]["path"] = "neexistuje.png"
    broken["layers"]["production"] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "broken_asset.json"
        path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_talking_head_manifest(manifest_path=path, assets_root=assets_dir(), strict=True)


def test_talking_head_manifest_falls_back_when_production_incomplete() -> None:
    manifest = load_talking_head_manifest()
    assert manifest["runtime"]["fallback_mode"] is True
    assert any("aktivuji fallback rig" in issue for issue in manifest["runtime"]["issues"])
