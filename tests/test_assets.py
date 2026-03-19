from __future__ import annotations

from kajovochat.resources.assets import load_asset_manifest, verify_asset_manifest


def test_asset_manifest_matches_files() -> None:
    assert verify_asset_manifest() == []


def test_asset_manifest_contains_talking_head_group() -> None:
    manifest = load_asset_manifest()
    assert "groups" in manifest
    assert "talking_head" in manifest["groups"]
