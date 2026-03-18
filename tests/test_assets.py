from __future__ import annotations

from kajovochat.resources.assets import verify_asset_manifest


def test_asset_manifest_matches_files() -> None:
    assert verify_asset_manifest() == []
