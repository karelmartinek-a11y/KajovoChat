from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_windows_exe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_windows_exe", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_windows_args_include_assets_and_entrypoint() -> None:
    module = _load_module()

    args = module.build_pyinstaller_args()

    assert "--windowed" in args
    assert str(module.ENTRY_POINT) == args[-1]
    joined = "\n".join(args)
    assert "kajovochat/resources/assets" in joined
    assert "assets_manifest.json" in joined
    assert "talking_head_manifest.json" in joined
    assert module.APP_NAME in args
