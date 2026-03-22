from __future__ import annotations

"""Build macOS .app bundle for Chatbot Kája using PyInstaller."""

import shutil
import sys
from pathlib import Path

from PyInstaller.__main__ import run as pyinstaller_run


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ChatbotKaja"
ENTRY_POINT = ROOT / "app_gui.py"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def _data_arg(source: Path, target: str) -> str:
    return f"{source}:{target}"


def main() -> None:
    if not ENTRY_POINT.exists():
        raise SystemExit(f"Chybí entry point: {ENTRY_POINT}")

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    add_datas = [
        _data_arg(ROOT / "kajovochat" / "resources" / "assets", "kajovochat/resources/assets"),
        _data_arg(ROOT / "kajovochat" / "resources" / "assets_manifest.json", "kajovochat/resources"),
        _data_arg(ROOT / "kajovochat" / "resources" / "talking_head_manifest.json", "kajovochat/resources"),
    ]

    args = [
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        APP_NAME,
        "--osx-bundle-identifier",
        "com.kajovo.chat",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(BUILD_DIR),
        "--paths",
        str(ROOT),
        "--collect-all",
        "PySide6",
        "--collect-all",
        "numpy",
        "--collect-all",
        "scipy",
        "--collect-all",
        "sounddevice",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "moderngl",
        "--collect-all",
        "openai",
        "--collect-all",
        "httpx",
        "--collect-all",
        "keyring",
    ]

    for data_arg in add_datas:
        args.extend(["--add-data", data_arg])

    args.append(str(ENTRY_POINT))
    pyinstaller_run(args)

    app_path = DIST_DIR / f"{APP_NAME}.app"
    if app_path.exists():
        print(f"Hotovo: {app_path}")
    else:
        print(f"Build dokončen, ale app bundle nebyl nalezen v {app_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
