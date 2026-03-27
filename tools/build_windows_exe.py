from __future__ import annotations

"""Build Windows .exe bundle for Chatbot Kája using PyInstaller."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "ChatbotKaja"
ENTRY_POINT = ROOT / "app_gui.py"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"


def _data_arg(source: Path, target: str) -> str:
    return f"{source};{target}"


def build_pyinstaller_args() -> list[str]:
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
    return args


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("Windows build je potreba spoustet primo na Windows. PyInstaller neni cross-compiler.")
    if not ENTRY_POINT.exists():
        raise SystemExit(f"Chybi entry point: {ENTRY_POINT}")

    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except Exception as exc:  # pragma: no cover - runtime only
        raise SystemExit(
            "Chybi PyInstaller. Nainstaluj build zavislosti prikazem: pip install -r requirements-build.txt"
        ) from exc

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    pyinstaller_run(build_pyinstaller_args())

    exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
    if exe_path.exists():
        print(f"Hotovo: {exe_path}")
    else:  # pragma: no cover - runtime only
        print(f"Build dokoncen, ale EXE nebylo nalezeno v {exe_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
