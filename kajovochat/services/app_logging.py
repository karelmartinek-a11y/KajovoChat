from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time
import signal
from pathlib import Path
from typing import Optional

from PySide6.QtCore import qInstallMessageHandler, QtMsgType

_INSTALLED = False

def _safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def install_app_logging(*, log_dir: Path, session_tag: str) -> dict:
    """Install robust logging + crash capture.

    - Writes a human-readable log file for the whole app lifetime.
    - Installs sys/thread exception hooks.
    - Enables faulthandler (helps with hard crashes/segfaults).
    - Captures Qt internal warnings via qInstallMessageHandler.

    Returns:
        dict with keys: app_log_path, faulthandler_path
    """
    global _INSTALLED
    if _INSTALLED:
        return {}

    _safe_mkdir(log_dir)

    app_log_path = log_dir / f"kajovochat_app_{session_tag}.log"
    fh_path = log_dir / f"kajovochat_faulthandler_{session_tag}.log"

    # Root logger config (idempotent: remove old handlers first).
    root = logging.getLogger()
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
        except Exception:
            pass
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    try:
        fh = logging.FileHandler(app_log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        # If file handler fails, we still keep stdout logs.
        pass

    log = logging.getLogger("kajovochat")
    log.info("logging_installed log_dir=%s", str(log_dir))

    # Faulthandler (best effort).
    try:
        f = open(fh_path, "a", encoding="utf-8")
        faulthandler.enable(file=f, all_threads=True)
        # Also dump stacks on fatal signals where supported.
        try:
            faulthandler.register(getattr(signal, "SIGTERM", 15), file=f, all_threads=True)
        except Exception:
            pass
        log.info("faulthandler_enabled path=%s", str(fh_path))
    except Exception as e:
        log.warning("faulthandler_enable_failed: %s", str(e))

    def _log_exception(prefix: str, exc_type, exc, tb) -> None:
        try:
            logging.getLogger("kajovochat.crash").error(
                "%s: unhandled exception", prefix, exc_info=(exc_type, exc, tb)
            )
        except Exception:
            pass

    def _sys_excepthook(exc_type, exc, tb) -> None:
        _log_exception("sys", exc_type, exc, tb)
        try:
            # Let default hook print too (useful when running from console).
            sys.__excepthook__(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = _sys_excepthook

    # threading.excepthook exists in Python 3.8+
    def _threading_excepthook(args) -> None:
        _log_exception(f"thread:{getattr(args, 'thread', None)}", args.exc_type, args.exc_value, args.exc_traceback)

    if hasattr(threading, "excepthook"):
        try:
            threading.excepthook = _threading_excepthook  # type: ignore[attr-defined]
        except Exception:
            pass

    # Qt message handler: route Qt warnings/errors into our log.
    def _qt_handler(mode: QtMsgType, context, message: str) -> None:
        name = "qt"
        try:
            if mode == QtMsgType.QtDebugMsg:
                logging.getLogger(name).debug(message)
            elif mode == QtMsgType.QtInfoMsg:
                logging.getLogger(name).info(message)
            elif mode == QtMsgType.QtWarningMsg:
                logging.getLogger(name).warning(message)
            elif mode == QtMsgType.QtCriticalMsg:
                logging.getLogger(name).error(message)
            elif mode == QtMsgType.QtFatalMsg:
                logging.getLogger(name).critical(message)
            else:
                logging.getLogger(name).info(message)
        except Exception:
            pass

    try:
        qInstallMessageHandler(_qt_handler)
    except Exception:
        pass

    _INSTALLED = True
    return {
        "app_log_path": str(app_log_path),
        "faulthandler_path": str(fh_path),
    }
