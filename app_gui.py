from __future__ import annotations

"""Convenience entrypoint for running the GUI.

The canonical entrypoint is: `python -m kajovochat`.
This file is kept for backwards compatibility with older launch scripts.
"""

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from kajovochat.main import MainWindow
from kajovochat.settings import AppSettings
from kajovochat.theme import app_stylesheet


def _install_excepthook() -> None:
    def _excepthook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            print(msg, file=sys.stderr)
        except Exception:
            pass
        try:
            tail = msg[-6000:] if len(msg) > 6000 else msg
            QMessageBox.critical(None, "Chatbot Kája – neočekávaná chyba", tail)
        except Exception:
            pass

    sys.excepthook = _excepthook


def main() -> int:
    settings = AppSettings.load()
    app = QApplication(sys.argv)
    app.setStyleSheet(app_stylesheet())
    _install_excepthook()

    w = MainWindow(settings)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
