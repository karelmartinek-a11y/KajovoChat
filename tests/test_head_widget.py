from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from kajovochat.widgets.head_widget import HeadWidget


ASSETS_DIR = Path(__file__).resolve().parents[1] / "kajovochat" / "resources" / "assets"


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_head_widget_accepts_lipsync_snapshot() -> None:
    _app()
    widget = HeadWidget(str(ASSETS_DIR / "head_photo.png"))
    widget.set_lipsync_snapshot(
        {
            "weights": {"closed": 0.1, "small": 0.1, "aa": 0.4, "ee": 0.2, "oo": 0.2},
        }
    )
    assert widget._mouth_energy > 0.0
    assert widget._aurora_bias > 0.0


def test_head_widget_error_reset_rect_exists_after_paint() -> None:
    app = _app()
    widget = HeadWidget(str(ASSETS_DIR / "head_photo.png"))
    widget.resize(640, 640)
    widget.set_state("error")
    widget.set_error_text("Test")
    widget.show()
    app.processEvents()
    widget.repaint()
    app.processEvents()
    assert widget._reset_rect.width() > 0.0


def test_head_widget_reports_render_backend() -> None:
    _app()
    widget = HeadWidget(str(ASSETS_DIR / "head_photo.png"))
    summary = widget.render_backend_summary()
    assert "backend" in summary or "fallback-2d" in summary or "gpu-opengl" in summary
