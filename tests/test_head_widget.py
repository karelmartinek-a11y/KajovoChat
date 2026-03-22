from __future__ import annotations

from PySide6.QtWidgets import QApplication

from kajovochat.widgets.head_widget import HeadWidget

def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_head_widget_accepts_lipsync_snapshot() -> None:
    _app()
    widget = HeadWidget("ignored")
    widget.set_lipsync_snapshot(
        {
            "weights": {"closed": 0.1, "small": 0.1, "aa": 0.4, "ee": 0.2, "oo": 0.2},
            "energy": 0.6,
        }
    )
    assert widget._mouth_energy > 0.0
    assert widget._aurora_bias > 0.0


def test_head_widget_error_reset_rect_exists_after_paint() -> None:
    app = _app()
    widget = HeadWidget("ignored")
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
    widget = HeadWidget("ignored")
    summary = widget.render_backend_summary()
    assert "backend=ekg-2d" == summary


def test_head_widget_terminal_keeps_last_ten_rows() -> None:
    _app()
    widget = HeadWidget("ignored")
    widget.set_terminal_text("\n".join(f"Řádek {index}" for index in range(14)))
    assert len(widget._terminal_lines) == 10
    assert widget._terminal_lines[0] == "Řádek 4"
    assert widget._terminal_lines[-1] == "Řádek 13"
