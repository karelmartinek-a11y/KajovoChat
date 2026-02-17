from __future__ import annotations

"""UI theme for Chatbot Kája.

Centralizes brand colors and the application-wide Qt stylesheet.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    # Brand colors (sampled from the provided logo)
    brand_blue: str = "#3CB0DB"
    brand_yellow: str = "#FCD237"
    navy: str = "#1A3451"
    bg: str = "#0B1220"
    surface: str = "#111A2B"
    surface_2: str = "#0E1726"
    text: str = "#EAF2F7"
    text_muted: str = "rgba(234, 242, 247, 170)"
    border: str = "rgba(255, 255, 255, 35)"


def app_stylesheet(t: Theme | None = None) -> str:
    t = t or Theme()
    # Notes:
    # - Keep focus rings visible for accessibility.
    # - Avoid overly bright whites on dark background.
    return (
        f"QWidget {{ background-color: {t.bg}; color: {t.text}; }}"
        f"QLabel {{ color: {t.text_muted}; }}"
        f"QToolTip {{ background-color: {t.surface}; color: {t.text}; border: 1px solid {t.border}; }}"

        # Inputs
        f"QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{"
        f"  background-color: rgba(255,255,255,12);"
        f"  border: 1px solid rgba(255,255,255,48);"
        f"  border-radius: 10px;"
        f"  padding: 8px 10px;"
        f"  color: {t.text};"
        f"}}"
        f"QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{"
        f"  border: 1px solid {t.brand_blue};"
        f"}}"
        f"QComboBox::drop-down {{ border: none; }}"

        # Buttons
        f"QPushButton {{"
        f"  padding: 9px 14px;"
        f"  background-color: rgba(255,255,255,14);"
        f"  border: 1px solid rgba(255,255,255,55);"
        f"  border-radius: 12px;"
        f"  color: {t.text};"
        f"}}"
        f"QPushButton:hover {{ background-color: rgba(255,255,255,20); }}"
        f"QPushButton:pressed {{ background-color: rgba(255,255,255,28); }}"
        f"QPushButton:disabled {{"
        f"  color: rgba(234,242,247,110);"
        f"  background-color: rgba(255,255,255,8);"
        f"  border-color: rgba(255,255,255,25);"
        f"}}"

        # Flat/primary action buttons (set via property)
        f"QPushButton[variant='primary'] {{"
        f"  background-color: rgba(60,176,219,22);"
        f"  border: 1px solid rgba(60,176,219,120);"
        f"  color: {t.text};"
        f"}}"
        f"QPushButton[variant='primary']:hover {{ background-color: rgba(60,176,219,30); }}"
        f"QPushButton[variant='danger'] {{"
        f"  background-color: rgba(252,210,55,16);"
        f"  border: 1px solid rgba(252,210,55,140);"
        f"  color: {t.text};"
        f"}}"
        f"QPushButton[variant='danger']:hover {{ background-color: rgba(252,210,55,22); }}"

        # Group boxes
        f"QGroupBox {{ border: 1px solid {t.border}; border-radius: 14px; margin-top: 12px; padding: 10px; }}"
        f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {t.text}; }}"
    )
