"""Matplotlib style for this notebook.

SIGNAL brief (academic profile), applied only to figures and docs.
There is no product UI, so the 4Geeks × Divinivid lockup is not used.
Arial and Helvetica are not installed in this environment; the stack
falls through to Arimo (metric-compatible sans) and then Liberation Sans.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = {
    "canvas": "#F7F8FA",
    "surface": "#FFFFFF",
    "ink": "#18202B",
    "muted": "#505E70",
    "border": "#788598",
    "subtle": "#E7ECF2",
    "action": "#B53A17",
    "on_action": "#FFFFFF",
    "info": "#204EA0",
    "success": "#236344",
    "danger": "#A32A26",
    "warning": "#765000",
}

FONT_STACK = ["Arial", "Helvetica", "Arimo", "Liberation Sans", "DejaVu Sans"]


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]

    def channel(value: float) -> float:
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(value) for value in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(relative_luminance(foreground), relative_luminance(background))
    darker = min(relative_luminance(foreground), relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def resolved_font_name() -> str:
    """First installed face in the SIGNAL stack. Hyphenated CSS families are not fontconfig patterns."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in installed:
            return name
    return "DejaVu Sans"


def apply_style() -> str:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["canvas"],
            "savefig.facecolor": COLORS["canvas"],
            "axes.facecolor": COLORS["surface"],
            "axes.edgecolor": COLORS["border"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.titleweight": "bold",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.dpi": 140,
        }
    )
    return resolved_font_name()


def contrast_report() -> dict:
    pairs = {
        "ink_on_canvas": (COLORS["ink"], COLORS["canvas"]),
        "ink_on_surface": (COLORS["ink"], COLORS["surface"]),
        "muted_on_canvas": (COLORS["muted"], COLORS["canvas"]),
        "on_action_on_action": (COLORS["on_action"], COLORS["action"]),
        "info_on_surface": (COLORS["info"], COLORS["surface"]),
        "action_on_surface": (COLORS["action"], COLORS["surface"]),
        "success_on_surface": (COLORS["success"], COLORS["surface"]),
        "danger_on_surface": (COLORS["danger"], COLORS["surface"]),
    }
    return {name: round(contrast_ratio(fg, bg), 2) for name, (fg, bg) in pairs.items()}
