"""Nature/IEEE grayscale plotting helpers.

All figures are saved as PNG only. No PDF writer is used anywhere in this package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", ">", "<"]
GRAY_LEVELS = ["0.10", "0.28", "0.45", "0.60", "0.75", "0.88"]


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 10.5,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, which="major", axis="both")
    ax.tick_params(which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def panel_labels(axes: Iterable[plt.Axes], x: float = -0.12, y: float = 1.04) -> None:
    for index, ax in enumerate(axes):
        ax.text(
            x,
            y,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
        )


def save_png(fig: plt.Figure, path: str | Path, close: bool = True) -> Path:
    """Save one publication-quality PNG and never create a PDF."""
    output = Path(path)
    if output.suffix.lower() != ".png":
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        bottom = float(getattr(fig, "_tight_bottom", 0.0))
        top = float(getattr(fig, "_tight_top", 1.0))
        fig.tight_layout(rect=[0.0, bottom, 1.0, top])
    except Exception:
        pass
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return output


def method_style(index: int) -> dict[str, object]:
    return {
        "color": GRAY_LEVELS[index % len(GRAY_LEVELS)],
        "linestyle": LINE_STYLES[index % len(LINE_STYLES)],
        "marker": MARKERS[index % len(MARKERS)],
        "linewidth": 1.35,
        "markersize": 4.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.8,
    }


apply_nature_style()
