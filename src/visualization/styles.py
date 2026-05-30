"""Consistent matplotlib styling for research-grade figures.

Apply once at the start of a script or notebook:

    from src.visualization.styles import apply_research_style
    apply_research_style()

Or use the context manager for isolated style scopes:

    with research_style_context():
        fig = plot_equity_curve(...)
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLORS = {
    "strategy": "#1f4e79",   # deep navy — primary series
    "benchmark": "#888888",  # neutral grey — benchmark/reference
    "positive": "#2e7d32",   # muted green — gains
    "negative": "#c62828",   # muted red — losses/drawdown
    "signal": "#f57c00",     # amber — signal / overlay
    "neutral": "#546e7a",    # slate — secondary series
    "grid": "#e0e0e0",       # very light grey — gridlines
}

# ---------------------------------------------------------------------------
# RC parameters
# ---------------------------------------------------------------------------

_RESEARCH_RC: dict[str, object] = {
    # Font
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    # Lines
    "lines.linewidth": 1.4,
    # Figure
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Save
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

# Standard figure widths (inches)
FIG_WIDTH_FULL = 12.0    # full-width single plot
FIG_WIDTH_HALF = 6.0     # half-width or small panel

# Standard figure height ratios
FIG_HEIGHT_STANDARD = 4.0   # default single-panel height
FIG_HEIGHT_TALL = 5.5       # equity curve + drawdown stacked
FIG_HEIGHT_HEATMAP = 4.5    # monthly heatmap


def apply_research_style(profile: str = "report") -> None:
    """Apply the research RC params globally for the current session.

    Args:
        profile: Render profile name — "report" (default) for PDF/publication
                 density, or "frontend" for browser-inspectable figures.
                 Sets the active profile for the duration of the session.
    """
    from src.visualization.render_profiles import (
        RENDER_PROFILES,
        set_active_profile,
    )
    set_active_profile(profile)
    mpl.rcParams.update(_RESEARCH_RC)
    profile_rc = RENDER_PROFILES[profile].get("rc", {})
    if profile_rc:
        mpl.rcParams.update(profile_rc)


@contextmanager
def research_style_context() -> Generator[None, None, None]:
    """Context manager that applies research style and restores original RC on exit."""
    with mpl.rc_context(_RESEARCH_RC):
        yield


def make_figure(
    nrows: int = 1,
    ncols: int = 1,
    width: float = FIG_WIDTH_FULL,
    height: float | None = None,
    height_ratios: list[float] | None = None,
    **kwargs: object,
) -> tuple[plt.Figure, plt.Axes | list[plt.Axes]]:
    """Create a styled figure with sensible defaults.

    Dimensions are automatically scaled by the active render profile's
    ``figsize_scale`` factor, so all figures grow uniformly when the
    frontend profile is active without requiring per-function changes.

    Returns (fig, axes) where axes is a single Axes for nrows=ncols=1,
    or a list of Axes otherwise.
    """
    if height is None:
        height = FIG_HEIGHT_STANDARD * nrows

    from src.visualization.render_profiles import get_figsize_scale
    scale = get_figsize_scale()
    scaled_w = width * scale
    scaled_h = height * scale

    gridspec_kw = {}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = height_ratios

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(scaled_w, scaled_h),
        gridspec_kw=gridspec_kw if gridspec_kw else None,
        **kwargs,
    )
    return fig, axes


def format_pct_axis(ax: plt.Axes, axis: str = "y") -> None:
    """Format the given axis ticks as percentages (e.g. 0.10 → '10%')."""
    formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def label_axes(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply title and axis labels with consistent style."""
    if title:
        ax.set_title(title, fontweight="semibold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
