"""Profile-aware typography scale for the visualization layer.

Provides centralised font-size constants for every semantic text category,
derived from the active render profile.  Plotting functions should call
``get_typography()`` instead of hardcoding local font sizes.

Usage::

    from src.visualization.typography import get_typography

    def plot_xxx(...):
        _t = get_typography()
        ax.legend(frameon=False, fontsize=_t.legend)
        ax.tick_params(labelsize=_t.tick)
        ax.text(..., fontsize=_t.annotation, ...)
        cbar.ax.tick_params(labelsize=_t.colorbar)

For heatmaps::

    from src.visualization.typography import get_typography, heatmap_cell_fontsize
    from src.visualization.render_profiles import ARCHETYPE_HEATMAP

    def plot_heatmap(...):
        _t = get_typography(ARCHETYPE_HEATMAP)
        font_size = heatmap_cell_fontsize(n_rows, n_cols)
        ax.text(j, i, f"{v:.2f}", fontsize=font_size, ...)
        ax.set_yticklabels(labels, fontsize=_t.tick)
        cbar.ax.tick_params(labelsize=_t.colorbar)

For dynamic density-based label sizing::

    tick_fs = scale_dynamic_fontsize(max(6.5, 10.0 - n * 0.3))
"""

from __future__ import annotations

from dataclasses import dataclass

from src.visualization.render_profiles import ARCHETYPE_DEFAULT, get_active_profile


# ---------------------------------------------------------------------------
# Typography scale dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypographyScale:
    """Font sizes (pt) for every semantic text category.

    Fields are named by their rendering role, not by the matplotlib API
    that applies them, so the same scale adapts across legend(fontsize=),
    tick_params(labelsize=), ax.text(fontsize=), etc.
    """

    annotation: float
    """Inline value labels: bar annotations, scatter callouts, metric overlays."""

    legend: float
    """Legend text: all ax.legend() calls."""

    tick: float
    """Explicit tick label overrides (set_xticklabels / tick_params(labelsize=))
    that live outside the global RC.  RC-managed ticks need no override."""

    colorbar: float
    """Colorbar tick labels and colorbar axis labels."""

    heatmap_cell: float
    """Base font size for heatmap cell value annotations.
    Actual size is bounded by matrix density — use heatmap_cell_fontsize()."""

    figure_text: float
    """Empty-state messages and free-floating fig.text() annotations."""

    small_label: float
    """Secondary axis labels: twin-axis y-labels, compact subplot titles,
    axis labels on crowded compound figures."""

    small_annotation: float
    """Tiny dense markers: regime labels, family header text, IC-bar labels
    on dense bar charts where annotation space is tight."""


# ---------------------------------------------------------------------------
# Profile-keyed scale tables
# ---------------------------------------------------------------------------

_TYPOGRAPHY: dict[str, TypographyScale] = {
    # Report profile — compact institutional publication density.
    # Values reflect the canonical baseline the codebase was calibrated for.
    "report": TypographyScale(
        annotation=8.5,
        legend=8.5,
        tick=8.5,
        colorbar=7.5,
        heatmap_cell=8.5,
        figure_text=10.0,
        small_label=8.0,
        small_annotation=7.0,
    ),
    # Frontend profile — browser-inspectable institutional quality.
    # All values scaled ~×1.4 from report to match the RC-level scaling
    # already applied (axes.titlesize 11→20, legend.fontsize 9→13).
    "frontend": TypographyScale(
        annotation=12.0,
        legend=12.0,
        tick=12.0,
        colorbar=10.5,
        heatmap_cell=12.0,
        figure_text=14.0,
        small_label=11.0,
        small_annotation=10.0,
    ),
}


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def get_typography(archetype: str = ARCHETYPE_DEFAULT) -> TypographyScale:
    """Return the typography scale for the active render profile.

    The ``archetype`` argument is accepted for call-site symmetry with
    ``get_archetype_hint()`` but does not alter the returned scale —
    all archetypes share the same font-size table within a profile.
    Archetype-specific canvas scaling is handled by ``make_figure()``.
    """
    profile = get_active_profile()
    return _TYPOGRAPHY.get(profile, _TYPOGRAPHY["report"])


def heatmap_cell_fontsize(n_rows: int, n_cols: int) -> float:
    """Return a profile-aware cell annotation font size bounded by matrix density.

    Preserves the existing report-profile density formula exactly::

        report_size = max(7.0, min(10.0, 120 / max(n_rows, n_cols)))

    For non-report profiles the report result is scaled proportionally,
    so denser heatmaps always produce smaller text regardless of profile.

    Args:
        n_rows: Number of rows in the heatmap matrix.
        n_cols: Number of columns in the heatmap matrix.

    Returns:
        Font size in pt, rounded to one decimal place.
    """
    density = 120.0 / max(n_rows, n_cols, 1)
    report_size = max(7.0, min(10.0, density))

    profile = get_active_profile()
    if profile == "report":
        return report_size

    scale = _TYPOGRAPHY[profile].heatmap_cell / _TYPOGRAPHY["report"].heatmap_cell
    return round(report_size * scale, 1)


def scale_dynamic_fontsize(report_value: float, field: str = "tick") -> float:
    """Scale a dynamically-computed report-profile font size to the active profile.

    Use this when a function computes a font size based on data dimensions
    (e.g. number of labels in a dense axis) and wants that result to scale
    with the render profile.

    The report result is passed through unchanged so existing density
    heuristics are respected in the report profile.

    Args:
        report_value: Font size computed for the report profile.
        field:        Which TypographyScale field to derive the scale factor from.
                      Defaults to "tick" — suitable for most label-density cases.

    Example::

        n = len(labels)
        tick_fs = scale_dynamic_fontsize(max(6.5, 10.0 - n * 0.3))
    """
    profile = get_active_profile()
    if profile == "report":
        return report_value
    report_base = getattr(_TYPOGRAPHY["report"], field)
    target_base = getattr(_TYPOGRAPHY.get(profile, _TYPOGRAPHY["report"]), field)
    scale = target_base / report_base
    return round(report_value * scale, 1)
