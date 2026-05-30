"""Return and risk distribution plots.

Histogram-based analytics for return distributions and the standard
monthly-return heatmap used in institutional research tearsheets.

Read-only: no data mutation, no backtest execution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_WIDTH_FULL,
    format_pct_axis,
    label_axes,
    make_figure,
)
from src.visualization.typography import (
    get_typography,
    scale_dynamic_fontsize,
)


def plot_return_distribution(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    bins: int = 60,
    kde: bool = True,
    title: str = "Return Distribution",
    strategy_label: str = "Strategy",
    benchmark_label: str = "Benchmark",
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of period returns with optional KDE overlay and benchmark comparison.

    Args:
        returns: Strategy period return series.
        benchmark: Optional benchmark return series for side-by-side comparison.
        bins: Number of histogram bins.
        kde: If True, overlay a kernel density estimate computed with numpy.
        title: Figure title.
        strategy_label: Legend label for strategy histogram.
        benchmark_label: Legend label for benchmark histogram (if provided).
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=3.8)

    r = returns.dropna()

    ax.hist(
        r,
        bins=bins,
        color=COLORS["strategy"],
        alpha=0.65,
        density=True,
        label=strategy_label,
        zorder=3,
    )

    if benchmark is not None:
        b = benchmark.dropna()
        ax.hist(
            b,
            bins=bins,
            color=COLORS["benchmark"],
            alpha=0.40,
            density=True,
            label=benchmark_label,
            zorder=2,
        )

    if kde:
        _overlay_kde(ax, r, color=COLORS["strategy"])
        if benchmark is not None:
            _overlay_kde(ax, benchmark.dropna(), color=COLORS["benchmark"], linestyle="--")

    # Vertical line at zero
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.9, linestyle="--", zorder=4)

    # Annotate mean and std
    mean_r = r.mean()
    std_r = r.std()
    ax.axvline(mean_r, color=COLORS["signal"], linewidth=1.1, linestyle=":", zorder=5,
               label=f"Mean: {mean_r:.3%}")

    format_pct_axis(ax, axis="x")
    ax.set_ylabel("Density")
    label_axes(ax, title=title, xlabel="Period Return")

    ax.legend(frameon=False)

    # Stat annotation
    _t = get_typography()
    ax.text(
        0.98, 0.96,
        f"μ = {mean_r:.3%}  σ = {std_r:.3%}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=_t.annotation,
        color=COLORS["neutral"],
    )

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_monthly_return_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
    save_path: str | None = None,
) -> plt.Figure:
    """Institutional-style calendar heatmap of monthly returns.

    Rows = years, columns = months (Jan–Dec).  Cells are colour-coded:
    green for positive, red for negative, white for missing.

    Args:
        returns: Daily (or any sub-monthly) return series with a DatetimeIndex.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    r = returns.dropna()
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError(
            "plot_monthly_return_heatmap requires a DatetimeIndex. "
            "Use to_datetime_index() first."
        )

    # Resample to monthly compound returns
    monthly = r.resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)

    # Pivot to year × month table
    monthly_df = monthly.to_frame("ret")
    monthly_df["year"] = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    # Ensure all 12 months are present (fill missing with NaN)
    pivot = pivot.reindex(columns=range(1, 13))

    years = pivot.index.tolist()
    n_years = len(years)

    fig, ax = make_figure(
        width=FIG_WIDTH_FULL,
        height=max(2.5, 0.45 * n_years + 1.2),
    )

    # --- Colour scale: symmetric around 0 ---
    abs_max = pivot.abs().max().max()
    if abs_max == 0 or np.isnan(abs_max):
        abs_max = 0.05
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    cmap = plt.cm.RdYlGn  # red–yellow–green, standard for return heatmaps

    # Draw cells
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    _t = get_typography()
    cell_fs = scale_dynamic_fontsize(max(6.5, 9.0 - n_years * 0.1), "heatmap_cell")
    for row_i, year in enumerate(years):
        for col_i, month in enumerate(range(1, 13)):
            val = pivot.loc[year, month]
            color = cmap(norm(val)) if not np.isnan(val) else (0.95, 0.95, 0.95, 1.0)
            rect = plt.Rectangle(
                [col_i, row_i], 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.8,
            )
            ax.add_patch(rect)
            if not np.isnan(val):
                text_color = "white" if abs(val) > abs_max * 0.55 else "#333333"
                ax.text(
                    col_i + 0.5, row_i + 0.5,
                    f"{val:.1%}",
                    ha="center", va="center",
                    fontsize=cell_fs,
                    color=text_color,
                )

    ax.set_xlim(0, 12)
    ax.set_ylim(0, n_years)
    ax.set_xticks([x + 0.5 for x in range(12)])
    ax.set_xticklabels(month_names, fontsize=_t.tick)
    ax.set_yticks([y + 0.5 for y in range(n_years)])
    ax.set_yticklabels(years, fontsize=_t.tick)
    ax.invert_yaxis()  # most recent year at top
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    cbar.ax.tick_params(labelsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _overlay_kde(
    ax: plt.Axes,
    data: pd.Series,
    color: str,
    linestyle: str = "-",
    n_points: int = 300,
) -> None:
    """Draw a simple Gaussian KDE over ``data`` on ``ax`` using numpy only."""
    if len(data) < 2:
        return
    bw = data.std() * (4.0 / (3.0 * len(data))) ** 0.2  # Silverman's rule
    x = np.linspace(data.min() - 3 * bw, data.max() + 3 * bw, n_points)
    # Evaluate KDE manually
    kde_vals = np.zeros_like(x)
    for xi in data:
        kde_vals += np.exp(-0.5 * ((x - xi) / bw) ** 2)
    kde_vals /= len(data) * bw * np.sqrt(2 * np.pi)
    ax.plot(x, kde_vals, color=color, linewidth=1.4, linestyle=linestyle, zorder=5)
