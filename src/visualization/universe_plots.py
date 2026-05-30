"""Publication-quality universe coverage and structure diagnostics.

All functions accept pre-computed DataFrames and return matplotlib Figures.
Read-only: no data loading, no model fitting, no recomputation.

Figures in this module establish universe research validity — they are
infrastructure diagnostics, not strategy performance charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.visualization.styles import (
    COLORS,
    FIG_WIDTH_FULL,
    FIG_HEIGHT_STANDARD,
    FIG_HEIGHT_TALL,
    label_axes,
    make_figure,
)
from src.visualization.utils import save_figure
from src.visualization.typography import get_typography, heatmap_cell_fontsize, scale_dynamic_fontsize


# ---------------------------------------------------------------------------
# Universe coverage heatmap
# ---------------------------------------------------------------------------


def plot_universe_coverage_heatmap(
    monthly_coverage: pd.DataFrame,
    title: str = "Universe Coverage by Asset and Month",
    save_path: str | None = None,
) -> plt.Figure:
    """Asset × month coverage heatmap.

    Each cell shows the fraction of trading days in that month for which
    the asset had a valid price.  Green = full coverage; red = data gap.
    Exposes structural data quality before any modelling begins.

    Args:
        monthly_coverage: (months × assets) DataFrame with values in [0, 1].
                          Index = monthly DatetimeIndex; columns = ticker symbols.
        title:            Figure title.
        save_path:        Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    df = monthly_coverage.dropna(how="all", axis=1)
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No coverage data", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_assets = len(df.columns)
    n_months = len(df)
    height = max(FIG_HEIGHT_STANDARD, n_assets * 0.5 + 1.5)
    fig, ax = make_figure(height=height, width=FIG_WIDTH_FULL)

    Z = df.values.T  # (n_assets, n_months)
    # Use cell-edge arrays so pcolormesh renders correctly for any n_assets >= 1.
    # shading="nearest" with a single-element y-array produces invisible cells.
    x_edges = np.arange(n_months + 1) - 0.5
    y_edges = np.arange(n_assets + 1) - 0.5
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax.grid(False)

    # Y-axis: asset tickers
    ax.set_yticks(np.arange(n_assets))
    ax.set_yticklabels(list(df.columns), fontsize=_t.tick)
    ax.yaxis.set_tick_params(length=0)

    # X-axis: year labels spaced annually
    year_positions = []
    year_labels = []
    prev_year = None
    for i, ts in enumerate(df.index):
        yr = ts.year
        if yr != prev_year:
            year_positions.append(i)
            year_labels.append(str(yr))
            prev_year = yr
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, fontsize=_t.small_annotation, rotation=0)

    # Separator lines between assets
    for i in range(n_assets):
        ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.6)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, aspect=20, pad=0.02)
    cbar.set_label("Monthly coverage fraction", fontsize=_t.colorbar)
    cbar.ax.tick_params(labelsize=_t.colorbar)

    label_axes(ax, title=title, xlabel="", ylabel="")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Asset availability timeline
# ---------------------------------------------------------------------------


def plot_asset_availability_timeline(
    prices: pd.DataFrame,
    title: str = "Rolling Asset Availability",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling count of assets with valid prices through time.

    Reveals structural changes in universe availability — asset additions,
    delistings, or persistent data gaps — before any cross-sectional
    computation that depends on full universe breadth.

    Args:
        prices:    Date × Asset price DataFrame (NaN = unavailable).
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    if prices.empty or prices.shape[1] == 0:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No price data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text,
                color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_total = prices.shape[1]
    daily_count = prices.notna().sum(axis=1)
    rolling_count = daily_count.rolling(21, min_periods=1).mean()

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD, width=FIG_WIDTH_FULL)

    vals = rolling_count.to_numpy()
    ax.fill_between(rolling_count.index, 0, vals,
                    color=COLORS["strategy"], alpha=0.25)
    ax.plot(rolling_count.index, vals,
            color=COLORS["strategy"], linewidth=1.3)

    ax.axhline(n_total, color=COLORS["grid"], linewidth=0.8, linestyle="--",
               label=f"Full universe ({n_total} assets)")

    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, n_total * 1.1)
    ax.legend(frameon=False, fontsize=get_typography().legend)
    label_axes(ax, title=title, ylabel="Assets with valid prices (21d avg)")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Cross-asset rolling volatility
# ---------------------------------------------------------------------------


def plot_cross_asset_volatility(
    vol_df: pd.DataFrame,
    title: str = "Cross-Asset Realised Volatility (63d Rolling)",
    save_path: str | None = None,
) -> plt.Figure:
    """Per-asset 63-day rolling annualised volatility through time.

    Establishes the macro volatility context and reveals which assets drive
    universe-level risk regimes.  Persistent divergence between assets
    identifies structural regime heterogeneity.

    Args:
        vol_df:    Date × Asset rolling volatility DataFrame (annualised).
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    df = vol_df.dropna(how="all")
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No volatility data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text,
                color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_assets = len(df.columns)
    # Extra height for external legend when many assets
    legend_rows = max(1, (n_assets + 3) // 4)
    height = FIG_HEIGHT_STANDARD + 0.3 * legend_rows
    fig, ax = make_figure(height=height, width=FIG_WIDTH_FULL)

    # Institutional qualitative palette
    _palette = [
        "#1f4e79", "#c0392b", "#27ae60", "#f39c12",
        "#8e44ad", "#16a085", "#2c3e50", "#d35400",
        "#2980b9", "#e74c3c",
    ]

    for i, col in enumerate(df.columns):
        series = df[col].dropna()
        if series.empty:
            continue
        color = _palette[i % len(_palette)]
        ax.plot(series.index, series.to_numpy(),
                color=color, linewidth=1.0, alpha=0.80, label=col)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.axhline(0, color=COLORS["grid"], linewidth=0.5, linestyle="--", alpha=0.5)

    ax.legend(
        frameon=False,
        fontsize=get_typography().legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(4, n_assets),
        handlelength=1.5,
        columnspacing=1.2,
    )
    label_axes(ax, title=title, ylabel="Annualised Volatility (63d rolling)")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Universe correlation heatmap
# ---------------------------------------------------------------------------


def plot_universe_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Cross-Asset Return Correlation Structure",
    save_path: str | None = None,
) -> plt.Figure:
    """Static pairwise correlation heatmap for the full universe.

    Establishes macro diversification structure: correlated clusters reduce
    effective cross-sectional breadth; orthogonal assets preserve ranking
    information content.  Divergent correlations between risk-on and
    risk-off assets confirm regime-diversified universe composition.

    Args:
        corr_df:   Asset × Asset Pearson correlation DataFrame.
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import math as _math

    _t = get_typography()
    n = len(corr_df)
    if n < 2:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "Need ≥ 2 assets for correlation", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    import matplotlib.colors as mcolors

    cell = max(0.5, min(0.9, 8.0 / n))
    size = max(FIG_HEIGHT_STANDARD, cell * n + 1.5)
    fig, ax = make_figure(width=size + 0.8, height=size)

    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

    Z = corr_df.values.copy().astype(float)
    im = ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
    ax.grid(False)

    font_size = heatmap_cell_fontsize(n, n)
    thresh = 0.6
    for i in range(n):
        for j in range(n):
            v = Z[i, j]
            if not _math.isnan(v):
                text_color = "white" if abs(v) > thresh else "#333333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=font_size, color=text_color)

    tick_fs = scale_dynamic_fontsize(max(6, 9 - n // 6))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(list(corr_df.columns), rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(list(corr_df.index), fontsize=tick_fs)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=_t.colorbar)
    cbar.set_label("Pearson r", fontsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig
