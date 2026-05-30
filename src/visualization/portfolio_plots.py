"""Portfolio-level visualization utilities.

These extend the existing backtest plot set with multi-asset specific charts:
    - Stacked allocation area chart (weight history)
    - Asset-level return contribution
    - Cross-sectional correlation heatmap
    - Weight heatmap, turnover, concentration, and rolling correlation

All functions accept plain DataFrames and return matplotlib Figures.
Read-only: no data mutation, no backtest execution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_WIDTH_FULL,
    format_pct_axis,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography, heatmap_cell_fontsize

_MAX_HEATMAP_COLS = 60  # downsample x-axis when more periods than this


# Ontology-aware semantic palette — groups assets by economic role so that
# regime rotations (equity → defensive, sector concentration, etc.) are
# immediately visible from colour clustering without reading the legend.
#   Broad equities     → cool blue family
#   International eq.  → green family
#   Rates / treasuries → purple family
#   Commodities        → gold family
#   Sectors (equity)   → warm orange/red family
_TICKER_COLORS: dict[str, str] = {
    # Broad US equities — cool blue spectrum
    "SPY": "#1B4F8A",
    "QQQ": "#2980B9",
    "IWM": "#5DADE2",
    "VTI": "#1A6FAA",
    "DIA": "#3498DB",
    # International equities — green family
    "EEM": "#1A7A4A",
    "VEA": "#27AE60",
    "VWO": "#52BE80",
    "EFA": "#1E8449",
    # Rates / treasuries — purple family
    "TLT": "#7D3C98",
    "IEF": "#9B59B6",
    "SHY": "#BB8FCE",
    "AGG": "#6C3483",
    "BND": "#A569BD",
    # Commodities — gold / amber family
    "GLD": "#D4A017",
    "SLV": "#B8860B",
    "USO": "#E67E22",
    "DBC": "#CA6F1E",
    # Equity sectors — warm orange/red family (perceptually warm, each distinct)
    "XLF": "#C0392B",   # financials — deep red
    "XLK": "#E8900A",   # tech — orange
    "XLE": "#A93226",   # energy — dark red
    "XLV": "#D35400",   # health — burnt orange
    "XLI": "#E74C3C",   # industrials — medium red
    "XLY": "#F1948A",   # consumer disc. — light red
    "XLP": "#F0B27A",   # consumer staples — peach
    # Credit / fixed income
    "HYG": "#884EA0",
    "LQD": "#76448A",
    "JNK": "#6E2F8A",
}

# Fallback qualitative palette for tickers not in _TICKER_COLORS
_ASSET_COLORS_FALLBACK = [
    "#1f4e79",
    "#c0392b",
    "#27ae60",
    "#f39c12",
    "#8e44ad",
    "#16a085",
    "#2c3e50",
]


def _get_asset_colors(symbols: list[str]) -> list[str]:
    """Return per-symbol colours using semantic palette with cyclic fallback."""
    fallback_pool = [c for c in _ASSET_COLORS_FALLBACK
                     if c not in {_TICKER_COLORS.get(s) for s in symbols}]
    if not fallback_pool:
        fallback_pool = _ASSET_COLORS_FALLBACK[:]
    fallback_cycle = (fallback_pool * ((len(symbols) // len(fallback_pool)) + 1))
    fi = 0
    result: list[str] = []
    for sym in symbols:
        if sym in _TICKER_COLORS:
            result.append(_TICKER_COLORS[sym])
        else:
            result.append(fallback_cycle[fi])
            fi += 1
    return result


def plot_weights(
    weights: pd.DataFrame,
    title: str = "Portfolio Weights",
    save_path: str | None = None,
) -> plt.Figure:
    """Stacked area chart of portfolio weight history.

    Best suited for long-only, fully-invested portfolios where weights sum
    to 1.  For sparse or variable-leverage portfolios the individual line
    variant is clearer — use plot_weight_lines() instead.

    Args:
        weights: Date × Asset weight DataFrame (values in [0, 1], rows ≈ 1).
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=4.0)

    symbols = list(weights.columns)
    colors = _get_asset_colors(symbols)

    ax.stackplot(
        weights.index,
        [weights[s].fillna(0.0) for s in symbols],
        labels=symbols,
        colors=colors,
        alpha=0.85,
    )

    format_pct_axis(ax)
    ax.set_ylim(0, None)
    label_axes(ax, title=title, ylabel="Weight")
    _t = get_typography()
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=len(symbols),
        frameon=False,
        fontsize=_t.legend,
        handlelength=1.1,
        columnspacing=0.9,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_weight_lines(
    weights: pd.DataFrame,
    title: str = "Asset Weights",
    save_path: str | None = None,
) -> plt.Figure:
    """Individual line chart for each asset's weight over time.

    More legible than the stacked area chart when weights are sparse
    or the portfolio is not fully invested.

    Args:
        weights: Date × Asset weight DataFrame.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=3.5)

    symbols = list(weights.columns)
    colors = _get_asset_colors(symbols)

    for sym, color in zip(symbols, colors, strict=False):
        ax.plot(
            weights.index,
            weights[sym].fillna(0.0),
            label=sym,
            color=color,
            linewidth=1.3,
        )

    ax.axhline(0, color=COLORS["grid"], linewidth=0.7)
    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Weight")
    ax.legend(frameon=False, ncol=min(4, len(symbols)), fontsize=get_typography().legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_asset_correlation(
    returns: pd.DataFrame,
    title: str = "Asset Return Correlation",
    save_path: str | None = None,
) -> plt.Figure:
    """Correlation heatmap across all assets.

    Args:
        returns: Date × Asset return DataFrame.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    corr = returns.corr()
    n = len(corr)

    fig, ax = make_figure(width=min(FIG_WIDTH_FULL, n * 1.1 + 1.5), height=n * 0.9 + 1.0)

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=-1, vmax=1)

    im = ax.imshow(corr.values, cmap=cmap, norm=norm, aspect="auto")

    _t = get_typography()
    cell_fs = heatmap_cell_fontsize(n, n)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "#333333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=cell_fs, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=_t.tick)
    ax.set_yticklabels(corr.index, fontsize=_t.tick)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_rolling_weights(
    weights: pd.DataFrame,
    resample_freq: str = "ME",
    title: str = "Monthly Portfolio Weights",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of portfolio weights resampled to a lower frequency.

    Useful for showing month-end rebalance decisions without the noise
    of forward-filled daily weights.

    Args:
        weights: Date × Asset weight DataFrame (daily or periodic).
        resample_freq: Pandas offset alias for the bar frequency (default 'ME').
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    sampled = weights.resample(resample_freq).last().fillna(0.0)
    symbols = list(sampled.columns)
    colors = _get_asset_colors(symbols)

    fig, ax = make_figure(height=3.5)

    bottom = np.zeros(len(sampled))
    for sym, color in zip(symbols, colors, strict=False):
        vals = sampled[sym].values
        ax.bar(
            sampled.index,
            vals,
            bottom=bottom,
            label=sym,
            color=color,
            alpha=0.85,
            width=20,  # ~20 calendar days wide for monthly bars
        )
        bottom = bottom + vals

    format_pct_axis(ax)
    ax.set_ylim(0, None)
    label_axes(ax, title=title, ylabel="Weight")
    ax.legend(frameon=False, ncol=min(4, len(symbols)), fontsize=get_typography().legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_weight_heatmap(
    weights: pd.DataFrame,
    title: str = "Weight Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of portfolio weights: assets on y-axis, time on x-axis.

    Downsamples to monthly snapshots to keep the chart readable.

    Args:
        weights: Date × Asset weight DataFrame.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    sampled = weights.resample("ME").last().fillna(0.0)
    n_assets = len(sampled.columns)
    n_periods = len(sampled)

    fig_w = max(FIG_WIDTH_FULL, n_periods * 0.35)
    fig_h = max(2.5, n_assets * 0.65 + 1.5)
    fig, ax = make_figure(width=fig_w, height=fig_h)

    im = ax.imshow(
        sampled.T.values,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    _t = get_typography()
    ax.set_yticks(range(n_assets))
    ax.set_yticklabels(sampled.columns, fontsize=_t.tick)

    step = max(1, n_periods // _MAX_HEATMAP_COLS)
    tick_positions = list(range(0, n_periods, step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [sampled.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right",
        fontsize=_t.tick,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=_t.colorbar)
    cbar.set_label("Weight", fontsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_turnover(
    weights: pd.DataFrame,
    title: str = "Portfolio Turnover",
    save_path: str | None = None,
) -> plt.Figure:
    """Daily portfolio turnover with a 21-day rolling average overlay.

    Turnover is defined as Σ|Δw_i| per day — twice the fraction of portfolio
    traded when going from one allocation to another.

    Args:
        weights: Date × Asset weight DataFrame.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    from src.visualization.diagnostics import compute_turnover

    turnover = compute_turnover(weights).fillna(0.0)
    rolling_avg = turnover.rolling(21, min_periods=1).mean()

    fig, ax = make_figure(height=3.5)
    ax.bar(
        turnover.index,
        turnover.values,
        color=COLORS["strategy"],
        alpha=0.5,
        width=1,
        label="Daily",
    )
    ax.plot(
        rolling_avg.index,
        rolling_avg.values,
        color=COLORS["negative"],
        linewidth=1.5,
        label="21d avg",
    )

    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Turnover (Σ|Δw|)")
    ax.legend(frameon=False, fontsize=get_typography().legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_weight_concentration(
    weights: pd.DataFrame,
    title: str = "Portfolio Concentration",
    save_path: str | None = None,
) -> plt.Figure:
    """Three-panel chart of HHI, max weight, and effective N over time.

    Args:
        weights: Date × Asset weight DataFrame.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    from src.visualization.diagnostics import compute_concentration_metrics

    metrics = compute_concentration_metrics(weights)

    fig, axes = make_figure(nrows=3, height=7.0, sharex=True)

    axes[0].plot(metrics.index, metrics["max_weight"], color=COLORS["strategy"], linewidth=1.3)
    format_pct_axis(axes[0])
    label_axes(axes[0], ylabel="Max Weight")
    axes[0].set_title(title, fontweight="semibold", pad=8)

    axes[1].plot(metrics.index, metrics["hhi"], color=COLORS["signal"], linewidth=1.3)
    label_axes(axes[1], ylabel="HHI")

    axes[2].plot(metrics.index, metrics["effective_n"], color=COLORS["positive"], linewidth=1.3)
    label_axes(axes[2], ylabel="Effective N")

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_asset_contribution(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    title: str = "Cumulative Asset Return Contribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Cumulative return contribution per asset using lagged weights.

    Contribution at time t = w_{t-1,i} * r_{t,i}.  Avoids look-ahead by
    shifting weights forward one period before multiplying.

    Args:
        returns: Date × Asset return DataFrame.
        weights: Date × Asset weight DataFrame (same shape as returns).
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    w_lagged = weights.shift(1).fillna(0.0)
    common_idx = returns.index.intersection(w_lagged.index)
    common_cols = returns.columns.intersection(w_lagged.columns)

    contrib = w_lagged.loc[common_idx, common_cols].multiply(
        returns.loc[common_idx, common_cols]
    )
    cum_contrib = (1.0 + contrib).cumprod() - 1.0

    symbols = list(cum_contrib.columns)
    colors = _get_asset_colors(symbols)

    fig, ax = make_figure(height=3.8)
    for sym, color in zip(symbols, colors, strict=False):
        ax.plot(cum_contrib.index, cum_contrib[sym], label=sym, color=color, linewidth=1.3)

    ax.axhline(0, color=COLORS["grid"], linewidth=0.7)
    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Cumulative Contribution")
    ax.legend(frameon=False, ncol=min(4, len(symbols)), fontsize=get_typography().legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
    title: str = "Rolling Average Pairwise Correlation",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling mean pairwise correlation across all assets.

    Useful for spotting regimes where diversification benefits shrink
    (correlations spike during crises) or expand.

    Args:
        returns: Date × Asset return DataFrame.
        window: Rolling window length in periods.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    from src.visualization.diagnostics import rolling_average_correlation

    avg_corr = rolling_average_correlation(returns, window=window)
    mean_val = avg_corr.mean()

    fig, ax = make_figure(height=3.5)
    ax.plot(avg_corr.index, avg_corr.values, color=COLORS["strategy"], linewidth=1.3)
    ax.axhline(0, color=COLORS["grid"], linewidth=0.7)
    ax.axhline(
        mean_val,
        color=COLORS["neutral"],
        linewidth=0.9,
        linestyle="--",
        label=f"Mean = {mean_val:.2f}",
    )
    ax.fill_between(avg_corr.index, avg_corr.values, 0, alpha=0.12, color=COLORS["strategy"])

    ax.set_ylim(-1, 1)
    label_axes(ax, title=title, ylabel="Avg Pairwise Correlation")
    ax.legend(frameon=False, fontsize=get_typography().legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig
