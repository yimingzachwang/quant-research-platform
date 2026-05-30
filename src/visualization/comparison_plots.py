"""Comparative visualization for multiple strategies.

Functions accept a dict[strategy_name, StrategyResult] and produce
overlay / side-by-side figures.  All functions are read-only — they never
execute backtests or mutate input data.

Follows the same style conventions as backtest_plots.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_HEIGHT_STANDARD,
    FIG_WIDTH_FULL,
    format_pct_axis,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography

# Qualitative palette for up to 8 strategies
_PALETTE = [
    "#1f4e79",  # deep navy
    "#c0392b",  # red
    "#27ae60",  # green
    "#f39c12",  # amber
    "#8e44ad",  # purple
    "#16a085",  # teal
    "#e67e22",  # orange
    "#2c3e50",  # dark slate
]


def _colors(n: int) -> list[str]:
    """Return n colours, cycling the palette if needed."""
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def plot_strategy_equity_curves(
    results: dict,  # dict[str, StrategyResult]
    title: str = "Strategy Comparison — Equity Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay equity curves for all strategies on a single axis.

    Args:
        results: dict mapping strategy name → StrategyResult.
        title: Figure title.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    colors = _colors(len(results))

    for (name, result), color in zip(results.items(), colors):
        ec = result.backtest["equity_curve"]
        ax.plot(ec.index, ec, label=name, color=color, linewidth=1.4)

    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.8, linestyle="-", zorder=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    label_axes(ax, title=title, ylabel="Growth of $1")
    ax.legend(frameon=False, fontsize=get_typography().legend)

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_strategy_drawdowns(
    results: dict,  # dict[str, StrategyResult]
    title: str = "Strategy Comparison — Drawdowns",
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay drawdown series for all strategies on a single axis.

    Args:
        results: dict mapping strategy name → StrategyResult.
        title: Figure title.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    colors = _colors(len(results))

    for (name, result), color in zip(results.items(), colors):
        dd = result.backtest["drawdown"]
        ax.plot(dd.index, dd, label=name, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(dd.index, dd, 0, alpha=0.07, color=color)

    ax.axhline(0, color=COLORS["grid"], linewidth=0.8)
    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Drawdown")
    ax.legend(frameon=False, fontsize=get_typography().legend)

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart comparing one metric across strategies.

    Args:
        metrics_df: DataFrame from metrics_table() — rows are strategies,
            columns are metric names.
        metric: Column name to plot.  Default 'sharpe_ratio'.
        title: Figure title.  Defaults to the metric name.
        save_path: If provided, saves the figure to this path.

    Returns:
        matplotlib Figure.
    """
    if metric not in metrics_df.columns:
        msg = f"Metric {metric!r} not in DataFrame columns: {list(metrics_df.columns)}"
        raise ValueError(msg)

    series = metrics_df[metric].sort_values(ascending=True)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in series]

    height = max(2.5, len(series) * 0.55)
    fig, ax = make_figure(height=height)

    bars = ax.barh(series.index, series.values, color=colors, alpha=0.85, height=0.6)

    # Annotate bar ends
    _t = get_typography()
    for bar, val in zip(bars, series.values):
        offset = 0.01 * (series.abs().max() or 1.0)
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + (offset if val >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha=ha,
            fontsize=_t.annotation,
        )

    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    ax.set_xlabel(metric.replace("_", " ").title())
    label_axes(ax, title=title or metric.replace("_", " ").title())

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_metrics_table(
    metrics_df: pd.DataFrame,
    title: str = "Strategy Metrics",
    save_path: str | None = None,
) -> plt.Figure:
    """Render a metrics DataFrame as a formatted matplotlib table figure.

    Useful for exporting a clean side-by-side summary alongside the
    equity curve and drawdown plots.

    Args:
        metrics_df: Output of metrics_table() — rows strategies, cols metrics.
        title: Figure title.
        save_path: If provided, saves to this path.

    Returns:
        matplotlib Figure.
    """
    # Format numeric values for display
    fmt = metrics_df.copy()
    pct_cols = {"annualized_return", "annualized_volatility", "max_drawdown", "hit_rate"}
    for col in fmt.columns:
        if col in pct_cols:
            fmt[col] = fmt[col].map(lambda v: f"{v:.1%}")
        else:
            fmt[col] = fmt[col].map(lambda v: f"{v:.3f}")

    n_rows, n_cols = fmt.shape
    height = max(2.0, n_rows * 0.45 + 1.0)
    fig, ax = make_figure(height=height)
    ax.axis("off")

    col_labels = [c.replace("_", "\n") for c in fmt.columns]
    table = ax.table(
        cellText=fmt.values,
        rowLabels=fmt.index.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    # Style header and row-label cells
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        if row == 0 or col == -1:
            cell.set_facecolor("#f0f4f8")
            cell.set_text_props(fontweight="semibold")

    label_axes(ax, title=title)
    fig.tight_layout()

    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig
