"""Visualization utilities for walk-forward validation results.

All functions accept WalkForwardResult objects (or plain DataFrames from
stability helpers) and return matplotlib Figures. Read-only: no data mutation.
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
from src.visualization.typography import get_typography

# Qualitative palette cycled across splits
_SPLIT_PALETTE = [
    "#1f4e79", "#c0392b", "#27ae60", "#f39c12",
    "#8e44ad", "#16a085", "#2c3e50", "#d35400",
]


def _split_color(i: int) -> str:
    return _SPLIT_PALETTE[i % len(_SPLIT_PALETTE)]


def plot_walk_forward_equity(
    wf_result,
    title: str = "Walk-Forward Equity Curves (Test Periods)",
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay equity curves for each test window on a single chart.

    Each split's equity curve is anchored at 1.0 at its test_start date,
    making splits visually comparable regardless of starting level.

    Args:
        wf_result: WalkForwardResult from run_walk_forward_validation().
        title: Figure title.
        save_path: Optional path to save the figure.
    """
    fig, ax = make_figure(height=4.0)

    for sr in wf_result.splits:
        color = _split_color(sr.split.split_index)
        label = f"Split {sr.split.split_index} ({sr.split.test_start.strftime('%Y-%m')})"
        ax.plot(
            sr.equity_curve.index,
            sr.equity_curve.values,
            color=color,
            linewidth=1.3,
            label=label,
            alpha=0.85,
        )

    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Cumulative Return (test period)")
    _t = get_typography()
    ax.legend(frameon=False, fontsize=_t.legend, ncol=max(1, min(4, len(wf_result.splits))))
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_walk_forward_stitched(
    wf_result,
    title: str = "Walk-Forward Stitched Equity Curve",
    save_path: str | None = None,
) -> plt.Figure:
    """Concatenate test-period equity curves into a single time series.

    Each test segment is chained from the terminal value of the previous one,
    giving a realistic out-of-sample equity curve with no gaps.

    Args:
        wf_result: WalkForwardResult from run_walk_forward_validation().
        title: Figure title.
        save_path: Optional path to save the figure.
    """
    if not wf_result.splits:
        fig, ax = make_figure()
        label_axes(ax, title=title)
        return fig

    # Chain equity curves: scale each so it starts at the prior end-value
    segments = []
    prev_end = 1.0
    for sr in wf_result.splits:
        scaled = sr.equity_curve * prev_end
        prev_end = float(scaled.iloc[-1])
        segments.append(scaled)

    stitched = pd.concat(segments)

    fig, ax = make_figure(height=4.0)
    ax.plot(stitched.index, stitched.values, color=COLORS["strategy"], linewidth=1.4)
    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")

    # Shade alternating test windows
    for i, sr in enumerate(wf_result.splits):
        if i % 2 == 0:
            ax.axvspan(
                sr.split.test_start,
                sr.split.test_end,
                alpha=0.06,
                color=COLORS["strategy"],
            )

    # Annotate terminal value
    final_val = float(stitched.iloc[-1])
    _t = get_typography()
    ax.annotate(
        f"Final: {final_val:.2f}×",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=_t.annotation,
        color=COLORS["strategy"],
        fontweight="semibold",
    )

    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Cumulative Return (stitched OOS)")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_split_sharpes(
    wf_result,
    title: str = "Out-of-Sample Sharpe by Split",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of Sharpe ratio per test split.

    Bars are colored green/red based on sign to quickly identify
    splits where the strategy underperformed.

    Args:
        wf_result: WalkForwardResult from run_walk_forward_validation().
        title: Figure title.
        save_path: Optional path to save the figure.
    """
    from src.validation.stability import rolling_sharpe_by_split

    sharpes = rolling_sharpe_by_split(wf_result)
    labels = [f"S{sr.split.split_index}\n{sr.split.test_start.strftime('%Y-%m')}"
              for sr in wf_result.splits]
    values = sharpes.values

    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in values]

    fig, ax = make_figure(width=max(8.0, len(values) * 1.2), height=3.5)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, alpha=0.8, width=0.6)

    _t = get_typography()
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.04 if v >= 0 else -0.12),
            f"{v:.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=_t.annotation,
        )

    mean_sharpe = float(np.mean(values))
    ax.axhline(0, color=COLORS["grid"], linewidth=0.8)
    ax.axhline(
        mean_sharpe,
        color=COLORS["neutral"],
        linewidth=0.9,
        linestyle="--",
        label=f"Mean = {mean_sharpe:.2f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_t.tick)
    label_axes(ax, title=title, ylabel="Sharpe Ratio")
    ax.legend(frameon=False, fontsize=_t.legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_metric_stability(
    wf_result,
    metric: str = "annualized_return",
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of any metric across test splits over time.

    Args:
        wf_result: WalkForwardResult from run_walk_forward_validation().
        metric: Key from the per-split metrics dict.
        title: Figure title (auto-generated if None).
        save_path: Optional path to save the figure.
    """
    from src.validation.stability import split_metrics_table

    table = split_metrics_table(wf_result)
    if metric not in table.columns:
        raise ValueError(f"Metric {metric!r} not found in split results.")

    values = table[metric]
    x_dates = table["test_start"]

    _title = title or f"{metric.replace('_', ' ').title()} by Split"
    fig, ax = make_figure(height=3.5)
    ax.plot(x_dates, values, marker="o", color=COLORS["strategy"], linewidth=1.3, markersize=5)
    _t = get_typography()
    ax.axhline(values.mean(), color=COLORS["neutral"], linewidth=0.9,
               linestyle="--", label=f"Mean = {values.mean():.3f}")
    ax.axhline(0, color=COLORS["grid"], linewidth=0.7)

    label_axes(ax, title=_title, ylabel=metric.replace("_", " ").title())
    ax.legend(frameon=False, fontsize=_t.legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_walk_forward_timeline(
    wf_result,
    title: str = "Walk-Forward Window Timeline",
    save_path: str | None = None,
) -> plt.Figure:
    """Gantt-style horizontal bar chart of train/test windows.

    Each split occupies a row.  Train windows are shown in neutral grey;
    test windows are green (positive OOS Sharpe) or red (negative).
    OOS Sharpe is annotated in white on each test bar.

    Args:
        wf_result: WalkForwardResult from run_walk_forward_validation().
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.dates as mdates

    splits = wf_result.splits
    if not splits:
        fig, ax = make_figure()
        label_axes(ax, title=title)
        return fig

    try:
        from src.validation.stability import split_metrics_table
        metrics_table = split_metrics_table(wf_result)
        sharpe_col = metrics_table["sharpe_ratio"].to_numpy() if "sharpe_ratio" in metrics_table.columns else None
    except Exception:
        sharpe_col = None

    n = len(splits)
    height = max(3.5, n * 0.7 + 1.2)
    fig, ax = make_figure(height=height, width=FIG_WIDTH_FULL)
    _t = get_typography()

    for i, sr in enumerate(splits):
        tr_s = mdates.date2num(pd.Timestamp(sr.split.train_start))
        tr_e = mdates.date2num(pd.Timestamp(sr.split.train_end))
        te_s = mdates.date2num(pd.Timestamp(sr.split.test_start))
        te_e = mdates.date2num(pd.Timestamp(sr.split.test_end))

        ax.barh(i, tr_e - tr_s, left=tr_s, height=0.55,
                color=COLORS["neutral"], alpha=0.4,
                label="Train" if i == 0 else "")

        sharpe = float(sharpe_col[i]) if sharpe_col is not None else None
        t_color = COLORS["positive"] if (sharpe is not None and sharpe >= 0) else COLORS["negative"]
        ax.barh(i, te_e - te_s, left=te_s, height=0.55,
                color=t_color, alpha=0.75,
                label="Test" if i == 0 else "")

        if sharpe is not None:
            mid_x = (te_s + te_e) / 2
            ax.text(mid_x, i, f"{sharpe:.2f}",
                    ha="center", va="center", fontsize=_t.small_annotation,
                    fontweight="semibold", color="white")

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"S{sr.split.split_index}" for sr in splits], fontsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="lower right")
    label_axes(ax, title=title, xlabel="", ylabel="")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_train_vs_test(
    wf_result,
    metric: str = "sharpe_ratio",
    train_results: "WalkForwardResult | None" = None,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Compare a metric across test splits (optionally vs. train splits).

    If ``train_results`` is provided its split values are shown alongside
    the test split values for a direct train/test gap analysis.

    Args:
        wf_result: WalkForwardResult for test periods.
        metric: Metric to compare.
        train_results: Optional WalkForwardResult for train periods.
        title: Figure title.
        save_path: Optional path to save the figure.
    """
    from src.validation.stability import split_metrics_table

    test_table = split_metrics_table(wf_result)
    if metric not in test_table.columns:
        raise ValueError(f"Metric {metric!r} not found.")

    _title = title or f"Train vs Test: {metric.replace('_', ' ').title()}"
    fig, ax = make_figure(height=3.5)

    x = np.arange(len(wf_result.splits))
    ax.bar(x - 0.2, test_table[metric].values, width=0.35,
           color=COLORS["strategy"], alpha=0.8, label="Test")

    if train_results is not None:
        train_table = split_metrics_table(train_results)
        if metric in train_table.columns:
            ax.bar(x + 0.2, train_table[metric].values, width=0.35,
                   color=COLORS["neutral"], alpha=0.8, label="Train")

    _t = get_typography()
    ax.axhline(0, color=COLORS["grid"], linewidth=0.8)
    labels = [f"S{sr.split.split_index}" for sr in wf_result.splits]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=_t.tick)
    label_axes(ax, title=_title, ylabel=metric.replace("_", " ").title())
    ax.legend(frameon=False, fontsize=_t.legend)
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig
