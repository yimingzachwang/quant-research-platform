"""Backtest diagnostic plots.

All functions accept the DataFrame produced by run_backtest() plus optional
customisation arguments.  They return a matplotlib Figure so the caller can
display, further annotate, or save it via save_figure().

None of these functions rerun a backtest or mutate input data.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_HEIGHT_TALL,
    format_pct_axis,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography


def plot_equity_curve(
    backtest: pd.DataFrame,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    strategy_label: str = "Strategy",
    benchmark_label: str = "Benchmark",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot cumulative equity curve with optional benchmark overlay.

    Args:
        backtest: DataFrame from run_backtest() — must contain 'equity_curve'.
        benchmark: Optional benchmark return series.  Will be rebased to 1.0.
        title: Figure title.
        strategy_label: Legend label for the strategy line.
        benchmark_label: Legend label for the benchmark line.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=3.5)

    ax.plot(
        backtest.index,
        backtest["equity_curve"],
        color=COLORS["strategy"],
        linewidth=1.6,
        label=strategy_label,
        zorder=3,
    )

    if benchmark is not None:
        bench_curve = (1.0 + benchmark).cumprod()
        # Rebase to match the strategy starting value
        bench_curve = bench_curve / bench_curve.iloc[0]
        ax.plot(
            bench_curve.index,
            bench_curve,
            color=COLORS["benchmark"],
            linewidth=1.2,
            linestyle="--",
            label=benchmark_label,
            zorder=2,
        )

    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.8, linestyle="-", zorder=1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    label_axes(ax, title=title, ylabel="Growth of $1")
    ax.legend(frameon=False)

    # Shade the area under the equity curve
    ax.fill_between(
        backtest.index,
        1.0,
        backtest["equity_curve"],
        where=backtest["equity_curve"] >= 1.0,
        alpha=0.08,
        color=COLORS["positive"],
        zorder=1,
    )
    ax.fill_between(
        backtest.index,
        1.0,
        backtest["equity_curve"],
        where=backtest["equity_curve"] < 1.0,
        alpha=0.08,
        color=COLORS["negative"],
        zorder=1,
    )

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_drawdown(
    backtest: pd.DataFrame,
    title: str = "Drawdown",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the drawdown series as a filled underwater chart.

    Args:
        backtest: DataFrame from run_backtest() — must contain 'drawdown'.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=2.8)

    dd = backtest["drawdown"]

    ax.fill_between(
        dd.index,
        dd,
        0,
        color=COLORS["negative"],
        alpha=0.55,
        linewidth=0,
        zorder=2,
    )
    ax.plot(dd.index, dd, color=COLORS["negative"], linewidth=0.8, zorder=3)
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, zorder=1)

    # Annotate max drawdown
    mdd_val = dd.min()
    mdd_idx = dd.idxmin()
    _t = get_typography()
    ax.annotate(
        f"Max DD: {mdd_val:.1%}",
        xy=(mdd_idx, mdd_val),
        xytext=(10, -14),
        textcoords="offset points",
        fontsize=_t.annotation,
        color=COLORS["negative"],
        arrowprops={"arrowstyle": "->", "color": COLORS["negative"], "lw": 0.8},
    )

    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Drawdown")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_rolling_sharpe(
    backtest: pd.DataFrame,
    window: int = 252,
    risk_free_rate: float = 0.0,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot annualized rolling Sharpe ratio.

    Args:
        backtest: DataFrame from run_backtest() — must contain 'net_return'.
        window: Rolling window in trading days (default 252 = 1 year).
        risk_free_rate: Annual risk-free rate for excess return computation.
        title: Figure title (auto-generated if None).
        save_path: If provided, saves the figure to this path.
    """
    r = backtest["net_return"]
    rf_per_period = risk_free_rate / 252
    excess = r - rf_per_period

    rolling_mean = excess.rolling(window).mean()
    rolling_std = excess.rolling(window).std()
    rolling_sr = (rolling_mean / rolling_std.replace(0, float("nan"))) * math.sqrt(252)

    if title is None:
        title = f"Rolling Sharpe Ratio ({window}d window)"

    fig, ax = make_figure(height=3.2)

    ax.plot(rolling_sr.index, rolling_sr, color=COLORS["strategy"], linewidth=1.4)
    ax.axhline(0.0, color=COLORS["neutral"], linewidth=0.8, linestyle="--")
    ax.axhline(1.0, color=COLORS["positive"], linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(-1.0, color=COLORS["negative"], linewidth=0.6, linestyle=":", alpha=0.7)

    # Shade positive/negative zones
    ax.fill_between(
        rolling_sr.index, rolling_sr, 0,
        where=rolling_sr >= 0, color=COLORS["positive"], alpha=0.07,
    )
    ax.fill_between(
        rolling_sr.index, rolling_sr, 0,
        where=rolling_sr < 0, color=COLORS["negative"], alpha=0.07,
    )

    label_axes(ax, title=title, ylabel="Sharpe Ratio (annualised)")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_rolling_volatility(
    backtest: pd.DataFrame,
    window: int = 63,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot annualized rolling realized volatility.

    Args:
        backtest: DataFrame from run_backtest() — must contain 'net_return'.
        window: Rolling window in trading days (default 63 = ~1 quarter).
        title: Figure title (auto-generated if None).
        save_path: If provided, saves the figure to this path.
    """
    r = backtest["net_return"]
    ann_vol = r.rolling(window).std() * math.sqrt(252)

    if title is None:
        title = f"Rolling Volatility ({window}d window, annualised)"

    fig, ax = make_figure(height=3.0)

    ax.plot(ann_vol.index, ann_vol, color=COLORS["neutral"], linewidth=1.3)
    ax.fill_between(ann_vol.index, ann_vol, 0, color=COLORS["neutral"], alpha=0.10)

    format_pct_axis(ax)
    label_axes(ax, title=title, ylabel="Volatility (annualised)")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_equity_and_drawdown(
    backtest: pd.DataFrame,
    benchmark: pd.Series | None = None,
    title: str = "Strategy Performance",
    strategy_label: str = "Strategy",
    benchmark_label: str = "Benchmark",
    save_path: str | None = None,
) -> plt.Figure:
    """Combined equity curve + drawdown panel — the standard research tearsheet layout.

    Args:
        backtest: DataFrame from run_backtest().
        benchmark: Optional benchmark return series, rebased to 1.0.
        title: Overall figure title.
        save_path: If provided, saves the figure to this path.
    """
    fig, axes = make_figure(
        nrows=2,
        height=FIG_HEIGHT_TALL,
        height_ratios=[2.8, 1.0],
        sharex=True,
    )
    ax_eq, ax_dd = axes

    # --- Equity curve ---
    ax_eq.plot(
        backtest.index,
        backtest["equity_curve"],
        color=COLORS["strategy"],
        linewidth=1.6,
        label=strategy_label,
        zorder=3,
    )
    if benchmark is not None:
        bench_curve = (1.0 + benchmark).cumprod()
        bench_curve = bench_curve / bench_curve.iloc[0]
        ax_eq.plot(
            bench_curve.index,
            bench_curve,
            color=COLORS["benchmark"],
            linewidth=1.2,
            linestyle="--",
            label=benchmark_label,
            zorder=2,
        )
    ax_eq.axhline(1.0, color=COLORS["grid"], linewidth=0.8)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}x"))
    label_axes(ax_eq, title=title, ylabel="Growth of $1")
    ax_eq.legend(frameon=False)

    # Annotate total return in bottom-right corner
    _t = get_typography()
    equity = backtest["equity_curve"].dropna()
    if len(equity) > 0:
        total_ret = float(equity.iloc[-1]) - 1.0
        color = COLORS["positive"] if total_ret >= 0 else COLORS["negative"]
        ax_eq.annotate(
            f"Total return: {total_ret:+.1%}",
            xy=(0.98, 0.04),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=_t.annotation,
            color=color,
            fontweight="semibold",
        )

    # --- Drawdown ---
    dd = backtest["drawdown"]
    ax_dd.fill_between(dd.index, dd, 0, color=COLORS["negative"], alpha=0.55, linewidth=0)
    ax_dd.plot(dd.index, dd, color=COLORS["negative"], linewidth=0.7)
    ax_dd.axhline(0, color=COLORS["neutral"], linewidth=0.7)

    # Annotate max drawdown
    mdd = float(dd.min())
    ax_dd.annotate(
        f"Max DD: {mdd:.1%}",
        xy=(0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=_t.annotation,
        color=COLORS["negative"],
    )

    format_pct_axis(ax_dd)
    label_axes(ax_dd, ylabel="Drawdown")

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig
