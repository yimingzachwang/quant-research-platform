"""Signal diagnostic plots.

Functions for visually inspecting strategy signals — price overlays,
moving averages, trade markers, and position state.

Read-only: no signal computation, no backtest execution.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.styles import (
    COLORS,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography


def plot_sma_strategy(
    close: pd.Series,
    fast_sma: pd.Series,
    slow_sma: pd.Series,
    signal: pd.Series | None = None,
    title: str = "SMA Strategy",
    fast_label: str = "Fast SMA",
    slow_label: str = "Slow SMA",
    n_markers: int = 50,
    save_path: str | None = None,
) -> plt.Figure:
    """Price chart with fast/slow SMAs and optional buy/sell markers.

    Args:
        close: Asset close price series (DatetimeIndex).
        fast_sma: Fast moving average series.
        slow_sma: Slow moving average series.
        signal: Optional signal series in {-1, 0, +1}.  Transitions from
            flat/short to long are shown as buy markers; long to flat/short
            as sell markers.
        title: Figure title.
        fast_label: Legend label for the fast SMA line.
        slow_label: Legend label for the slow SMA line.
        n_markers: Maximum number of buy/sell markers to render (downsampled
            if signal changes too frequently to keep chart legible).
        save_path: If provided, saves the figure to this path.
    """
    nrows = 2 if signal is not None else 1
    height_ratios = [3.5, 1.0] if signal is not None else None
    fig, axes = make_figure(
        nrows=nrows,
        height=4.5 if signal is not None else 3.5,
        height_ratios=height_ratios,
        sharex=True,
    )
    ax_price = axes[0] if signal is not None else axes

    # --- Price + SMAs ---
    ax_price.plot(
        close.index, close,
        color=COLORS["neutral"], linewidth=0.9, label="Price", alpha=0.8, zorder=2,
    )
    ax_price.plot(
        fast_sma.index, fast_sma,
        color=COLORS["signal"], linewidth=1.3, label=fast_label, zorder=3,
    )
    ax_price.plot(
        slow_sma.index, slow_sma,
        color=COLORS["strategy"], linewidth=1.3, label=slow_label, zorder=3,
    )

    # --- Buy/sell markers from signal transitions ---
    if signal is not None:
        transitions = signal.diff().fillna(0)
        buy_dates = transitions[transitions > 0].index
        sell_dates = transitions[transitions < 0].index

        # Downsample markers if too dense
        if len(buy_dates) > n_markers:
            step = max(1, len(buy_dates) // n_markers)
            buy_dates = buy_dates[::step]
        if len(sell_dates) > n_markers:
            step = max(1, len(sell_dates) // n_markers)
            sell_dates = sell_dates[::step]

        buy_prices = close.reindex(buy_dates).dropna()
        sell_prices = close.reindex(sell_dates).dropna()

        ax_price.scatter(
            buy_prices.index, buy_prices,
            marker="^", color=COLORS["positive"], s=45, zorder=5, label="Buy", linewidths=0,
        )
        ax_price.scatter(
            sell_prices.index, sell_prices,
            marker="v", color=COLORS["negative"], s=45, zorder=5, label="Sell", linewidths=0,
        )

    ax_price.set_ylabel("Price")
    ax_price.legend(frameon=False, ncol=2)
    ax_price.set_title(title, fontweight="semibold", pad=8)

    # --- Signal state panel ---
    if signal is not None:
        ax_sig = axes[1]
        _plot_signal_state_on_axis(ax_sig, signal)
        ax_sig.set_ylabel("Position")

    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


def plot_position_state(
    signal: pd.Series,
    title: str = "Position State",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize long / flat / short position state as a filled bar chart.

    Long (+1): green fill above zero.
    Flat (0): nothing.
    Short (-1): red fill below zero.

    Args:
        signal: Position or signal series with values in {-1, 0, +1} or
            continuous weights.
        title: Figure title.
        save_path: If provided, saves the figure to this path.
    """
    fig, ax = make_figure(height=2.5)
    _plot_signal_state_on_axis(ax, signal)
    label_axes(ax, title=title, ylabel="Position")
    fig.tight_layout()
    if save_path:
        from src.visualization.utils import save_figure
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plot_signal_state_on_axis(ax: plt.Axes, signal: pd.Series) -> None:
    """Render a signal/position series as a filled state chart on ``ax``."""
    ax.fill_between(
        signal.index, signal, 0,
        where=signal > 0, color=COLORS["positive"], alpha=0.55, linewidth=0, label="Long",
    )
    ax.fill_between(
        signal.index, signal, 0,
        where=signal < 0, color=COLORS["negative"], alpha=0.55, linewidth=0, label="Short",
    )
    ax.step(signal.index, signal, color=COLORS["neutral"], linewidth=0.7, where="post")
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="-")
    _t = get_typography()
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Flat", "Long"], fontsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper right")
