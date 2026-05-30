"""Comparative allocation research visualizations.

Supports Phase 2.5: controlled comparison of equal-weight vs softmax
allocation policies across the same signal/universe/features.

All functions accept pre-computed DataFrames/dicts loaded from experiment
artefacts — no data loading, no model fitting, no recomputation.

Figures expose tradeoffs and diagnostics, not performance dashboards.
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
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography, scale_dynamic_fontsize
from src.visualization.utils import save_figure

# Qualitative palette indexed by scheme label
_SCHEME_PALETTE = {
    "Equal Weight": "#1f4e79",      # deep navy
    "Softmax τ=0.5": "#c0392b",    # red (most concentrated)
    "Softmax τ=1.0": "#e67e22",    # amber
    "Softmax τ=2.0": "#27ae60",    # green (closest to EW)
}
_DEFAULT_PALETTE = ["#1f4e79", "#c0392b", "#e67e22", "#27ae60", "#8e44ad", "#16a085"]


def _scheme_colors(labels: list[str]) -> list[str]:
    return [_SCHEME_PALETTE.get(lbl, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])
            for i, lbl in enumerate(labels)]


# ---------------------------------------------------------------------------
# Equity curve overlay
# ---------------------------------------------------------------------------


def plot_equity_comparison(
    equity_dict: dict[str, pd.Series],
    title: str = "Allocation Policy Comparison — Equity Curves",
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay equity curves for all allocation schemes on a single axis.

    Args:
        equity_dict: scheme_label → cumulative equity curve Series.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    labels = list(equity_dict.keys())
    colors = _scheme_colors(labels)

    for (lbl, curve), color in zip(equity_dict.items(), colors, strict=False):
        lw = 1.8 if "Equal Weight" in lbl else 1.2
        ax.plot(curve.index, curve.values, color=color, linewidth=lw, label=lbl,
                alpha=0.9)

    ax.axhline(1.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    label_axes(ax, title=title)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Concentration evolution comparison
# ---------------------------------------------------------------------------


def plot_hhi_comparison(
    hhi_dict: dict[str, pd.Series],
    window: int = 63,
    title: str = "HHI Through Time by Allocation Scheme",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling HHI time-series for all allocation schemes.

    HHI = Σwᵢ² per period. Equal-weight across k assets gives HHI = 1/k.
    Higher HHI ↔ more concentrated. The equal-weight baseline is the
    reference; all softmax curves should be at or above it.

    Args:
        hhi_dict: scheme_label → raw daily HHI Series.
        window: Rolling mean window in trading days.
        save_path: Optional save path.
    """
    _t = get_typography()
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    labels = list(hhi_dict.keys())
    colors = _scheme_colors(labels)
    min_p = max(1, window // 4)

    for (lbl, hhi), color in zip(hhi_dict.items(), colors, strict=False):
        rolled = hhi.rolling(window, min_periods=min_p).mean()
        lw = 1.8 if "Equal Weight" in lbl else 1.2
        ax.plot(rolled.index, rolled.values, color=color, linewidth=lw,
                label=f"{lbl} (mean {hhi.mean():.3f})", alpha=0.9)

    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_ylabel(f"HHI ({window}d rolling mean)", fontsize=_t.small_label)
    label_axes(ax, title=title)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_breadth_entropy_comparison(
    breadth_dict: dict[str, pd.Series],
    eff_n_dict: dict[str, pd.Series],
    window: int = 63,
    title: str = "Effective Breadth & Entropy N by Allocation Scheme",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel comparison: effective breadth (1/HHI) and entropy effective-N.

    Both panels tell the same diversification story from different angles.
    """
    _t = get_typography()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.7),
                                    sharex=True)

    labels = list(breadth_dict.keys())
    colors = _scheme_colors(labels)
    min_p = max(1, window // 4)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=_t.tick)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    for (lbl, br), color in zip(breadth_dict.items(), colors, strict=False):
        rolled = br.rolling(window, min_periods=min_p).mean()
        lw = 1.8 if "Equal Weight" in lbl else 1.2
        ax1.plot(rolled.index, rolled.values, color=color, linewidth=lw,
                 label=lbl, alpha=0.9)

    ax1.set_ylabel(f"Effective breadth ({window}d)", fontsize=_t.small_label)
    ax1.legend(frameon=False, fontsize=_t.legend)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left")

    for (lbl, en), color in zip(eff_n_dict.items(), colors, strict=False):
        rolled = en.rolling(window, min_periods=min_p).mean()
        lw = 1.8 if "Equal Weight" in lbl else 1.2
        ax2.plot(rolled.dropna().index, rolled.dropna().values, color=color,
                 linewidth=lw, label=lbl, alpha=0.9)

    ax2.set_ylabel(f"Entropy effective-N ({window}d)", fontsize=_t.small_label)
    ax2.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Turnover comparison
# ---------------------------------------------------------------------------


def plot_turnover_comparison(
    turnover_dict: dict[str, pd.Series],
    window: int = 21,
    title: str = "Portfolio Turnover by Allocation Scheme",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling turnover comparison: reveals cost drag by allocation scheme.

    Higher turnover amplifies transaction-cost drag. Softmax with low τ
    may be more sensitive to daily score fluctuations, increasing turnover
    relative to equal-weight.
    """
    _t = get_typography()
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    labels = list(turnover_dict.keys())
    colors = _scheme_colors(labels)
    min_p = max(1, window // 4)

    for (lbl, to), color in zip(turnover_dict.items(), colors, strict=False):
        rolled = to.rolling(window, min_periods=min_p).mean()
        lw = 1.8 if "Equal Weight" in lbl else 1.2
        mean_to = float(to.mean())
        ax.plot(rolled.index, rolled.values, color=color, linewidth=lw,
                label=f"{lbl} (mean {mean_to:.4f})", alpha=0.9)

    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_ylabel(f"Daily turnover ({window}d rolling)", fontsize=_t.small_label)
    label_axes(ax, title=title)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Sharpe vs concentration scatter
# ---------------------------------------------------------------------------


def plot_sharpe_vs_concentration(
    summary_df: pd.DataFrame,
    title: str = "Risk-Adjusted Return vs Concentration (Mean HHI)",
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter: Sharpe ratio vs mean HHI, with turnover as bubble size.

    Reveals whether concentration improves risk-adjusted returns or merely
    adds idiosyncratic risk without commensurate reward.

    Args:
        summary_df: DataFrame with index=scheme_label, columns including:
            sharpe_ratio, mean_hhi, mean_turnover (optional).
    """
    _t = get_typography()
    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    labels = list(summary_df.index)
    colors = _scheme_colors(labels)

    hhi_vals = summary_df["mean_hhi"].values
    sharpe_vals = summary_df["sharpe_ratio"].values

    # Bubble size from turnover when available
    turnover_vals = summary_df.get("mean_turnover", pd.Series(0.01, index=summary_df.index)).values
    sizes = np.clip(turnover_vals * 5000, 80, 600)

    for i, (lbl, color) in enumerate(zip(labels, colors, strict=False)):
        ax.scatter(hhi_vals[i], sharpe_vals[i], s=sizes[i], color=color,
                   alpha=0.85, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(lbl, (hhi_vals[i], sharpe_vals[i]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=scale_dynamic_fontsize(_t.small_annotation, len(labels)),
                    color=color)

    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7, linestyle="--")
    ax.set_xlabel("Mean HHI (concentration)", fontsize=_t.small_label)
    ax.set_ylabel("Sharpe ratio", fontsize=_t.small_label)
    ax.tick_params(labelsize=_t.tick)
    ax.grid(alpha=0.2, linewidth=0.5)
    label_axes(ax, title=title)

    if "mean_turnover" in summary_df.columns:
        ax.text(0.98, 0.02, "Bubble size ∝ mean turnover",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_t.small_annotation, color=COLORS["neutral"])

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Metrics comparison bar chart
# ---------------------------------------------------------------------------


def plot_allocation_metrics_bar(
    summary_df: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Allocation Policy Performance Comparison",
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing key metrics across allocation schemes.

    Args:
        summary_df: Rows = schemes, columns = metric names.
        metrics: Subset of columns to plot. Defaults to standard set.
    """
    _t = get_typography()
    if metrics is None:
        metrics = ["sharpe_ratio", "annualized_return", "max_drawdown",
                   "mean_hhi", "mean_turnover"]
    cols = [m for m in metrics if m in summary_df.columns]
    if not cols:
        fig, ax = make_figure()
        ax.text(0.5, 0.5, "No metrics to display", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n_metrics = len(cols)
    n_schemes = len(summary_df)
    labels = list(summary_df.index)
    _scheme_colors(labels)

    fig, axes = plt.subplots(1, n_metrics,
                              figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 0.9))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, cols, strict=False):
        vals = summary_df[metric].values
        bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                      for v in vals]
        ax.bar(range(n_schemes), vals, color=bar_colors, alpha=0.85, width=0.6)
        ax.set_xticks(range(n_schemes))
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=scale_dynamic_fontsize(_t.tick, n_schemes))
        ax.axhline(0, color=COLORS["neutral"], linewidth=0.7)
        ax.set_title(metric.replace("_", "\n"), fontsize=_t.small_annotation,
                     fontweight="semibold")
        ax.tick_params(labelsize=_t.tick)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

        for i, val in enumerate(vals):
            va = "bottom" if val >= 0 else "top"
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02 if ax.get_ylim()[0] != ax.get_ylim()[1] else 0.005
            ax.text(i, val + (offset if val >= 0 else -offset),
                    f"{val:.3f}", ha="center", va=va,
                    fontsize=_t.small_annotation)

    fig.suptitle(title, fontsize=_t.small_label, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Concentration vs temperature parameter
# ---------------------------------------------------------------------------


def plot_concentration_vs_temperature(
    summary_df: pd.DataFrame,
    title: str = "Allocation Concentration vs Softmax Temperature",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel: mean HHI and effective-N vs τ parameter.

    Quantifies the mechanical concentration effect of temperature.
    Expected monotonic relationship: lower τ → higher HHI, lower eff-N.

    Args:
        summary_df: Requires columns: temperature (float or NaN for EW),
            mean_hhi, mean_eff_n.
    """
    _t = get_typography()
    df = summary_df.copy()

    has_temp = "temperature" in df.columns and df["temperature"].notna().any()
    if not has_temp:
        fig, ax = make_figure()
        ax.text(0.5, 0.5, "Temperature data not available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    sm_df = df[df["temperature"].notna()].sort_values("temperature")
    if sm_df.empty:
        fig, ax = make_figure()
        ax.text(0.5, 0.5, "No softmax experiments found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ew_hhi = df.loc[df["temperature"].isna(), "mean_hhi"].values[0] if df["temperature"].isna().any() else None
    ew_eff_n = df.loc[df["temperature"].isna(), "mean_eff_n"].values[0] if df["temperature"].isna().any() else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 0.85))
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=_t.tick)
        ax.grid(alpha=0.2, linewidth=0.5)

    temps = sm_df["temperature"].values

    ax1.plot(temps, sm_df["mean_hhi"].values, "o-",
             color=COLORS["negative"], linewidth=1.5, markersize=7)
    if ew_hhi is not None:
        ax1.axhline(ew_hhi, color=COLORS["neutral"], linewidth=1.0, linestyle="--",
                    label=f"Equal weight ({ew_hhi:.3f})")
        ax1.legend(frameon=False, fontsize=_t.legend)
    ax1.set_xlabel("Softmax temperature (τ)", fontsize=_t.small_label)
    ax1.set_ylabel("Mean HHI", fontsize=_t.small_label)
    ax1.set_title("Concentration (HHI) vs τ", fontsize=_t.small_label,
                  fontweight="semibold", loc="left")

    if "mean_eff_n" in sm_df.columns:
        ax2.plot(temps, sm_df["mean_eff_n"].values, "o-",
                 color=COLORS["positive"], linewidth=1.5, markersize=7)
        if ew_eff_n is not None:
            ax2.axhline(ew_eff_n, color=COLORS["neutral"], linewidth=1.0, linestyle="--",
                        label=f"Equal weight ({ew_eff_n:.2f})")
            ax2.legend(frameon=False, fontsize=_t.legend)
        ax2.set_xlabel("Softmax temperature (τ)", fontsize=_t.small_label)
        ax2.set_ylabel("Entropy effective-N", fontsize=_t.small_label)
        ax2.set_title("Diversification (eff-N) vs τ", fontsize=_t.small_label,
                      fontweight="semibold", loc="left")

    fig.suptitle(title, fontsize=_t.small_label, fontweight="bold")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Dispersion vs concentration regime
# ---------------------------------------------------------------------------


def plot_dispersion_concentration_overlay(
    cs_std_dict: dict[str, pd.Series],
    hhi_dict: dict[str, pd.Series],
    scheme_label: str,
    window: int = 63,
    title: str = "Prediction Dispersion vs Allocation Concentration",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel: prediction CS std (top) vs HHI (bottom) for one scheme.

    Reveals whether periods of prediction dispersion collapse (low CS std)
    correspond to concentration spikes or reversals.  Shaded stress zones
    highlight co-movement.

    Args:
        cs_std_dict: scheme_label → CS prediction std Series.
        hhi_dict:    scheme_label → HHI Series.
        scheme_label: Which scheme to render.
    """
    _t = get_typography()
    cs_std = cs_std_dict.get(scheme_label)
    hhi = hhi_dict.get(scheme_label)

    if cs_std is None or hhi is None:
        fig, ax = make_figure()
        ax.text(0.5, 0.5, f"No data for scheme '{scheme_label}'",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    min_p = max(1, window // 4)
    std_roll = cs_std.rolling(window, min_periods=min_p).mean()
    hhi_roll = hhi.rolling(window, min_periods=min_p).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.6),
                                    sharex=True)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=_t.tick)

    ax1.fill_between(std_roll.index, std_roll.values,
                     color=COLORS["strategy"], alpha=0.3)
    ax1.plot(std_roll.index, std_roll.values,
             color=COLORS["strategy"], linewidth=1.2)
    mean_std = float(std_roll.dropna().mean())
    ax1.axhline(mean_std, color=COLORS["neutral"], linewidth=0.8, linestyle="--",
                label=f"Mean σ = {mean_std:.4f}")
    ax1.set_ylabel(f"CS prediction σ ({window}d)", fontsize=_t.small_label)
    ax1.legend(frameon=False, fontsize=_t.legend)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left")

    ax2.fill_between(hhi_roll.index, hhi_roll.values,
                     color=COLORS["negative"], alpha=0.25)
    ax2.plot(hhi_roll.index, hhi_roll.values,
             color=COLORS["negative"], linewidth=1.2)
    mean_hhi = float(hhi_roll.dropna().mean())
    ax2.axhline(mean_hhi, color=COLORS["neutral"], linewidth=0.8, linestyle="--",
                label=f"Mean HHI = {mean_hhi:.3f}")
    ax2.set_ylabel(f"HHI ({window}d rolling)", fontsize=_t.small_label)
    ax2.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Confidence calibration comparison
# ---------------------------------------------------------------------------


def plot_calibration_comparison(
    calibration_dict: dict[str, dict],
    title: str = "Confidence Calibration: Quintile Returns by Scheme",
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay quintile mean returns for all schemes on a grouped bar chart.

    Reveals whether calibration (monotonic quintile ordering) is consistent
    across allocation schemes, or whether some schemes disrupt rank ordering.

    Args:
        calibration_dict: scheme_label → calibration_data dict (from
            _prepare_allocation_research_diagnostics).
    """
    _t = get_typography()
    valid = {k: v for k, v in calibration_dict.items()
             if isinstance(v, dict) and v.get("quintile_returns") is not None}

    if not valid:
        fig, ax = make_figure()
        ax.text(0.5, 0.5, "No calibration data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    labels = list(valid.keys())
    colors = _scheme_colors(labels)
    n_schemes = len(labels)
    qr_ref = next(iter(valid.values()))["quintile_returns"]
    n_q = len(qr_ref)
    x = np.arange(n_q)
    width = 0.8 / n_schemes

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD))
    ax.tick_params(labelsize=_t.tick)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    for i, (lbl, color) in enumerate(zip(labels, colors, strict=False)):
        qr = valid[lbl]["quintile_returns"]
        offset = (i - n_schemes / 2 + 0.5) * width
        ax.bar(x + offset, qr.values, width=width * 0.9, color=color,
               alpha=0.8, label=lbl)

    ax.axhline(0, color=COLORS["neutral"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_q)], fontsize=_t.tick)
    ax.set_ylabel("Mean realized forward return", fontsize=_t.small_label)
    ax.legend(frameon=False, fontsize=_t.legend)
    label_axes(ax, title=title)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig
