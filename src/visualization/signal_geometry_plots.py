"""Signal geometry comparative visualizations.

Supports Phase 3A: controlled comparison of prediction geometry and confidence
structure across Ridge regularization strengths (α sweep) on the 15-ETF universe.

All functions accept pre-computed diagnostics dicts loaded from experiment
artefacts — no data loading, no model fitting, no recomputation.

Figures expose geometry, calibration, robustness, and instability diagnostics,
not performance dashboards.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_HEIGHT_STANDARD,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography
from src.visualization.utils import save_figure

# Ordered α palette: darker = more regularised (α=0.5), lighter = less (α=0.01)
_ALPHA_PALETTE = {
    "α=0.50": "#1f4e79",   # deep navy — baseline, strong regularization
    "α=0.10": "#2980b9",   # medium blue
    "α=0.05": "#e67e22",   # amber — boundary zone
    "α=0.01": "#c0392b",   # red — minimal regularization / risk zone
}
_DEFAULT_PALETTE = ["#1f4e79", "#2980b9", "#e67e22", "#c0392b",
                    "#27ae60", "#8e44ad"]


def _alpha_color(label: str, idx: int = 0) -> str:
    return _ALPHA_PALETTE.get(label, _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)])


def _alpha_colors(labels: list[str]) -> list[str]:
    return [_alpha_color(lbl, i) for i, lbl in enumerate(labels)]


# ---------------------------------------------------------------------------
# 1. Prediction dispersion sweep (bar)
# ---------------------------------------------------------------------------


def plot_dispersion_sweep(
    dispersion_by_alpha: dict[str, dict],
    title: str = "Prediction Geometry by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart comparing mean CS std and top-minus-bottom spread across α values.

    Args:
        dispersion_by_alpha: {alpha_label: {"mean_cs_std": float,
                                             "mean_cs_spread": float,
                                             "min_cs_std": float,
                                             "max_cs_std": float}}
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not dispersion_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No dispersion data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    labels = list(dispersion_by_alpha.keys())
    colors = _alpha_colors(labels)
    cs_std = [dispersion_by_alpha[k].get("mean_cs_std", float("nan")) for k in labels]
    cs_spread = [dispersion_by_alpha[k].get("mean_cs_spread", float("nan")) for k in labels]

    fig, axes = make_figure(ncols=2, height=FIG_HEIGHT_STANDARD)
    ax_std, ax_spread = axes

    x = np.arange(len(labels))
    bars_std = ax_std.bar(x, cs_std, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars_std, cs_std, strict=False):
        if not math.isnan(val):
            ax_std.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cs_std, default=0) * 0.02,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=_t.annotation)

    bars_spread = ax_spread.bar(x, cs_spread, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars_spread, cs_spread, strict=False):
        if not math.isnan(val):
            ax_spread.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + max(cs_spread, default=0) * 0.02,
                           f"{val:.4f}", ha="center", va="bottom", fontsize=_t.annotation)

    for ax, metric_label in [(ax_std, "Mean Cross-Sectional Prediction σ"),
                              (ax_spread, "Mean Top-Minus-Bottom Spread")]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=_t.tick)
        ax.tick_params(axis="y", labelsize=_t.tick)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        label_axes(ax, "", "", metric_label)

    fig.suptitle(title, fontsize=_t.small_label, fontweight="bold", y=1.01)
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Prediction dispersion evolution (time-series overlay)
# ---------------------------------------------------------------------------


def plot_dispersion_evolution(
    cs_std_by_alpha: dict[str, pd.Series],
    window: int = 63,
    title: str = "Prediction Dispersion Evolution by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling cross-sectional prediction σ time-series, one line per α.

    Args:
        cs_std_by_alpha: {alpha_label: daily CS std Series}
        window: Rolling window for smoothing.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not cs_std_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No dispersion data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    labels = list(cs_std_by_alpha.keys())
    colors = _alpha_colors(labels)

    for lbl, color in zip(labels, colors, strict=False):
        series = cs_std_by_alpha[lbl]
        if series is None or series.empty:
            continue
        rolled = series.rolling(window, min_periods=max(1, window // 4)).mean()
        lw = 1.8 if "0.50" in lbl else 1.2
        ax.plot(rolled.index, rolled.values, color=color, linewidth=lw, label=lbl, alpha=0.9)

    ax.set_ylabel("Cross-Sectional Prediction σ (rolling)", fontsize=_t.small_label)
    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    label_axes(ax, title, "", "")

    ax.annotate(f"{window}d rolling mean", xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=_t.annotation, color=COLORS["neutral"])

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Calibration sweep (grouped quintile bars)
# ---------------------------------------------------------------------------


def plot_calibration_sweep(
    calibration_by_alpha: dict[str, dict],
    title: str = "Confidence Calibration by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped quintile bar chart showing calibration quality across α values.

    Args:
        calibration_by_alpha: {alpha_label: {"quintile_returns": pd.Series (index Q1..Q5),
                                              "monotonic_up": bool,
                                              "top_minus_bottom_spread": float}}
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not calibration_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No calibration data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    labels = list(calibration_by_alpha.keys())
    colors = _alpha_colors(labels)

    # Determine quintile labels from first available entry
    q_labels: list[str] = []
    for d in calibration_by_alpha.values():
        qr = d.get("quintile_returns")
        if qr is None:
            continue
        if isinstance(qr, dict):
            qr = pd.Series(qr)
        if not qr.empty:
            q_labels = list(qr.index)
            break
    if not q_labels:
        q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    n_q = len(q_labels)
    n_alpha = len(labels)
    total_width = 0.75
    bar_width = total_width / n_alpha
    x = np.arange(n_q)

    for i, (lbl, color) in enumerate(zip(labels, colors, strict=False)):
        d = calibration_by_alpha[lbl]
        qr = d.get("quintile_returns")
        if qr is None:
            continue
        if isinstance(qr, dict):
            qr = pd.Series(qr)
        vals = [float(qr.get(q, float("nan"))) for q in q_labels]
        offset = (i - n_alpha / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, width=bar_width * 0.9, color=color, alpha=0.8,
                      edgecolor="white", linewidth=0.4, label=lbl)

    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=_t.tick)
    ax.tick_params(axis="y", labelsize=_t.tick)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    label_axes(ax, title, "Prediction Quintile", "Mean 21-day Forward Return")

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Walk-forward stability heatmap (split × α)
# ---------------------------------------------------------------------------


def plot_wf_stability_heatmap(
    split_sharpe_by_alpha: dict[str, list[float]],
    split_labels: list[str] | None = None,
    title: str = "Walk-Forward Sharpe Stability by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap: rows = WF splits, columns = α values, cells = split Sharpe.

    Args:
        split_sharpe_by_alpha: {alpha_label: [sharpe_split_0, sharpe_split_1, ...]}
        split_labels: Optional list of split period labels.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not split_sharpe_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No walk-forward data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    alpha_labels = list(split_sharpe_by_alpha.keys())
    n_splits = max(len(v) for v in split_sharpe_by_alpha.values())
    if split_labels is None:
        split_labels = [f"Split {i+1}" for i in range(n_splits)]

    matrix = np.full((n_splits, len(alpha_labels)), float("nan"))
    for j, lbl in enumerate(alpha_labels):
        vals = split_sharpe_by_alpha[lbl]
        for i, v in enumerate(vals):
            if i < n_splits:
                matrix[i, j] = v

    fig, ax = make_figure(height=max(FIG_HEIGHT_STANDARD, 0.55 * n_splits + 1.2))

    vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -1.0
    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    abs_max = max(abs(vmin), abs(vmax), 0.1)
    cmap = plt.cm.RdYlGn

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax, label="OOS Sharpe", fraction=0.025, pad=0.03)

    ax.set_xticks(np.arange(len(alpha_labels)))
    ax.set_xticklabels(alpha_labels, fontsize=_t.tick)
    ax.set_yticks(np.arange(n_splits))
    ax.set_yticklabels(split_labels[:n_splits], fontsize=_t.tick)

    for i in range(n_splits):
        for j in range(len(alpha_labels)):
            val = matrix[i, j]
            if not math.isnan(val):
                text_color = "white" if abs(val) > abs_max * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=_t.annotation, color=text_color, fontweight="bold")

    ax.set_xlabel("Regularization Strength (α)", fontsize=_t.small_label)
    ax.set_ylabel("Walk-Forward Split", fontsize=_t.small_label)
    label_axes(ax, title, "", "")

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Robustness tradeoff scatter (dispersion vs Sharpe)
# ---------------------------------------------------------------------------


def plot_robustness_tradeoff(
    summary_by_alpha: dict[str, dict],
    title: str = "Signal Geometry vs Walk-Forward Robustness",
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter: mean CS std (x) vs OOS Sharpe (y), bubble = WF Sharpe std.

    Args:
        summary_by_alpha: {alpha_label: {"mean_cs_std": float, "oos_mean_sharpe": float,
                                          "oos_sharpe_std": float, "mean_turnover": float}}
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not summary_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No summary data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    labels = list(summary_by_alpha.keys())
    colors = _alpha_colors(labels)

    for lbl, color in zip(labels, colors, strict=False):
        d = summary_by_alpha[lbl]
        x = d.get("mean_cs_std", float("nan"))
        y = d.get("oos_mean_sharpe", float("nan"))
        std_val = d.get("oos_sharpe_std", None)
        if math.isnan(x) or math.isnan(y):
            continue
        size = max(80, min(600, (std_val or 0.3) * 800)) if std_val else 200
        ax.scatter(x, y, s=size, color=color, alpha=0.85, edgecolors="white", linewidths=1.0,
                   zorder=5, label=lbl)
        ax.annotate(lbl, (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=_t.annotation, color=color)

    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
    ax.tick_params(labelsize=_t.tick)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    label_axes(ax, title, "Mean Cross-Sectional Prediction σ",
               "OOS Mean Sharpe Ratio")

    ax.annotate("Bubble size ∝ WF Sharpe std (instability)",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=_t.annotation, color=COLORS["neutral"])

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Turnover by alpha (rolling overlay)
# ---------------------------------------------------------------------------


def plot_turnover_by_alpha(
    turnover_by_alpha: dict[str, pd.Series],
    window: int = 21,
    title: str = "Portfolio Turnover by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling daily turnover overlay across α values.

    Args:
        turnover_by_alpha: {alpha_label: daily turnover Series}
        window: Rolling window in days.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not turnover_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No turnover data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    labels = list(turnover_by_alpha.keys())
    colors = _alpha_colors(labels)

    for lbl, color in zip(labels, colors, strict=False):
        series = turnover_by_alpha[lbl]
        if series is None or series.empty:
            continue
        rolled = series.rolling(window, min_periods=1).mean()
        lw = 1.8 if "0.50" in lbl else 1.2
        ax.plot(rolled.index, rolled.values, color=color, linewidth=lw, label=lbl, alpha=0.9)

    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    label_axes(ax, title, "", f"Daily Turnover ({window}d rolling mean)")

    ax.annotate("Higher turnover → more transaction cost drag",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=_t.annotation, color=COLORS["neutral"])

    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Intra-basket spread geometry
# ---------------------------------------------------------------------------


def plot_intrabasket_geometry(
    dispersion_by_alpha: dict[str, dict],
    title: str = "Intra-Basket Score Geometry by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """2-panel: mean CS std and estimated intra-basket spread vs α.

    The intra-basket spread is estimated as (top-minus-bottom spread) × (k/N)
    where k=top-5 and N=15 — the typical fraction of range captured by the
    selected basket relative to the full universe.

    Args:
        dispersion_by_alpha: Same dict as plot_dispersion_sweep.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not dispersion_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No dispersion data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    k_over_N = 5 / 15  # top-5 from 15: fraction of full universe spread captured

    labels = list(dispersion_by_alpha.keys())
    colors = _alpha_colors(labels)
    x = np.arange(len(labels))
    cs_std = [dispersion_by_alpha[k].get("mean_cs_std", float("nan")) for k in labels]
    cs_spread = [dispersion_by_alpha[k].get("mean_cs_spread", float("nan")) for k in labels]
    intrabasket = [s * k_over_N if not math.isnan(s) else float("nan") for s in cs_spread]

    fig, axes = make_figure(ncols=2, height=FIG_HEIGHT_STANDARD)
    ax_cs, ax_ib = axes

    for ax, vals, ylabel in [
        (ax_cs, cs_std, "Mean CS Prediction σ"),
        (ax_ib, intrabasket, "Est. Intra-Basket Spread (×k/N)"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        max_val = max((v for v in vals if not math.isnan(v)), default=0)
        for bar, val in zip(bars, vals, strict=False):
            if not math.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max_val * 0.02,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=_t.annotation)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=_t.tick)
        ax.tick_params(axis="y", labelsize=_t.tick)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        label_axes(ax, "", "", ylabel)

    # Softmax activation threshold annotation on intra-basket panel
    ax_ib.axhline(0.01, color=COLORS["negative"], linewidth=1.0, linestyle=":",
                  label="~softmax threshold (1bp)")
    ax_ib.legend(frameon=False, fontsize=_t.annotation, loc="upper left")

    fig.suptitle(title, fontsize=_t.small_label, fontweight="bold", y=1.01)
    if save_path:
        save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Concentration emergence (rolling HHI after softmax re-evaluation)
# ---------------------------------------------------------------------------


def plot_concentration_emergence(
    hhi_by_alpha: dict[str, pd.Series],
    window: int = 63,
    title: str = "Concentration Emergence by Regularization Strength",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling HHI time-series across α values showing whether softmax
    can concentrate weight as α decreases.

    Args:
        hhi_by_alpha: {alpha_label: HHI Series (daily)}
        window: Rolling window for smoothing.
        title: Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    if not hhi_by_alpha:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No HHI data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.annotation, color=COLORS["neutral"])
        label_axes(ax, title, "", "")
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
    labels = list(hhi_by_alpha.keys())
    colors = _alpha_colors(labels)

    for lbl, color in zip(labels, colors, strict=False):
        series = hhi_by_alpha[lbl]
        if series is None or series.empty:
            continue
        rolled = series.rolling(window, min_periods=max(1, window // 4)).mean()
        lw = 1.8 if "0.50" in lbl else 1.2
        ax.plot(rolled.index, rolled.values, color=color, linewidth=lw, label=lbl, alpha=0.9)

    # Equal-weight reference for top-5 (1/5 = 0.2)
    ax.axhline(0.2, color=COLORS["grid"], linewidth=1.0, linestyle="--", label="EW reference (1/k)")
    ax.tick_params(labelsize=_t.tick)
    ax.legend(frameon=False, fontsize=_t.legend, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    label_axes(ax, title, "", f"Mean HHI ({window}d rolling)")

    ax.annotate("Higher HHI = more concentration. EW top-5 = 0.200",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=_t.annotation, color=COLORS["neutral"])

    if save_path:
        save_figure(fig, save_path)
    return fig
