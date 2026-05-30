"""Publication-quality allocation research visualization.

Supports ML-aware portfolio construction research: concentration dynamics,
prediction dispersion, and confidence calibration diagnostics.

All functions accept pre-computed DataFrames/dicts and return matplotlib Figures.
Read-only: no data loading, no model fitting, no recomputation.

Figures in this module are diagnostics-first — they reveal whether confidence
signals carry economic information and how allocation concentration evolves
through time. They are not performance attribution charts.
"""

from __future__ import annotations

import math as _math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.styles import (
    COLORS,
    FIG_HEIGHT_STANDARD,
    FIG_HEIGHT_TALL,
    FIG_WIDTH_FULL,
    label_axes,
    make_figure,
)
from src.visualization.typography import get_typography
from src.visualization.utils import save_figure

# ---------------------------------------------------------------------------
# Allocation concentration evolution
# ---------------------------------------------------------------------------


def plot_concentration_evolution(
    weights: pd.DataFrame,
    window: int = 63,
    title: str = "Allocation Concentration Through Time",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling HHI, effective breadth, and effective-N entropy through time.

    Three-panel figure revealing concentration dynamics across the backtest:
      Panel 1 — rolling Herfindahl-Hirschman Index (HHI = Σwᵢ²). Higher values
                indicate more concentrated allocation; equal-weight across k assets
                yields HHI = 1/k.
      Panel 2 — rolling effective breadth (1/HHI): how many equally-effective
                bets the allocation implies.
      Panel 3 — rolling entropy-based effective N = exp(H) where H = -Σwᵢln(wᵢ).
                N* = 1 indicates full concentration; N* = k is uniform.

    Args:
        weights:   Date × Asset weight DataFrame.
        window:    Rolling window in trading days (default 63 ≈ one quarter).
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    w = weights.fillna(0.0)
    abs_w = w.abs()
    active_mask = (abs_w > 1e-10).any(axis=1)

    if not active_mask.any():
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No active allocation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    # ── Compute concentration series ──
    hhi = (abs_w ** 2).sum(axis=1)
    eff_breadth = (1.0 / hhi.replace(0.0, float("nan"))).fillna(0.0)

    def _row_effective_n(row: pd.Series) -> float:
        pos = row[row > 1e-10]
        if pos.empty:
            return float("nan")
        total = pos.sum()
        if total <= 0:
            return float("nan")
        p = pos / total
        h = float(-(p * p.apply(_math.log)).sum())
        return float(_math.exp(h))

    eff_n = abs_w.apply(_row_effective_n, axis=1)

    min_periods = max(1, window // 4)
    hhi_roll = hhi.rolling(window, min_periods=min_periods).mean()
    breadth_roll = eff_breadth.rolling(window, min_periods=min_periods).mean()
    eff_n_roll = eff_n.rolling(window, min_periods=min_periods).mean()

    # ── Layout ──
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_TALL * 1.3),
                              sharex=True)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=_t.tick)

    # Panel 1: HHI
    ax1 = axes[0]
    ax1.fill_between(hhi_roll.index, hhi_roll.values,
                     color=COLORS["negative"], alpha=0.25)
    ax1.plot(hhi_roll.index, hhi_roll.values,
             color=COLORS["negative"], linewidth=1.2)
    mean_hhi = float(hhi_roll.dropna().mean())
    ax1.axhline(mean_hhi, color=COLORS["neutral"], linewidth=0.9, linestyle="--",
                label=f"Mean HHI = {mean_hhi:.3f}")
    ax1.set_ylabel(f"HHI ({window}d roll)", fontsize=_t.small_label)
    ax1.legend(frameon=False, fontsize=_t.legend)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left")

    # Panel 2: Effective breadth
    ax2 = axes[1]
    ax2.fill_between(breadth_roll.index, breadth_roll.values,
                     color=COLORS["strategy"], alpha=0.25)
    ax2.plot(breadth_roll.index, breadth_roll.values,
             color=COLORS["strategy"], linewidth=1.2)
    mean_breadth = float(breadth_roll.dropna().mean())
    ax2.axhline(mean_breadth, color=COLORS["neutral"], linewidth=0.9, linestyle="--",
                label=f"Mean breadth = {mean_breadth:.2f}")
    ax2.set_ylabel(f"Effective breadth ({window}d)", fontsize=_t.small_label)
    ax2.legend(frameon=False, fontsize=_t.legend)

    # Panel 3: Effective N (entropy-based)
    ax3 = axes[2]
    ax3.fill_between(eff_n_roll.dropna().index, eff_n_roll.dropna().values,
                     color=COLORS["positive"], alpha=0.2)
    ax3.plot(eff_n_roll.dropna().index, eff_n_roll.dropna().values,
             color=COLORS["positive"], linewidth=1.2)
    mean_eff_n = float(eff_n_roll.dropna().mean())
    ax3.axhline(mean_eff_n, color=COLORS["neutral"], linewidth=0.9, linestyle="--",
                label=f"Mean eff-N = {mean_eff_n:.2f}")
    ax3.set_ylabel(f"Effective N (entropy, {window}d)", fontsize=_t.small_label)
    ax3.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Prediction dispersion
# ---------------------------------------------------------------------------


def plot_prediction_dispersion(
    score_wide: pd.DataFrame,
    window: int = 63,
    stress_mask: pd.Series | None = None,
    title: str = "Cross-Sectional Prediction Dispersion",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling cross-sectional prediction spread and top-minus-bottom separation.

    Two-panel figure revealing prediction dispersion dynamics:
      Panel 1 — rolling cross-sectional standard deviation of raw prediction scores.
                Low std indicates score compression — the model assigns similar scores
                across all assets, reducing effective ranking information.
      Panel 2 — rolling top-minus-bottom score spread (max − min per row).
                Near-zero values indicate ranking collapse — when scores compress
                toward a single value, allocation becomes effectively arbitrary.

    Both panels use a 63-day rolling mean to reveal structural regimes rather
    than daily noise.

    Args:
        score_wide: Date × Asset raw prediction score DataFrame.
        window:     Rolling window in trading days (default 63).
        stress_mask: Optional boolean Series for stress regime shading.
        title:      Figure title.
        save_path:  Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()
    sw = score_wide.dropna(how="all")

    if sw.empty or len(sw) < 5:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "Insufficient prediction data", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    cs_std = sw.std(axis=1)
    cs_spread = sw.max(axis=1) - sw.min(axis=1)

    min_periods = max(1, window // 4)
    std_roll = cs_std.rolling(window, min_periods=min_periods).mean()
    spread_roll = cs_spread.rolling(window, min_periods=min_periods).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.7),
                                    sharex=True)
    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=_t.tick)

    # Stress regime shading
    if stress_mask is not None:
        _aligned_mask = stress_mask.reindex(sw.index, fill_value=False)
        for ax in (ax1, ax2):
            _prev = False
            _t0 = None
            for dt, is_stress in _aligned_mask.items():
                if is_stress and not _prev:
                    _t0 = dt
                elif not is_stress and _prev and _t0 is not None:
                    ax.axvspan(_t0, dt, color=COLORS["negative"], alpha=0.06)
                    _t0 = None
                _prev = is_stress

    # Panel 1: cross-sectional std
    ax1.fill_between(std_roll.index, std_roll.values,
                     color=COLORS["strategy"], alpha=0.3)
    ax1.plot(std_roll.index, std_roll.values,
             color=COLORS["strategy"], linewidth=1.2)
    mean_std = float(std_roll.dropna().mean())
    ax1.axhline(mean_std, color=COLORS["neutral"], linewidth=0.9, linestyle="--",
                label=f"Mean σ = {mean_std:.4f}")
    ax1.set_ylabel(f"CS prediction σ ({window}d roll)", fontsize=_t.small_label)
    ax1.legend(frameon=False, fontsize=_t.legend)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left")

    # Panel 2: top-minus-bottom spread
    ax2.fill_between(spread_roll.index, spread_roll.values,
                     color=COLORS["positive"], alpha=0.25)
    ax2.plot(spread_roll.index, spread_roll.values,
             color=COLORS["positive"], linewidth=1.2)
    mean_spread = float(spread_roll.dropna().mean())
    ax2.axhline(mean_spread, color=COLORS["neutral"], linewidth=0.9, linestyle="--",
                label=f"Mean spread = {mean_spread:.4f}")
    ax2.set_ylabel(f"Top−Bottom score spread ({window}d)", fontsize=_t.small_label)
    ax2.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


def plot_confidence_calibration(
    calibration_data: dict,
    title: str = "Confidence Calibration: Prediction Score vs Realized Return",
    save_path: str | None = None,
) -> plt.Figure:
    """Mean realized return by prediction-score quintile.

    Reveals whether prediction magnitude contains economic information — a
    monotonically increasing pattern (left-to-right) confirms that stronger
    cross-sectional scores correspond to stronger realized forward returns.

    The bottom panel shows cumulative separation between the top and bottom
    quintile groups through time when monthly time-series data is available.

    Args:
        calibration_data: Dict with keys from _prepare_allocation_research_diagnostics:
            quintile_returns:       pd.Series indexed Q0..Q4 (mean realized return)
            quintile_counts:        pd.Series indexed Q0..Q4 (observation count)
            monotonic_up:           bool — is Q0 < Q1 < ... < Q4?
            top_minus_bottom_spread: float — Q4 mean minus Q0 mean return
            group_monthly:          optional pd.DataFrame (Month × [top, mid, bottom])
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    _t = get_typography()

    qr = calibration_data.get("quintile_returns")
    qc = calibration_data.get("quintile_counts")
    monotonic = calibration_data.get("monotonic_up", False)
    spread = calibration_data.get("top_minus_bottom_spread", float("nan"))
    group_monthly = calibration_data.get("group_monthly")  # optional

    if qr is None or len(qr) < 2:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "Insufficient calibration data", ha="center", va="center",
                transform=ax.transAxes, fontsize=_t.figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    has_monthly = (
        group_monthly is not None
        and isinstance(group_monthly, pd.DataFrame)
        and not group_monthly.empty
        and len(group_monthly) >= 6
    )

    if has_monthly:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD))

    # ── Panel 1: Bar chart of mean return by quintile ──
    n_q = len(qr)
    x = np.arange(n_q)

    # Colour gradient: red (low score) → green (high score)
    _neg = np.array([0.80, 0.20, 0.20])  # approximate COLORS["negative"] RGB
    _pos = np.array([0.16, 0.58, 0.27])  # approximate COLORS["positive"] RGB
    bar_colors = [
        tuple((_neg * (1 - t) + _pos * t).tolist())
        for t in np.linspace(0, 1, n_q)
    ]
    ret_vals = qr.values
    bars = ax1.bar(x, ret_vals, color=bar_colors, alpha=0.85, width=0.6, edgecolor="white",
                   linewidth=0.5)

    for bar, val in zip(bars, ret_vals, strict=False):
        sign = 1 if val >= 0 else -1
        offset = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.015 if ax1.get_ylim()[0] != ax1.get_ylim()[1] else 0.002
        ax1.text(bar.get_x() + bar.get_width() / 2, val + sign * abs(offset),
                 f"{val:.4f}", ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=_t.small_annotation)

    ax1.axhline(0, color=COLORS["grid"], linewidth=0.8)
    ax1.set_xticks(x)
    xlabels = [f"Q{i + 1}\n(n={int(qc.iloc[i]) if qc is not None else ''})"
               for i in range(n_q)]
    ax1.set_xticklabels(xlabels, fontsize=_t.tick)
    ax1.set_ylabel("Mean realized forward return", fontsize=_t.small_label)
    ax1.tick_params(labelsize=_t.tick)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Monotonicity annotation
    if monotonic:
        verdict = f"Monotonically increasing — spread = {spread:.4f}"
        verdict_color = COLORS["positive"]
    elif not _math.isnan(spread):
        verdict = f"Non-monotonic — spread = {spread:.4f}"
        verdict_color = COLORS["negative"]
    else:
        verdict = "Insufficient data"
        verdict_color = COLORS["neutral"]

    ax1.annotate(
        verdict,
        xy=(0.02, 0.96), xycoords="axes fraction",
        ha="left", va="top",
        fontsize=_t.small_annotation,
        color=verdict_color,
        fontweight="semibold",
    )
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left")

    # ── Panel 2: Cumulative top vs bottom group evolution ──
    if has_monthly:
        gm = group_monthly
        top_cum = (1 + gm.get("top", pd.Series())).cumprod()
        bot_cum = (1 + gm.get("bottom", pd.Series())).cumprod()

        ax2.fill_between(top_cum.index, top_cum.values,
                         color=COLORS["positive"], alpha=0.2, label="Top quintile")
        ax2.plot(top_cum.index, top_cum.values,
                 color=COLORS["positive"], linewidth=1.2, label="Top")
        ax2.fill_between(bot_cum.index, bot_cum.values,
                         color=COLORS["negative"], alpha=0.2)
        ax2.plot(bot_cum.index, bot_cum.values,
                 color=COLORS["negative"], linewidth=1.2, label="Bottom")
        ax2.axhline(1.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
        ax2.set_ylabel("Cumulative return (monthly, gross)", fontsize=_t.small_label)
        ax2.tick_params(labelsize=_t.tick)
        ax2.legend(frameon=False, fontsize=_t.legend)
        ax2.grid(axis="y", alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig
