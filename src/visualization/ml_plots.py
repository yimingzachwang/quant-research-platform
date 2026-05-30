"""ML research diagnostic plots.

All functions accept plain pandas objects (Series, DataFrames) produced by
src.ml.diagnostics and return a matplotlib Figure.  Read-only: no data
mutation, no model fitting, no recomputation.

Follows existing visualization conventions from src.visualization.styles:
    - make_figure / label_axes / COLORS from styles.py
    - save_figure from utils.py
    - matplotlib only — no seaborn, no dashboards, no interactive systems

Note: plot_signal_turnover is named distinctly from
src.visualization.portfolio_plots.plot_turnover (which operates on full
portfolio backtest output) to avoid a naming collision on import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.visualization.styles import (
    COLORS,
    FIG_WIDTH_FULL,
    FIG_WIDTH_HALF,
    FIG_HEIGHT_STANDARD,
    format_pct_axis,
    label_axes,
    make_figure,
)
from src.visualization.utils import save_figure
from src.visualization.typography import get_typography, heatmap_cell_fontsize, scale_dynamic_fontsize


# ---------------------------------------------------------------------------
# Presentation-layer feature labels — single source of truth via families
# ---------------------------------------------------------------------------

def _feat_label(name: str) -> str:
    """Return a publication-ready feature label.

    Delegates to src.features.families.generate_feature_label() which is
    the canonical label authority.  Importing inside the function keeps the
    module dependency optional (plots still work if families is unavailable).
    """
    try:
        from src.features.families import generate_feature_label
        return generate_feature_label(name)
    except Exception:
        return name


def _bar_annotation_ylim(
    vals: "np.ndarray",
    padding_frac: float = 0.20,
) -> "tuple[float, float]":
    """Return (ymin, ymax) with symmetric annotation headroom for annotated bar charts.

    Ensures top/bottom annotations cannot clip against axis boundaries.
    Always includes 0 in the range so the zero-line stays visible.

    Args:
        vals:         Array of bar values (may contain NaN).
        padding_frac: Fraction of data span to add on each end.
    """
    valid = vals[~np.isnan(vals)] if hasattr(vals, "__len__") else np.array([])
    lo = float(min(0.0, valid.min())) if len(valid) else -0.1
    hi = float(max(0.0, valid.max())) if len(valid) else 0.1
    span = max(hi - lo, 1e-6)
    return lo - padding_frac * span, hi + padding_frac * span


# ---------------------------------------------------------------------------
# Prediction diagnostics
# ---------------------------------------------------------------------------


def plot_prediction_vs_actual(
    actual: pd.Series,
    predicted: pd.Series,
    title: str = "Prediction vs Actual",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel figure: time-series overlay and scatter plot.

    Top panel: actual (grey) and predicted (amber) over time.
    Bottom panel: scatter of predicted vs actual with a zero-axes cross.

    Args:
        actual:    Observed target series.
        predicted: Model prediction series aligned to the same index.
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    # Align
    df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()

    fig, (ax_ts, ax_sc) = make_figure(nrows=2, height=FIG_HEIGHT_STANDARD * 1.8)

    # — Time series —
    ax_ts.plot(df.index, df["actual"], color=COLORS["benchmark"],
               linewidth=1.1, label="Actual", alpha=0.75)
    ax_ts.plot(df.index, df["predicted"], color=COLORS["signal"],
               linewidth=1.1, label="Predicted", alpha=0.85)
    ax_ts.axhline(0.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
    _t = get_typography()
    label_axes(ax_ts, title=title, ylabel="Value")
    ax_ts.legend(frameon=False, fontsize=_t.legend)

    # — Scatter —
    ax_sc.scatter(df["predicted"], df["actual"],
                  color=COLORS["strategy"], alpha=0.35, s=10, linewidths=0)
    # Zero cross-hairs
    ax_sc.axhline(0.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
    ax_sc.axvline(0.0, color=COLORS["grid"], linewidth=0.7, linestyle="--")
    # 45-degree reference line
    lim = max(abs(df["predicted"].max()), abs(df["actual"].max()),
              abs(df["predicted"].min()), abs(df["actual"].min())) * 1.05
    ax_sc.plot([-lim, lim], [-lim, lim], color=COLORS["neutral"],
               linewidth=0.8, linestyle=":", label="y = x")
    label_axes(ax_sc, xlabel="Predicted", ylabel="Actual")
    ax_sc.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_prediction_distribution(
    predictions: pd.Series,
    bins: int = 40,
    title: str = "Prediction Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of model predictions.

    Includes a vertical line at zero and mean/median annotations.

    Args:
        predictions: Series of model predictions.
        bins:        Number of histogram bins.  Default 40.
        title:       Figure title.
        save_path:   Optional save path.

    Returns:
        matplotlib Figure.
    """
    preds = predictions.dropna()

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    ax.hist(preds.to_numpy(), bins=bins, color=COLORS["strategy"],
            alpha=0.75, edgecolor="white", linewidth=0.4)

    mean_val = float(preds.mean())
    median_val = float(preds.median())

    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--", label="zero")
    ax.axvline(mean_val, color=COLORS["signal"], linewidth=1.2, linestyle="-",
               label=f"mean={mean_val:.4f}")
    ax.axvline(median_val, color=COLORS["neutral"], linewidth=1.2, linestyle="-.",
               label=f"median={median_val:.4f}")

    label_axes(ax, title=title, xlabel="Prediction", ylabel="Count")
    ax.legend(frameon=False, fontsize=get_typography().legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_information_coefficient(
    ic_series: pd.Series,
    rolling_window: int = 6,
    title: str = "Information Coefficient (IC)",
    save_path: str | None = None,
) -> plt.Figure:
    """Time-series plot of cross-sectional IC with rolling mean overlay.

    Bars show per-date IC values; line shows rolling mean IC.
    Coloured bars: positive IC = strategy colour, negative = negative colour.

    Args:
        ic_series:      Date-indexed IC series from information_coefficient().
        rolling_window: Window for the rolling mean line.  Default 21.
        title:          Figure title.
        save_path:      Optional save path.

    Returns:
        matplotlib Figure.
    """
    ic = ic_series.dropna()

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    colors_bar = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                  for v in ic.to_numpy()]
    ax.bar(ic.index, ic.to_numpy(), color=colors_bar, alpha=0.55, width=1.5)

    if len(ic) >= rolling_window:
        rolling_mean = ic.rolling(rolling_window, min_periods=rolling_window).mean()
        ax.plot(rolling_mean.index, rolling_mean.to_numpy(),
                color=COLORS["strategy"], linewidth=1.5,
                label=f"{rolling_window}-mo rolling IC")

    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")

    mean_ic = float(ic.mean())
    ax.axhline(mean_ic, color=COLORS["signal"], linewidth=1.0, linestyle=":",
               label=f"mean IC={mean_ic:.4f}")

    label_axes(ax, title=title, ylabel="IC (Pearson)")
    ax.legend(frameon=False, fontsize=get_typography().legend)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# IC regime
# ---------------------------------------------------------------------------


_STRESS_MIN_DAYS = 21   # discard stress runs shorter than this many trading days
_STRESS_GAP_DAYS = 10   # merge two runs separated by a gap ≤ this many trading days


def _shade_stress_regimes(
    ax: "plt.Axes",
    dates: pd.DatetimeIndex,
    stress_mask: "pd.Series | None",
) -> None:
    """Shade high-stress periods on ax as restrained light-grey bands.

    Applies minimum-duration and gap-merge filtering so the overlay stays
    visually coherent rather than fragmenting into dozens of narrow bands.

    Args:
        ax:           Axes to shade.
        dates:        Full date index of the plotted series.
        stress_mask:  Boolean Series aligned to ``dates``.  True = stress period.
    """
    if stress_mask is None or stress_mask.empty:
        return
    try:
        mask = stress_mask.reindex(dates).fillna(False)
        vals = mask.to_numpy()
        idx = mask.index

        # Pass 1 — collect raw contiguous stress runs as (start_i, end_i) pairs
        runs: list[list[int]] = []
        in_run = False
        for i, v in enumerate(vals):
            if v and not in_run:
                in_run = True
                runs.append([i, i])
            elif v and in_run:
                runs[-1][1] = i
            else:
                in_run = False

        # Pass 2 — merge adjacent runs whose gap is ≤ _STRESS_GAP_DAYS
        merged: list[list[int]] = []
        for run in runs:
            if merged and run[0] - merged[-1][1] <= _STRESS_GAP_DAYS:
                merged[-1][1] = run[1]
            else:
                merged.append(list(run))

        # Pass 3 — filter out runs shorter than _STRESS_MIN_DAYS
        final = [(s, e) for s, e in merged if e - s + 1 >= _STRESS_MIN_DAYS]

        for s, e in final:
            ax.axvspan(idx[s], idx[e], color="#888888", alpha=0.07, linewidth=0,
                       label="_nolegend_")
    except Exception:
        pass


def plot_ic_regime(
    ic_daily: pd.Series,
    title: str = "Rolling IC Through Time",
    save_path: str | None = None,
    stress_mask: "pd.Series | None" = None,
) -> plt.Figure:
    """Continuous 63-day rolling IC for regime visibility.

    Shows when the model's directional accuracy was persistently positive
    (green fill) or persistently negative (red fill).  Reveals sustained
    periods of model effectiveness vs breakdown across market regimes.

    Args:
        ic_daily:    Daily-granularity rolling Pearson IC series.
        title:       Figure title.
        save_path:   Optional save path.
        stress_mask: Optional boolean Series; True = high-volatility stress period.
                     Shaded as light grey bands on the plot.

    Returns:
        matplotlib Figure.
    """
    ic = ic_daily.dropna()
    if ic.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No IC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD, width=FIG_WIDTH_FULL)

    # Regime shading behind the IC line
    _shade_stress_regimes(ax, ic.index, stress_mask)

    vals = ic.to_numpy()
    ax.plot(ic.index, vals, color=COLORS["strategy"], linewidth=1.2, alpha=0.9)
    ax.fill_between(ic.index, 0.0, vals,
                    where=vals >= 0, color=COLORS["positive"], alpha=0.22)
    ax.fill_between(ic.index, 0.0, vals,
                    where=vals < 0, color=COLORS["negative"], alpha=0.22)
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")

    mean_ic = float(ic.mean())
    ax.axhline(mean_ic, color=COLORS["signal"], linewidth=1.0, linestyle=":",
               label=f"mean IC = {mean_ic:.4f}")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#888888", alpha=0.35, linewidth=0,
                         label="high-vol stress"))
    ax.legend(handles=handles, frameon=False, fontsize=get_typography().legend)

    label_axes(ax, title=title, ylabel="IC (63d rolling mean)")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_rolling_directional_accuracy(
    da_series: pd.Series,
    title: str = "Rolling Directional Accuracy (126d)",
    save_path: str | None = None,
    stress_mask: "pd.Series | None" = None,
    is_panel: bool = False,
) -> plt.Figure:
    """126-day rolling directional accuracy / IC consistency through time.

    Fill above 50% is green (the model called direction correctly more often
    than not); fill below 50% is red (active signal degradation).
    The 50% dashed line is the random-chance baseline.

    For cross-sectional panel mode the series represents the fraction of
    positive-IC days in the rolling window (same 50% null hypothesis).

    Args:
        da_series:   Date-indexed rolling rate in [0, 1].
        title:       Figure title.
        save_path:   Optional save path.
        stress_mask: Optional boolean Series; True = high-volatility stress.
                     Shaded as light grey bands.
        is_panel:    True for cross-sectional panel mode — changes y-axis label
                     from "Directional Accuracy" to "IC Consistency".

    Returns:
        matplotlib Figure.
    """
    da = da_series.dropna()
    if da.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No directional accuracy data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD, width=FIG_WIDTH_FULL)

    _shade_stress_regimes(ax, da.index, stress_mask)

    vals = da.to_numpy()
    ax.plot(da.index, vals, color=COLORS["strategy"], linewidth=1.2, alpha=0.9)
    ax.fill_between(da.index, 0.5, vals,
                    where=vals >= 0.5, color=COLORS["positive"], alpha=0.22)
    ax.fill_between(da.index, 0.5, vals,
                    where=vals < 0.5, color=COLORS["negative"], alpha=0.22)

    ax.axhline(0.5, color=COLORS["grid"], linewidth=0.9, linestyle="--",
               label="50% (random baseline)")
    mean_da = float(da.mean())
    ax.axhline(mean_da, color=COLORS["signal"], linewidth=1.0, linestyle=":",
               label=f"mean = {mean_da:.1%}")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#888888", alpha=0.35, linewidth=0,
                         label="high-vol stress"))
    ax.legend(handles=handles, frameon=False, fontsize=get_typography().legend)
    ylabel = "IC Consistency (126d rolling)" if is_panel else "Directional Accuracy (126d rolling)"
    label_axes(ax, title=title, ylabel=ylabel)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Stability diagnostics
# ---------------------------------------------------------------------------


def plot_split_metric_stability(
    metric_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of a chosen metric across walk-forward splits.

    Bars are coloured by sign: positive = strategy colour, negative = negative.
    A horizontal dashed line marks zero.

    Args:
        metric_df: DataFrame from split_metric_table() or split_metrics_table().
                   Index is split index; metric columns are floats.
        metric:    Column name to plot.  Default "sharpe_ratio".
        title:     Figure title.  Defaults to the metric name.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If metric column not found in metric_df.
    """
    if metric not in metric_df.columns:
        raise ValueError(
            f"metric {metric!r} not found in DataFrame columns: "
            f"{list(metric_df.columns)}"
        )

    vals = metric_df[metric].dropna()
    if title is None:
        title = f"Split Stability: {metric}"

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    vals_np = vals.to_numpy(dtype=float)
    bar_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in vals_np]
    bars = ax.bar(
        vals.index.astype(str), vals_np,
        color=bar_colors, alpha=0.75, edgecolor="white", linewidth=0.4,
    )

    _t = get_typography()
    for bar, v in zip(bars, vals_np):
        if v >= 0:
            y, va = v + 0.025, "bottom"
        else:
            y, va = v + 0.18, "bottom"  # inside bar, clear of lower axis region
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{v:.2f}",
            ha="center",
            va=va,
            fontsize=_t.small_annotation,
            color="#333333",
        )

    # Annotation headroom — prevents labels from clipping axis boundary on both ends
    _ylo, _yhi = _bar_annotation_ylim(vals_np, padding_frac=0.22)
    ax.set_ylim(_ylo, _yhi)

    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")

    mean_val = float(vals.mean())
    ax.axhline(mean_val, color=COLORS["signal"], linewidth=1.0, linestyle=":",
               label=f"mean={mean_val:.2f}")

    label_axes(ax, title=title, xlabel="Split", ylabel=metric)
    ax.legend(frameon=False, fontsize=_t.legend)
    ax.tick_params(axis="x", labelsize=_t.tick)
    ax.tick_params(axis="y", labelsize=_t.tick)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_coefficient_stability(
    coeff_df: pd.DataFrame,
    top_n: int | None = None,
    title: str = "Coefficient Stability",
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of linear model coefficients across splits.

    Each feature shows mean ± 1 std as error bars.  Bars are coloured by
    sign of the mean: positive = positive colour, negative = negative colour.

    Args:
        coeff_df: DataFrame from coefficient_stability() with columns
                  mean, std, sign_consistency, min, max; index = feature names.
        top_n:    If set, shows only the top_n features by |mean|.
        title:    Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.

    Raises:
        ValueError: If coeff_df is missing required columns.
    """
    required = {"mean", "std"}
    missing = required - set(coeff_df.columns)
    if missing:
        raise ValueError(f"coeff_df missing required columns: {missing}")

    df = coeff_df.copy()
    if top_n is not None:
        df = df.reindex(df["mean"].abs().nlargest(top_n).index)

    # Sort by mean for visual clarity
    df = df.sort_values("mean")

    fig, ax = make_figure(height=max(FIG_HEIGHT_STANDARD, 0.35 * len(df) + 1.0),
                          width=FIG_WIDTH_FULL)

    y_pos = np.arange(len(df))
    means = df["mean"].to_numpy(dtype=float)
    stds = df["std"].fillna(0.0).to_numpy(dtype=float)

    bar_colors = [COLORS["positive"] if m >= 0 else COLORS["negative"] for m in means]

    ax.barh(y_pos, means, xerr=stds, color=bar_colors, alpha=0.75,
            error_kw={"elinewidth": 1.0, "ecolor": COLORS["neutral"], "capsize": 3},
            edgecolor="white", linewidth=0.4)

    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_feat_label(n) for n in df.index.tolist()], fontsize=get_typography().tick)
    label_axes(ax, title=title, xlabel="Coefficient (mean ± std)")

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Feature correlation
# ---------------------------------------------------------------------------


def plot_feature_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    feature_families: "dict[str, list[str]] | None" = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Pairwise Pearson correlation heatmap for a feature matrix.

    Intended for the research diagnostics workflow: visualises multicollinearity
    and redundant information content across engineered features.

    When feature_families is provided, features are reordered by family and
    separator lines are drawn between family groups.

    Args:
        corr_df:         Feature × Feature correlation DataFrame from X.corr().
        title:           Figure title.
        feature_families: Optional dict of family_name → [feature_names].
                          When provided, features are reordered to group families
                          together and family separator lines are drawn.
        save_path:       Optional save path.

    Returns:
        matplotlib Figure.
    """
    import math as _math

    if corr_df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    # --- Family-ordered reindexing ---
    ordered_cols = list(corr_df.columns)
    family_boundaries: list[int] = []
    family_labels_pos: list[tuple[float, str]] = []

    if feature_families:
        reordered: list[str] = []
        for fam, members in feature_families.items():
            present = [m for m in members if m in corr_df.columns]
            if present:
                if reordered:
                    family_boundaries.append(len(reordered))
                mid = len(reordered) + len(present) / 2 - 0.5
                family_labels_pos.append((mid, fam))
                reordered.extend(present)
        # Append any features not in any family
        remaining = [c for c in corr_df.columns if c not in reordered]
        if remaining:
            if reordered:
                family_boundaries.append(len(reordered))
            reordered.extend(remaining)
        if reordered:
            ordered_cols = reordered
            corr_df = corr_df.reindex(index=ordered_cols, columns=ordered_cols)

    n = len(corr_df)
    cell = max(0.5, min(1.0, 8.0 / n))
    size = max(FIG_HEIGHT_STANDARD, cell * n + 1.5)

    fig, ax = make_figure(width=size + 0.8, height=size)

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    vals = corr_df.values.copy().astype(float)
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto")

    _t = get_typography()
    font_size = heatmap_cell_fontsize(n, n)
    thresh = 0.6
    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            if not _math.isnan(v):
                text_color = "white" if abs(v) > thresh else "#333333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=font_size, color=text_color)

    # --- Family separator lines ---
    for boundary in family_boundaries:
        pos = boundary - 0.5
        ax.axhline(pos, color="white", linewidth=2.0, alpha=0.9)
        ax.axvline(pos, color="white", linewidth=2.0, alpha=0.9)

    tick_fs = scale_dynamic_fontsize(max(6, 9 - n // 6), "tick")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_feat_label(c) for c in corr_df.columns],
                       rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels([_feat_label(c) for c in corr_df.index], fontsize=tick_fs)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=_t.colorbar)
    cbar.set_label("Pearson r", fontsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()

    # --- Family annotations: placed AFTER tight_layout to avoid collision ---
    # tight_layout computes the minimum left margin for tick labels.  We then
    # expand it further and use fig.text() to place family labels in the new
    # free space — guaranteed clear of the tick-label area.
    if family_labels_pos:
        try:
            from src.features.families import FEATURE_FAMILY_COLORS
            fam_fs = scale_dynamic_fontsize(max(6, 8 - n // 10), "small_annotation")
            max_chars = max(len(fam) for _, fam in family_labels_pos)
            # Estimate inches needed for the widest family label + gap
            extra_in = max_chars * fam_fs * 0.60 / 72.0 + 0.12
            extra_frac = extra_in / fig.get_figwidth()
            base_left = fig.subplotpars.left
            fig.subplots_adjust(left=min(0.50, base_left + extra_frac))
            inv_fig = fig.transFigure.inverted()
            for pos, fam in family_labels_pos:
                color = FEATURE_FAMILY_COLORS.get(fam, "#666666")
                # Convert data-y to figure-y after layout is finalised
                _, y_fig = inv_fig.transform(ax.transData.transform((0, pos)))
                # Centre horizontally inside the newly created extra margin
                x_fig = extra_frac / 2.0
                fig.text(x_fig, y_fig, fam,
                         fontsize=fam_fs, color=color, fontweight="bold",
                         ha="center", va="center")
        except Exception:
            pass

    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------


def plot_residuals(
    actual: pd.Series,
    predicted: pd.Series,
    rolling_window: int = 63,
    title: str = "Residual Diagnostics",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel residual diagnostic figure.

    Top panel: histogram of residuals (actual − predicted) with mean and zero lines.
    Bottom panel: rolling residual mean — exposes systematic bias drift through time.
    A rolling mean persistently above or below zero indicates the model is
    systematically over- or under-predicting in that market regime.

    Args:
        actual:         Observed target series.
        predicted:      Model prediction series aligned to the same index.
        rolling_window: Window for the rolling residual mean.  Default 63 (3 months).
        title:          Figure title.
        save_path:      Optional save path.

    Returns:
        matplotlib Figure.
    """
    df = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    residuals = df["actual"] - df["predicted"]

    fig, (ax_hist, ax_roll) = make_figure(nrows=2, height=FIG_HEIGHT_STANDARD * 1.8)

    # — Residual distribution —
    ax_hist.hist(
        residuals.to_numpy(), bins=40,
        color=COLORS["strategy"], alpha=0.70, edgecolor="white", linewidth=0.4,
    )
    mean_r = float(residuals.mean())
    ax_hist.axvline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--", label="zero")
    ax_hist.axvline(
        mean_r, color=COLORS["signal"], linewidth=1.2, linestyle="-",
        label=f"mean={mean_r:.4f}",
    )
    label_axes(ax_hist, title=title, xlabel="Residual (actual − predicted)", ylabel="Count")
    ax_hist.legend(frameon=False, fontsize=get_typography().legend)

    # — Rolling residual mean —
    rolling_mean = residuals.rolling(rolling_window, min_periods=rolling_window // 2).mean()
    vals = rolling_mean.to_numpy()
    ax_roll.plot(rolling_mean.index, vals, color=COLORS["strategy"], linewidth=1.2)
    ax_roll.fill_between(
        rolling_mean.index, 0.0, vals,
        where=vals > 0, color=COLORS["positive"], alpha=0.25,
    )
    ax_roll.fill_between(
        rolling_mean.index, 0.0, vals,
        where=vals <= 0, color=COLORS["negative"], alpha=0.25,
    )
    ax_roll.axhline(0.0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    label_axes(
        ax_roll,
        ylabel=f"{rolling_window}d rolling residual mean",
    )

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Coefficient evolution
# ---------------------------------------------------------------------------


def plot_coefficient_evolution(
    coeff_df: pd.DataFrame,
    title: str = "Coefficient Evolution Across Splits",
    split_labels: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart of each feature's coefficient trajectory across walk-forward splits.

    Each feature is one line; the x-axis is the split index (chronological order).
    A horizontal dashed line at zero separates positive from negative contributions.
    Legend is placed below the plotting area to avoid overlap with data.

    Args:
        coeff_df:     (n_splits × n_features) DataFrame from _collect_wf_coefficients().
                      Index = split index (int), columns = feature names.
        title:        Figure title.
        split_labels: Optional list of calendar-style x-axis labels (e.g. "2017-01").
                      Falls back to the DataFrame index as strings if None.
        save_path:    Optional save path.

    Returns:
        matplotlib Figure.
    """
    df = coeff_df.dropna(how="all")
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No coefficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_feats = len(df.columns)
    n_splits = len(df)

    # Extra height for external legend below + comfortable typography
    legend_rows = max(1, (n_feats + 3) // 4)
    fig, ax = make_figure(
        height=FIG_HEIGHT_STANDARD + 0.42 * legend_rows + 0.3,
        width=FIG_WIDTH_FULL,
    )

    color_cycle = [
        COLORS.get("strategy", "#2c6fad"),
        COLORS.get("signal", "#d4a017"),
        COLORS.get("positive", "#2e7d32"),
        COLORS.get("negative", "#c62828"),
        COLORS.get("benchmark", "#6e6e6e"),
        COLORS.get("neutral", "#9e9e9e"),
        "#7b4397",
        "#0097a7",
    ]

    x_labels = split_labels if split_labels and len(split_labels) == n_splits \
        else [str(v) for v in df.index]

    for i, col in enumerate(df.columns):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(
            x_labels,
            df[col].to_numpy(dtype=float),
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            color=color,
            label=_feat_label(col),
            alpha=0.88,
        )

    _t = get_typography()
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    label_axes(ax, title=title, xlabel="", ylabel="Coefficient")
    ax.tick_params(axis="x", labelsize=_t.tick)
    ax.tick_params(axis="y", labelsize=_t.tick)
    ax.yaxis.label.set_size(_t.small_label)

    # Legend below axes — no overlap with data region; legible at institutional print scale
    ax.legend(
        frameon=False,
        fontsize=_t.legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, n_feats),
        handlelength=1.6,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    fig.tight_layout()
    # 300 DPI for this publication-quality appendix figure
    fig._save_dpi = 300  # type: ignore[attr-defined]
    if save_path:
        from src.visualization.utils import save_figure as _sf
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------


def plot_signal_turnover(
    turnover: pd.Series,
    title: str = "Signal Turnover",
    save_path: str | None = None,
) -> plt.Figure:
    """Time-series plot of per-period portfolio turnover.

    Includes a horizontal dashed line at mean turnover.

    Named plot_signal_turnover (not plot_turnover) to avoid a naming
    collision with src.visualization.portfolio_plots.plot_turnover, which
    operates on portfolio backtest output.

    Args:
        turnover:  Per-period turnover pd.Series from signal_turnover().
        title:     Figure title.
        save_path: Optional save path.

    Returns:
        matplotlib Figure.
    """
    to = turnover.dropna()

    fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)

    ax.fill_between(to.index, 0.0, to.to_numpy(),
                    color=COLORS["strategy"], alpha=0.35)
    ax.plot(to.index, to.to_numpy(),
            color=COLORS["strategy"], linewidth=1.2)

    mean_to = float(to.mean()) if len(to) > 0 else float("nan")
    if not np.isnan(mean_to):
        ax.axhline(mean_to, color=COLORS["signal"], linewidth=1.0,
                   linestyle="--", label=f"mean={mean_to:.4f}")
        ax.legend(frameon=False, fontsize=get_typography().legend)

    label_axes(ax, title=title, ylabel="Turnover (Σ|Δw|)")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Feature regime heatmap
# ---------------------------------------------------------------------------


def plot_feature_heatmap(
    feature_df: pd.DataFrame,
    title: str = "Feature Regime Behaviour",
    save_path: str | None = None,
) -> plt.Figure:
    """Z-score feature heatmap: features on y-axis, time on x-axis.

    Each column is z-score normalised independently and clipped to ±3σ.
    Reveals when features were in extreme states and how the feature
    environment evolved across regimes — the visual complement to the
    coefficient stability chart.

    Args:
        feature_df: DataFrame with datetime index and feature columns.
        title:      Figure title.
        save_path:  Optional save path.

    Returns:
        matplotlib Figure.
    """
    df = feature_df.dropna(how="all")
    if df.empty or len(df.columns) < 1:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No feature data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    z = (df - df.mean()) / df.std().replace(0, 1)
    z = z.clip(-3, 3)

    n_features = len(z.columns)
    height = max(FIG_HEIGHT_STANDARD, n_features * 0.42 + 1.2)
    fig, ax = make_figure(height=height, width=FIG_WIDTH_FULL)

    Z = z.values.T  # (n_features, n_dates)
    # shading='nearest': C must have shape (len(Y), len(X))
    mesh = ax.pcolormesh(
        z.index, np.arange(n_features), Z,
        cmap="RdYlBu_r", vmin=-3, vmax=3, shading="nearest",
    )

    _t = get_typography()
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([_feat_label(c) for c in z.columns], fontsize=_t.small_annotation)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(False)

    for i in range(n_features):
        ax.axhline(i - 0.5, color="white", linewidth=0.4, alpha=0.5)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, aspect=20, pad=0.02)
    cbar.set_label("z-score (±3σ clipped)", fontsize=_t.colorbar)
    cbar.ax.tick_params(labelsize=_t.colorbar)

    label_axes(ax, title=title, xlabel="", ylabel="")
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Coefficient sign heatmap
# ---------------------------------------------------------------------------


def plot_coefficient_sign_heatmap(
    coeff_df: pd.DataFrame,
    title: str = "Coefficient Sign Stability",
    split_labels: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of model coefficients across walk-forward splits.

    x-axis: split index (chronological).
    y-axis: feature.
    Colour: blue = negative coefficient, red = positive; saturation encodes
    magnitude on a symmetric diverging scale.  Sign reversals across the
    x-axis are the primary diagnostic for regime-specific learning.

    Args:
        coeff_df:     (n_splits × n_features) DataFrame from _collect_wf_coefficients().
                      Index = split index, columns = feature names.
        title:        Figure title.
        split_labels: Optional list of x-axis labels (e.g. test start dates).
                      Falls back to "S{index}" if None.
        save_path:    Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    df = coeff_df.dropna(how="all")
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No coefficient data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_splits = len(df)
    n_features = len(df.columns)
    height = max(FIG_HEIGHT_STANDARD, n_features * 0.55 + 1.2)
    width = max(6.0, n_splits * 1.1 + 2.5)
    fig, ax = make_figure(height=height, width=width)

    Z = df.values.T.astype(float)  # (n_features, n_splits)
    vmax = float(np.nanmax(np.abs(Z)))
    if vmax < 1e-10 or np.isnan(vmax):
        vmax = 1.0

    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]

    im = ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")

    _t = get_typography()
    font_size = heatmap_cell_fontsize(n_features, n_splits)
    for i in range(n_features):
        for j in range(n_splits):
            v = Z[i, j]
            if not np.isnan(v):
                bg_norm = norm(v)
                text_color = "white" if abs(bg_norm - 0.5) > 0.28 else "#333333"
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=font_size, color=text_color)

    x_labels = split_labels if split_labels and len(split_labels) == n_splits \
        else [f"S{idx}" for idx in df.index]
    ax.set_xticks(range(n_splits))
    ax.set_xticklabels(x_labels, fontsize=_t.tick, rotation=30, ha="right")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([_feat_label(c) for c in df.columns], fontsize=_t.tick)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=_t.colorbar)
    cbar.set_label("Coefficient", fontsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Per-feature split IC heatmap
# ---------------------------------------------------------------------------


def plot_feature_ic_heatmap(
    feature_ic_df: pd.DataFrame,
    title: str = "Per-Feature Pearson IC by Walk-Forward Split",
    split_labels: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of per-feature Pearson IC across walk-forward test splits.

    x-axis: split (chronological); y-axis: feature.
    Warm colours (green) = positive IC; cool colours (red) = negative IC.
    Cells are annotated with the IC value.  Reveals which features drove
    predictive power in which regimes.

    Args:
        feature_ic_df: (n_splits × n_features) DataFrame.
                       Index = split index (int), columns = feature names.
        title:         Figure title.
        split_labels:  Optional list of temporal x-axis labels (e.g. "2019-01").
                       Falls back to "S{index}" if None or length mismatch.
        save_path:     Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    df = feature_ic_df.dropna(how="all")
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No per-feature IC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_splits = len(df)
    n_features = len(df.columns)
    height = max(FIG_HEIGHT_STANDARD, n_features * 0.55 + 1.2)
    width = max(6.0, n_splits * 1.1 + 2.5)
    fig, ax = make_figure(height=height, width=width)

    Z = df.values.T.astype(float)  # (n_features, n_splits)

    vabs = float(np.nanmax(np.abs(Z)))
    if vabs < 1e-10 or np.isnan(vabs):
        vabs = 0.1
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]

    im = ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")

    _t = get_typography()
    font_size = heatmap_cell_fontsize(n_features, n_splits)
    for i in range(n_features):
        for j in range(n_splits):
            v = Z[i, j]
            if not np.isnan(v):
                bg_norm = norm(v)
                text_color = "white" if abs(bg_norm - 0.5) > 0.3 else "#333333"
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=font_size, color=text_color)

    x_labels = split_labels if split_labels and len(split_labels) == n_splits \
        else [f"S{idx}" for idx in df.index]
    ax.set_xticks(range(n_splits))
    ax.set_xticklabels(x_labels, fontsize=_t.tick, rotation=30, ha="right")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([_feat_label(c) for c in df.columns], fontsize=_t.tick)
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.ax.tick_params(labelsize=_t.colorbar)
    cbar.set_label("Pearson IC", fontsize=_t.colorbar)

    label_axes(ax, title=title)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_feature_family_ic(
    feature_ic_df: pd.DataFrame,
    feature_families: dict[str, list[str]],
    title: str = "Feature IC by Family",
    split_labels: list[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart of mean per-family Pearson IC across walk-forward splits.

    Shows the average IC across all features within each family, per split.
    Positive IC = family provided predictive signal in that regime.
    Negative IC = family systematically predicted in the wrong direction.

    Args:
        feature_ic_df: (n_splits × n_features) DataFrame from _collect_wf_feature_ic.
        feature_families: Dict of family_name → [feature_name, ...].
                          Only families with features present in feature_ic_df.columns
                          are rendered.  Extra feature names are silently ignored.
        title:           Figure title.
        split_labels:    Optional temporal x-axis labels (e.g. "2019-01").
        save_path:       Optional save path.

    Returns:
        matplotlib Figure.
    """
    from src.features.families import FEATURE_FAMILY_COLORS

    df = feature_ic_df.dropna(how="all")
    if df.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No IC data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_splits = len(df)
    x_labels = (
        split_labels if split_labels and len(split_labels) == n_splits
        else [f"S{idx}" for idx in df.index]
    )

    # Compute per-family mean IC for each split
    family_ic: dict[str, pd.Series] = {}
    for family, members in feature_families.items():
        present = [m for m in members if m in df.columns]
        if not present:
            continue
        family_ic[family] = df[present].mean(axis=1)

    if not family_ic:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No family members found in IC data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_families = len(family_ic)
    fig_height = max(FIG_HEIGHT_STANDARD, 3.5)
    fig, ax = make_figure(height=fig_height)

    x = np.arange(n_splits)
    bar_width = 0.8 / n_families
    offsets = np.linspace(-(n_families - 1) / 2, (n_families - 1) / 2, n_families) * bar_width

    _t = get_typography()
    for offset, (family, ic_series) in zip(offsets, family_ic.items()):
        color = FEATURE_FAMILY_COLORS.get(family, COLORS["neutral"])
        values = ic_series.values
        bars = ax.bar(x + offset, values, bar_width * 0.9, label=family,
                      color=color, alpha=0.8, linewidth=0)
        for bar, v in zip(bars, values):
            if not np.isnan(v) and abs(v) > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if v >= 0 else -0.003),
                    f"{v:+.2f}",
                    ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=_t.small_annotation,
                    color="#444444",
                )

    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=_t.tick, rotation=30, ha="right")
    ax.set_ylabel("Mean IC (Pearson)", fontsize=_t.small_label)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    ax.tick_params(axis="y", labelsize=_t.tick)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, color=COLORS["neutral"])
    ax.set_axisbelow(True)

    legend_cols = min(3, n_families)
    ax.legend(
        frameon=False, fontsize=_t.legend, ncol=legend_cols,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        handlelength=1.2,
    )

    label_axes(ax, title=title)
    extra_bottom = 0.08 + 0.06 * ((n_families - 1) // legend_cols)
    fig.subplots_adjust(bottom=extra_bottom + 0.12)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Cross-sectional concentration & breadth diagnostics (G-X-SYNC-7)
# ---------------------------------------------------------------------------


def plot_portfolio_concentration(
    weights: pd.DataFrame,
    window: int = 63,
    title: str = "Portfolio Concentration & Breadth",
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling concentration and effective breadth for a multi-asset portfolio.

    Top panel: rolling count of held assets (non-zero weight rows).
    Bottom panel: effective number of holdings N* = exp(H) where H is the
    Shannon entropy of the normalised weight vector.  N* = 1 indicates full
    concentration; N* = k indicates equal weighting across k assets.

    Args:
        weights:    Date × asset weight DataFrame (NaN or 0 = not held).
        window:     Rolling window length in trading days.
        title:      Figure title.
        save_path:  Optional save path.

    Returns:
        matplotlib Figure.
    """
    w = weights.copy().fillna(0.0)
    if w.empty:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No weight data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    # Active position count (non-zero weights per day)
    n_held = (w.abs() > 1e-8).sum(axis=1).astype(float)
    n_held_roll = n_held.rolling(window, min_periods=max(1, window // 4)).mean()

    # Effective N via weight entropy: N* = exp(-sum(w_i * log(w_i)))
    def _effective_n(row: pd.Series) -> float:
        pos = row[row > 1e-8]
        if pos.empty:
            return float("nan")
        p = pos / pos.sum()
        return float(np.exp(-float((p * np.log(p)).sum())))

    eff_n = w.apply(_effective_n, axis=1)
    eff_n_roll = eff_n.rolling(window, min_periods=max(1, window // 4)).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.6),
                                    sharex=True)

    _t = get_typography()
    # Top: rolling held-asset count
    ax1.fill_between(n_held_roll.index, n_held_roll.values,
                     color=COLORS["strategy"], alpha=0.3)
    ax1.plot(n_held_roll.index, n_held_roll.values,
             color=COLORS["strategy"], linewidth=1.2)
    ax1.set_ylabel(f"Holdings count ({window}d avg)", fontsize=_t.small_label)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(labelsize=_t.tick)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left", pad=4)

    # Bottom: effective N*
    ax2.fill_between(eff_n_roll.index, eff_n_roll.values,
                     color=COLORS["signal"], alpha=0.3)
    ax2.plot(eff_n_roll.index, eff_n_roll.values,
             color=COLORS["signal"], linewidth=1.2, label=f"Effective N* ({window}d avg)")
    n_assets = w.shape[1]
    ax2.axhline(n_assets, color=COLORS["grid"], linewidth=0.8, linestyle="--",
                label=f"Max breadth ({n_assets} assets)")
    ax2.axhline(1.0, color=COLORS["negative"], linewidth=0.8, linestyle=":",
                label="Full concentration (N*=1)")
    ax2.set_ylabel("Effective holdings N*", fontsize=_t.small_label)
    ax2.tick_params(labelsize=_t.tick)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=_t.legend, loc="upper right")

    fig.tight_layout(h_pad=0.6)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Regime-conditional IC by feature family (Step 2 — regime interpretation)
# ---------------------------------------------------------------------------


def plot_ic_by_vol_regime(
    family_ic_by_regime: dict[str, dict[str, float]],
    feature_families: "dict[str, list[str]] | None" = None,
    title: str = "Feature Family IC by Volatility Regime",
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing per-family mean IC in high- vs low-vol regimes.

    Each feature family appears as a pair of bars — one for high-volatility
    test splits and one for low-volatility test splits.  The comparison exposes
    which hypothesis families provided cross-sectional signal preferentially in
    stressed vs calm market environments.

    Args:
        family_ic_by_regime: {"high_vol": {"Trend": float, ...},
                              "low_vol":  {"Trend": float, ...}}
                             Produced by src.reporting.regime.compute_regime_stats().
        feature_families:    Optional family → members dict; used to enforce a
                             canonical display order.  Ignored if not provided.
        title:               Figure title.
        save_path:           Optional save path.

    Returns:
        matplotlib Figure.
    """
    from src.features.families import FEATURE_FAMILY_COLORS

    high_ic = family_ic_by_regime.get("high_vol", {})
    low_ic = family_ic_by_regime.get("low_vol", {})

    _order = ["Trend", "Volatility", "Mean-Reversion", "Market Structure", "Relative Strength"]
    all_fams = sorted(
        set(high_ic) | set(low_ic),
        key=lambda f: _order.index(f) if f in _order else len(_order),
    )

    if not all_fams:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No regime IC data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    n_fams = len(all_fams)
    fig, ax = make_figure(height=max(3.0, FIG_HEIGHT_STANDARD), width=FIG_WIDTH_FULL)

    x = np.arange(n_fams)
    bar_w = 0.35

    high_vals = [high_ic.get(f, float("nan")) for f in all_fams]
    low_vals = [low_ic.get(f, float("nan")) for f in all_fams]

    for i, fam in enumerate(all_fams):
        base_color = FEATURE_FAMILY_COLORS.get(fam, COLORS["neutral"])
        hv = high_vals[i]
        lv = low_vals[i]

        if not np.isnan(hv):
            bar_h = ax.bar(x[i] - bar_w / 2, hv, bar_w, color=base_color,
                           alpha=0.85, linewidth=0, label="High-vol" if i == 0 else "")
            if abs(hv) > 0.005:
                ax.text(bar_h[0].get_x() + bar_h[0].get_width() / 2,
                        hv + (0.003 if hv >= 0 else -0.003),
                        f"{hv:+.3f}", ha="center",
                        va="bottom" if hv >= 0 else "top", fontsize=_t.small_annotation, color="#333333")

        if not np.isnan(lv):
            bar_l = ax.bar(x[i] + bar_w / 2, lv, bar_w, color=base_color,
                           alpha=0.38, linewidth=0.8, edgecolor=base_color,
                           label="Low-vol" if i == 0 else "")
            if abs(lv) > 0.005:
                ax.text(bar_l[0].get_x() + bar_l[0].get_width() / 2,
                        lv + (0.003 if lv >= 0 else -0.003),
                        f"{lv:+.3f}", ha="center",
                        va="bottom" if lv >= 0 else "top", fontsize=_t.small_annotation, color="#333333")

    _t = get_typography()
    ax.axhline(0, color=COLORS["neutral"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_fams, fontsize=_t.tick)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, color=COLORS["neutral"])
    ax.set_axisbelow(True)

    # Annotation headroom — prevents tallest bar label from clipping top boundary
    _all_vals = np.array([v for v in (high_vals + low_vals) if not np.isnan(v)])
    if len(_all_vals):
        _ylo, _yhi = _bar_annotation_ylim(_all_vals, padding_frac=0.22)
        ax.set_ylim(_ylo, _yhi)

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(facecolor="#555555", alpha=0.85, label="High-vol splits"),
        mpatches.Patch(facecolor="#555555", alpha=0.38, edgecolor="#555555",
                       linewidth=0.8, label="Low-vol splits"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False, fontsize=_t.legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handlelength=1.5,
        handletextpad=0.5,
    )
    fig.suptitle(title, fontsize=_t.small_label, fontweight="bold", y=0.97)

    ax.set_ylabel("Mean IC (Pearson)", fontsize=_t.small_label)
    ax.tick_params(axis="y", labelsize=_t.tick)
    fig.tight_layout()
    fig.subplots_adjust(top=0.83)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_prediction_strength(
    prediction_strength: dict,
    title: str = "Prediction Strength by Score Group",
    save_path: str | None = None,
) -> plt.Figure:
    """Two-panel confidence & outcome monotonicity diagnostic.

    Panel 1 (top): bar chart — mean realized N-day forward return for each
    prediction rank group (top/mid/bottom thirds).  Left-to-right monotonic
    ordering is the primary ML legitimacy signal: score magnitude, not merely
    sign, carries cross-sectional economic content.

    Panel 2 (bottom): cumulative return curves — top, mid, bottom groups over
    time.  Persistent separation between top and bottom confirms durable signal
    strength; convergence marks regimes of prediction-strength collapse.

    Args:
        prediction_strength: dict returned by _prepare_prediction_strength().
                             Expected keys: group_mean_returns, group_monthly,
                             ls_spread, is_monotonic, n_obs, horizon.
        title:               Figure title prefix.
        save_path:           Optional save path.

    Returns:
        matplotlib Figure.
    """
    group_means = prediction_strength.get("group_mean_returns") or {}
    group_monthly: "pd.DataFrame | None" = prediction_strength.get("group_monthly")
    is_monotonic = prediction_strength.get("is_monotonic")
    ls_spread = prediction_strength.get("ls_spread")
    horizon = prediction_strength.get("horizon", 21)
    n_obs = prediction_strength.get("n_obs", 0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_FULL, 5.5),
        gridspec_kw={"height_ratios": [1, 1.8]},
    )

    # ── Panel 1: mean return bars ──────────────────────────────────────────
    groups = ["top", "mid", "bottom"]
    labels = ["Top (high score)", "Mid", "Bottom (low score)"]
    bar_colors = [
        COLORS.get("positive", "#2e7d32"),
        COLORS.get("neutral", "#9e9e9e"),
        COLORS.get("negative", "#c62828"),
    ]
    means = [group_means.get(g, float("nan")) for g in groups]

    x = np.arange(len(groups))
    _t = get_typography()
    bars = ax1.bar(x, means, color=bar_colors, alpha=0.82, linewidth=0, width=0.45)
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.0003 if val >= 0 else -0.0003),
                f"{val:.2%}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=_t.annotation, color="#222222",
            )

    # Annotation headroom — prevents top label from clipping against axis boundary
    _valid_means = np.array([v for v in means if not np.isnan(v)])
    if len(_valid_means):
        _ylo, _yhi = _bar_annotation_ylim(_valid_means, padding_frac=0.22)
        ax1.set_ylim(_ylo, _yhi)

    ax1.axhline(0, color=COLORS.get("grid", "#cccccc"), linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=_t.tick)
    ax1.set_ylabel(f"Mean {horizon}D fwd return", fontsize=_t.small_label)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax1.tick_params(axis="y", labelsize=_t.tick)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5, color=COLORS.get("grid", "#cccccc"))
    ax1.set_axisbelow(True)

    mono_str = "monotonic" if is_monotonic else "non-monotonic"
    spread_str = f"   L/S spread: {ls_spread:.2%}" if ls_spread is not None else ""
    ax1.set_title(
        f"{title}  ({n_obs} monthly obs.  ·  {mono_str}{spread_str})",
        fontsize=_t.small_label, fontweight="bold", loc="left", pad=4,
    )

    # ── Panel 2: cumulative return curves ────────────────────────────────
    if group_monthly is not None and not group_monthly.empty:
        cum_colors = {
            "top": COLORS.get("positive", "#2e7d32"),
            "mid": COLORS.get("neutral", "#9e9e9e"),
            "bottom": COLORS.get("negative", "#c62828"),
        }
        cum_labels = {"top": "Top group", "mid": "Mid group", "bottom": "Bottom group"}
        for grp in ["top", "mid", "bottom"]:
            if grp not in group_monthly.columns:
                continue
            vals = group_monthly[grp].dropna()
            if vals.empty:
                continue
            cum = (1.0 + vals).cumprod()
            ax2.plot(
                cum.index, cum.values,
                color=cum_colors[grp], linewidth=1.4,
                label=cum_labels[grp], alpha=0.9,
            )
        ax2.axhline(1.0, color=COLORS.get("grid", "#cccccc"), linewidth=0.8,
                    linestyle="--", alpha=0.7)
        ax2.set_ylabel("Cumulative return", fontsize=_t.small_label)
        ax2.tick_params(labelsize=_t.tick)
        ax2.grid(axis="y", alpha=0.3, linewidth=0.5,
                 color=COLORS.get("grid", "#cccccc"))
        ax2.set_axisbelow(True)
        ax2.legend(frameon=False, fontsize=_t.legend, loc="upper left",
                   handlelength=1.2)
    else:
        ax2.text(0.5, 0.5, "Cumulative data unavailable",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=_t.figure_text, color=COLORS.get("neutral", "#9e9e9e"))

    fig.tight_layout(h_pad=0.8)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_ranking_geometry(
    ranking_geo: dict,
    title: str = "Cross-Sectional Ranking Geometry",
    save_path: str | None = None,
) -> "plt.Figure":
    """Four-panel ranking-state diagnostic covering signal geometry and temporal stability.

    Panel 1 — Prediction Dispersion (S1 + S5 combined):
        Rolling IQR of predicted scores (left axis, filled) and rolling IC standard
        deviation (right axis, dashed line).  Distinguishes compression (low IQR) from
        erraticity (high IC std at near-zero mean IC).

    Panel 2 — Score Discrimination (S2):
        Rolling top-group minus bottom-group score spread.  Near-zero indicates the
        model assigns similar scores to top and bottom ranked assets — selection is
        arbitrary.  Sustained positive values confirm ranking conviction.

    Panel 3 — Realized Discrimination (S3):
        Rolling realized forward-return spread between top- and bottom-ranked asset
        groups.  Economic confirmation of whether score separation translates to outcome
        separation.  Pre-cost gross diagnostic — not a deployable return.

    Panel 4 — Rank Persistence (S4):
        Monthly Spearman autocorrelation of asset score rankings.  High values indicate
        stable model convictions; near-zero marks regimes where rankings flip arbitrarily
        each rebalance, amplifying turnover without informational content.

    Args:
        ranking_geo: Dict produced by _prepare_ranking_geometry() containing optional
                     keys: rolling_score_iqr, rolling_ic_std, rolling_score_spread,
                     rolling_realized_spread, rank_persistence.
        title:       Figure title (top-left panel header).
        save_path:   Optional save path.

    Returns:
        matplotlib Figure.
    """
    iqr_series = ranking_geo.get("rolling_score_iqr")
    ic_std = ranking_geo.get("rolling_ic_std")
    score_spread = ranking_geo.get("rolling_score_spread")
    realized_spread = ranking_geo.get("rolling_realized_spread")
    rank_pers = ranking_geo.get("rank_persistence")

    def _has(s: object) -> bool:
        return s is not None and hasattr(s, "__len__") and len(s) >= 10  # type: ignore[arg-type]

    # Determine which panels have data
    panels: list[str] = []
    if _has(iqr_series):
        panels.append("iqr")
    if _has(score_spread):
        panels.append("spread")
    if _has(realized_spread):
        panels.append("realized")
    if _has(rank_pers):
        panels.append("persistence")

    if not panels:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "Insufficient ranking geometry data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        if save_path:
            save_figure(fig, save_path, close=False)
        return fig

    n = len(panels)
    height_ratios = [1.5 if p == "iqr" else 1.0 for p in panels]
    fig_height = FIG_HEIGHT_STANDARD * (0.55 + 0.65 * n)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(FIG_WIDTH_FULL, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n == 1:
        axes = [axes]

    _t = get_typography()
    for ax_i, panel in enumerate(panels):
        ax = axes[ax_i]

        if panel == "iqr":
            # Left axis: rolling IQR
            ax.fill_between(iqr_series.index, iqr_series.values,  # type: ignore[union-attr]
                            alpha=0.18, color=COLORS["strategy"])
            ax.plot(iqr_series.index, iqr_series.values,  # type: ignore[union-attr]
                    linewidth=1.3, color=COLORS["strategy"],
                    label="Score IQR (63d avg)")
            ax.set_ylabel("Score IQR", fontsize=_t.small_label)
            ax.tick_params(labelsize=_t.tick)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left", pad=4)

            if _has(ic_std):
                ax2 = ax.twinx()
                ax2.plot(ic_std.index, ic_std.values,  # type: ignore[union-attr]
                         linewidth=1.0, color=COLORS["signal"],
                         linestyle="--", alpha=0.80, label="IC std (63d)")
                ax2.set_ylabel("IC std", fontsize=_t.colorbar, color=COLORS["signal"])
                ax2.tick_params(labelsize=_t.colorbar, colors=COLORS["signal"])
                ax2.spines["right"].set_visible(True)
                ax2.spines["right"].set_color(COLORS["signal"])
                ax2.spines["right"].set_alpha(0.5)
                lines1, labs1 = ax.get_legend_handles_labels()
                lines2, labs2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labs1 + labs2,
                          frameon=False, fontsize=_t.legend, loc="upper left")
            else:
                ax.legend(frameon=False, fontsize=_t.legend)

        elif panel == "spread":
            ax.fill_between(score_spread.index, score_spread.values,  # type: ignore[union-attr]
                            alpha=0.18, color=COLORS["positive"])
            ax.plot(score_spread.index, score_spread.values,  # type: ignore[union-attr]
                    linewidth=1.3, color=COLORS["positive"],
                    label="Top-bottom score spread (63d avg)")
            ax.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
            ax.set_ylabel("Score spread", fontsize=_t.small_label)
            ax.tick_params(labelsize=_t.tick)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=_t.legend)

        elif panel == "realized":
            pos_mask = realized_spread.values >= 0  # type: ignore[union-attr]
            ax.fill_between(realized_spread.index, realized_spread.values,  # type: ignore[union-attr]
                            where=pos_mask, alpha=0.22, color=COLORS["positive"])
            ax.fill_between(realized_spread.index, realized_spread.values,  # type: ignore[union-attr]
                            where=~pos_mask, alpha=0.22, color=COLORS["negative"])
            ax.plot(realized_spread.index, realized_spread.values,  # type: ignore[union-attr]
                    linewidth=1.0, color=COLORS["neutral"],
                    label="Realized top-bottom spread (63d avg, pre-cost)")
            ax.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
            format_pct_axis(ax)
            ax.set_ylabel("Return spread", fontsize=_t.small_label)
            ax.tick_params(labelsize=_t.tick)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=_t.legend)

        elif panel == "persistence":
            bar_colors = [
                COLORS["positive"] if float(v) >= 0 else COLORS["negative"]
                for v in rank_pers.values  # type: ignore[union-attr]
            ]
            ax.bar(rank_pers.index, rank_pers.values,  # type: ignore[union-attr]
                   width=20, color=bar_colors, alpha=0.50, zorder=2)
            rp_roll = rank_pers.rolling(3, min_periods=2).mean()  # type: ignore[union-attr]
            ax.plot(rp_roll.index, rp_roll.values,
                    linewidth=1.3, color=COLORS["strategy"],
                    label="3-month rolling mean", zorder=3)
            ax.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--", zorder=1)
            ax.set_ylabel("Rank autocorr.", fontsize=_t.small_label)
            ax.set_ylim(-1.05, 1.05)
            ax.tick_params(labelsize=_t.tick)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=_t.legend)

    fig.tight_layout(h_pad=0.5)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


# ---------------------------------------------------------------------------
# Feature contribution diagnostics (Phase II — C1 & C2)
# ---------------------------------------------------------------------------


def plot_feature_contribution_heatmap(
    contribution_df: "pd.DataFrame",
    feature_families: "dict[str, list[str]] | None" = None,
    title: str = "Feature Contribution Through Time",
    save_path: str | None = None,
) -> "plt.Figure":
    """Date × Feature contribution heatmap (Phase II, C1).

    Visualises realised predictive influence (coef × z-score) through time.
    Features are grouped by family; separator lines mark family boundaries.
    Diverging RdBu_r colormap: red = positive contribution (model predicts
    high return for this feature state), blue = negative.

    Args:
        contribution_df: Date × Feature DataFrame of rolling contribution values.
        feature_families: {family → [feature_names]} for grouping.
        title:           Figure title.
        save_path:       Optional save path.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.colors as mcolors

    df = contribution_df.dropna(how="all")
    if df.empty or df.shape[1] < 1:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    # --- Family-ordered reindexing ---
    ordered_cols = list(df.columns)
    family_boundaries: list[int] = []
    family_labels_pos: list[tuple[float, str]] = []

    if feature_families:
        reordered: list[str] = []
        for fam, members in feature_families.items():
            present = [m for m in members if m in df.columns]
            if present:
                if reordered:
                    family_boundaries.append(len(reordered))
                mid = len(reordered) + len(present) / 2 - 0.5
                family_labels_pos.append((mid, fam))
                reordered.extend(present)
        remaining = [c for c in df.columns if c not in reordered]
        if remaining:
            if reordered:
                family_boundaries.append(len(reordered))
            reordered.extend(remaining)
        if reordered:
            ordered_cols = reordered
            df = df[ordered_cols]

    n_features = len(df.columns)
    height = max(FIG_HEIGHT_STANDARD, n_features * 0.44 + 1.5)
    fig, ax = make_figure(height=height, width=FIG_WIDTH_FULL)

    Z = df.values.T.astype(float)  # (n_features, n_dates)

    # Symmetric colour scale using 98th percentile to reduce outlier influence
    finite_vals = Z[np.isfinite(Z)]
    vabs = float(np.percentile(np.abs(finite_vals), 98)) if len(finite_vals) > 0 else 0.1
    if vabs < 1e-10:
        vabs = 0.1

    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    mesh = ax.pcolormesh(
        df.index, np.arange(n_features), Z,
        cmap="RdBu_r", norm=norm, shading="nearest",
    )

    # Family separator lines (horizontal between groups)
    for boundary in family_boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.8, alpha=0.9)

    _t = get_typography()
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([_feat_label(c) for c in df.columns], fontsize=_t.small_annotation)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(False)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.75, aspect=20, pad=0.02)
    cbar.set_label("Contribution (coef × z)", fontsize=_t.colorbar)
    cbar.ax.tick_params(labelsize=_t.colorbar)

    label_axes(ax, title=title, xlabel="", ylabel="")
    fig.tight_layout()

    # Family labels — placed after tight_layout to avoid collision with tick labels
    if family_labels_pos:
        try:
            from src.features.families import FEATURE_FAMILY_COLORS
            fam_fs = scale_dynamic_fontsize(max(6, 8 - n_features // 8), "small_annotation")
            max_chars = max(len(fam) for _, fam in family_labels_pos)
            extra_in = max_chars * fam_fs * 0.60 / 72.0 + 0.12
            extra_frac = extra_in / fig.get_figwidth()
            base_left = fig.subplotpars.left
            fig.subplots_adjust(left=min(0.50, base_left + extra_frac))
            inv_fig = fig.transFigure.inverted()
            for pos, fam in family_labels_pos:
                color = FEATURE_FAMILY_COLORS.get(fam, "#666666")
                _, y_fig = inv_fig.transform(
                    ax.transData.transform((df.index[len(df.index) // 2], pos))
                )
                x_fig = extra_frac / 2.0
                fig.text(x_fig, y_fig, fam,
                         fontsize=fam_fs, color=color, fontweight="bold",
                         ha="center", va="center")
        except Exception:
            pass

    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_family_contribution_timeline(
    family_contrib_df: "pd.DataFrame",
    family_share_df: "pd.DataFrame",
    title: str = "Family Contribution Timeline",
    save_path: str | None = None,
) -> "plt.Figure":
    """Two-panel family contribution timeline (Phase II, C2).

    Top panel: Signed rolling family contributions (line chart) — shows which
    family drove the prediction signal and in which direction.  Persistent
    negative contributions identify families contributing counter-cyclically.

    Bottom panel: Normalized absolute family contribution share (stacked area)
    — shows which family dominated prediction at each point regardless of sign.

    Args:
        family_contrib_df: Date × Family DataFrame of signed rolling contributions.
        family_share_df:   Date × Family DataFrame of normalized absolute shares.
        title:             Figure title.
        save_path:         Optional save path.

    Returns:
        matplotlib Figure.
    """
    try:
        from src.features.families import FEATURE_FAMILY_COLORS as _FAM_COLORS
    except Exception:
        _FAM_COLORS: dict = {}

    def _has(df: object) -> bool:
        return df is not None and hasattr(df, "__len__") and len(df) >= 10  # type: ignore[arg-type]

    if not _has(family_contrib_df) and not _has(family_share_df):
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "No family contribution data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    # --- Top panel: signed family contributions ---
    if _has(family_contrib_df):
        for col in family_contrib_df.columns:  # type: ignore[union-attr]
            color = _FAM_COLORS.get(col, "#666666")
            ax1.plot(
                family_contrib_df.index,  # type: ignore[union-attr]
                family_contrib_df[col].values,  # type: ignore[union-attr]
                linewidth=1.3, color=color, label=col, alpha=0.85,
            )
        _t = get_typography()
        ax1.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Signed contribution", fontsize=_t.small_label)
        ax1.tick_params(labelsize=_t.tick)
        ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left", pad=4)
        # Shared legend above ax1 — serves both panels; keeps all canvases unobstructed
        n_fams = len(family_contrib_df.columns)  # type: ignore[union-attr]
        ax1.legend(
            frameon=False, fontsize=_t.legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(4, n_fams),
            handlelength=1.2,
            columnspacing=1.0,
        )

    # --- Bottom panel: normalized absolute share (stacked area) ---
    if _has(family_share_df):
        families = list(family_share_df.columns)  # type: ignore[union-attr]
        bottom = np.zeros(len(family_share_df))  # type: ignore[arg-type]
        for col in families:
            color = _FAM_COLORS.get(col, "#666666")
            vals = family_share_df[col].values  # type: ignore[union-attr]
            ax2.fill_between(
                family_share_df.index,  # type: ignore[union-attr]
                bottom, bottom + vals,
                color=color, alpha=0.65, linewidth=0,
            )
            bottom = bottom + vals
        _t = get_typography()
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("Contribution share", fontsize=_t.small_label)
        ax2.tick_params(labelsize=_t.tick)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax2.set_axisbelow(True)
        # No separate legend: family colours match the shared legend on ax1

    fig.tight_layout(h_pad=0.6)
    fig.subplots_adjust(top=0.88)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig


def plot_score_dispersion(
    score_wide: "pd.DataFrame",
    window: int = 63,
    title: str = "Score Dispersion & Ranking Conviction",
    stress_mask: "pd.Series | None" = None,
    save_path: str | None = None,
) -> "plt.Figure":
    """Score-dispersion and ranking-conviction diagnostics for a cross-sectional portfolio.

    Replaces the degenerate N*/holdings-count panel (uninformative under hard
    top-N equal-weight construction) with two signal-level diagnostics:

    Top panel — Cross-sectional score spread (IQR, rolling mean):
        IQR = 75th − 25th percentile of predicted scores across assets per day.
        Wide IQR → model assigns well-separated ranks; narrow IQR → model
        is indifferent, and any top-N selection is arbitrary.

    Bottom panel — Ranking entropy (rolling mean):
        Shannon entropy of the rank-probability vector (rank / sum(ranks)).
        High entropy → scores are evenly distributed and rankings are
        uncertain. Low entropy → a few assets dominate the ranking.
        Complements the IQR by capturing distributional shape, not just spread.

    Args:
        score_wide: Date × Asset DataFrame of predicted scores (wide format,
                    MultiIndex unstacked).  Rows with all-NaN are dropped.
        window:     Rolling mean window in trading days (default 63).
        title:      Figure title.
        stress_mask: Optional boolean Series aligned to score_wide.index for
                     stress-regime shading.
        save_path:  Optional save path.
    """
    sw = score_wide.dropna(how="all")
    if sw.empty or sw.shape[1] < 2:
        fig, ax = make_figure(height=FIG_HEIGHT_STANDARD)
        ax.text(0.5, 0.5, "Insufficient score data", ha="center", va="center",
                transform=ax.transAxes, fontsize=get_typography().figure_text, color=COLORS["neutral"])
        label_axes(ax, title=title)
        fig.tight_layout()
        return fig

    # --- Per-date cross-sectional IQR (score spread) ---
    iqr = sw.quantile(0.75, axis=1) - sw.quantile(0.25, axis=1)
    iqr_roll = iqr.rolling(window, min_periods=max(1, window // 4)).mean()

    # --- Per-date top-minus-median spread (ranking conviction) ---
    # How far above the cross-sectional median is the top-scored asset per day.
    # Wide spread → model has strong conviction about its top pick.
    # Near-zero → top asset barely beats the median; selection is arbitrary.
    top_minus_med = sw.max(axis=1) - sw.median(axis=1)
    top_minus_med_roll = top_minus_med.rolling(window, min_periods=max(1, window // 4)).mean()

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH_FULL, FIG_HEIGHT_STANDARD * 1.6),
        sharex=True,
    )

    # Stress shading (shared helper)
    if stress_mask is not None:
        _shade_stress_regimes(ax1, sw.index, stress_mask)
        _shade_stress_regimes(ax2, sw.index, stress_mask)

    _t = get_typography()
    # Top: IQR
    ax1.fill_between(iqr_roll.index, iqr_roll.values,
                     color=COLORS["strategy"], alpha=0.25)
    ax1.plot(iqr_roll.index, iqr_roll.values,
             color=COLORS["strategy"], linewidth=1.2,
             label=f"Score IQR ({window}d rolling mean)")
    ax1.set_ylabel(f"Score IQR ({window}d avg)", fontsize=_t.small_label)
    ax1.tick_params(labelsize=_t.tick)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title(title, fontsize=_t.small_label, fontweight="bold", loc="left", pad=4)
    ax1.legend(frameon=False, fontsize=_t.legend)

    # Bottom: top-minus-median conviction spread
    ax2.fill_between(top_minus_med_roll.index, top_minus_med_roll.values,
                     color=COLORS["signal"], alpha=0.25)
    ax2.plot(top_minus_med_roll.index, top_minus_med_roll.values,
             color=COLORS["signal"], linewidth=1.2,
             label=f"Top-minus-median ({window}d rolling mean)")
    ax2.axhline(0, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax2.set_ylabel(f"Top-vs-median spread ({window}d avg)", fontsize=_t.small_label)
    ax2.tick_params(labelsize=_t.tick)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=_t.legend, loc="upper right")

    fig.tight_layout(h_pad=0.6)
    if save_path:
        save_figure(fig, save_path, close=False)
    return fig
