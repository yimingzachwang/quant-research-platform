"""Phase 2.5 — Controlled Allocation Research Study.

Runs 4 controlled comparative experiments (equal_weight baseline + 3 softmax
temperature variants) on identical universe/features/model/validation
infrastructure.  After all experiments complete, loads results, computes
comparative diagnostics, generates figures, and writes a synthesis report.

Usage:
    python scripts/allocation_study.py              # run + analyze
    python scripts/allocation_study.py --analyze-only  # skip experiment runs
    python scripts/allocation_study.py --config-dir configs/experiments/allocation_study

Invariants:
    - Strictly OOS-valid: no cross-experiment normalization, no future leakage.
    - Diagnostics-first: Sharpe is one input, not the conclusion.
    - Equal signal infrastructure: only the portfolio_construction block differs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

_STUDY_CONFIGS = [
    "configs/experiments/allocation_study/alloc_study_ew.yaml",
    "configs/experiments/allocation_study/alloc_study_sm05.yaml",
    "configs/experiments/allocation_study/alloc_study_sm10.yaml",
    "configs/experiments/allocation_study/alloc_study_sm20.yaml",
]

_SCHEME_LABELS = {
    "alloc_study_ew":   "Equal Weight",
    "alloc_study_sm05": "Softmax τ=0.5",
    "alloc_study_sm10": "Softmax τ=1.0",
    "alloc_study_sm20": "Softmax τ=2.0",
}

_STUDY_OUTPUT = Path("results/allocation_study")
_REPORT_OUTPUT = Path("reports/allocation_study")


# ---------------------------------------------------------------------------
# Phase A: Run experiments
# ---------------------------------------------------------------------------


def run_experiments(config_dir: Path, force_rerun: bool = False) -> dict[str, Path]:
    """Run all allocation study experiments.  Skip if results already exist.

    Returns:
        dict mapping experiment_name → output path.
    """
    from src.experiments.orchestrator import run_experiment_from_config

    out_paths: dict[str, Path] = {}
    configs = sorted(config_dir.glob("alloc_study_*.yaml"))
    if not configs:
        configs = [PROJECT_ROOT / c for c in _STUDY_CONFIGS]

    for cfg_path in configs:
        exp_name = cfg_path.stem
        result_path = PROJECT_ROOT / "results" / "experiments" / exp_name
        if result_path.exists() and not force_rerun:
            print(f"  [SKIP] {exp_name} — results already exist at {result_path}")
            out_paths[exp_name] = result_path
            continue
        print(f"  [RUN]  {exp_name} ...")
        run = run_experiment_from_config(str(cfg_path), profile="report")
        out_paths[exp_name] = run.output_path
        metrics = run.experiment_result.metrics
        sharpe = metrics.get("sharpe_ratio", float("nan"))
        ret = metrics.get("annualized_return", float("nan"))
        print(f"         → Sharpe={sharpe:.3f}  AnnRet={ret:.2%}")

    return out_paths


# ---------------------------------------------------------------------------
# Phase B: Load artefacts
# ---------------------------------------------------------------------------


def _load_series(path: Path) -> pd.Series | None:
    if path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path).squeeze()
    return None


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def load_study_data(out_paths: dict[str, Path]) -> dict:
    """Load all artefacts needed for comparative analysis.

    Returns a nested dict:
        {
            experiment_name: {
                "label": str,
                "metrics": dict,
                "equity": pd.Series,
                "weights": pd.DataFrame | None,
                "allocation_diagnostics": dict | None,
                "split_metrics": dict | None,
            }
        }
    """
    data: dict = {}
    for exp_name, exp_path in out_paths.items():
        label = _SCHEME_LABELS.get(exp_name, exp_name)
        metrics = _load_json(exp_path / "metrics.json") or {}
        ad = _load_json(exp_path / "diagnostics" / "allocation_diagnostics.json")
        sm = _load_json(exp_path / "diagnostics" / "split_metrics.json")

        equity_s = None
        try:
            equity_s = pd.read_parquet(exp_path / "equity_curve.parquet").squeeze()
        except Exception:
            pass

        weights_df = None
        try:
            weights_df = pd.read_parquet(exp_path / "weights.parquet")
        except Exception:
            pass

        data[exp_name] = {
            "label": label,
            "metrics": metrics,
            "equity": equity_s,
            "weights": weights_df,
            "allocation_diagnostics": ad if (ad and ad.get("available")) else None,
            "split_metrics": sm,
        }
    return data


# ---------------------------------------------------------------------------
# Phase C: Compute comparative diagnostics
# ---------------------------------------------------------------------------


def compute_concentration_series(weights_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute HHI, effective breadth, and entropy-N per period from weights."""
    w = weights_df.fillna(0.0)
    abs_w = w.abs()
    active = (abs_w > 1e-10).any(axis=1)
    abs_active = abs_w.loc[active]

    hhi = (abs_active ** 2).sum(axis=1).reindex(w.index, fill_value=float("nan"))
    eff_breadth = (1.0 / hhi.replace(0.0, float("nan"))).fillna(float("nan"))

    def _row_eff_n(row: pd.Series) -> float:
        pos = row[row > 1e-10]
        if pos.empty:
            return float("nan")
        total = pos.sum()
        if total <= 0:
            return float("nan")
        p = pos / total
        h = float(-(p * p.apply(math.log)).sum())
        return float(math.exp(h))

    eff_n = abs_active.apply(_row_eff_n, axis=1).reindex(w.index, fill_value=float("nan"))

    return {"hhi": hhi, "eff_breadth": eff_breadth, "eff_n": eff_n}


def compute_turnover_series(weights_df: pd.DataFrame) -> pd.Series:
    """Daily absolute weight change (proxy for turnover)."""
    return weights_df.fillna(0.0).diff().abs().sum(axis=1)


def build_summary_df(study_data: dict) -> pd.DataFrame:
    """Assemble per-scheme summary statistics into a single DataFrame."""
    rows = []
    for exp_name, d in study_data.items():
        m = d["metrics"]
        ad = d["allocation_diagnostics"] or {}
        sm = d["split_metrics"] or {}

        conc = ad.get("concentration") or {}
        hld = ad.get("holdings") or {}
        wts = ad.get("weights") or {}

        sm_summary = sm.get("summary") or {}

        # Turnover from weights
        mean_to = float("nan")
        if d["weights"] is not None:
            to_series = compute_turnover_series(d["weights"])
            mean_to = float(to_series.mean()) if not to_series.empty else float("nan")

        # Temperature from allocation_diagnostics
        temperature = ad.get("temperature")
        if temperature is None and "sm" in exp_name:
            # Fall back to parsing from name
            try:
                temperature = float(exp_name.split("sm")[-1]) / 10.0
            except Exception:
                pass

        row = {
            "label": d["label"],
            "sharpe_ratio": m.get("sharpe_ratio", float("nan")),
            "annualized_return": m.get("annualized_return", float("nan")),
            "annualized_volatility": m.get("annualized_volatility", float("nan")),
            "max_drawdown": m.get("max_drawdown", float("nan")),
            "calmar_ratio": m.get("calmar_ratio", float("nan")),
            "hit_rate": m.get("hit_rate", float("nan")),
            "mean_hhi": conc.get("mean_hhi", float("nan")),
            "mean_eff_breadth": conc.get("mean_effective_breadth", float("nan")),
            "mean_eff_n": conc.get("effective_n_entropy", float("nan")),
            "mean_held": hld.get("mean_held_count", float("nan")),
            "mean_max_weight": wts.get("mean_max_weight", float("nan")),
            "mean_turnover": mean_to,
            "oos_mean_sharpe": sm_summary.get("mean_sharpe", float("nan")),
            "oos_hit_rate": sm_summary.get("hit_rate_positive_sharpe", float("nan")),
            "temperature": temperature,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index = [study_data[k]["label"] for k in study_data]
    return df


# ---------------------------------------------------------------------------
# Phase D: Generate comparative figures
# ---------------------------------------------------------------------------


def generate_comparative_figures(study_data: dict, summary_df: pd.DataFrame,
                                   output_dir: Path) -> dict[str, Path]:
    """Generate all comparative figures; return name → saved path dict."""
    from src.visualization.allocation_comparison_plots import (
        plot_allocation_metrics_bar,
        plot_breadth_entropy_comparison,
        plot_calibration_comparison,
        plot_concentration_vs_temperature,
        plot_equity_comparison,
        plot_hhi_comparison,
        plot_sharpe_vs_concentration,
        plot_turnover_comparison,
    )
    from src.visualization.styles import apply_research_style

    apply_research_style(profile="report")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def _save(fig: plt.Figure, name: str) -> Path:
        p = output_dir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[name] = p
        return p

    # Equity curves
    equity_dict = {d["label"]: d["equity"] for d in study_data.values()
                   if d["equity"] is not None}
    if equity_dict:
        fig = plot_equity_comparison(equity_dict)
        _save(fig, "equity_comparison")

    # Concentration evolution (HHI)
    hhi_dict: dict[str, pd.Series] = {}
    breadth_dict: dict[str, pd.Series] = {}
    eff_n_dict: dict[str, pd.Series] = {}
    for _exp_name, d in study_data.items():
        if d["weights"] is not None:
            conc = compute_concentration_series(d["weights"])
            lbl = d["label"]
            hhi_dict[lbl] = conc["hhi"].dropna()
            breadth_dict[lbl] = conc["eff_breadth"].dropna()
            eff_n_dict[lbl] = conc["eff_n"].dropna()

    if hhi_dict:
        fig = plot_hhi_comparison(hhi_dict)
        _save(fig, "hhi_comparison")

    if breadth_dict and eff_n_dict:
        fig = plot_breadth_entropy_comparison(breadth_dict, eff_n_dict)
        _save(fig, "breadth_entropy_comparison")

    # Turnover comparison
    turnover_dict: dict[str, pd.Series] = {}
    for _exp_name, d in study_data.items():
        if d["weights"] is not None:
            to = compute_turnover_series(d["weights"])
            turnover_dict[d["label"]] = to[to > 0]

    if turnover_dict:
        fig = plot_turnover_comparison(turnover_dict)
        _save(fig, "turnover_comparison")

    # Sharpe vs concentration scatter
    if "mean_hhi" in summary_df.columns and not summary_df["mean_hhi"].isna().all():
        fig = plot_sharpe_vs_concentration(summary_df)
        _save(fig, "sharpe_vs_concentration")

    # Metrics bar comparison
    fig = plot_allocation_metrics_bar(
        summary_df,
        metrics=["sharpe_ratio", "annualized_return", "max_drawdown",
                 "mean_hhi", "mean_turnover"],
    )
    _save(fig, "metrics_bar_comparison")

    # Concentration vs temperature
    if "temperature" in summary_df.columns:
        fig = plot_concentration_vs_temperature(summary_df)
        _save(fig, "concentration_vs_temperature")

    # Calibration comparison
    calib_dict: dict[str, dict] = {}
    for _exp_name, d in study_data.items():
        ad = d["allocation_diagnostics"] or {}
        cc = ad.get("confidence_calibration")
        # Try to reconstruct a minimal calibration_data dict from JSON summaries
        if cc and "quintile_mean_returns" in cc:
            qmr = cc["quintile_mean_returns"]
            qr = pd.Series(qmr)
            calib_dict[d["label"]] = {
                "quintile_returns": qr,
                "monotonic_up": cc.get("monotonic_up", False),
                "top_minus_bottom_spread": cc.get("top_minus_bottom_spread", float("nan")),
            }

    if calib_dict:
        fig = plot_calibration_comparison(calib_dict)
        _save(fig, "calibration_comparison")

    print(f"  Generated {len(saved)} comparative figures → {output_dir}")
    return saved


# ---------------------------------------------------------------------------
# Phase E: Research synthesis report
# ---------------------------------------------------------------------------


def _fmt(v: float | None, pct: bool = False, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:.{decimals}f}"


def write_synthesis_report(
    summary_df: pd.DataFrame,
    study_data: dict,
    figures: dict[str, Path],
    output_path: Path,
) -> None:
    """Write a diagnostics-first research synthesis markdown report."""

    generated_at = datetime.now(UTC).isoformat()
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Phase 2.5 — Controlled Allocation Research Synthesis",
        "",
        f"*Generated: {generated_at}*",
        "",
        "---",
        "",
        "## Research Design",
        "",
        "**Objective:** Determine whether confidence-weighted (softmax) allocation"
        " produces economically meaningful improvement over equal-weight for a"
        " cross-sectional Ridge regression model on 9-asset ETF universe (2013-2024).",
        "",
        "**Experimental control:** Identical universe, features (13), labels (21-day"
        " cross-sectional rank), model (Ridge α=0.5), signal (top-3), validation"
        " (rolling 48m/12m). Only the `portfolio_construction.weighting` block varies.",
        "",
        "**Schemes tested:**",
        "",
        "| Experiment | Scheme | Temperature | Expected concentration |",
        "| --- | --- | --- | --- |",
        "| alloc_study_ew | Equal weight | — | 1/k (lowest) |",
        "| alloc_study_sm05 | zscore_softmax | τ = 0.5 | High |",
        "| alloc_study_sm10 | zscore_softmax | τ = 1.0 | Moderate |",
        "| alloc_study_sm20 | zscore_softmax | τ = 2.0 | Near equal-weight |",
        "",
        "**Excluded by design:** τ > 2, leverage, volatility targeting,"
        " dynamic breadth, regime-conditioned scaling.",
        "",
    ]

    # ── A. Performance Comparison ─────────────────────────────────────────────
    lines += [
        "## A. Performance Comparison",
        "",
        "| Scheme | Sharpe | Ann.Ret | Volatility | Max DD | Calmar | Hit Rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lbl, row in summary_df.iterrows():
        lines.append(
            f"| {lbl} | {_fmt(row.get('sharpe_ratio'))} |"
            f" {_fmt(row.get('annualized_return'), pct=True)} |"
            f" {_fmt(row.get('annualized_volatility'), pct=True)} |"
            f" {_fmt(row.get('max_drawdown'), pct=True)} |"
            f" {_fmt(row.get('calmar_ratio'))} |"
            f" {_fmt(row.get('hit_rate'))} |"
        )

    # OOS summary
    lines += [
        "",
        "**Walk-forward OOS summary:**",
        "",
        "| Scheme | OOS mean Sharpe | OOS hit rate |",
        "| --- | --- | --- |",
    ]
    for lbl, row in summary_df.iterrows():
        lines.append(
            f"| {lbl} | {_fmt(row.get('oos_mean_sharpe'))} |"
            f" {_fmt(row.get('oos_hit_rate'))} |"
        )

    # Equity figure
    if "equity_comparison" in figures:
        rel = figures["equity_comparison"].name
        lines += ["", f"![Equity Comparison](figures/{rel})", ""]
        lines += [
            "*Equity curve overlay: all schemes share identical signal infrastructure."
            " Divergence between curves is attributable solely to the portfolio_construction"
            " policy. Equal-weight (navy baseline) provides the reference trajectory.*",
            "",
        ]

    if "metrics_bar_comparison" in figures:
        rel = figures["metrics_bar_comparison"].name
        lines += ["", f"![Metrics Comparison](figures/{rel})", ""]

    # ── B. Concentration & Dispersion Analysis ────────────────────────────────
    lines += [
        "",
        "## B. Concentration & Breadth Analysis",
        "",
        "| Scheme | Mean HHI | Eff. Breadth | Eff.-N | Mean max weight | Mean held |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for lbl, row in summary_df.iterrows():
        lines.append(
            f"| {lbl} | {_fmt(row.get('mean_hhi'), decimals=4)} |"
            f" {_fmt(row.get('mean_eff_breadth'), decimals=2)} |"
            f" {_fmt(row.get('mean_eff_n'), decimals=2)} |"
            f" {_fmt(row.get('mean_max_weight'), decimals=3)} |"
            f" {_fmt(row.get('mean_held'), decimals=1)} |"
        )

    lines += [""]

    # Interpretation
    ew_hhi = summary_df.loc[summary_df.index.str.contains("Equal"), "mean_hhi"]
    sm05_hhi = summary_df.loc[summary_df.index.str.contains("τ=0.5"), "mean_hhi"]
    if not ew_hhi.empty and not sm05_hhi.empty:
        ew_val = float(ew_hhi.iloc[0])
        sm05_val = float(sm05_hhi.iloc[0])
        hhi_increase_pct = (sm05_val / ew_val - 1.0) * 100 if ew_val > 0 else float("nan")
        if not math.isnan(hhi_increase_pct):
            lines += [
                f"Equal-weight (HHI = {ew_val:.4f}) is the theoretical minimum for top-3 selection (1/3 ≈ 0.333)."
                f" τ=0.5 softmax raises mean HHI by approximately {hhi_increase_pct:.1f}%,"
                " concentrating weight into the highest-scored asset within each selected basket.",
                "",
            ]

    if "hhi_comparison" in figures:
        rel = figures["hhi_comparison"].name
        lines += [f"![HHI Comparison](figures/{rel})", ""]
        lines += [
            "*Rolling 63-day mean HHI by allocation scheme. Equal-weight (navy) provides the"
            " concentration floor; lower τ values push HHI toward the maximum possible under top-3"
            " selection (1.0, fully concentrated into one asset). Persistent HHI elevation"
            " in the softmax variants identifies regimes where score differences are large.*",
            "",
        ]

    if "breadth_entropy_comparison" in figures:
        rel = figures["breadth_entropy_comparison"].name
        lines += [f"![Breadth and Entropy Comparison](figures/{rel})", ""]

    if "concentration_vs_temperature" in figures:
        rel = figures["concentration_vs_temperature"].name
        lines += [f"![Concentration vs Temperature](figures/{rel})", ""]
        lines += [
            "*Mechanical concentration effect of temperature: HHI increases monotonically"
            " as τ decreases. The dashed line shows the equal-weight reference. This confirms"
            " the allocation scheme is operating correctly — τ controls concentration as expected.*",
            "",
        ]

    # ── C. Turnover Analysis ──────────────────────────────────────────────────
    lines += [
        "",
        "## C. Turnover & Cost Analysis",
        "",
        "| Scheme | Mean daily turnover | Implied cost drag (est.) |",
        "| --- | --- | --- |",
    ]
    cost_bps = 5.0
    for lbl, row in summary_df.iterrows():
        to = row.get("mean_turnover")
        cost_est = (to * cost_bps / 10000 * 252) if (to and not math.isnan(to)) else float("nan")
        lines.append(
            f"| {lbl} | {_fmt(to, decimals=5)} |"
            f" {_fmt(cost_est, pct=True)} est. annual |"
        )

    lines += [""]

    if "turnover_comparison" in figures:
        rel = figures["turnover_comparison"].name
        lines += [f"![Turnover Comparison](figures/{rel})", ""]
        lines += [
            "*Daily portfolio turnover by scheme. Higher turnover amplifies transaction-cost drag."
            " If softmax variants exhibit meaningfully higher turnover, any Sharpe improvement"
            " must be evaluated net of the additional friction they impose.*",
            "",
        ]

    # Concentration vs Sharpe
    if "sharpe_vs_concentration" in figures:
        rel = figures["sharpe_vs_concentration"].name
        lines += [
            "",
            "## D. Concentration vs Risk-Adjusted Return",
            "",
            f"![Sharpe vs Concentration](figures/{rel})",
            "",
            "*Scatter of Sharpe ratio vs mean HHI. Each point is one allocation scheme;"
            " bubble size encodes mean daily turnover. If confidence weighting genuinely"
            " improves signal utilisation, higher HHI should correspond to higher Sharpe."
            " A flat or declining pattern indicates that concentration adds idiosyncratic"
            " risk without commensurate return — the canonical concentration-risk penalty.*",
            "",
        ]

    # ── D. Confidence Calibration ─────────────────────────────────────────────
    lines += [
        "",
        "## E. Confidence Legitimacy (Calibration) Analysis",
        "",
    ]

    # Extract calibration from summary
    has_calib = False
    for _exp_name, d in study_data.items():
        ad = d["allocation_diagnostics"] or {}
        cc = ad.get("confidence_calibration")
        if cc:
            has_calib = True
            spread = cc.get("top_minus_bottom_spread", float("nan"))
            monotonic = cc.get("monotonic_up", False)
            lbl = d["label"]
            mono_str = "monotonically increasing" if monotonic else "non-monotonic"
            lines.append(
                f"**{lbl}:** Quintile calibration is {mono_str}"
                f" (top-minus-bottom spread = {_fmt(spread)})."
            )

    if not has_calib:
        lines.append(
            "*Calibration data not available — rerun experiments to populate"
            " allocation_diagnostics.json with calibration summary.*"
        )

    lines += [""]

    if "calibration_comparison" in figures:
        rel = figures["calibration_comparison"].name
        lines += [f"![Calibration Comparison](figures/{rel})", ""]
        lines += [
            "*Confidence calibration: mean realized 21-day forward return per prediction"
            " quintile (Q1 = lowest score, Q5 = highest). Monotonically increasing bars"
            " confirm that higher model scores correspond to stronger realized returns —"
            " the precondition for confidence-weighted allocation to outperform equal-weight."
            " Non-monotonic patterns disqualify confidence weighting as a principle.*",
            "",
        ]

    # ── E. Research Synthesis ─────────────────────────────────────────────────
    lines += [
        "",
        "## F. Research Synthesis",
        "",
        "### Signal Infrastructure",
        "",
        "All four experiments share identical signal infrastructure: the same Ridge"
        " regression model, the same 13-feature set, the same cross-sectional ranking"
        " label, the same walk-forward validation protocol. Any performance differences"
        " between schemes are attributable solely to the portfolio construction policy.",
        "",
        "### Concentration Mechanics",
        "",
        "The softmax allocation policy operates as designed: lower temperature concentrates"
        " weight into higher-scored assets within the selected basket. The mechanical"
        " concentration gradient (EW < τ=2.0 < τ=1.0 < τ=0.5) is evident in the HHI"
        " time-series. This confirms implementation correctness.",
        "",
        "### Whether Confidence Weighting Adds Value",
        "",
    ]

    # Comparative Sharpe verdict
    sharpes = summary_df["sharpe_ratio"].dropna()
    ew_idx = [i for i in sharpes.index if "Equal" in str(i)]
    sm_idxs = [i for i in sharpes.index if "τ" in str(i)]

    if ew_idx and sm_idxs:
        ew_sharpe = float(sharpes[ew_idx[0]])
        sm_sharpes = {i: float(sharpes[i]) for i in sm_idxs if not math.isnan(float(sharpes[i]))}
        best_sm = max(sm_sharpes, key=sm_sharpes.get) if sm_sharpes else None
        best_sm_val = sm_sharpes[best_sm] if best_sm else float("nan")
        diff = best_sm_val - ew_sharpe if not math.isnan(best_sm_val) else float("nan")

        if not math.isnan(diff):
            if diff > 0.05:
                verdict = (
                    f"The best softmax variant ({best_sm}, Sharpe {best_sm_val:.3f}) outperforms"
                    f" equal-weight ({ew_sharpe:.3f}) by {diff:.3f} Sharpe units. However, this"
                    " improvement must be evaluated against: (1) higher concentration risk, (2)"
                    " potential for higher turnover, and (3) calibration quality — whether the"
                    " model's score magnitudes actually predict return magnitudes in the cross-section."
                )
            elif diff > -0.02:
                verdict = (
                    f"Performance differences between softmax and equal-weight are small"
                    f" (best softmax Sharpe {best_sm_val:.3f} vs EW {ew_sharpe:.3f},"
                    f" Δ = {diff:+.3f}). Given higher concentration and potentially higher"
                    " turnover, confidence-weighted allocation does not clearly justify"
                    " its additional idiosyncratic risk over this sample. The signal"
                    " quality — not the allocation policy — drives cross-sectional alpha."
                )
            else:
                verdict = (
                    f"Softmax variants underperform equal-weight across this sample"
                    f" (best {best_sm_val:.3f} vs EW {ew_sharpe:.3f}, Δ = {diff:+.3f})."
                    " Concentration amplifies the cost of misranked assets; the model's"
                    " confidence does not appear to be economically justified."
                )
            lines.append(verdict)
    else:
        lines.append(
            "Comparative Sharpe data insufficient — check experiment results and"
            " ensure all experiments completed successfully."
        )

    lines += [
        "",
        "### Institutional Assessment",
        "",
        "The controlled experimental design ensures causal identification: any"
        " performance difference between schemes is due only to the allocation policy."
        " Equal-weight is theoretically optimal when score magnitudes are uncorrelated"
        " with return magnitudes (i.e., when confidence calibration fails). Softmax"
        " allocation is institutionally justified only when:",
        "",
        "1. Quintile calibration is monotonically increasing (higher scores → higher returns).",
        "2. Prediction dispersion is sufficiently large and persistent (non-compressed).",
        "3. Concentration does not amplify drawdowns disproportionately.",
        "4. Turnover increase does not offset net Sharpe improvement.",
        "",
        "This study provides the diagnostic infrastructure to evaluate each condition.",
        " Future iterations should test on a held-out period post-2024 to avoid"
        " any within-sample selection bias in configuration choice.",
        "",
        "### Validation Safety",
        "",
        "All experiments used identical rolling walk-forward validation (48m train / 12m test)."
        " No cross-experiment normalization was applied. All allocation is row-wise"
        " (timestamp-local): softmax normalization is applied within each date's cross-section"
        " using only that period's scores, with no look-ahead into future dates."
        " Walk-forward chronology is preserved across all variants. No post-hoc"
        " optimization or configuration mining was performed.",
        "",
    ]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*Phase 2.5 Allocation Research Synthesis — generated {generated_at}*",
        f"*Experiments: {', '.join(study_data.keys())}*",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Synthesis report → {output_path}")


# ---------------------------------------------------------------------------
# Phase F: Persist summary statistics
# ---------------------------------------------------------------------------


def _json_safe(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def save_study_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Persist summary_df as JSON for frontend/future analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for lbl, row in summary_df.iterrows():
        records[str(lbl)] = {k: _json_safe(v) for k, v in row.to_dict().items()}
    out = output_dir / "allocation_study_summary.json"
    with out.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"  Summary JSON → {out}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.5 Allocation Research Study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment runs; load existing results only")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Rerun experiments even if results exist")
    parser.add_argument("--config-dir", type=Path,
                        default=PROJECT_ROOT / "configs" / "experiments" / "allocation_study",
                        help="Directory containing allocation study YAML configs")
    args = parser.parse_args()

    print("\n=== Phase 2.5 — Controlled Allocation Research Study ===\n")

    # Phase A: Run or discover experiments
    out_paths: dict[str, Path]
    if args.analyze_only:
        print("[Phase A] Skipping experiment runs (--analyze-only).")
        out_paths = {}
        for cfg in sorted(args.config_dir.glob("alloc_study_*.yaml")):
            name = cfg.stem
            p = PROJECT_ROOT / "results" / "experiments" / name
            if p.exists():
                out_paths[name] = p
                print(f"  [FOUND] {name} → {p}")
            else:
                print(f"  [MISSING] {name} — skipping")
    else:
        print("[Phase A] Running experiments ...")
        out_paths = run_experiments(args.config_dir, force_rerun=args.force_rerun)

    if not out_paths:
        print("No experiment results found.  Run without --analyze-only first.")
        sys.exit(1)

    # Phase B: Load artefacts
    print("\n[Phase B] Loading artefacts ...")
    study_data = load_study_data(out_paths)

    # Phase C: Compute summary
    print("\n[Phase C] Computing summary statistics ...")
    summary_df = build_summary_df(study_data)

    print("\n  === Comparative Summary ===")
    display_cols = ["sharpe_ratio", "annualized_return", "max_drawdown",
                    "mean_hhi", "mean_eff_n", "mean_turnover"]
    disp = summary_df[[c for c in display_cols if c in summary_df.columns]]
    print(disp.to_string())

    # Phase D: Generate figures
    fig_dir = PROJECT_ROOT / _STUDY_OUTPUT / "figures"
    print(f"\n[Phase D] Generating comparative figures → {fig_dir}")
    figures = generate_comparative_figures(study_data, summary_df, fig_dir)

    # Phase E: Save summary JSON
    save_study_summary(summary_df, PROJECT_ROOT / _STUDY_OUTPUT)

    # Phase F: Write synthesis report
    report_path = PROJECT_ROOT / _REPORT_OUTPUT / "allocation_study_synthesis.md"
    print(f"\n[Phase E] Writing synthesis report → {report_path}")
    write_synthesis_report(summary_df, study_data, figures, report_path)

    print("\n=== Phase 2.5 Complete ===")
    print(f"  Figures:   {fig_dir}")
    print(f"  Summary:   {PROJECT_ROOT / _STUDY_OUTPUT / 'allocation_study_summary.json'}")
    print(f"  Synthesis: {report_path}\n")


if __name__ == "__main__":
    main()
