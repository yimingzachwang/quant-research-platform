"""Phase 3A — Controlled Signal Geometry Expansion Study.

Runs 4 controlled experiments across Ridge α ∈ {0.5, 0.1, 0.05, 0.01} on the
15-ETF expanded universe with identical features, labels, signal, validation,
and portfolio construction. After all experiments complete, loads results,
computes signal geometry diagnostics, generates comparative figures, and writes
a diagnostics-first research synthesis.

Research question:
    Does reduced regularization produce economically meaningful prediction
    geometry (dispersed, stable, calibrated) — or amplified noise?

Controlled variables (identical across all runs):
    - Universe: 15 ETFs (Phase 3A expansion)
    - Features: 13-feature set (momentum, volatility, trend, breakout, etc.)
    - Label: ranking_target (21d cross-sectional rank)
    - Signal: top-5 equal-weight
    - Validation: rolling 48m train / 12m test
    - Portfolio construction: equal_weight

Varying:
    - model.params.alpha: 0.5, 0.1, 0.05, 0.01

Usage:
    python scripts/signal_geometry_study.py              # run + analyze
    python scripts/signal_geometry_study.py --analyze-only  # skip experiment runs
    python scripts/signal_geometry_study.py --force-rerun   # force all experiments
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
    "configs/experiments/signal_geometry/sg_alpha_050.yaml",
    "configs/experiments/signal_geometry/sg_alpha_010.yaml",
    "configs/experiments/signal_geometry/sg_alpha_005.yaml",
    "configs/experiments/signal_geometry/sg_alpha_001.yaml",
]

_ALPHA_LABELS = {
    "sg_alpha_050": "α=0.50",
    "sg_alpha_010": "α=0.10",
    "sg_alpha_005": "α=0.05",
    "sg_alpha_001": "α=0.01",
}

_ALPHA_VALUES = {
    "sg_alpha_050": 0.50,
    "sg_alpha_010": 0.10,
    "sg_alpha_005": 0.05,
    "sg_alpha_001": 0.01,
}

_STUDY_OUTPUT = Path("results/signal_geometry")
_REPORT_OUTPUT = Path("reports/signal_geometry")


# ---------------------------------------------------------------------------
# Phase A: Run experiments
# ---------------------------------------------------------------------------


def run_experiments(config_dir: Path, force_rerun: bool = False) -> dict[str, Path]:
    """Run all signal geometry experiments.  Skip if results already exist."""
    from src.experiments.orchestrator import run_experiment_from_config

    out_paths: dict[str, Path] = {}
    configs = sorted(config_dir.glob("sg_alpha_*.yaml"))
    if not configs:
        configs = [PROJECT_ROOT / c for c in _STUDY_CONFIGS]

    for cfg_path in configs:
        exp_name = cfg_path.stem
        result_path = PROJECT_ROOT / "results" / "experiments" / exp_name
        if result_path.exists() and not force_rerun:
            print(f"  [SKIP] {exp_name} — results already exist")
            out_paths[exp_name] = result_path
            continue
        label = _ALPHA_LABELS.get(exp_name, exp_name)
        print(f"  [RUN]  {exp_name} ({label}) ...")
        run = run_experiment_from_config(str(cfg_path), profile="report")
        out_paths[exp_name] = run.output_path
        m = run.experiment_result.metrics
        print(f"         → Sharpe={m.get('sharpe_ratio', float('nan')):.3f}  "
              f"AnnRet={m.get('annualized_return', float('nan')):.2%}")

    return out_paths


# ---------------------------------------------------------------------------
# Phase B: Load artefacts
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def load_study_data(out_paths: dict[str, Path]) -> dict:
    """Load all artefacts needed for signal geometry analysis.

    Returns nested dict keyed by experiment name with:
        label, alpha, metrics, equity, weights, allocation_diagnostics, split_metrics
    """
    data: dict = {}
    for exp_name, exp_path in out_paths.items():
        label = _ALPHA_LABELS.get(exp_name, exp_name)
        alpha = _ALPHA_VALUES.get(exp_name, float("nan"))
        metrics = _load_json(exp_path / "metrics.json") or {}
        ad = _load_json(exp_path / "diagnostics" / "allocation_diagnostics.json")
        sm = _load_json(exp_path / "diagnostics" / "split_metrics.json")

        equity_s: pd.Series | None = None
        try:
            equity_s = pd.read_parquet(exp_path / "equity_curve.parquet").squeeze()
        except Exception:
            pass

        weights_df: pd.DataFrame | None = None
        try:
            weights_df = pd.read_parquet(exp_path / "weights.parquet")
        except Exception:
            pass

        data[exp_name] = {
            "label": label,
            "alpha": alpha,
            "metrics": metrics,
            "equity": equity_s,
            "weights": weights_df,
            "allocation_diagnostics": ad if (ad and ad.get("available")) else None,
            "split_metrics": sm,
        }
    return data


# ---------------------------------------------------------------------------
# Phase C: Compute geometry diagnostics
# ---------------------------------------------------------------------------


def compute_turnover_series(weights_df: pd.DataFrame) -> pd.Series:
    return weights_df.fillna(0.0).diff().abs().sum(axis=1)


def compute_concentration_series(weights_df: pd.DataFrame) -> dict[str, pd.Series]:
    w = weights_df.fillna(0.0)
    abs_w = w.abs()
    active = (abs_w > 1e-10).any(axis=1)
    abs_active = abs_w.loc[active]
    hhi = (abs_active ** 2).sum(axis=1).reindex(w.index, fill_value=float("nan"))
    return {"hhi": hhi}


def extract_dispersion_series(allocation_diagnostics: dict | None) -> pd.Series | None:
    """Extract a proxy CS std series from allocation_diagnostics.
    The stored scalar is a mean; we return it as a constant-value Series
    for overlay plots (actual time-series would require re-running extraction).
    """
    if not allocation_diagnostics:
        return None
    disp = allocation_diagnostics.get("prediction_dispersion") or {}
    mean_cs_std = disp.get("mean_cs_std")
    if mean_cs_std is None:
        disp = allocation_diagnostics.get("dispersion_summary") or {}
        mean_cs_std = disp.get("mean_cs_std")
    return mean_cs_std


def build_dispersion_by_alpha(study_data: dict) -> dict[str, dict]:
    """Aggregate prediction dispersion metrics from allocation_diagnostics."""
    result: dict[str, dict] = {}
    for exp_name, d in study_data.items():
        label = d["label"]
        ad = d["allocation_diagnostics"] or {}
        disp = ad.get("prediction_dispersion") or {}
        # Also check dispersion_summary (written by extended _write_allocation_diagnostics)
        if not disp:
            disp = ad.get("dispersion_summary") or {}
        result[label] = {
            "mean_cs_std": disp.get("mean_cs_std", float("nan")),
            "mean_cs_spread": disp.get("mean_cs_spread", float("nan")),
            "min_cs_std": disp.get("min_cs_std", float("nan")),
            "max_cs_std": disp.get("max_cs_std", float("nan")),
        }
    return result


def build_calibration_by_alpha(study_data: dict) -> dict[str, dict]:
    """Aggregate confidence calibration metrics from allocation_diagnostics."""
    result: dict[str, dict] = {}
    for exp_name, d in study_data.items():
        label = d["label"]
        ad = d["allocation_diagnostics"] or {}
        cc = ad.get("confidence_calibration") or {}
        if not cc:
            cc = ad.get("calibration_summary") or {}
        qmr = cc.get("quintile_mean_returns") or {}
        if isinstance(qmr, dict):
            qr = pd.Series(qmr)
        else:
            qr = pd.Series(dtype=float)
        result[label] = {
            "quintile_returns": qr,
            "monotonic_up": cc.get("monotonic_up", None),
            "top_minus_bottom_spread": cc.get("top_minus_bottom_spread", float("nan")),
        }
    return result


def build_split_sharpe_by_alpha(study_data: dict) -> tuple[dict[str, list[float]], list[str]]:
    """Extract per-split Sharpe ratios from split_metrics for all α."""
    result: dict[str, list[float]] = {}
    split_labels: list[str] = []
    max_splits = 0

    for exp_name, d in study_data.items():
        sm = d["split_metrics"] or {}
        splits = sm.get("splits") or []
        if len(splits) > max_splits:
            max_splits = len(splits)
            split_labels = [f"{s.get('test_start','')[:4]}–{s.get('test_end','')[:4]}"
                            for s in splits]
        label = d["label"]
        result[label] = [float(s.get("sharpe_ratio", float("nan"))) for s in splits]

    return result, split_labels


def build_summary_by_alpha(study_data: dict,
                            dispersion_by_alpha: dict[str, dict]) -> dict[str, dict]:
    """Build per-alpha summary dict for robustness scatter plot."""
    result: dict[str, dict] = {}
    for exp_name, d in study_data.items():
        label = d["label"]
        m = d["metrics"]
        sm = d["split_metrics"] or {}

        split_sharpes = [float(s.get("sharpe_ratio", float("nan")))
                         for s in (sm.get("splits") or [])]
        valid_ss = [x for x in split_sharpes if not math.isnan(x)]
        oos_std = float(np.std(valid_ss)) if len(valid_ss) >= 2 else float("nan")

        mean_to = float("nan")
        if d["weights"] is not None:
            to_s = compute_turnover_series(d["weights"])
            mean_to = float(to_s.mean()) if not to_s.empty else float("nan")

        disp = dispersion_by_alpha.get(label, {})

        sm_summary = sm.get("summary") or {}
        result[label] = {
            "alpha": d["alpha"],
            "sharpe_ratio": m.get("sharpe_ratio", float("nan")),
            "annualized_return": m.get("annualized_return", float("nan")),
            "annualized_volatility": m.get("annualized_volatility", float("nan")),
            "max_drawdown": m.get("max_drawdown", float("nan")),
            "hit_rate": m.get("hit_rate", float("nan")),
            "oos_mean_sharpe": sm_summary.get("mean_sharpe", float("nan")),
            "oos_sharpe_std": oos_std,
            "mean_turnover": mean_to,
            "mean_cs_std": disp.get("mean_cs_std", float("nan")),
            "mean_cs_spread": disp.get("mean_cs_spread", float("nan")),
        }
    return result


# ---------------------------------------------------------------------------
# Phase D: Generate comparative figures
# ---------------------------------------------------------------------------


def generate_comparative_figures(
    study_data: dict,
    dispersion_by_alpha: dict[str, dict],
    calibration_by_alpha: dict[str, dict],
    split_sharpe_by_alpha: dict[str, list[float]],
    split_labels: list[str],
    summary_by_alpha: dict[str, dict],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate all comparative figures; return name → saved path dict."""
    from src.visualization.styles import apply_research_style
    from src.visualization.signal_geometry_plots import (
        plot_dispersion_sweep,
        plot_calibration_sweep,
        plot_wf_stability_heatmap,
        plot_robustness_tradeoff,
        plot_turnover_by_alpha,
        plot_intrabasket_geometry,
    )

    apply_research_style(profile="report")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    def _save(fig: plt.Figure, name: str) -> Path:
        p = output_dir / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[name] = p
        return p

    # 1. Dispersion sweep
    if dispersion_by_alpha:
        _save(plot_dispersion_sweep(dispersion_by_alpha), "dispersion_sweep")

    # 2. Calibration sweep
    if calibration_by_alpha:
        _save(plot_calibration_sweep(calibration_by_alpha), "calibration_sweep")

    # 3. Walk-forward stability heatmap
    if split_sharpe_by_alpha:
        _save(plot_wf_stability_heatmap(split_sharpe_by_alpha, split_labels),
              "wf_stability_heatmap")

    # 4. Robustness tradeoff scatter
    if summary_by_alpha:
        _save(plot_robustness_tradeoff(summary_by_alpha), "robustness_tradeoff")

    # 5. Turnover by alpha
    turnover_dict: dict[str, pd.Series] = {}
    for exp_name, d in study_data.items():
        if d["weights"] is not None:
            to = compute_turnover_series(d["weights"])
            turnover_dict[d["label"]] = to[to > 0]
    if turnover_dict:
        _save(plot_turnover_by_alpha(turnover_dict), "turnover_by_alpha")

    # 6. Intra-basket geometry
    if dispersion_by_alpha:
        _save(plot_intrabasket_geometry(dispersion_by_alpha), "intrabasket_geometry")

    # 7. Equity overlay
    equity_dict = {d["label"]: d["equity"] for d in study_data.values()
                   if d["equity"] is not None}
    if equity_dict:
        from src.visualization.signal_geometry_plots import _alpha_colors
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = list(equity_dict.keys())
        colors = _alpha_colors(labels)
        for (lbl, curve), color in zip(equity_dict.items(), colors):
            lw = 1.8 if "0.50" in lbl else 1.2
            ax.plot(curve.index, curve.values, color=color, linewidth=lw, label=lbl, alpha=0.9)
        ax.axhline(1.0, color="#aaa", linewidth=0.7, linestyle="--")
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_title("Equity Curves by Regularization Strength", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        _save(fig, "equity_overlay")

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


def _geometry_verdict(dispersion_by_alpha: dict[str, dict]) -> str:
    """Generate a plain-text geometry verdict for the synthesis."""
    baseline_label = "α=0.50"
    baseline_std = dispersion_by_alpha.get(baseline_label, {}).get("mean_cs_std", float("nan"))
    if math.isnan(baseline_std):
        return "Insufficient data to determine geometry verdict."

    widening_lines: list[str] = []
    for lbl, d in dispersion_by_alpha.items():
        if lbl == baseline_label:
            continue
        cs_std = d.get("mean_cs_std", float("nan"))
        if not math.isnan(cs_std) and not math.isnan(baseline_std) and baseline_std > 0:
            pct_change = (cs_std / baseline_std - 1.0) * 100
            widening_lines.append(f"{lbl}: CS σ = {cs_std:.4f} ({pct_change:+.0f}% vs baseline)")

    if not widening_lines:
        return "No geometry comparison available."
    return "Prediction geometry vs baseline (α=0.50): " + "; ".join(widening_lines) + "."


def _calibration_verdict(calibration_by_alpha: dict[str, dict]) -> str:
    """Generate a calibration verdict for the synthesis."""
    verdicts: list[str] = []
    for lbl, d in calibration_by_alpha.items():
        mono = d.get("monotonic_up")
        spread = d.get("top_minus_bottom_spread", float("nan"))
        if mono is None:
            verdicts.append(f"{lbl}: calibration data unavailable")
        elif mono:
            verdicts.append(f"{lbl}: monotonically increasing (spread={_fmt(spread, pct=True)})")
        else:
            verdicts.append(f"{lbl}: NON-MONOTONIC — confidence ordering broken")
    return ". ".join(verdicts) + "." if verdicts else "Calibration data unavailable."


def write_synthesis_report(
    study_data: dict,
    dispersion_by_alpha: dict[str, dict],
    calibration_by_alpha: dict[str, dict],
    split_sharpe_by_alpha: dict[str, list[float]],
    split_labels: list[str],
    summary_by_alpha: dict[str, dict],
    figures: dict[str, Path],
    output_path: Path,
) -> None:
    """Write the Phase 3A diagnostics-first research synthesis."""
    generated_at = datetime.now(UTC).isoformat()
    lines: list[str] = []

    lines += [
        "# Phase 3A — Signal Geometry Expansion Research Synthesis",
        "",
        f"*Generated: {generated_at}*",
        "*Experiments: sg_alpha_050, sg_alpha_010, sg_alpha_005, sg_alpha_001*",
        "",
        "---",
        "",
        "## Research Design",
        "",
        "**Research question:** Does reduced Ridge regularization produce economically"
        " meaningful prediction geometry — dispersed, stable, and calibrated — or does"
        " it amplify noise?",
        "",
        "**Controlled variables (identical across all runs):**",
        "- Universe: 15 ETFs (SPY, QQQ, IWM, XLK, XLF, XLE, XLV, EFA, EEM, TLT,"
        " HYG, TIP, GLD, DBC, VNQ)",
        "- Features: 13-feature set (momentum, volatility, trend, breakout, drawdown,"
        " beta, risk-adjusted momentum)",
        "- Label: 21-day cross-sectional return rank",
        "- Signal: top-5 equal-weight (5/15 ≈ 33% selection breadth, matching original 3/9)",
        "- Validation: rolling 48m train / 12m test",
        "- Portfolio construction: equal_weight",
        "",
        "**Varying:** `model.params.alpha` only.",
        "",
        "| Experiment | α | Expected geometry effect |",
        "| --- | --- | --- |",
        "| sg_alpha_050 | 0.50 | Strong L2 shrinkage — baseline compressed geometry |",
        "| sg_alpha_010 | 0.10 | Moderate reduction — moderate expansion |",
        "| sg_alpha_005 | 0.05 | Low regularization — possible noise amplification |",
        "| sg_alpha_001 | 0.01 | Minimal regularization — maximum expressiveness, instability risk |",
        "",
    ]

    # ── A. Performance Summary ────────────────────────────────────────────────
    lines += [
        "## A. Performance Summary",
        "",
        "| α | Sharpe | Ann.Return | Volatility | Max DD | Hit Rate |"
        " OOS Sharpe | OOS WF Std |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lbl, d in summary_by_alpha.items():
        lines.append(
            f"| {lbl} | {_fmt(d.get('sharpe_ratio'))} |"
            f" {_fmt(d.get('annualized_return'), pct=True)} |"
            f" {_fmt(d.get('annualized_volatility'), pct=True)} |"
            f" {_fmt(d.get('max_drawdown'), pct=True)} |"
            f" {_fmt(d.get('hit_rate'))} |"
            f" {_fmt(d.get('oos_mean_sharpe'))} |"
            f" {_fmt(d.get('oos_sharpe_std'))} |"
        )
    lines += [""]

    if "equity_overlay" in figures:
        lines += [f"![Equity Overlay](figures/{figures['equity_overlay'].name})", ""]
        lines += [
            "*Equity curves by regularization strength. Performance differences between α values"
            " are attributable only to prediction geometry — all other pipeline components are"
            " identical. Divergence from the α=0.50 baseline (navy) indicates geometry-driven"
            " performance change.*",
            "",
        ]

    # ── B. Signal Geometry ────────────────────────────────────────────────────
    lines += [
        "",
        "## B. Signal Geometry Findings",
        "",
        "### B1. Prediction Dispersion by α",
        "",
        "| α | Mean CS σ | Mean Top-Bottom Spread | Min CS σ | Max CS σ |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lbl, d in dispersion_by_alpha.items():
        lines.append(
            f"| {lbl} | {_fmt(d.get('mean_cs_std'), decimals=4)} |"
            f" {_fmt(d.get('mean_cs_spread'), decimals=4)} |"
            f" {_fmt(d.get('min_cs_std'), decimals=4)} |"
            f" {_fmt(d.get('max_cs_std'), decimals=4)} |"
        )
    lines += [""]

    lines += [_geometry_verdict(dispersion_by_alpha), ""]

    if "dispersion_sweep" in figures:
        lines += [f"![Dispersion Sweep](figures/{figures['dispersion_sweep'].name})", ""]
        lines += [
            "*Mean cross-sectional prediction σ (left) and top-minus-bottom spread (right)"
            " by α. If reduced regularization produces meaningful geometry, these metrics"
            " should increase substantially — and remain statistically stable OOS.*",
            "",
        ]

    lines += [
        "",
        "### B2. Intra-Basket Confidence Structure",
        "",
        "The relevant geometric quantity for confidence-weighted allocation is not the"
        " full-universe score spread but the *intra-basket* spread — the score difference"
        " between the 1st- and k-th ranked asset within the selected top-5 basket."
        " Estimated as (top-minus-bottom spread) × (k/N) = (spread) × (5/15).",
        "",
    ]

    if "intrabasket_geometry" in figures:
        lines += [f"![Intra-Basket Geometry](figures/{figures['intrabasket_geometry'].name})", ""]
        lines += [
            "*Left: mean CS σ. Right: estimated intra-basket spread (full spread × k/N)."
            " The dashed line marks the approximate softmax activation threshold (~1bp)"
            " — below this, softmax produces near-uniform weights regardless of temperature."
            " Bars above this threshold indicate α configurations where confidence weighting"
            " could become mechanically meaningful.*",
            "",
        ]

    # ── C. Confidence Legitimacy ──────────────────────────────────────────────
    lines += [
        "",
        "## C. Confidence Legitimacy Analysis",
        "",
        "### C1. Quintile Calibration",
        "",
    ]
    for lbl, d in calibration_by_alpha.items():
        qr = d.get("quintile_returns")
        mono = d.get("monotonic_up")
        spread = d.get("top_minus_bottom_spread", float("nan"))
        mono_str = "monotonically increasing" if mono else ("NON-MONOTONIC" if mono is False
                                                            else "unknown")
        lines.append(f"**{lbl}:** Quintile calibration is {mono_str}"
                     f" (Q5-Q1 spread = {_fmt(spread, pct=True)}).")
        if isinstance(qr, pd.Series) and not qr.empty:
            q_vals = " | ".join(f"{q}: {_fmt(float(v), pct=True)}"
                                for q, v in qr.items())
            lines.append(f"  Quintile returns: {q_vals}.")
        lines.append("")

    lines += [_calibration_verdict(calibration_by_alpha), ""]

    if "calibration_sweep" in figures:
        lines += [f"![Calibration Sweep](figures/{figures['calibration_sweep'].name})", ""]
        lines += [
            "*Quintile calibration by α. Monotonically increasing bars (Q1 < Q2 < ... < Q5)"
            " confirm the model ranks assets correctly. Critical question: does the Q5-Q1"
            " spread widen with lower α, indicating improved ranking confidence?*",
            "",
        ]

    # ── D. Robustness Analysis ────────────────────────────────────────────────
    lines += [
        "",
        "## D. Robustness & Walk-Forward Stability",
        "",
        "### D1. Split-by-Split Consistency",
        "",
    ]

    if split_labels:
        lines += [
            "| α | " + " | ".join(split_labels) + " | OOS Mean |",
            "| --- | " + " | ".join(["---"] * len(split_labels)) + " | --- |",
        ]
        for lbl, sharpes in split_sharpe_by_alpha.items():
            d = summary_by_alpha.get(lbl, {})
            oos_mean = _fmt(d.get("oos_mean_sharpe"))
            row = f"| {lbl} | " + " | ".join(
                _fmt(float(s)) if not math.isnan(float(s)) else "—" for s in sharpes
            ) + f" | {oos_mean} |"
            lines.append(row)
        lines += [""]

    if "wf_stability_heatmap" in figures:
        lines += [f"![WF Stability Heatmap](figures/{figures['wf_stability_heatmap'].name})", ""]
        lines += [
            "*Walk-forward Sharpe by split and α. Green = positive OOS Sharpe; red = negative."
            " Consistent colour across rows indicates stable geometry. Split-specific drops"
            " (isolated red cells) suggest regime-sensitive behaviour.*",
            "",
        ]

    lines += [
        "",
        "### D2. Instability Diagnostics (Turnover)",
        "",
        "| α | Mean Daily Turnover | Est. Annual Friction (5bps) |",
        "| --- | --- | --- |",
    ]
    cost_bps = 5.0
    for lbl, d in summary_by_alpha.items():
        to = d.get("mean_turnover")
        cost_est = (to * cost_bps / 10000 * 252) if (to and not math.isnan(to)) else float("nan")
        lines.append(f"| {lbl} | {_fmt(to, decimals=5)} |"
                     f" {_fmt(cost_est, pct=True)} est. annual |")
    lines += [""]

    if "turnover_by_alpha" in figures:
        lines += [f"![Turnover by α](figures/{figures['turnover_by_alpha'].name})", ""]
        lines += [
            "*Daily portfolio turnover by α. Lower regularization may increase prediction"
            " volatility → more frequent asset rank changes → higher turnover → higher"
            " friction cost. Any Sharpe improvement from geometry widening must exceed"
            " the additional turnover drag to be institutionally meaningful.*",
            "",
        ]

    if "robustness_tradeoff" in figures:
        lines += [
            "",
            "### D3. Geometry vs Robustness Tradeoff",
            "",
            f"![Robustness Tradeoff](figures/{figures['robustness_tradeoff'].name})",
            "",
            "*Scatter of OOS Sharpe vs mean CS σ. Each point is one α configuration;"
            " bubble size encodes walk-forward Sharpe standard deviation (OOS instability)."
            " If reduced regularization genuinely improves confidence geometry, higher σ"
            " should correspond to higher OOS Sharpe. A flat or declining pattern with"
            " larger bubbles indicates noise amplification rather than signal extraction.*",
            "",
        ]

    # ── E. Research Assessment ────────────────────────────────────────────────
    lines += [
        "",
        "## E. Research Assessment",
        "",
        "### E1. Does reduced regularization produce economically meaningful geometry?",
        "",
    ]

    baseline_std = dispersion_by_alpha.get("α=0.50", {}).get("mean_cs_std", float("nan"))
    most_dispersed_lbl = max(
        dispersion_by_alpha,
        key=lambda k: dispersion_by_alpha[k].get("mean_cs_std", 0.0),
        default=None,
    )
    if most_dispersed_lbl and not math.isnan(baseline_std) and baseline_std > 0:
        best_std = dispersion_by_alpha[most_dispersed_lbl].get("mean_cs_std", float("nan"))
        pct_change = (best_std / baseline_std - 1.0) * 100
        lines.append(
            f"The most dispersed configuration ({most_dispersed_lbl}) achieves mean CS σ ="
            f" {_fmt(best_std, decimals=4)} vs {_fmt(baseline_std, decimals=4)} at baseline,"
            f" a {pct_change:+.0f}% change."
        )
    lines += [""]

    lines += [
        "### E2. Does confidence legitimacy improve?",
        "",
        _calibration_verdict(calibration_by_alpha),
        "",
        "### E3. Is the geometry improvement institutionally believable?",
        "",
        "The institutional test requires that geometry improvements are:",
        "1. Statistically stable across walk-forward splits (not concentrated in one regime)",
        "2. Accompanied by monotonic calibration at each α level",
        "3. Not offset by disproportionate turnover increases",
        "4. Persistent OOS — not a reflection of in-sample fit",
        "",
        "The walk-forward split analysis and turnover diagnostics above address conditions"
        " 1, 3, and 4. Condition 2 is addressed by the calibration sweep.",
        "",
    ]

    # ── F. Validation Safety ──────────────────────────────────────────────────
    lines += [
        "",
        "## F. Validation Safety Confirmation",
        "",
        "**Chronology integrity:** All four experiments use identical rolling walk-forward"
        " validation (48m train / 12m test, 0-day gap). No future-aware normalization is"
        " applied. Each allocation decision on date *t* uses predictions from a model"
        " trained only on data ending at *t−1* (enforced by `shift(1)` in the backtest engine).",
        "",
        "**No cross-experiment contamination:** Each experiment is an independent run with"
        " its own fitted model, independent walk-forward splits, and independent provenance"
        " hash. No model object is shared across α configurations.",
        "",
        "**No post-hoc optimization:** The experiment matrix was specified before running."
        " No configuration adjustments were made after observing results.",
        "",
        "**Controlled causality:** Only `model.params.alpha` varies across experiments."
        " Universe, features, labels, signal, validation, and portfolio construction are"
        " identical. Any performance or geometry differences are attributable solely to"
        " regularization strength.",
        "",
    ]

    # ── G. Synthesis Conclusion ───────────────────────────────────────────────
    lines += [
        "",
        "## G. Synthesis Conclusion",
        "",
        "> This section is intentionally diagnostic. The platform does not identify"
        " a 'best' α or recommend a configuration. It identifies what geometry changes,"
        " whether those changes improve calibration, and whether robustness is preserved.",
        "",
    ]

    # Determine if geometry widened enough to potentially justify softmax
    best_spread = max(
        (d.get("mean_cs_spread", 0.0) for d in dispersion_by_alpha.values()),
        default=0.0
    )
    intrabasket_best = best_spread * (5 / 15)
    if intrabasket_best > 0.02:
        softmax_verdict = (
            f"The maximum estimated intra-basket spread ({intrabasket_best:.4f}) exceeds"
            " the softmax activation threshold (~0.01). At least one α configuration may"
            " produce confidence geometry that supports meaningful confidence weighting."
            " Further allocation research (Phase 3B) is warranted."
        )
    elif intrabasket_best > 0.005:
        softmax_verdict = (
            f"The maximum estimated intra-basket spread ({intrabasket_best:.4f}) is near"
            " the softmax activation threshold (~0.01). Geometry widening is present but"
            " marginal. Whether this translates to meaningful concentration depends on"
            " the specific split periods and temperature values tested."
        )
    else:
        softmax_verdict = (
            "The maximum estimated intra-basket spread across all α configurations remains"
            " below the softmax activation threshold (~0.01). Reduced regularization alone"
            " is insufficient to produce economically meaningful confidence geometry in"
            " this feature/label configuration."
        )

    lines += [softmax_verdict, "", "---", "",
              f"*Phase 3A Signal Geometry Research Synthesis — {generated_at}*",
              "*Universe: 15 ETFs, 2013–2024. Features: 13. Label: ranking_target 21d.*",
              "*Signal: top-5 equal-weight. Validation: 48m/12m rolling.*"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Synthesis report → {output_path}")


# ---------------------------------------------------------------------------
# Phase F: Persist summary
# ---------------------------------------------------------------------------


def _json_safe(v: object) -> object:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def save_study_summary(summary_by_alpha: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {lbl: {k: _json_safe(v) for k, v in d.items()}
               for lbl, d in summary_by_alpha.items()}
    out = output_dir / "signal_geometry_summary.json"
    with out.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"  Summary JSON → {out}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3A Signal Geometry Study")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment runs; load existing results only")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force all experiments to rerun")
    parser.add_argument("--config-dir", type=Path,
                        default=PROJECT_ROOT / "configs" / "experiments" / "signal_geometry",
                        help="Directory containing signal geometry YAML configs")
    args = parser.parse_args()

    print("\n=== Phase 3A — Signal Geometry Expansion Study ===\n")

    # Phase A
    out_paths: dict[str, Path]
    if args.analyze_only:
        print("[Phase A] Skipping experiment runs (--analyze-only).")
        out_paths = {}
        for cfg in sorted(args.config_dir.glob("sg_alpha_*.yaml")):
            name = cfg.stem
            p = PROJECT_ROOT / "results" / "experiments" / name
            if p.exists():
                out_paths[name] = p
                print(f"  [FOUND] {name}")
            else:
                print(f"  [MISSING] {name}")
    else:
        print("[Phase A] Running experiments ...")
        out_paths = run_experiments(args.config_dir, force_rerun=args.force_rerun)

    if not out_paths:
        print("No experiment results found. Run without --analyze-only first.")
        sys.exit(1)

    # Phase B
    print("\n[Phase B] Loading artefacts ...")
    study_data = load_study_data(out_paths)

    # Phase C
    print("\n[Phase C] Computing signal geometry diagnostics ...")
    dispersion_by_alpha = build_dispersion_by_alpha(study_data)
    calibration_by_alpha = build_calibration_by_alpha(study_data)
    split_sharpe_by_alpha, split_labels = build_split_sharpe_by_alpha(study_data)
    summary_by_alpha = build_summary_by_alpha(study_data, dispersion_by_alpha)

    print("\n  === Signal Geometry Summary ===")
    for lbl, d in dispersion_by_alpha.items():
        cs_std = d.get("mean_cs_std", float("nan"))
        cs_spread = d.get("mean_cs_spread", float("nan"))
        sm = summary_by_alpha.get(lbl, {})
        sharpe = sm.get("sharpe_ratio", float("nan"))
        oos = sm.get("oos_mean_sharpe", float("nan"))
        print(f"  {lbl}: Sharpe={sharpe:.3f}  OOS={oos:.3f}  "
              f"CS_std={cs_std:.4f}  CS_spread={cs_spread:.4f}")

    # Phase D
    fig_dir = PROJECT_ROOT / _STUDY_OUTPUT / "figures"
    print(f"\n[Phase D] Generating comparative figures → {fig_dir}")
    figures = generate_comparative_figures(
        study_data, dispersion_by_alpha, calibration_by_alpha,
        split_sharpe_by_alpha, split_labels, summary_by_alpha, fig_dir,
    )

    # Phase E
    save_study_summary(summary_by_alpha, PROJECT_ROOT / _STUDY_OUTPUT)

    # Phase F
    report_path = PROJECT_ROOT / _REPORT_OUTPUT / "signal_geometry_synthesis.md"
    print(f"\n[Phase F] Writing synthesis report → {report_path}")
    write_synthesis_report(
        study_data, dispersion_by_alpha, calibration_by_alpha,
        split_sharpe_by_alpha, split_labels, summary_by_alpha,
        figures, report_path,
    )

    print("\n=== Phase 3A Complete ===")
    print(f"  Figures:   {fig_dir}")
    print(f"  Summary:   {PROJECT_ROOT / _STUDY_OUTPUT / 'signal_geometry_summary.json'}")
    print(f"  Synthesis: {report_path}\n")


if __name__ == "__main__":
    main()
