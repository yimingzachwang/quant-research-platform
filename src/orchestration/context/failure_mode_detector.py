"""Detects common strategy failure modes from diagnostic artefacts.

Returns a list of FailureMode objects sorted by severity.  All detection is
deterministic and rule-based — no LLM involvement.
"""

from __future__ import annotations

from typing import Any

from src.orchestration.api.schemas import FailureMode
from src.orchestration.context.context_schema import (
    DRAWDOWN_SEVERE,
    IC_MEANINGFUL,
    OOS_HIT_RATE_WEAK,
    SEVERITY_CRITICAL,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)

_SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}


def detect_failure_modes(
    metrics: dict[str, Any] | None = None,
    split_metrics: dict[str, Any] | None = None,
    ml_model_diagnostics: dict[str, Any] | None = None,
    backtest_diagnostics: dict[str, Any] | None = None,
    alignment_diagnostics: dict[str, Any] | None = None,
) -> list[FailureMode]:
    """Run all failure-mode detectors and return sorted results."""
    modes: list[FailureMode] = []

    if metrics:
        modes.extend(_check_performance(metrics))
    if split_metrics:
        modes.extend(_check_validation(split_metrics))
    if ml_model_diagnostics:
        modes.extend(_check_ml_signal(ml_model_diagnostics))
    if alignment_diagnostics:
        modes.extend(_check_alignment(alignment_diagnostics))
    if backtest_diagnostics:
        modes.extend(_check_backtest(backtest_diagnostics))

    return sorted(modes, key=lambda m: _SEVERITY_ORDER.get(m.severity, 99))


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _check_performance(metrics: dict[str, Any]) -> list[FailureMode]:
    modes = []
    sharpe = metrics.get("sharpe_ratio")
    dd = metrics.get("max_drawdown")
    ret = metrics.get("annualized_return")

    if sharpe is not None and sharpe < 0:
        modes.append(FailureMode(
            name="negative_sharpe",
            severity=SEVERITY_CRITICAL,
            description="Full-period Sharpe ratio is negative — strategy destroys risk-adjusted value.",
            evidence={"sharpe_ratio": sharpe},
        ))
    elif sharpe is not None and sharpe < 0.3:
        modes.append(FailureMode(
            name="weak_sharpe",
            severity=SEVERITY_WARNING,
            description=f"Sharpe ratio {sharpe:.3f} is below 0.30 — marginal risk-adjusted return.",
            evidence={"sharpe_ratio": sharpe},
        ))

    if dd is not None and dd <= DRAWDOWN_SEVERE:
        modes.append(FailureMode(
            name="severe_drawdown",
            severity=SEVERITY_CRITICAL,
            description=f"Max drawdown {dd * 100:.1f}% exceeds -40% — severe capital impairment risk.",
            evidence={"max_drawdown": dd},
        ))

    if ret is not None and ret < 0:
        modes.append(FailureMode(
            name="negative_return",
            severity=SEVERITY_CRITICAL,
            description="Annualized return is negative — strategy loses capital in expectation.",
            evidence={"annualized_return": ret},
        ))

    return modes


def _check_validation(split_metrics: dict[str, Any]) -> list[FailureMode]:
    modes = []
    summary = split_metrics.get("summary", {})
    splits = split_metrics.get("splits", [])

    hit_rate = summary.get("hit_rate_positive_sharpe")
    if hit_rate is not None and hit_rate < OOS_HIT_RATE_WEAK:
        modes.append(FailureMode(
            name="poor_oos_consistency",
            severity=SEVERITY_CRITICAL,
            description=(
                f"OOS Sharpe hit rate {hit_rate * 100:.0f}% is below 50% — "
                "signal is not consistent across walk-forward splits."
            ),
            evidence={"hit_rate_positive_sharpe": hit_rate},
        ))

    mean_sharpe = summary.get("mean_sharpe")
    std_sharpe = summary.get("std_sharpe")
    if mean_sharpe is not None and std_sharpe is not None and std_sharpe > 0:
        cv = abs(std_sharpe / mean_sharpe) if mean_sharpe != 0 else float("inf")
        if cv > 3.0:
            modes.append(FailureMode(
                name="high_split_sharpe_variance",
                severity=SEVERITY_WARNING,
                description=(
                    f"Coefficient of variation {cv:.1f}x for OOS Sharpe across splits — "
                    "results are highly regime-dependent."
                ),
                evidence={"mean_sharpe": mean_sharpe, "std_sharpe": std_sharpe},
            ))

    # split records use "sharpe_ratio" (the OOS test-window Sharpe)
    sharpe_vals = [s.get("sharpe_ratio") for s in splits if s.get("sharpe_ratio") is not None]
    if sharpe_vals:
        worst = min(sharpe_vals)
        if worst < -1.0:
            modes.append(FailureMode(
                name="catastrophic_split",
                severity=SEVERITY_CRITICAL,
                description=f"Worst split Sharpe {worst:.2f} — at least one period is catastrophic.",
                evidence={"worst_split_sharpe": worst},
            ))

    return modes


def _check_ml_signal(ml_model_diagnostics: dict[str, Any]) -> list[FailureMode]:
    modes = []
    ic_summary = ml_model_diagnostics.get("ic_summary", {})
    coef_stability_list = ml_model_diagnostics.get("coefficient_stability_summary") or []
    if isinstance(coef_stability_list, dict):
        coef_stability_list = []

    mean_ic = ic_summary.get("mean_ic")
    if mean_ic is not None and mean_ic < IC_MEANINGFUL:
        sev = SEVERITY_CRITICAL if mean_ic <= 0 else SEVERITY_WARNING
        modes.append(FailureMode(
            name="weak_ic",
            severity=sev,
            description=(
                f"Mean IC {mean_ic:.4f} is below meaningful threshold ({IC_MEANINGFUL}) — "
                "model predictions lack directional accuracy."
            ),
            evidence={"mean_ic": mean_ic},
        ))

    n_reversals = sum(1 for x in coef_stability_list if x.get("sign_consistency", 1) < 0.5)
    n_total = len(coef_stability_list)
    if n_total and n_total > 0:
        reversal_pct = n_reversals / n_total
        if reversal_pct > 0.5:
            modes.append(FailureMode(
                name="coefficient_instability",
                severity=SEVERITY_WARNING,
                description=(
                    f"{n_reversals}/{n_total} features ({reversal_pct * 100:.0f}%) "
                    "exhibit sign reversals across splits — learned relationships are unstable."
                ),
                evidence={"n_sign_reversal_features": n_reversals, "n_features": n_total},
            ))

    da = ml_model_diagnostics.get("directional_accuracy")
    if da is not None and da < 0.48:
        modes.append(FailureMode(
            name="poor_directional_accuracy",
            severity=SEVERITY_WARNING,
            description=f"Directional accuracy {da * 100:.1f}% — model predicts return direction correctly less than 48% of the time.",
            evidence={"directional_accuracy": da},
        ))

    return modes


def _check_alignment(alignment_diagnostics: dict[str, Any]) -> list[FailureMode]:
    modes = []
    loss_pct = alignment_diagnostics.get("alignment_loss_pct")
    if loss_pct is not None and loss_pct > 30:
        modes.append(FailureMode(
            name="high_alignment_loss",
            severity=SEVERITY_WARNING,
            description=(
                f"Feature-return alignment dropped {loss_pct:.1f}% of samples — "
                "significant data loss during label construction."
            ),
            evidence={"alignment_loss_pct": loss_pct},
        ))
    return modes


def _check_backtest(backtest_diagnostics: dict[str, Any]) -> list[FailureMode]:
    modes = []
    raw_turnover = backtest_diagnostics.get("monthly_avg_turnover")
    # monthly_avg_turnover may be a list of {date, value} dicts or a scalar
    if isinstance(raw_turnover, list) and raw_turnover:
        vals = [r.get("value", 0) for r in raw_turnover if isinstance(r, dict)]
        turnover = sum(vals) / len(vals) if vals else None
    elif isinstance(raw_turnover, (int, float)):
        turnover = raw_turnover
    else:
        turnover = None
    if turnover is not None and turnover > 1.5:
        modes.append(FailureMode(
            name="high_turnover",
            severity=SEVERITY_INFO,
            description=(
                f"Monthly average turnover {turnover:.2f}x — high transaction cost drag likely. "
                "Verify net-of-cost performance."
            ),
            evidence={"monthly_avg_turnover": turnover},
        ))
    return modes
