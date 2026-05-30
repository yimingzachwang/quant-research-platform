"""Summarizes ML model diagnostics for LLM context."""

from __future__ import annotations

from typing import Any

from src.orchestration.context.context_schema import (
    DA_STRONG,
    DA_WEAK,
    IC_MEANINGFUL,
    IC_STRONG,
)


def summarize_ml_diagnostics(
    ml_model_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert ml_model_diagnostics.json into a structured ML summary."""
    if not ml_model_diagnostics:
        return {"available": False}

    model_type = ml_model_diagnostics.get("model_type")
    ic_summary = ml_model_diagnostics.get("ic_summary", {})
    # coefficient_stability_summary is a list of per-feature dicts
    coef_stability_list = ml_model_diagnostics.get("coefficient_stability_summary") or []
    if isinstance(coef_stability_list, dict):
        coef_stability_list = []
    ml_model_diagnostics.get("prediction_strength", {})
    ranking_geom = ml_model_diagnostics.get("ranking_geometry", {})
    fc = ml_model_diagnostics.get("feature_contributions", {})

    mean_ic = ic_summary.get("mean_ic")
    ic_t_stat = ic_summary.get("ic_t_stat")  # may be absent
    pct_positive = ic_summary.get("pct_positive_ic") or ic_summary.get("pct_positive_months")
    da = ml_model_diagnostics.get("directional_accuracy")

    n_stable_features = sum(
        1 for x in coef_stability_list if x.get("sign_consistency", 0) >= 0.7
    )
    n_sign_reversals = sum(
        1 for x in coef_stability_list if x.get("sign_consistency", 0) < 0.5
    )

    mean_hhi = fc.get("mean_hhi")
    dominant_family = fc.get("dominant_family")
    n_transitions = fc.get("n_family_transitions")
    most_volatile = fc.get("most_volatile_feature")

    return {
        "available": True,
        "model_type": model_type,
        "ic": {
            "mean_ic": _round(mean_ic),
            "ic_t_stat": _round(ic_t_stat),
            "pct_positive_months_pct": _pct(pct_positive),
            "ic_tier": _ic_tier(mean_ic),
        },
        "directional_accuracy": {
            "value_pct": _pct(da),
            "tier": _da_tier(da),
        },
        "coefficient_stability": {
            "n_features": len(coef_stability_list),
            "n_stable_features": n_stable_features,
            "n_sign_reversal_features": n_sign_reversals,
        },
        "feature_contributions": {
            "dominant_family": dominant_family,
            "n_family_transitions": n_transitions,
            "mean_hhi": _round(mean_hhi),
            "most_volatile_feature": most_volatile,
            "concentration_tier": _hhi_tier(mean_hhi),
        },
        "ranking_geometry": {
            "mean_score_iqr": _round(ranking_geom.get("mean_score_iqr")),
            "mean_rank_autocorr": _round(ranking_geom.get("mean_rank_autocorr")),
            "mean_realized_spread_pct": _pct(ranking_geom.get("mean_realized_spread")),
        },
    }


def summarize_feature_context(
    feature_summary: dict[str, Any] | None,
    feature_families: dict[str, Any] | None,
) -> dict[str, Any]:
    """Produce a lightweight feature context block for LLM consumption."""
    if not feature_summary and not feature_families:
        return {}

    # feature_families.json has shape {"families": {...}, "n_families": N, ...}
    fam_map: dict[str, list] = {}
    if feature_families:
        raw = feature_families.get("families") if isinstance(feature_families, dict) else feature_families
        if isinstance(raw, dict):
            fam_map = {k: v for k, v in raw.items() if isinstance(v, list)}

    families = list(fam_map.keys())
    n_features_by_family = {fam: len(feats) for fam, feats in fam_map.items()}

    n_rows = feature_summary.get("n_rows_before_alignment") if feature_summary else None
    n_rows_after = feature_summary.get("n_rows_after_alignment") if feature_summary else None

    return {
        "feature_families": families,
        "n_features_by_family": n_features_by_family,
        "total_features": sum(n_features_by_family.values()),
        "n_sample_rows_before_alignment": n_rows,
        "n_sample_rows_after_alignment": n_rows_after,
    }


def _pct(v: float | None) -> str | None:
    if v is None:
        return None
    return f"{v * 100:.2f}%"


def _round(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v, 4)


def _ic_tier(ic: float | None) -> str:
    if ic is None:
        return "unknown"
    if ic >= IC_STRONG:
        return "strong"
    if ic >= IC_MEANINGFUL:
        return "meaningful"
    if ic > 0:
        return "marginal"
    return "negative"


def _da_tier(da: float | None) -> str:
    if da is None:
        return "unknown"
    if da >= DA_STRONG:
        return "strong"
    if da >= DA_WEAK:
        return "marginal"
    return "weak"


def _hhi_tier(hhi: float | None) -> str:
    if hhi is None:
        return "unknown"
    if hhi >= 0.50:
        return "highly_concentrated"
    if hhi >= 0.30:
        return "moderately_concentrated"
    return "diversified"
