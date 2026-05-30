"""Lightweight vol-regime classification for ML research interpretation.

Derives simple, interpretable market-condition tags from existing experimental
data — no external databases, no latent-state models, no probabilistic engines.

The single public function compute_regime_stats() is called once by the
orchestrator after _prepare_ml_diagnostics().  It returns a flat dict consumed
by the visualization layer (plot_ic_by_vol_regime) and the markdown reporter
(_regime_conditional_behavior).

Regime dimensions implemented
------------------------------
vol_regime (split-level):
    Each walk-forward test window is classified "high_vol" or "low_vol" by
    comparing its mean cross-asset 21D realised vol to the median across all
    test windows.  Median split → roughly equal regime populations, unlike the
    existing stress_mask (2σ threshold) which flags only extreme events.

family_ic_by_regime:
    The (n_splits × n_features) IC DataFrame is partitioned by vol regime.
    Feature ICs within each regime are aggregated to the family level using
    get_family_for_name() from src.features.families.

No regime dimension requires refitting, reloading prices beyond the existing
in-memory panel, or introducing any new configuration parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.validation.walk_forward import WalkForwardResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_vol_levels(
    prices: pd.DataFrame,
    wf: WalkForwardResult,
) -> dict[int, str]:
    """Classify each WF split's test period as 'high_vol' or 'low_vol'.

    Computes mean cross-asset 21D rolling annualised vol during each split's
    test window, then classifies splits above/below the median vol across all
    test windows.  Returns {} when fewer than 2 splits have price data.
    """
    rets = prices.pct_change()
    rolling_vol = rets.rolling(21).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean(axis=1)

    split_vols: dict[int, float] = {}
    for sr in wf.splits:
        s = sr.split
        test_vol = avg_vol.loc[s.test_start : s.test_end].dropna()
        if len(test_vol) > 0:
            split_vols[s.split_index] = float(test_vol.mean())

    if len(split_vols) < 2:
        return {}

    median_vol = float(np.median(list(split_vols.values())))
    return {
        idx: ("high_vol" if v > median_vol else "low_vol")
        for idx, v in split_vols.items()
    }


def _family_ic_by_regime(
    feature_ic_df: pd.DataFrame,
    split_vol_levels: dict[int, str],
) -> dict[str, dict[str, float]]:
    """Mean IC per feature family per vol regime.

    Args:
        feature_ic_df:    (n_splits × n_features) IC DataFrame.
                          Index = split_index (int).
        split_vol_levels: {split_index: 'high_vol' | 'low_vol'}.

    Returns:
        {"high_vol": {"Trend": 0.12, ...}, "low_vol": {"Trend": 0.05, ...}}
        Empty inner dicts when no splits exist for that regime.
    """
    try:
        from src.features.families import get_family_for_name
    except Exception:
        return {}

    result: dict[str, dict[str, float]] = {"high_vol": {}, "low_vol": {}}

    for regime in ("high_vol", "low_vol"):
        split_ids = [idx for idx, r in split_vol_levels.items() if r == regime]
        if not split_ids:
            continue

        subset = feature_ic_df.loc[
            feature_ic_df.index.isin(split_ids)
        ].dropna(how="all")
        if subset.empty:
            continue

        # Accumulate IC values per family
        family_vals: dict[str, list[float]] = {}
        for col in subset.columns:
            fam = get_family_for_name(col)
            if fam and fam != "Other":
                vals = subset[col].dropna().tolist()
                family_vals.setdefault(fam, []).extend(vals)

        result[regime] = {
            fam: float(np.mean(vals))
            for fam, vals in family_vals.items()
            if vals
        }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_regime_stats(
    prices: pd.DataFrame,
    wf: Any,
    ml_data: dict,
) -> dict:
    """Compute lightweight vol-regime statistics for report interpretation.

    Called once in the orchestrator after _prepare_ml_diagnostics().
    All sub-operations are best-effort; failures leave the key absent.

    Args:
        prices:  Full aligned price panel (dates × assets).
        wf:      WalkForwardResult or None.
        ml_data: Pre-computed diagnostics dict from _prepare_ml_diagnostics().

    Returns:
        {
            "split_vol_levels":  {split_idx: "high_vol" | "low_vol"},
            "high_vol_frac":     float — fraction of splits classified high-vol,
            "n_high_vol_splits": int,
            "n_low_vol_splits":  int,
            "family_ic_by_regime": {
                "high_vol": {"Trend": float, ...},
                "low_vol":  {"Trend": float, ...},
            },
            "dominant_family": {
                "high_vol": str,  # family with highest mean IC in high-vol splits
                "low_vol":  str,
            },
        }
    """
    result: dict = {}

    if wf is None or not hasattr(wf, "splits") or not wf.splits:
        return result

    # --- Vol regime classification at split level ---
    try:
        split_levels = _split_vol_levels(prices, wf)
        if split_levels:
            result["split_vol_levels"] = split_levels
            n_high = sum(1 for v in split_levels.values() if v == "high_vol")
            n_low = len(split_levels) - n_high
            result["n_high_vol_splits"] = n_high
            result["n_low_vol_splits"] = n_low
            result["high_vol_frac"] = n_high / len(split_levels)
    except Exception:
        pass

    # --- Feature family IC disaggregated by vol regime ---
    feature_ic_df = ml_data.get("feature_ic_splits")
    split_levels = result.get("split_vol_levels", {})
    if feature_ic_df is not None and not feature_ic_df.empty and split_levels:
        try:
            family_ic = _family_ic_by_regime(feature_ic_df, split_levels)
            # Only store if at least one regime has data
            if any(family_ic.values()):
                result["family_ic_by_regime"] = family_ic

                # Dominant family: highest mean IC per regime
                dominant: dict[str, str] = {}
                for regime, fam_ics in family_ic.items():
                    if fam_ics:
                        dominant[regime] = max(fam_ics, key=fam_ics.get)
                if dominant:
                    result["dominant_family"] = dominant
        except Exception:
            pass

    return result
