"""Tests for orchestration.context using real canonical experiment artefacts."""

import pytest

from src.orchestration.context.context_builder import build_context
from src.orchestration.context.metric_summarizer import summarize_metrics
from src.orchestration.context.validation_summarizer import summarize_validation
from src.orchestration.context.ml_diagnostic_summarizer import (
    summarize_ml_diagnostics,
    summarize_feature_context,
)
from src.orchestration.context.failure_mode_detector import detect_failure_modes


# ---------------------------------------------------------------------------
# Unit tests for individual summarizers
# ---------------------------------------------------------------------------


def test_summarize_metrics_empty():
    result = summarize_metrics({})
    assert result == {}


def test_summarize_metrics_basic():
    m = {
        "sharpe_ratio": 0.55,
        "annualized_return": 0.10,
        "annualized_volatility": 0.18,
        "max_drawdown": -0.25,
        "calmar_ratio": 0.40,
        "hit_rate": 0.52,
    }
    result = summarize_metrics(m)
    assert result["sharpe_tier"] == "good"
    assert result["drawdown_severity"] == "elevated"
    assert "9.76%" in result["annualized_return_pct"] or "10.00%" in result["annualized_return_pct"]


def test_sharpe_tiers():
    assert summarize_metrics({"sharpe_ratio": 1.5})["sharpe_tier"] == "excellent"
    assert summarize_metrics({"sharpe_ratio": 0.7})["sharpe_tier"] == "good"
    assert summarize_metrics({"sharpe_ratio": 0.2})["sharpe_tier"] == "weak"
    assert summarize_metrics({"sharpe_ratio": -0.1})["sharpe_tier"] == "negative"


def test_summarize_validation_none():
    result = summarize_validation(None)
    assert result == {"available": False}


def test_summarize_validation_basic():
    split_metrics = {
        "n_splits": 5,
        "summary": {
            "n_splits": 5,
            "mean_sharpe": 0.6,
            "std_sharpe": 0.4,
            "hit_rate_positive_sharpe": 0.8,
            "mean_annualized_return": 0.09,
            "worst_max_drawdown": -0.30,
        },
        "splits": [
            {"oos_sharpe": 0.5},
            {"oos_sharpe": 0.8},
            {"oos_sharpe": 0.3},
            {"oos_sharpe": 0.9},
            {"oos_sharpe": 0.4},
        ],
    }
    result = summarize_validation(split_metrics)
    assert result["available"] is True
    assert result["n_splits"] == 5
    assert result["consistency_tier"] == "strong"
    assert result["n_negative_sharpe_splits"] == 0


def test_summarize_ml_diagnostics_none():
    result = summarize_ml_diagnostics(None)
    assert result == {"available": False}


def test_detect_failure_modes_no_failures():
    modes = detect_failure_modes(
        metrics={"sharpe_ratio": 1.2, "annualized_return": 0.12, "max_drawdown": -0.15}
    )
    assert modes == []


def test_detect_failure_modes_negative_sharpe():
    modes = detect_failure_modes(metrics={"sharpe_ratio": -0.3})
    names = [m.name for m in modes]
    assert "negative_sharpe" in names


def test_detect_failure_modes_severe_drawdown():
    modes = detect_failure_modes(metrics={"sharpe_ratio": 0.5, "max_drawdown": -0.50})
    names = [m.name for m in modes]
    assert "severe_drawdown" in names


def test_detect_failure_modes_weak_oos():
    modes = detect_failure_modes(split_metrics={
        "summary": {"hit_rate_positive_sharpe": 0.3},
        "splits": [],
    })
    names = [m.name for m in modes]
    assert "poor_oos_consistency" in names


def test_failure_modes_sorted_by_severity():
    modes = detect_failure_modes(
        metrics={"sharpe_ratio": -0.3, "max_drawdown": -0.50},
        split_metrics={"summary": {"hit_rate_positive_sharpe": 0.3}, "splits": []},
    )
    critical = [m for m in modes if m.severity == "critical"]
    warnings = [m for m in modes if m.severity == "warning"]
    # All critical before all warnings in returned list
    if critical and warnings:
        last_critical_idx = max(modes.index(m) for m in critical)
        first_warning_idx = min(modes.index(m) for m in warnings)
        assert last_critical_idx < first_warning_idx


# ---------------------------------------------------------------------------
# Integration tests against real canonical experiment data
# ---------------------------------------------------------------------------


def test_build_context_canonical_multi_asset():
    ctx = build_context("canonical_ml_multi_asset")
    assert ctx.experiment_name == "canonical_ml_multi_asset"
    assert ctx.strategy_name != ""
    assert ctx.performance.get("sharpe_ratio") is not None
    assert ctx.validation.get("available") is True
    assert ctx.ml_diagnostics.get("available") is True
    assert len(ctx.available_plots) > 0


def test_build_context_failure_modes_canonical():
    ctx = build_context("canonical_ml_multi_asset")
    # severe_drawdown should be flagged (max_dd ~-45%)
    names = [fm["name"] for fm in ctx.failure_modes]
    assert "severe_drawdown" in names


def test_build_context_feature_summary():
    ctx = build_context("canonical_ml_multi_asset")
    assert "feature_families" in ctx.feature_summary
    families = ctx.feature_summary["feature_families"]
    assert "Trend" in families


def test_build_context_ml_diagnostics_tiers():
    ctx = build_context("canonical_ml_multi_asset")
    ic = ctx.ml_diagnostics.get("ic", {})
    assert ic.get("ic_tier") in ("marginal", "meaningful", "strong", "negative", "unknown")


def test_build_context_showcase():
    ctx = build_context("canonical_ml_showcase")
    assert ctx.experiment_name == "canonical_ml_showcase"
    assert ctx.ml_diagnostics.get("available") is True


# ---------------------------------------------------------------------------
# Fix 3 — per_split_sharpes extraction regression
# ---------------------------------------------------------------------------


def test_per_split_sharpes_non_empty():
    """per_split_sharpes must be populated from splits[].sharpe_ratio."""
    ctx = build_context("canonical_ml_multi_asset")
    per_split = ctx.validation.get("per_split_sharpes", [])
    assert len(per_split) > 0, "per_split_sharpes must not be empty"


def test_per_split_sharpes_count_matches_n_splits():
    ctx = build_context("canonical_ml_multi_asset")
    n_splits = ctx.validation.get("n_splits", 0)
    per_split = ctx.validation.get("per_split_sharpes", [])
    assert len(per_split) == n_splits


def test_per_split_sharpes_are_floats():
    ctx = build_context("canonical_ml_multi_asset")
    for v in ctx.validation.get("per_split_sharpes", []):
        assert isinstance(v, float), f"Expected float, got {type(v)}: {v}"


def test_n_negative_sharpe_splits_correct():
    """n_negative_sharpe_splits must be derived from actual split Sharpes."""
    ctx = build_context("canonical_ml_multi_asset")
    per_split = ctx.validation.get("per_split_sharpes", [])
    expected_negatives = sum(1 for v in per_split if v < 0)
    assert ctx.validation.get("n_negative_sharpe_splits") == expected_negatives


def test_catastrophic_split_detector_uses_sharpe_ratio():
    """catastrophic_split failure mode must fire when a split Sharpe < -1.0."""
    from src.orchestration.context.failure_mode_detector import detect_failure_modes

    split_metrics = {
        "summary": {"hit_rate_positive_sharpe": 0.4},
        "splits": [
            {"sharpe_ratio": -1.5},
            {"sharpe_ratio": 0.8},
        ],
    }
    modes = detect_failure_modes(split_metrics=split_metrics)
    names = [m.name for m in modes]
    assert "catastrophic_split" in names


# ---------------------------------------------------------------------------
# Fix 4 — universe extraction regression
# ---------------------------------------------------------------------------


def test_universe_tickers_populated():
    """asset_tickers must be non-empty for the canonical multi-asset experiment."""
    ctx = build_context("canonical_ml_multi_asset")
    tickers = ctx.universe_summary.get("asset_tickers", [])
    assert len(tickers) > 0, "universe asset_tickers must not be empty"


def test_universe_n_assets_correct():
    ctx = build_context("canonical_ml_multi_asset")
    assert ctx.universe_summary.get("n_assets", 0) > 0


def test_universe_tickers_are_strings():
    ctx = build_context("canonical_ml_multi_asset")
    for ticker in ctx.universe_summary.get("asset_tickers", []):
        assert isinstance(ticker, str) and len(ticker) > 0


def test_universe_coverage_pct_present():
    ctx = build_context("canonical_ml_multi_asset")
    assert ctx.universe_summary.get("mean_coverage_pct") is not None


def test_universe_tickers_serialized_in_context():
    """Universe tickers must survive context serialization."""
    from src.orchestration.context.context_builder import _context_to_dict

    ctx = build_context("canonical_ml_multi_asset")
    d = _context_to_dict(ctx)
    tickers = d.get("universe_summary", {}).get("asset_tickers", [])
    assert len(tickers) > 0


# ---------------------------------------------------------------------------
# Fix 5 — null-field pruning regression
# ---------------------------------------------------------------------------


def test_context_dict_has_no_none_values():
    """No None value must appear anywhere in the serialized context dict."""
    from src.orchestration.context.context_builder import _context_to_dict

    def _find_nones(obj, path=""):
        if obj is None:
            return [path]
        if isinstance(obj, dict):
            found = []
            for k, v in obj.items():
                found.extend(_find_nones(v, f"{path}.{k}"))
            return found
        if isinstance(obj, list):
            found = []
            for i, item in enumerate(obj):
                found.extend(_find_nones(item, f"{path}[{i}]"))
            return found
        return []

    ctx = build_context("canonical_ml_multi_asset")
    d = _context_to_dict(ctx)
    none_paths = _find_nones(d)
    assert none_paths == [], f"None values found at: {none_paths}"


def test_context_dict_has_no_empty_dicts():
    from src.orchestration.context.context_builder import _context_to_dict

    def _find_empty_dicts(obj, path=""):
        if isinstance(obj, dict):
            found = []
            if not obj and path:
                found.append(path)
            for k, v in obj.items():
                found.extend(_find_empty_dicts(v, f"{path}.{k}"))
            return found
        if isinstance(obj, list):
            found = []
            for i, item in enumerate(obj):
                found.extend(_find_empty_dicts(item, f"{path}[{i}]"))
            return found
        return []

    ctx = build_context("canonical_ml_multi_asset")
    d = _context_to_dict(ctx)
    empty_dict_paths = _find_empty_dicts(d)
    assert empty_dict_paths == [], f"Empty dicts found at: {empty_dict_paths}"


def test_prune_nulls_removes_none():
    from src.orchestration.context.context_builder import _prune_nulls

    result = _prune_nulls({"a": 1, "b": None, "c": {"d": None, "e": 2}})
    assert "b" not in result
    assert "d" not in result["c"]
    assert result["a"] == 1
    assert result["c"]["e"] == 2


def test_prune_nulls_removes_empty_collections():
    from src.orchestration.context.context_builder import _prune_nulls

    result = _prune_nulls({"a": [], "b": {}, "c": [1, 2], "d": {"x": 1}})
    assert "a" not in result
    assert "b" not in result
    assert result["c"] == [1, 2]
    assert result["d"] == {"x": 1}


def test_prune_nulls_preserves_zero_and_false():
    from src.orchestration.context.context_builder import _prune_nulls

    result = _prune_nulls({"zero": 0, "false": False, "empty_str": "", "none": None})
    assert result["zero"] == 0
    assert result["false"] is False
    assert result["empty_str"] == ""
    assert "none" not in result


# ---------------------------------------------------------------------------
# Fix 5 — primary-only plot filtering regression
# ---------------------------------------------------------------------------


def test_available_plots_are_primary_only():
    """build_context must only include primary-importance plots in available_plots."""
    ctx = build_context("canonical_ml_multi_asset")
    for plot in ctx.available_plots:
        assert plot["importance"] == "primary", (
            f"Non-primary plot included in context: {plot['name']} (importance={plot['importance']})"
        )


def test_available_plots_non_empty():
    ctx = build_context("canonical_ml_multi_asset")
    assert len(ctx.available_plots) > 0, "available_plots must not be empty after primary filtering"


def test_available_plots_fewer_than_all():
    """Primary filtering must reduce total plot count from the full plot index."""
    from src.orchestration.retrieval.plot_retriever import get_plot_index

    ctx = build_context("canonical_ml_multi_asset")
    all_plots = get_plot_index("canonical_ml_multi_asset")
    assert len(ctx.available_plots) < len(all_plots), (
        "Primary-only filter should exclude some secondary plots"
    )


# ---------------------------------------------------------------------------
# Fix 4 — context hash provenance regression
# ---------------------------------------------------------------------------


def test_context_hash_is_deterministic():
    """SHA256 of the same context dict must be stable across two calls."""
    import hashlib
    import json

    from src.orchestration.context.context_builder import _context_to_dict

    ctx = build_context("canonical_ml_multi_asset")
    d = _context_to_dict(ctx)
    h1 = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    assert h1 == h2


def test_context_hash_is_64_hex_chars():
    import hashlib
    import json

    from src.orchestration.context.context_builder import _context_to_dict

    ctx = build_context("canonical_ml_multi_asset")
    d = _context_to_dict(ctx)
    h = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
