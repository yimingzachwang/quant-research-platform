"""End-to-end tests for the top-level research_api using real artefacts."""

import pytest

from src.orchestration.api.research_api import (
    build_llm_context,
    compare,
    diff,
    find_experiments,
    get_experiment_plots,
    get_experiment_summary,
    list_all_experiments,
    list_artefacts,
    list_experiment_summaries,
    load_experiment,
    rank,
    rank_experiments_by_sharpe,
    retrieve_all_diagnostics,
    retrieve_artefact,
    run_llm_review,
)
from src.orchestration.llm.review_schema import PROVIDER_STUB

_CANONICAL = "canonical_ml_multi_asset"
_SHOWCASE = "canonical_ml_showcase"


def test_list_all_experiments():
    exps = list_all_experiments()
    assert _CANONICAL in exps
    assert _SHOWCASE in exps


def test_find_experiments_by_tag():
    # "momentum" tag exists in the example_momentum_rotation registry entry
    found = find_experiments(tag="momentum")
    assert len(found) >= 1


def test_find_experiments_no_filter():
    found = find_experiments()
    assert _CANONICAL in found


def test_get_experiment_summary():
    s = get_experiment_summary(_CANONICAL)
    assert s is not None
    assert s.sharpe_ratio is not None
    assert s.has_ml is True


def test_get_experiment_summary_missing():
    assert get_experiment_summary("nonexistent_xyz") is None


def test_list_experiment_summaries():
    summaries = list_experiment_summaries()
    names = [s.experiment_name for s in summaries]
    assert _CANONICAL in names


def test_rank_experiments_by_sharpe():
    ranked = rank_experiments_by_sharpe()
    sharpes = [s.sharpe_ratio for s in ranked if s.sharpe_ratio is not None]
    assert sharpes == sorted(sharpes, reverse=True)


def test_load_experiment_metadata_only():
    bundle = load_experiment(_CANONICAL, include_timeseries=False)
    assert "metadata" in bundle
    assert "metrics" in bundle
    assert "equity_curve" not in bundle


def test_load_experiment_with_timeseries():
    bundle = load_experiment(_CANONICAL, include_timeseries=True)
    assert "equity_curve" in bundle
    assert bundle["equity_curve"] is not None


def test_retrieve_artefact():
    m = retrieve_artefact(_CANONICAL, "metadata")
    assert m is not None
    assert "experiment_name" in m


def test_retrieve_all_diagnostics():
    d = retrieve_all_diagnostics(_CANONICAL)
    assert "ml_model_diagnostics" in d
    assert "split_metrics" in d


def test_list_artefacts():
    artefacts = list_artefacts(_CANONICAL)
    assert len(artefacts) > 5
    keys = [a.key for a in artefacts]
    assert "metadata" in keys


def test_get_experiment_plots():
    plots = get_experiment_plots(_CANONICAL)
    assert len(plots) > 0


def test_get_experiment_plots_primary_only():
    primary = get_experiment_plots(_CANONICAL, primary_only=True)
    assert all(p.importance == "primary" for p in primary)


def test_compare():
    rows = compare([_CANONICAL, _SHOWCASE])
    assert len(rows) == 2
    names = [r["experiment_name"] for r in rows]
    assert _CANONICAL in names
    assert _SHOWCASE in names


def test_diff():
    d = diff(_CANONICAL, _SHOWCASE)
    assert "metric_diffs" in d
    assert "sharpe_ratio" in d["metric_diffs"]
    assert d["metric_diffs"]["sharpe_ratio"]["delta"] is not None


def test_rank():
    rows = rank()
    assert len(rows) >= 2
    sharpes = [r["sharpe_ratio"] for r in rows if r["sharpe_ratio"] is not None]
    assert sharpes == sorted(sharpes, reverse=True)


def test_build_llm_context_no_persist():
    ctx = build_llm_context(_CANONICAL, persist=False)
    assert ctx.experiment_name == _CANONICAL
    assert ctx.ml_diagnostics.get("available") is True


def test_run_llm_review_stub():
    review = run_llm_review(
        _CANONICAL,
        provider=PROVIDER_STUB,
        persist_context=False,
        persist_review=False,
    )
    assert review.provider == PROVIDER_STUB
    assert review.experiment_name == _CANONICAL
    assert "STUB" in review.review_text
    assert isinstance(review.flags, list)


def test_run_llm_review_stub_showcase():
    review = run_llm_review(
        _SHOWCASE,
        provider=PROVIDER_STUB,
        persist_context=False,
        persist_review=False,
    )
    assert review.experiment_name == _SHOWCASE
