"""Tests for orchestration.registry using real canonical experiments."""



from src.orchestration.registry.artefact_registry import (
    ALL_ARTEFACTS,
    get_spec,
    list_keys,
)
from src.orchestration.registry.experiment_registry import (
    find_by_strategy,
    find_by_tag,
    get_summary,
    list_all,
    list_summaries,
    rank_by_sharpe,
)

# ---------------------------------------------------------------------------
# Artefact registry (static, no I/O)
# ---------------------------------------------------------------------------


def test_artefact_registry_has_core_keys():
    keys = list_keys("core")
    assert "metadata" in keys
    assert "metrics" in keys
    assert "equity_curve" in keys


def test_artefact_registry_has_diagnostics_keys():
    keys = list_keys("diagnostics")
    assert "backtest_diagnostics" in keys
    assert "ml_model_diagnostics" in keys
    assert "split_metrics" in keys


def test_artefact_registry_get_spec():
    spec = get_spec("metadata")
    assert spec is not None
    assert spec.filename == "metadata.json"
    assert spec.group == "core"


def test_artefact_registry_get_spec_missing():
    assert get_spec("nonexistent_key_xyz") is None


def test_artefact_registry_all_have_descriptions():
    for spec in ALL_ARTEFACTS:
        assert spec.description, f"{spec.key} has no description"


# ---------------------------------------------------------------------------
# Experiment registry (uses real results/experiments/ on disk)
# ---------------------------------------------------------------------------


def test_list_all_includes_canonical():
    experiments = list_all()
    assert "canonical_ml_multi_asset" in experiments
    assert "canonical_ml_showcase" in experiments


def test_get_summary_canonical_multi_asset():
    summary = get_summary("canonical_ml_multi_asset")
    assert summary is not None
    assert summary.experiment_name == "canonical_ml_multi_asset"
    assert summary.sharpe_ratio is not None
    assert summary.has_ml is True
    assert summary.has_validation is True


def test_get_summary_canonical_showcase():
    summary = get_summary("canonical_ml_showcase")
    assert summary is not None
    assert summary.has_ml is True


def test_get_summary_nonexistent():
    assert get_summary("nonexistent_experiment_xyz_abc") is None


def test_list_summaries_not_empty():
    summaries = list_summaries()
    assert len(summaries) >= 2


def test_list_summaries_all_have_names():
    for s in list_summaries():
        assert s.experiment_name
        assert s.strategy_name


def test_rank_by_sharpe_descending():
    ranked = rank_by_sharpe(descending=True)
    sharpes = [s.sharpe_ratio for s in ranked if s.sharpe_ratio is not None]
    assert sharpes == sorted(sharpes, reverse=True)


def test_rank_by_sharpe_ascending():
    ranked = rank_by_sharpe(descending=False)
    sharpes = [s.sharpe_ratio for s in ranked if s.sharpe_ratio is not None]
    assert sharpes == sorted(sharpes)


def test_find_by_tag():
    # "momentum" is a tag present in example_momentum_rotation's registry entry
    results = find_by_tag("momentum")
    assert len(results) >= 1
    # All returned names must have the tag in the registry
    for name in results:
        summary = get_summary(name)
        assert summary is not None  # experiment must exist on disk


def test_find_by_strategy_pattern():
    results = find_by_strategy("ridge")
    assert len(results) >= 1
    for name in results:
        summary = get_summary(name)
        assert summary is not None
        assert "ridge" in summary.strategy_name.lower()
