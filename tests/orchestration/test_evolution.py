"""Tests for Phase 2B-B — Research Evolution Chains.

Validates:
- lineage metadata persistence and loading
- parent-child resolution
- deterministic chain ordering
- evolution step derivation
- chain serialization
- persistence correctness
- API integration
- missing-parent handling
- context hash preservation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.orchestration.api.schemas import (
    EvolutionStep,
    ExperimentLineage,
    ResearchEvolutionChain,
)
from src.orchestration.evolution.evolution_builder import (
    _chain_to_dict,
    _generate_evolution_summary,
    _lineage_to_dict,
    _step_from_contexts,
    build_evolution_chain,
    build_evolution_step,
    load_lineage,
    persist_evolution_chain,
    register_lineage,
    resolve_chain,
)

_ROOT = "canonical_ml_showcase"
_CHILD = "canonical_ml_multi_asset"


# ---------------------------------------------------------------------------
# 1. Lineage metadata persistence
# ---------------------------------------------------------------------------


def test_register_lineage_writes_file(tmp_path):
    lin = register_lineage(
        "canonical_ml_showcase",
        parent_experiment=None,
        iteration_reason="Baseline",
        experiments_base=tmp_path,
    )
    path = tmp_path / "canonical_ml_showcase" / "lineage.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["experiment_name"] == "canonical_ml_showcase"
    assert data["parent_experiment"] is None


def test_register_lineage_writes_parent(tmp_path):
    register_lineage(
        "canonical_ml_multi_asset",
        parent_experiment="canonical_ml_showcase",
        iteration_reason="Expand universe for OOS improvement",
        derived_from_iteration=True,
        experiments_base=tmp_path,
    )
    path = tmp_path / "canonical_ml_multi_asset" / "lineage.json"
    data = json.loads(path.read_text())
    assert data["parent_experiment"] == "canonical_ml_showcase"
    assert data["iteration_reason"] == "Expand universe for OOS improvement"
    assert data["derived_from_iteration"] is True


def test_register_lineage_returns_dataclass(tmp_path):
    lin = register_lineage("canonical_ml_showcase", None, experiments_base=tmp_path)
    assert isinstance(lin, ExperimentLineage)
    assert lin.experiment_name == "canonical_ml_showcase"


def test_register_lineage_has_registered_at(tmp_path):
    lin = register_lineage("canonical_ml_showcase", None, experiments_base=tmp_path)
    # Should parse without error
    datetime.fromisoformat(lin.registered_at)


# ---------------------------------------------------------------------------
# 2. Lineage loading
# ---------------------------------------------------------------------------


def test_load_lineage_returns_none_if_missing(tmp_path):
    result = load_lineage("nonexistent_experiment", tmp_path)
    assert result is None


def test_load_lineage_round_trips(tmp_path):
    register_lineage(
        "canonical_ml_showcase",
        parent_experiment=None,
        iteration_reason="Baseline",
        context_hash="abc123",
        experiments_base=tmp_path,
    )
    lin = load_lineage("canonical_ml_showcase", tmp_path)
    assert lin is not None
    assert lin.experiment_name == "canonical_ml_showcase"
    assert lin.parent_experiment is None
    assert lin.iteration_reason == "Baseline"
    assert lin.context_hash == "abc123"


def test_load_lineage_parent_preserved(tmp_path):
    register_lineage(
        "canonical_ml_multi_asset",
        parent_experiment="canonical_ml_showcase",
        experiments_base=tmp_path,
    )
    lin = load_lineage("canonical_ml_multi_asset", tmp_path)
    assert lin is not None
    assert lin.parent_experiment == "canonical_ml_showcase"


# ---------------------------------------------------------------------------
# 3. Parent-child resolution and deterministic ordering
# ---------------------------------------------------------------------------


def test_resolve_chain_single_node(tmp_path):
    register_lineage("canonical_ml_showcase", parent_experiment=None,
                     experiments_base=tmp_path)
    # Use real experiments dir so list_experiments finds them
    from src.orchestration.utils.filesystem import experiments_root
    chain = resolve_chain("canonical_ml_showcase", tmp_path)
    assert chain == ["canonical_ml_showcase"]


def test_resolve_chain_two_nodes(tmp_path):
    # Mirror real experiment dirs into tmp_path so list_experiments works
    (tmp_path / "canonical_ml_showcase").mkdir()
    (tmp_path / "canonical_ml_multi_asset").mkdir()
    # Metadata stubs so list_experiments recognises them
    (tmp_path / "canonical_ml_showcase" / "metadata.json").write_text("{}")
    (tmp_path / "canonical_ml_multi_asset" / "metadata.json").write_text("{}")

    register_lineage("canonical_ml_showcase", None, experiments_base=tmp_path)
    register_lineage("canonical_ml_multi_asset", "canonical_ml_showcase",
                     experiments_base=tmp_path)

    chain = resolve_chain("canonical_ml_showcase", tmp_path)
    assert chain == ["canonical_ml_showcase", "canonical_ml_multi_asset"]


def test_resolve_chain_stops_at_leaf(tmp_path):
    for name in ("exp_a", "exp_b", "exp_c"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")

    register_lineage("exp_a", None, experiments_base=tmp_path)
    register_lineage("exp_b", "exp_a", experiments_base=tmp_path)
    register_lineage("exp_c", "exp_b", experiments_base=tmp_path)

    chain = resolve_chain("exp_a", tmp_path)
    assert chain == ["exp_a", "exp_b", "exp_c"]


def test_resolve_chain_deterministic_on_multiple_children(tmp_path):
    for name in ("root", "child_a", "child_b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")

    register_lineage("root", None, experiments_base=tmp_path)
    register_lineage("child_a", "root", experiments_base=tmp_path)
    register_lineage("child_b", "root", experiments_base=tmp_path)

    chain1 = resolve_chain("root", tmp_path)
    chain2 = resolve_chain("root", tmp_path)
    assert chain1 == chain2  # deterministic
    assert chain1[0] == "root"
    assert len(chain1) == 2  # takes lexicographically first child


def test_resolve_chain_missing_parent_returns_root_only(tmp_path):
    (tmp_path / "orphan").mkdir()
    (tmp_path / "orphan" / "metadata.json").write_text("{}")
    # No lineage registered — no children found
    chain = resolve_chain("orphan", tmp_path)
    assert chain == ["orphan"]


# ---------------------------------------------------------------------------
# 4. Evolution step derivation from real contexts
# ---------------------------------------------------------------------------


def test_build_evolution_step_root_returns_step():
    step = build_evolution_step(
        curr_name=_ROOT,
        prev_name=None,
        lineage=None,
    )
    assert isinstance(step, EvolutionStep)
    assert step.experiment_name == _ROOT
    assert step.key_improvements == []


def test_build_evolution_step_root_sets_direction():
    from src.orchestration.api.schemas import ExperimentLineage

    lin = ExperimentLineage(
        experiment_name=_ROOT, parent_experiment=None,
        created_at="", registered_at="",
        iteration_reason="Baseline SPY strategy",
        derived_from_iteration=False, derived_from_comparison=False,
        context_hash="",
    )
    step = build_evolution_step(_ROOT, None, lin)
    assert "Baseline" in step.research_direction or "SPY" in step.research_direction


def test_step_from_contexts_canonical():
    """Derive an EvolutionStep from showcase→multi_asset context diff."""
    step = _step_from_contexts(
        curr_name=_CHILD,
        prev_name=_ROOT,
        research_direction="Expand to multi-asset panel",
        experiments_base=None,
    )
    assert isinstance(step, EvolutionStep)
    assert step.experiment_name == _CHILD
    # showcase→multi_asset: poor_oos_consistency and high_split_sharpe_variance resolved
    resolved_text = " ".join(step.key_improvements)
    assert "poor_oos_consistency" in resolved_text or "high_split_sharpe_variance" in resolved_text
    # severe_drawdown introduced
    risk_text = " ".join(step.new_risks)
    assert "severe_drawdown" in risk_text
    # catastrophic_split persists
    assert "catastrophic_split" in step.persistent_failures


def test_step_from_contexts_validation_changes():
    step = _step_from_contexts(_CHILD, _ROOT, "", None)
    # mean_oos_sharpe improved: -0.32 → 0.645
    val_text = " ".join(step.validation_changes)
    assert "mean_oos_sharpe" in val_text
    assert "improved" in val_text


# ---------------------------------------------------------------------------
# 5. _generate_evolution_summary — deterministic
# ---------------------------------------------------------------------------


def _make_step(name, improvements=None, risks=None, persistent=None, val=None, direction=""):
    return EvolutionStep(
        experiment_name=name,
        key_improvements=improvements or [],
        new_risks=risks or [],
        persistent_failures=persistent or [],
        validation_changes=val or [],
        research_direction=direction,
    )


def test_generate_summary_single_step():
    steps = [_make_step("exp_a", persistent=["catastrophic_split"])]
    summary = _generate_evolution_summary(["exp_a"], steps)
    assert "exp_a" in summary
    assert "catastrophic_split" in summary


def test_generate_summary_two_steps():
    steps = [
        _make_step("exp_a"),
        _make_step("exp_b",
                   improvements=["Resolved failure mode: poor_oos_consistency"],
                   risks=["New failure mode: severe_drawdown"],
                   val=["mean_oos_sharpe improved: -0.32 → 0.645 (Δ+0.965)"],
                   direction="Expand universe"),
    ]
    summary = _generate_evolution_summary(["exp_a", "exp_b"], steps)
    assert "exp_a" in summary
    assert "exp_b" in summary
    assert "poor_oos_consistency" in summary or "severe_drawdown" in summary or "mean_oos_sharpe" in summary


def test_generate_summary_is_deterministic():
    steps = [_make_step("a"), _make_step("b", improvements=["x"], risks=["y"])]
    s1 = _generate_evolution_summary(["a", "b"], steps)
    s2 = _generate_evolution_summary(["a", "b"], steps)
    assert s1 == s2


def test_generate_summary_empty():
    s = _generate_evolution_summary([], [])
    assert len(s) > 0


# ---------------------------------------------------------------------------
# 6. build_evolution_chain integration
# ---------------------------------------------------------------------------


def test_build_evolution_chain_single(tmp_path):
    (tmp_path / _ROOT).mkdir()
    (tmp_path / _ROOT / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, "Baseline", experiments_base=tmp_path)
    # For the root step, load real context
    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    assert isinstance(chain, ResearchEvolutionChain)
    assert chain.root_experiment == _ROOT
    assert chain.experiments == [_ROOT]
    assert len(chain.steps) == 1


def test_build_evolution_chain_two_nodes(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")

    register_lineage(_ROOT, None, "Baseline", experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, "Multi-asset expansion", experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    assert chain.experiments == [_ROOT, _CHILD]
    assert len(chain.steps) == 2


def test_build_evolution_chain_generated_at_is_iso(tmp_path):
    (tmp_path / _ROOT).mkdir()
    (tmp_path / _ROOT / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    datetime.fromisoformat(chain.generated_at)


def test_build_evolution_chain_summary_non_empty(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, "Test", experiments_base=tmp_path)
    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    assert len(chain.evolution_summary) > 20


# ---------------------------------------------------------------------------
# 7. Persistence correctness
# ---------------------------------------------------------------------------


def test_persist_evolution_chain_writes_json(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, "Expansion", experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    persist_evolution_chain(chain, evolution_base=tmp_path)

    json_path = tmp_path / _ROOT / "evolution_chain.json"
    assert json_path.exists()


def test_persist_evolution_chain_writes_md(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    persist_evolution_chain(chain, evolution_base=tmp_path)

    md_path = tmp_path / _ROOT / "evolution_chain.md"
    assert md_path.exists()


def test_persist_json_has_required_fields(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    persist_evolution_chain(chain, evolution_base=tmp_path)

    data = json.loads((tmp_path / _ROOT / "evolution_chain.json").read_text())
    for field in ("root_experiment", "experiments", "generated_at",
                  "evolution_summary", "steps"):
        assert field in data, f"Missing field: {field}"


def test_persist_steps_are_serialized(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    persist_evolution_chain(chain, evolution_base=tmp_path)

    data = json.loads((tmp_path / _ROOT / "evolution_chain.json").read_text())
    assert isinstance(data["steps"], list)
    for step in data["steps"]:
        for f in ("experiment_name", "key_improvements", "new_risks",
                  "persistent_failures", "validation_changes", "research_direction"):
            assert f in step


def test_persist_md_contains_root(tmp_path):
    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")
    register_lineage(_ROOT, None, experiments_base=tmp_path)
    register_lineage(_CHILD, _ROOT, experiments_base=tmp_path)

    chain = build_evolution_chain(_ROOT, experiments_base=tmp_path)
    persist_evolution_chain(chain, evolution_base=tmp_path)

    md = (tmp_path / _ROOT / "evolution_chain.md").read_text()
    assert _ROOT in md
    assert _CHILD in md


# ---------------------------------------------------------------------------
# 8. Missing-parent handling
# ---------------------------------------------------------------------------


def test_chain_with_no_lineage_returns_single_node(tmp_path):
    (tmp_path / "orphan").mkdir()
    (tmp_path / "orphan" / "metadata.json").write_text("{}")
    # No lineage registered — chain contains only root
    chain = build_evolution_chain("orphan", experiments_base=tmp_path)
    assert chain.experiments == ["orphan"]
    assert len(chain.steps) == 1


def test_load_lineage_missing_file_returns_none(tmp_path):
    assert load_lineage("does_not_exist", tmp_path) is None


def test_build_step_missing_context_returns_empty_step():
    """If context cannot be loaded, step should return empty lists rather than raise."""
    step = _step_from_contexts(
        curr_name="nonexistent_exp",
        prev_name="also_nonexistent",
        research_direction="test",
        experiments_base=None,
    )
    assert isinstance(step, EvolutionStep)
    assert step.key_improvements == []
    assert step.new_risks == []


# ---------------------------------------------------------------------------
# 9. Context hash preservation
# ---------------------------------------------------------------------------


def test_register_lineage_preserves_context_hash(tmp_path):
    register_lineage(
        "canonical_ml_showcase",
        None,
        context_hash="deadbeef" * 8,
        experiments_base=tmp_path,
    )
    lin = load_lineage("canonical_ml_showcase", tmp_path)
    assert lin.context_hash == "deadbeef" * 8


def test_lineage_dict_has_context_hash():
    from src.orchestration.api.schemas import ExperimentLineage
    lin = ExperimentLineage(
        experiment_name="x", parent_experiment=None,
        created_at="t", registered_at="t",
        iteration_reason=None, derived_from_iteration=False,
        derived_from_comparison=False, context_hash="abc",
    )
    d = _lineage_to_dict(lin)
    assert d["context_hash"] == "abc"


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------


def test_api_register_experiment_lineage(tmp_path):
    from src.orchestration.api.research_api import register_experiment_lineage

    lin = register_experiment_lineage(
        "canonical_ml_showcase",
        parent_experiment=None,
        iteration_reason="Baseline",
        base=tmp_path,
    )
    assert isinstance(lin, ExperimentLineage)


def test_api_build_research_evolution_chain_no_persist(tmp_path):
    from src.orchestration.api.research_api import (
        build_research_evolution_chain,
        register_experiment_lineage,
    )

    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")

    register_experiment_lineage(_ROOT, None, base=tmp_path)
    register_experiment_lineage(_CHILD, _ROOT, "Multi-asset expansion", base=tmp_path)

    chain = build_research_evolution_chain(
        _ROOT, base=tmp_path, persist=False, evolution_base=tmp_path
    )
    assert isinstance(chain, ResearchEvolutionChain)
    assert chain.experiments == [_ROOT, _CHILD]
    assert not (tmp_path / _ROOT / "evolution_chain.json").exists()


def test_api_build_research_evolution_chain_persist(tmp_path):
    from src.orchestration.api.research_api import (
        build_research_evolution_chain,
        register_experiment_lineage,
    )

    for name in (_ROOT, _CHILD):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metadata.json").write_text("{}")

    register_experiment_lineage(_ROOT, None, base=tmp_path)
    register_experiment_lineage(_CHILD, _ROOT, "Expansion", base=tmp_path)

    build_research_evolution_chain(
        _ROOT, base=tmp_path, persist=True, evolution_base=tmp_path
    )
    assert (tmp_path / _ROOT / "evolution_chain.json").exists()
    assert (tmp_path / _ROOT / "evolution_chain.md").exists()
