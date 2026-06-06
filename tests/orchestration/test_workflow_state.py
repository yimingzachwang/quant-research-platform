"""Tests for research_api.get_research_workflow_state (read-only inspection).

No quant engine, no LLM, no real experiments — only on-disk presence checks
against isolated tmp directories.
"""

from __future__ import annotations

import json

from src.orchestration.api.research_api import get_research_workflow_state

_EXP = "exp_a"


def _dirs(tmp_path):
    return {
        "base": tmp_path / "experiments",
        "llm_base": tmp_path / "llm",
        "configs_base": tmp_path / "configs",
        "reports_base": tmp_path / "reports",
    }


def test_empty_state_all_absent(tmp_path):
    st = get_research_workflow_state(_EXP, **_dirs(tmp_path))
    assert st["experiment_name"] == _EXP
    assert st["baseline_artefacts_exist"] is False
    assert st["context_ready"] is False
    assert st["metadata_exists"] is False
    assert st["metrics_exists"] is False
    assert st["review_exists"] is False
    assert st["proposal_exists"] is False
    assert st["draft_exists"] is False
    assert st["latest_draft_id"] is None
    assert st["latest_draft_approved"] is False
    assert st["rendered_yaml_exists"] is False
    assert st["revised_artefacts_exist"] is False


def test_baseline_only(tmp_path):
    d = _dirs(tmp_path)
    (d["base"] / _EXP).mkdir(parents=True)
    (d["base"] / _EXP / "metadata.json").write_text("{}")
    (d["base"] / _EXP / "metrics.json").write_text("{}")

    st = get_research_workflow_state(_EXP, **d)
    assert st["baseline_artefacts_exist"] is True
    assert st["context_ready"] is True
    assert st["review_exists"] is False
    assert st["draft_exists"] is False


def test_picks_latest_draft_and_approval(tmp_path):
    d = _dirs(tmp_path)
    (d["base"] / _EXP).mkdir(parents=True)
    (d["base"] / _EXP / "metadata.json").write_text("{}")
    (d["base"] / _EXP / "metrics.json").write_text("{}")
    llm_exp = d["llm_base"] / _EXP
    llm_exp.mkdir(parents=True)
    (llm_exp / "llm_review.json").write_text("{}")
    (llm_exp / "iteration_proposal.json").write_text("{}")
    # Two drafts; the newer (by generated_at) wins and is approved.
    (llm_exp / "draft_old.json").write_text(json.dumps({
        "draft_id": "old", "approved": False, "proposed_name": "exp_a_v2",
        "generated_at": "2026-01-01T00:00:00+00:00",
    }))
    (llm_exp / "draft_new.json").write_text(json.dumps({
        "draft_id": "new", "approved": True, "proposed_name": "exp_a_v2",
        "generated_at": "2026-02-01T00:00:00+00:00",
    }))

    st = get_research_workflow_state(_EXP, **d)
    assert st["review_exists"] is True
    assert st["proposal_exists"] is True
    assert st["draft_exists"] is True
    assert st["latest_draft_id"] == "new"
    assert st["latest_draft_approved"] is True
    assert st["proposed_name"] == "exp_a_v2"
    assert st["rendered_yaml_exists"] is False


def test_rendered_yaml_and_revised_artefacts(tmp_path):
    d = _dirs(tmp_path)
    (d["base"] / _EXP).mkdir(parents=True)
    (d["base"] / _EXP / "metadata.json").write_text("{}")
    (d["base"] / _EXP / "metrics.json").write_text("{}")
    llm_exp = d["llm_base"] / _EXP
    llm_exp.mkdir(parents=True)
    (llm_exp / "draft_x.json").write_text(json.dumps({
        "draft_id": "x", "approved": True, "proposed_name": "exp_a_v2",
        "generated_at": "2026-02-01T00:00:00+00:00",
    }))
    # Rendered YAML for the proposed name.
    d["configs_base"].mkdir(parents=True)
    (d["configs_base"] / "exp_a_v2.yaml").write_text("name: exp_a_v2\n")
    # Revised experiment artefacts + plots.
    rev = d["base"] / "exp_a_v2"
    (rev / "plots").mkdir(parents=True)
    (rev / "metadata.json").write_text("{}")

    st = get_research_workflow_state(_EXP, **d)
    assert st["rendered_yaml_exists"] is True
    assert st["rendered_yaml_path"].endswith("exp_a_v2.yaml")
    assert st["revised_artefacts_exist"] is True
    assert st["plots_dir"] is not None and st["plots_dir"].endswith("plots")
