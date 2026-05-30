"""Tests for Phase 3 — Config Synthesis (draft_schema, draft_generator,
draft_validator, yaml_renderer, intent integration, router integration).

All LLM calls are mocked.  No quant engine is invoked.
All disk I/O uses tmp_path fixtures.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from src.orchestration.config_generation.draft_schema import (
    DraftChange,
    DraftValidationResult,
    ExperimentDraft,
    apply_changes,
    compute_draft_hash,
)
from src.orchestration.config_generation.draft_validator import (
    approve_draft,
    validate_draft,
)
from src.orchestration.config_generation.yaml_renderer import render_to_yaml
from src.orchestration.intents.intent_parser import _rule_based_parse, parse
from src.orchestration.intents.intent_schema import (
    GenerateDraftIntent,
)
from src.orchestration.router.workflow_router import route

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict[str, Any] = {
    "version": "2",
    "name": "canonical_ml_showcase",
    "universe": {"tickers": ["SPY"]},
    "date_range": {"start": "2013-01-01", "end": "2024-12-31"},
    "model": {"type": "RidgeRegression", "params": {"alpha": 0.5}},
    "labels": {"type": "forward_returns", "params": {"horizon": 21}},
    "signal": {"type": "sign", "params": {}},
    "validation": {
        "type": "rolling",
        "parameters": {"train_months": 48, "test_months": 12, "gap_days": 0},
    },
    "execution": {"transaction_cost_bps": 5},
    "portfolio_construction": {
        "weighting": {
            "scheme": "equal_weight",
            "prediction_normalization": "none",
            "temperature": None,
        }
    },
    "features": {
        "ticker": "SPY",
        "entries": [
            {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
            {"name": "mom_60", "type": "momentum", "params": {"lookback": 60}},
        ],
    },
    "output": {
        "base_dir": "results/experiments",
        "registry_path": "results/experiments/registry.json",
        "register": True,
        "save_plots": True,
    },
}


def _make_config_file(tmp_path: Path, name: str = "canonical_ml_showcase") -> Path:
    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{name}.yaml"
    path.write_text(yaml.dump(_BASE_CONFIG), encoding="utf-8")
    return cfg_dir


def _make_draft(
    base_experiment: str = "canonical_ml_showcase",
    proposed_name: str = "canonical_ml_showcase_v2",
    changes: list[DraftChange] | None = None,
    approved: bool = False,
) -> ExperimentDraft:
    if changes is None:
        changes = [
            DraftChange(
                section="model",
                field="params.alpha",
                current_value=0.5,
                proposed_value=1.0,
                rationale="increase regularization",
            )
        ]
    draft_hash = compute_draft_hash(base_experiment, proposed_name, changes)
    return ExperimentDraft(
        draft_id=str(uuid.uuid4()),
        draft_hash=draft_hash,
        base_experiment=base_experiment,
        source_proposal_hash="abc123",
        proposed_name=proposed_name,
        changes=changes,
        generated_at="2026-05-29T00:00:00+00:00",
        approved=approved,
    )


# ---------------------------------------------------------------------------
# DraftChange dataclass
# ---------------------------------------------------------------------------


class TestDraftChange:
    def test_fields_preserved(self):
        c = DraftChange(
            section="model",
            field="params.alpha",
            current_value=0.5,
            proposed_value=1.0,
            rationale="more regularization",
        )
        assert c.section == "model"
        assert c.field == "params.alpha"
        assert c.current_value == 0.5
        assert c.proposed_value == 1.0
        assert c.rationale == "more regularization"

    def test_current_value_can_be_none(self):
        c = DraftChange(
            section="features",
            field="entries.add",
            current_value=None,
            proposed_value={"name": "vol_1m", "type": "Volatility", "params": {}},
            rationale="add vol feature",
        )
        assert c.current_value is None

    def test_asdict_roundtrip(self):
        c = DraftChange(
            section="labels",
            field="params.horizon",
            current_value=21,
            proposed_value=42,
            rationale="longer horizon",
        )
        d = dataclasses.asdict(c)
        assert d["proposed_value"] == 42


# ---------------------------------------------------------------------------
# ExperimentDraft dataclass
# ---------------------------------------------------------------------------


class TestExperimentDraft:
    def test_default_not_approved(self):
        draft = _make_draft()
        assert draft.approved is False
        assert draft.approved_at is None

    def test_to_dict_includes_all_fields(self):
        draft = _make_draft()
        d = draft.to_dict()
        assert "draft_id" in d
        assert "draft_hash" in d
        assert "base_experiment" in d
        assert "source_proposal_hash" in d
        assert "proposed_name" in d
        assert "changes" in d
        assert "generated_at" in d
        assert "approved" in d
        assert "approved_at" in d

    def test_to_dict_changes_are_serialisable(self):
        draft = _make_draft()
        d = draft.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["changes"][0]["section"] == "model"


# ---------------------------------------------------------------------------
# compute_draft_hash
# ---------------------------------------------------------------------------


class TestComputeDraftHash:
    def test_returns_12_char_hex(self):
        changes = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        h = compute_draft_hash("exp_a", "exp_a_v2", changes)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        changes = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        h1 = compute_draft_hash("exp_a", "exp_a_v2", changes)
        h2 = compute_draft_hash("exp_a", "exp_a_v2", changes)
        assert h1 == h2

    def test_sensitive_to_proposed_name(self):
        changes = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        h1 = compute_draft_hash("exp_a", "exp_a_v2", changes)
        h2 = compute_draft_hash("exp_a", "exp_a_v3", changes)
        assert h1 != h2

    def test_sensitive_to_proposed_value(self):
        c1 = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        c2 = [DraftChange("model", "params.alpha", 0.5, 2.0, "r")]
        assert compute_draft_hash("e", "e_v2", c1) != compute_draft_hash("e", "e_v2", c2)

    def test_sensitive_to_base_experiment(self):
        changes = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        h1 = compute_draft_hash("exp_a", "exp_a_v2", changes)
        h2 = compute_draft_hash("exp_b", "exp_a_v2", changes)
        assert h1 != h2

    def test_empty_changes(self):
        h = compute_draft_hash("exp_a", "exp_a_v2", [])
        assert len(h) == 12


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------


class TestApplyChanges:
    def test_does_not_mutate_base(self):
        base = {"model": {"params": {"alpha": 0.5}}}
        changes = [DraftChange("model", "params.alpha", 0.5, 1.0, "r")]
        apply_changes(base, changes)
        assert base["model"]["params"]["alpha"] == 0.5

    def test_scalar_path_update(self):
        base = {"model": {"params": {"alpha": 0.5}}}
        changes = [DraftChange("model", "params.alpha", 0.5, 2.0, "r")]
        out = apply_changes(base, changes)
        assert out["model"]["params"]["alpha"] == 2.0

    def test_top_level_type_update(self):
        base = {"model": {"type": "RidgeRegression", "params": {}}}
        changes = [DraftChange("model", "type", "RidgeRegression", "Lasso", "r")]
        out = apply_changes(base, changes)
        assert out["model"]["type"] == "Lasso"

    def test_entries_add(self):
        base = {"features": {"entries": [{"name": "mom", "type": "momentum", "params": {}}]}}
        new_feat = {"name": "vol", "type": "rolling_volatility", "params": {"window": 21}}
        changes = [DraftChange("features", "entries.add", None, new_feat, "r")]
        out = apply_changes(base, changes)
        assert len(out["features"]["entries"]) == 2
        assert out["features"]["entries"][1]["name"] == "vol"

    def test_entries_remove(self):
        base = {
            "features": {
                "entries": [
                    {"name": "mom", "type": "momentum", "params": {}},
                    {"name": "vol", "type": "rolling_volatility", "params": {}},
                ]
            }
        }
        changes = [DraftChange("features", "entries.remove", None, "mom", "r")]
        out = apply_changes(base, changes)
        assert len(out["features"]["entries"]) == 1
        assert out["features"]["entries"][0]["name"] == "vol"

    def test_entries_remove_nonexistent_is_noop(self):
        base = {"features": {"entries": [{"name": "vol", "type": "rolling_volatility", "params": {}}]}}
        changes = [DraftChange("features", "entries.remove", None, "gone", "r")]
        out = apply_changes(base, changes)
        assert len(out["features"]["entries"]) == 1

    def test_multiple_changes_applied_in_order(self):
        base = {"model": {"params": {"alpha": 0.5, "max_iter": 500}}}
        changes = [
            DraftChange("model", "params.alpha", 0.5, 1.0, "r"),
            DraftChange("model", "params.max_iter", 500, 2000, "r"),
        ]
        out = apply_changes(base, changes)
        assert out["model"]["params"]["alpha"] == 1.0
        assert out["model"]["params"]["max_iter"] == 2000

    def test_creates_missing_section(self):
        base: dict = {}
        changes = [DraftChange("execution", "transaction_cost_bps", None, 10, "r")]
        out = apply_changes(base, changes)
        assert out["execution"]["transaction_cost_bps"] == 10


# ---------------------------------------------------------------------------
# validate_draft
# ---------------------------------------------------------------------------


class TestValidateDraft:
    def test_unknown_change_path_is_invalid(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        changes = [DraftChange("model", "params.unknown_field", None, 99, "r")]
        draft = _make_draft(changes=changes)
        result = validate_draft(draft, configs_base=cfg_dir)
        assert not result.is_valid
        assert any("Unknown change path" in e for e in result.errors)

    def test_name_collision_is_invalid(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        # Proposed name == base experiment name → collision
        draft = _make_draft(proposed_name="canonical_ml_showcase")
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_showcase"],
        ):
            result = validate_draft(draft, configs_base=cfg_dir)
        assert not result.is_valid
        assert any("already exists" in e for e in result.errors)

    def test_missing_base_config_is_invalid(self, tmp_path):
        draft = _make_draft(base_experiment="nonexistent_experiment")
        result = validate_draft(draft, configs_base=tmp_path / "configs" / "experiments")
        assert not result.is_valid
        assert any("Base config not found" in e for e in result.errors)

    def test_version_1_config_is_invalid(self, tmp_path):
        cfg_dir = tmp_path / "configs" / "experiments"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        old_cfg = {"version": "1", "name": "old_exp"}
        (cfg_dir / "old_exp.yaml").write_text(yaml.dump(old_cfg), encoding="utf-8")
        draft = _make_draft(base_experiment="old_exp", proposed_name="old_exp_v2")
        result = validate_draft(draft, configs_base=cfg_dir)
        assert not result.is_valid
        assert any("config version 2" in e for e in result.errors)

    def test_valid_draft_passes(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = _make_draft()
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_showcase"],
        ):
            result = validate_draft(draft, configs_base=cfg_dir)
        assert result.is_valid
        assert result.errors == []

    def test_invalid_model_type_caught_by_validator(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        changes = [DraftChange("model", "type", "RidgeRegression", "BogusModelXYZ", "r")]
        draft = _make_draft(changes=changes)
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=["canonical_ml_showcase"],
        ):
            result = validate_draft(draft, configs_base=cfg_dir)
        assert not result.is_valid

    def test_result_is_draft_validation_result(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = _make_draft()
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=[],
        ):
            result = validate_draft(draft, configs_base=cfg_dir)
        assert isinstance(result, DraftValidationResult)

    def test_rendered_yaml_with_metadata_is_valid_base(self, tmp_path):
        """Phase 4 chaining: a rendered YAML (which has a top-level 'metadata' block)
        must be usable as the base experiment for a subsequent draft validation."""
        cfg_dir = _make_config_file(tmp_path)
        # Simulate the metadata block that render_to_yaml() injects
        rendered = dict(_BASE_CONFIG)
        rendered["metadata"] = {
            "draft_hash": "abc123def456",
            "base_experiment": "canonical_ml_showcase",
            "source_proposal_hash": "xyz789",
            "experiment_hash": "aabbcc112233",
        }
        rendered_name = "canonical_ml_showcase_v2"
        rendered["name"] = rendered_name
        (cfg_dir / f"{rendered_name}.yaml").write_text(
            yaml.dump(rendered), encoding="utf-8"
        )
        draft = _make_draft(
            base_experiment=rendered_name,
            proposed_name="canonical_ml_showcase_v3",
        )
        with patch(
            "src.orchestration.config_generation.draft_validator.list_all",
            return_value=[rendered_name],
        ):
            result = validate_draft(draft, configs_base=cfg_dir)
        assert result.is_valid, (
            f"Chained draft validation failed — validate_ml_config() may be "
            f"rejecting the orchestration 'metadata' key. Errors: {result.errors}"
        )


# ---------------------------------------------------------------------------
# approve_draft
# ---------------------------------------------------------------------------


class TestApproveDraft:
    def test_sets_approved_true(self):
        draft = _make_draft(approved=False)
        approved = approve_draft(draft)
        assert approved.approved is True

    def test_sets_approved_at(self):
        draft = _make_draft(approved=False)
        approved = approve_draft(draft)
        assert approved.approved_at is not None
        assert "2026" in approved.approved_at or approved.approved_at.startswith("20")

    def test_recomputes_hash(self):
        draft = _make_draft(approved=False)
        approved = approve_draft(draft)
        expected_hash = compute_draft_hash(
            draft.base_experiment, draft.proposed_name, draft.changes
        )
        assert approved.draft_hash == expected_hash

    def test_does_not_mutate_original(self):
        draft = _make_draft(approved=False)
        original_hash = draft.draft_hash
        approve_draft(draft)
        assert draft.approved is False
        assert draft.draft_hash == original_hash

    def test_preserves_all_other_fields(self):
        draft = _make_draft()
        approved = approve_draft(draft)
        assert approved.draft_id == draft.draft_id
        assert approved.base_experiment == draft.base_experiment
        assert approved.proposed_name == draft.proposed_name
        assert approved.changes == draft.changes
        assert approved.source_proposal_hash == draft.source_proposal_hash
        assert approved.generated_at == draft.generated_at


# ---------------------------------------------------------------------------
# render_to_yaml
# ---------------------------------------------------------------------------


class TestRenderToYaml:
    def test_unapproved_draft_raises(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = _make_draft(approved=False)
        with pytest.raises(ValueError, match="not been approved"):
            render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)

    def test_dry_run_returns_yaml_string(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            yaml_str = render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        assert isinstance(yaml_str, str)
        assert yaml_str.strip()

    def test_dry_run_does_not_write_file(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        output = cfg_dir / "canonical_ml_showcase_v2.yaml"
        assert not output.exists()

    def test_non_dry_run_writes_file(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            render_to_yaml(draft, configs_base=cfg_dir, dry_run=False)
        output = cfg_dir / "canonical_ml_showcase_v2.yaml"
        assert output.exists()

    def test_yaml_contains_provenance_header(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            yaml_str = render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        assert "# Generated by Quant Research Platform" in yaml_str
        assert draft.base_experiment in yaml_str
        assert draft.draft_hash in yaml_str

    def test_yaml_applies_changes(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        changes = [DraftChange("model", "params.alpha", 0.5, 2.5, "test")]
        draft = approve_draft(_make_draft(changes=changes))
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            yaml_str = render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        # yaml.safe_load ignores comment lines — parse the full string directly
        parsed = yaml.safe_load(yaml_str)
        assert parsed["model"]["params"]["alpha"] == 2.5

    def test_yaml_sets_proposed_name(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            yaml_str = render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "canonical_ml_showcase_v2"

    def test_yaml_embeds_metadata(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        draft = approve_draft(_make_draft())
        with patch(
            "src.orchestration.config_generation.yaml_renderer.validate_ml_config"
        ):
            yaml_str = render_to_yaml(draft, configs_base=cfg_dir, dry_run=True)
        parsed = yaml.safe_load(yaml_str)
        assert "metadata" in parsed
        assert parsed["metadata"]["draft_hash"] == draft.draft_hash
        assert parsed["metadata"]["base_experiment"] == draft.base_experiment

    def test_missing_base_config_raises(self, tmp_path):
        draft = approve_draft(_make_draft(base_experiment="no_such_exp"))
        with pytest.raises(Exception):
            render_to_yaml(draft, configs_base=tmp_path / "configs" / "experiments", dry_run=True)


# ---------------------------------------------------------------------------
# generate_draft (unit — LLM mocked)
# ---------------------------------------------------------------------------


class TestGenerateDraft:
    _VALID_LLM_RESPONSE = json.dumps({
        "proposed_name": "canonical_ml_showcase_v2",
        "changes": [
            {
                "section": "model",
                "field": "params.alpha",
                "proposed_value": 1.0,
                "rationale": "increase regularization",
            }
        ],
    })

    def _make_proposal_file(self, tmp_path: Path) -> Path:
        llm_dir = tmp_path / "results" / "llm_reviews" / "canonical_ml_showcase"
        llm_dir.mkdir(parents=True, exist_ok=True)
        proposal = {
            "experiment_name": "canonical_ml_showcase",
            "generated_at": "2026-05-29T00:00:00+00:00",
            "context_hash": "abc123456789",
            "research_focus": "regularization sweep",
            "rationale": "explore stronger regularization",
            "supporting_evidence": [],
            "suggested_experiments": ["try alpha=1.0"],
            "instability_signals": [],
            "validation_concerns": [],
            "feature_risks": [],
            "confidence": "medium",
            "provider": "stub",
            "model": None,
            "prompt_template": "default",
        }
        (llm_dir / "iteration_proposal.json").write_text(
            json.dumps(proposal), encoding="utf-8"
        )
        return tmp_path / "results" / "llm_reviews"

    def test_returns_experiment_draft(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        mock_resp = MagicMock()
        mock_resp.text = self._VALID_LLM_RESPONSE

        with patch(
            "src.orchestration.config_generation.draft_generator.call_llm",
            return_value=mock_resp,
        ):
            from src.orchestration.config_generation.draft_generator import generate_draft
            draft = generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )

        assert isinstance(draft, ExperimentDraft)
        assert draft.approved is False
        assert draft.proposed_name == "canonical_ml_showcase_v2"

    def test_fills_current_value_from_config(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        mock_resp = MagicMock()
        mock_resp.text = self._VALID_LLM_RESPONSE

        with patch(
            "src.orchestration.config_generation.draft_generator.call_llm",
            return_value=mock_resp,
        ):
            from src.orchestration.config_generation.draft_generator import generate_draft
            draft = generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )

        # current_value comes from base config (0.5), NOT from LLM (which didn't supply it)
        change = draft.changes[0]
        assert change.current_value == 0.5
        assert change.proposed_value == 1.0

    def test_persists_draft_to_disk(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        mock_resp = MagicMock()
        mock_resp.text = self._VALID_LLM_RESPONSE

        with patch(
            "src.orchestration.config_generation.draft_generator.call_llm",
            return_value=mock_resp,
        ):
            from src.orchestration.config_generation.draft_generator import generate_draft
            draft = generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )

        draft_path = llm_dir / "canonical_ml_showcase" / f"draft_{draft.draft_id}.json"
        assert draft_path.exists()

    def test_missing_config_raises(self, tmp_path):
        llm_dir = self._make_proposal_file(tmp_path)
        from src.orchestration.config_generation.draft_generator import generate_draft
        with pytest.raises(FileNotFoundError, match="No YAML config found"):
            generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=tmp_path / "empty",
            )

    def test_missing_proposal_raises(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        empty_llm_dir = tmp_path / "results" / "llm_reviews"
        empty_llm_dir.mkdir(parents=True, exist_ok=True)
        from src.orchestration.config_generation.draft_generator import generate_draft
        with pytest.raises(FileNotFoundError, match="No IterationProposal found"):
            generate_draft(
                "canonical_ml_showcase",
                llm_base=empty_llm_dir,
                configs_base=cfg_dir,
            )

    def test_invalid_llm_json_raises(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        mock_resp = MagicMock()
        mock_resp.text = "not json at all {{{"

        with patch(
            "src.orchestration.config_generation.draft_generator.call_llm",
            return_value=mock_resp,
        ):
            from src.orchestration.config_generation.draft_generator import generate_draft
            with pytest.raises(ValueError, match="invalid JSON"):
                generate_draft(
                    "canonical_ml_showcase",
                    llm_base=llm_dir,
                    configs_base=cfg_dir,
                )

    def test_markdown_fenced_json_is_parsed(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        mock_resp = MagicMock()
        mock_resp.text = f"```json\n{self._VALID_LLM_RESPONSE}\n```"

        with patch(
            "src.orchestration.config_generation.draft_generator.call_llm",
            return_value=mock_resp,
        ):
            from src.orchestration.config_generation.draft_generator import generate_draft
            draft = generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )
        assert draft.proposed_name == "canonical_ml_showcase_v2"

    def test_proposal_hash_mismatch_raises(self, tmp_path):
        cfg_dir = _make_config_file(tmp_path)
        llm_dir = self._make_proposal_file(tmp_path)
        from src.orchestration.config_generation.draft_generator import generate_draft
        with pytest.raises(ValueError, match="hash mismatch"):
            generate_draft(
                "canonical_ml_showcase",
                proposal_hash="wrong_hash_value",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )

    def test_version1_config_raises(self, tmp_path):
        cfg_dir = tmp_path / "configs" / "experiments"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        old_cfg = {"version": "1", "name": "canonical_ml_showcase"}
        (cfg_dir / "canonical_ml_showcase.yaml").write_text(yaml.dump(old_cfg), encoding="utf-8")
        llm_dir = self._make_proposal_file(tmp_path)
        from src.orchestration.config_generation.draft_generator import generate_draft
        with pytest.raises(ValueError, match="config version 2"):
            generate_draft(
                "canonical_ml_showcase",
                llm_base=llm_dir,
                configs_base=cfg_dir,
            )


# ---------------------------------------------------------------------------
# GenerateDraftIntent — schema
# ---------------------------------------------------------------------------


class TestGenerateDraftIntentSchema:
    def test_frozen(self):
        intent = GenerateDraftIntent(experiment_name="exp_a")
        with pytest.raises((TypeError, AttributeError)):
            intent.experiment_name = "mutated"  # type: ignore[misc]

    def test_default_provider(self):
        intent = GenerateDraftIntent(experiment_name="exp_a")
        assert intent.provider == "anthropic"
        assert intent.model is None

    def test_custom_model(self):
        intent = GenerateDraftIntent(
            experiment_name="exp_a", provider="openai", model="gpt-4o"
        )
        assert intent.provider == "openai"
        assert intent.model == "gpt-4o"


# ---------------------------------------------------------------------------
# GenerateDraftIntent — parser (rule-based)
# ---------------------------------------------------------------------------

_KNOWN = ["canonical_ml_showcase", "canonical_ml_multi_asset"]


class TestGenerateDraftIntentParser:
    def test_draft_keyword(self):
        result = _rule_based_parse("generate draft for canonical_ml_showcase", _KNOWN)
        assert isinstance(result, GenerateDraftIntent)
        assert result.experiment_name == "canonical_ml_showcase"

    def test_create_config_draft(self):
        result = _rule_based_parse(
            "create config draft for canonical_ml_multi_asset", _KNOWN
        )
        assert isinstance(result, GenerateDraftIntent)
        assert result.experiment_name == "canonical_ml_multi_asset"

    def test_synthesize_config(self):
        result = _rule_based_parse(
            "synthesize config for canonical_ml_showcase", _KNOWN
        )
        assert isinstance(result, GenerateDraftIntent)

    def test_no_experiment_name_returns_none(self):
        result = _rule_based_parse("generate draft", _KNOWN)
        assert result is None

    def test_draft_does_not_trigger_review(self):
        result = _rule_based_parse("generate draft for canonical_ml_showcase", _KNOWN)
        from src.orchestration.intents.intent_schema import ReviewExperimentIntent
        assert not isinstance(result, ReviewExperimentIntent)

    def test_parse_returns_generate_draft_intent(self):
        result = parse("generate draft for canonical_ml_showcase", known_experiments=_KNOWN)
        assert isinstance(result, GenerateDraftIntent)
        assert result.experiment_name == "canonical_ml_showcase"


# ---------------------------------------------------------------------------
# GenerateDraftIntent — router
# ---------------------------------------------------------------------------

_API_MODULE = "src.orchestration.router.workflow_router._api"


class TestGenerateDraftRouter:
    def test_routes_to_generate_experiment_draft(self):
        intent = GenerateDraftIntent(experiment_name="exp_a", provider="stub")
        sentinel = MagicMock(name="ExperimentDraft")
        with patch(f"{_API_MODULE}.generate_experiment_draft", return_value=sentinel) as m:
            result = route(intent)
        assert result.success
        assert result.api_function == "generate_experiment_draft"
        assert result.result is sentinel
        m.assert_called_once_with("exp_a", provider="stub", model=None)

    def test_routes_with_model_override(self):
        intent = GenerateDraftIntent(
            experiment_name="exp_b", provider="openai", model="gpt-4o"
        )
        with patch(f"{_API_MODULE}.generate_experiment_draft", return_value=None) as m:
            route(intent)
        m.assert_called_once_with("exp_b", provider="openai", model="gpt-4o")

    def test_api_error_returns_failed_result(self):
        intent = GenerateDraftIntent(experiment_name="exp_a")
        with patch(
            f"{_API_MODULE}.generate_experiment_draft",
            side_effect=FileNotFoundError("no proposal"),
        ):
            result = route(intent)
        assert not result.success
        assert "FileNotFoundError" in result.error


# ---------------------------------------------------------------------------
# research_api exports
# ---------------------------------------------------------------------------


class TestResearchApiExports:
    def test_generate_experiment_draft_exported(self):
        from src.orchestration.api.research_api import generate_experiment_draft
        assert callable(generate_experiment_draft)

    def test_validate_experiment_draft_exported(self):
        from src.orchestration.api.research_api import validate_experiment_draft
        assert callable(validate_experiment_draft)

    def test_approve_experiment_draft_exported(self):
        from src.orchestration.api.research_api import approve_experiment_draft
        assert callable(approve_experiment_draft)

    def test_render_draft_to_yaml_exported(self):
        from src.orchestration.api.research_api import render_draft_to_yaml
        assert callable(render_draft_to_yaml)

    def test_orchestration_init_exports_new_api(self):
        import src.orchestration as orch
        assert hasattr(orch, "generate_experiment_draft")
        assert hasattr(orch, "validate_experiment_draft")
        assert hasattr(orch, "approve_experiment_draft")
        assert hasattr(orch, "render_draft_to_yaml")


# ---------------------------------------------------------------------------
# Config generation __init__ exports
# ---------------------------------------------------------------------------


class TestConfigGenerationInit:
    def test_all_symbols_exported(self):
        from src.orchestration.config_generation import (
            DraftChange,
            DraftValidationResult,
            ExperimentDraft,
            apply_changes,
            approve_draft,
            compute_draft_hash,
            generate_draft,
            render_to_yaml,
            validate_draft,
        )
        assert all([
            DraftChange, DraftValidationResult, ExperimentDraft,
            apply_changes, approve_draft, compute_draft_hash,
            generate_draft, render_to_yaml, validate_draft,
        ])


# ---------------------------------------------------------------------------
# Canonical examples coverage
# ---------------------------------------------------------------------------


class TestCanonicalExamples:
    def test_all_generate_draft_examples_parse_correctly(self):
        from src.orchestration.intents.intent_examples import CANONICAL_EXAMPLES
        draft_examples = [
            (text, expected)
            for text, expected, _ in CANONICAL_EXAMPLES
            if expected == "GenerateDraftIntent"
        ]
        assert len(draft_examples) >= 3, "Expected at least 3 GenerateDraftIntent examples"
        for text, expected in draft_examples:
            result = _rule_based_parse(text, _KNOWN)
            assert result is not None, f"Unrecognised: {text!r}"
            assert type(result).__name__ == expected, (
                f"{text!r}: expected {expected}, got {type(result).__name__}"
            )
