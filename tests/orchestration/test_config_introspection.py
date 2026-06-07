"""Tests for the config-introspection and config-change layer.

Covers all 20 required scenarios:
  * inspect_experiment_config: compact summary, not-found
  * list_changeable_config_fields: allowed paths and operations
  * list_available_features: schema types only, family info, filtering
  * list_supported_models: schema types, current-model highlight, no invention
  * generate_config_change_draft: set scalar, add/remove/replace feature, multi-change
  * Error paths: invalid feature type, absent remove, duplicate add, schema-incompatible
  * No LLM call in any introspection or change tool
  * Draft is unapproved, no render/execute/approval side effects
  * MCP envelope contract on all 5 new tools
  * Operator manual routing rules present for config tools
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml

from src.experiments.ml_config import get_valid_feature_types, get_valid_model_types
from src.mcp import zeto_server as zeto
from src.orchestration.api import research_api as api

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "version": "2",
    "name": "test_exp",
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

_MCP_MOD = "src.mcp.zeto_server._api"
_LLM_MOD = "src.orchestration.config_generation.draft_generator.call_llm"


def _make_config(tmp_path: Path, name: str = "test_exp") -> Path:
    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{name}.yaml").write_text(yaml.dump(_BASE_CONFIG), encoding="utf-8")
    return cfg_dir


def _draft_dirs(tmp_path: Path) -> dict:
    return {
        "base": tmp_path / "results" / "experiments",
        "reports_base": tmp_path / "reports",
        "llm_base": tmp_path / "results" / "llm_reviews",
        "configs_base": tmp_path / "configs" / "experiments",
    }


# ---------------------------------------------------------------------------
# 1–2: inspect_experiment_config
# ---------------------------------------------------------------------------


class TestInspectExperimentConfig:
    def test_compact_summary_no_raw_yaml(self, tmp_path):
        """Compact envelope: model/features/validation present; raw YAML never returned."""
        _make_config(tmp_path)
        result = api.inspect_experiment_config(
            "test_exp",
            configs_base=tmp_path / "configs" / "experiments",
        )
        assert result["status"] == "ok"
        assert result["model"]["type"] == "RidgeRegression"
        assert result["features"]["count"] == 2
        assert set(result["features"]["entries"]) == {"mom_20", "mom_60"}
        assert result["validation"]["type"] == "rolling"
        assert "changeable_paths" in result
        assert len(result["changeable_paths"]) > 0
        # Never returns raw YAML string.
        for val in result.values():
            assert not (isinstance(val, str) and val.strip().startswith("version:")), (
                "inspect must not return raw YAML"
            )

    def test_not_found_returns_error_status(self, tmp_path):
        """Missing experiment returns config_not_found, not an exception."""
        result = api.inspect_experiment_config(
            "does_not_exist",
            configs_base=tmp_path / "configs" / "experiments",
        )
        assert result["status"] == "config_not_found"
        assert "errors" in result
        assert result["errors"]


# ---------------------------------------------------------------------------
# 3: list_changeable_config_fields
# ---------------------------------------------------------------------------


class TestListChangeableConfigFields:
    def test_allowed_paths_and_operations_present(self, tmp_path):
        """Every field has a field_path, operations list, and type; features.entries included."""
        _make_config(tmp_path)
        result = api.list_changeable_config_fields(
            "test_exp",
            configs_base=tmp_path / "configs" / "experiments",
        )
        assert result["status"] == "ok"
        fields = result["fields"]
        assert len(fields) > 0

        paths = {f["field_path"] for f in fields}
        # Scalar whitelisted paths must be present.
        assert "model.type" in paths
        assert "model.params.alpha" in paths
        assert "features.entries" in paths

        for f in fields:
            assert "operations" in f and f["operations"]
            assert "type" in f or "field_path" in f

        # Current value populated when experiment name given.
        alpha_field = next(f for f in fields if f["field_path"] == "model.params.alpha")
        assert alpha_field["current_value"] == 0.5

        # Feature entry count returned for known experiment.
        assert result["feature_entry_count"] == 2


# ---------------------------------------------------------------------------
# 4–5: list_available_features
# ---------------------------------------------------------------------------


class TestListAvailableFeatures:
    def test_schema_types_only_with_family_info(self, tmp_path):
        """Only schema-valid types returned; each has type, family, required_params."""
        _make_config(tmp_path)
        result = api.list_available_features(
            "test_exp",
            configs_base=tmp_path / "configs" / "experiments",
        )
        assert result["status"] == "ok"
        valid_types = get_valid_feature_types()
        returned_types = {f["type"] for f in result["features"]}

        # All returned types must be schema-valid; no invented names.
        assert returned_types.issubset(valid_types)
        # All schema types must appear.
        assert returned_types == valid_types

        for f in result["features"]:
            assert "family" in f and f["family"]
            assert "required_params" in f
            assert isinstance(f["required_params"], list)
            assert "currently_used" in f
            assert "operations" in f
            assert {"add", "remove", "replace"}.issubset(set(f["operations"]))

        # currently_used_count populated when experiment name given.
        assert result["currently_used_count"] is not None
        used_types = {f["type"] for f in result["features"] if f["currently_used"]}
        assert "momentum" in used_types

    def test_family_and_query_filters_narrow_results(self):
        """family and query filters return a strict subset of all types."""
        all_result = api.list_available_features()
        all_types = {f["type"] for f in all_result["features"]}

        # family filter — "volatility" family should hit only vol types.
        vol_result = api.list_available_features(family="volatility")
        vol_types = {f["type"] for f in vol_result["features"]}
        assert vol_types.issubset(all_types)
        # All returned have family containing "volatility".
        for f in vol_result["features"]:
            assert "volatility" in f["family"].lower()

        # query filter — "sma" should match exactly sma.
        sma_result = api.list_available_features(query="sma")
        assert any(f["type"] == "sma" for f in sma_result["features"])
        for f in sma_result["features"]:
            assert "sma" in f["type"].lower() or "sma" in f["family"].lower()

        # No experiment name → currently_used is None for all items.
        for f in all_result["features"]:
            assert f["currently_used"] is None


# ---------------------------------------------------------------------------
# 6: list_supported_models
# ---------------------------------------------------------------------------


class TestListSupportedModels:
    def test_schema_models_no_invention(self, tmp_path):
        """Only schema-valid model types; is_current flag correct; no invented names."""
        _make_config(tmp_path)
        result = api.list_supported_models(
            "test_exp",
            configs_base=tmp_path / "configs" / "experiments",
        )
        assert result["status"] == "ok"
        valid_models = get_valid_model_types()
        returned_names = {m["name"] for m in result["supported_models"]}

        # Must not invent names.
        assert returned_names.issubset(valid_models)
        assert returned_names == valid_models

        # current_model populated from experiment config.
        assert result["current_model"] == "RidgeRegression"

        # Only the current model has is_current=True.
        current_flag = {m["name"]: m["is_current"] for m in result["supported_models"]}
        assert current_flag["RidgeRegression"] is True
        assert all(not v for k, v in current_flag.items() if k != "RidgeRegression")

        assert result["model_switching_supported"] is True


# ---------------------------------------------------------------------------
# 7–10: generate_config_change_draft happy paths
# ---------------------------------------------------------------------------


class TestGenerateConfigChangeDraftHappyPath:
    def test_set_scalar_alpha(self, tmp_path):
        """Set model.params.alpha deterministically; no LLM; draft is created."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{"field_path": "model.params.alpha", "operation": "set", "value": 2.0}],
            reason="unit test",
            base=d["base"],
            llm_base=d["llm_base"],
            configs_base=d["configs_base"],
            reports_base=d["reports_base"],
        )
        assert result["status"] == "ok"
        draft = result["draft"]
        assert len(draft.changes) == 1
        assert draft.changes[0].section == "model"
        assert draft.changes[0].field == "params.alpha"
        assert draft.changes[0].proposed_value == 2.0
        assert draft.changes[0].current_value == 0.5

    def test_add_feature(self, tmp_path):
        """Add a new sma feature; validates required params; draft has entries.add change."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                "value": {"name": "sma_50", "type": "sma", "params": {"window": 50}},
            }],
            reason="unit test",
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok"
        changes = result["draft"].changes
        assert any(c.field == "entries.add" for c in changes)
        add_change = next(c for c in changes if c.field == "entries.add")
        assert add_change.proposed_value["name"] == "sma_50"
        assert add_change.proposed_value["type"] == "sma"

    def test_remove_feature(self, tmp_path):
        """Remove existing feature by name; draft has entries.remove change."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "remove",
                "value": "mom_60",
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok"
        changes = result["draft"].changes
        assert any(c.field == "entries.remove" for c in changes)
        remove_change = next(c for c in changes if c.field == "entries.remove")
        assert remove_change.proposed_value == "mom_60"

    def test_replace_feature(self, tmp_path):
        """Replace emits remove + add in order; new type validated against schema."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "replace",
                "old_value": "mom_20",
                "value": {
                    "name": "sma_20",
                    "type": "sma",
                    "params": {"window": 20},
                },
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok"
        changes = result["draft"].changes
        fields = [c.field for c in changes]
        # Must produce remove then add.
        assert "entries.remove" in fields
        assert "entries.add" in fields
        assert fields.index("entries.remove") < fields.index("entries.add")

    def test_multi_change_batch(self, tmp_path):
        """Multiple changes in one call; all applied atomically; correct draft produced."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[
                {"field_path": "model.params.alpha", "operation": "set", "value": 1.0},
                {
                    "field_path": "features.entries",
                    "operation": "add",
                    "value": {"name": "sma_100", "type": "sma", "params": {"window": 100}},
                },
            ],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok"
        fields = [c.field for c in result["draft"].changes]
        assert "params.alpha" in fields
        assert "entries.add" in fields


# ---------------------------------------------------------------------------
# 11: Unapproved draft
# ---------------------------------------------------------------------------


class TestDraftIsUnapproved:
    def test_draft_unapproved_no_side_effects(self, tmp_path):
        """Draft is never auto-approved; no execute/render/approval called."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        with (
            patch(f"src.orchestration.config_generation.draft_validator.approve_draft") as m_approve,
            patch(f"src.orchestration.api.research_api.render_draft_to_yaml", create=True) as m_render,
        ):
            result = api.generate_config_change_draft(
                "test_exp",
                changes=[{"field_path": "model.params.alpha", "operation": "set", "value": 3.0}],
                **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
            )
        assert result["status"] == "ok"
        assert result["draft"].approved is False
        m_approve.assert_not_called()
        m_render.assert_not_called()


# ---------------------------------------------------------------------------
# 12–15: Error paths
# ---------------------------------------------------------------------------


class TestGenerateConfigChangeDraftErrors:
    def test_invalid_feature_type_rejected(self, tmp_path):
        """Invented feature type is refused; status is invalid_changes."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                "value": {"name": "bad_feat", "type": "invented_feature_xyz", "params": {}},
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert result["errors"]
        assert any("invented_feature_xyz" in e for e in result["errors"])

    def test_remove_absent_feature_rejected(self, tmp_path):
        """Removing a feature name that does not exist in the config is refused."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "remove",
                "value": "nonexistent_feature",
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert any("nonexistent_feature" in e for e in result["errors"])

    def test_duplicate_add_rejected(self, tmp_path):
        """Adding a feature whose name already exists in the config is refused."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                "value": {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert any("mom_20" in e for e in result["errors"])

    def test_schema_incompatible_change_rejected(self, tmp_path):
        """Setting train_months to a non-positive value fails validate_ml_config."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "validation.parameters.train_months",
                "operation": "set",
                "value": -1,
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "schema_incompatible"
        assert result["errors"]


# ---------------------------------------------------------------------------
# 16: No LLM call in any introspection or change-draft tool
# ---------------------------------------------------------------------------


class TestNoLlmCall:
    def test_all_introspection_tools_make_no_llm_call(self, tmp_path):
        """None of the 5 new tools calls the LLM."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        with patch(_LLM_MOD) as m_llm:
            api.inspect_experiment_config(
                "test_exp",
                configs_base=d["configs_base"],
            )
            api.list_changeable_config_fields(
                "test_exp",
                configs_base=d["configs_base"],
            )
            api.list_available_features(
                "test_exp",
                configs_base=d["configs_base"],
            )
            api.list_supported_models(
                "test_exp",
                configs_base=d["configs_base"],
            )
            api.generate_config_change_draft(
                "test_exp",
                changes=[{"field_path": "model.params.alpha", "operation": "set", "value": 1.5}],
                **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
            )
        m_llm.assert_not_called()


# ---------------------------------------------------------------------------
# 17: MCP envelope contract
# ---------------------------------------------------------------------------

_CONTRACT_KEYS = {"ok", "stage", "display", "data", "next_suggested_action"}


class TestMcpEnvelopeContract:
    def test_all_new_mcp_tools_return_valid_envelope(self, tmp_path):
        """All 5 new MCP tools return the standard compact envelope."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)

        with patch(f"{_MCP_MOD}.inspect_experiment_config") as m_inspect:
            m_inspect.return_value = {
                "status": "ok",
                "model": {"type": "RidgeRegression"},
                "features": {"count": 2, "ticker": "SPY", "entries": ["mom_20", "mom_60"]},
                "validation": {"type": "rolling"},
                "experiment_name": "test_exp",
                "config_path": "configs/experiments/test_exp.yaml",
                "labels": {}, "signal": {}, "execution": {}, "universe": {},
                "changeable_paths": ["model.params.alpha"],
            }
            out = zeto.inspect_experiment_config("test_exp")
        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "experiment_config_inspected"

        with patch(f"{_MCP_MOD}.list_changeable_config_fields") as m_lcf:
            m_lcf.return_value = {
                "status": "ok",
                "fields": [{"field_path": "model.params.alpha", "operations": ["set"], "type": "float"}],
                "feature_entry_count": 2,
                "experiment_name": None,
            }
            out = zeto.list_changeable_config_fields()
        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "changeable_config_fields_listed"

        with patch(f"{_MCP_MOD}.list_available_features") as m_laf:
            m_laf.return_value = {
                "status": "ok",
                "features": [{"type": "sma", "family": "moving_average",
                               "required_params": ["window"], "currently_used": None,
                               "operations": ["add", "remove", "replace"]}],
                "total_types": 1,
                "currently_used_count": None,
                "current_feature_names": None,
                "experiment_name": None,
            }
            out = zeto.list_available_features()
        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "available_features_listed"

        with patch(f"{_MCP_MOD}.list_supported_models") as m_lsm:
            m_lsm.return_value = {
                "status": "ok",
                "current_model": None,
                "supported_models": [{"name": "RidgeRegression", "required_params": [],
                                       "optional_params": ["alpha"], "is_current": False}],
                "model_switching_supported": True,
                "experiment_name": None,
            }
            out = zeto.list_supported_models()
        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "supported_models_listed"

    def test_generate_config_change_draft_mcp_envelope(self, tmp_path):
        """MCP generate_config_change_draft returns correct envelope; next action is validate."""
        from src.orchestration.config_generation.draft_schema import (
            DraftChange,
            ExperimentDraft,
            compute_draft_hash,
        )
        changes = [DraftChange(
            section="model", field="params.alpha",
            current_value=0.5, proposed_value=2.0,
            rationale="test",
        )]
        stub_draft = ExperimentDraft(
            draft_id="d1",
            draft_hash=compute_draft_hash("test_exp", "test_exp_v2", changes),
            base_experiment="test_exp",
            source_proposal_hash="",
            proposed_name="test_exp_v2",
            changes=changes,
            generated_at="2026-01-01T00:00:00+00:00",
        )
        with patch(f"{_MCP_MOD}.generate_config_change_draft") as m_gcd:
            m_gcd.return_value = {"status": "ok", "draft": stub_draft}
            out = zeto.generate_config_change_draft(
                "test_exp",
                changes=json.dumps([
                    {"field_path": "model.params.alpha", "operation": "set", "value": 2.0}
                ]),
            )
        assert _CONTRACT_KEYS.issubset(out)
        assert out["ok"] is True
        assert out["stage"] == "config_change_draft_generated"
        assert out["next_suggested_action"] == "validate_experiment_draft"
        assert out["data"]["approved"] is False


# ---------------------------------------------------------------------------
# 18: No arbitrary file access
# ---------------------------------------------------------------------------


class TestNoArbitraryFileAccess:
    def test_list_available_features_uses_no_shell(self):
        """list_available_features calls no subprocess or shell tools."""
        import subprocess
        import src.orchestration.api.research_api as _mod
        with patch.object(subprocess, "run") as m_sub, \
             patch.object(subprocess, "Popen") as m_popen:
            api.list_available_features()
        m_sub.assert_not_called()
        m_popen.assert_not_called()

    def test_list_supported_models_uses_no_shell(self):
        """list_supported_models calls no subprocess or shell tools."""
        import subprocess
        with patch.object(subprocess, "run") as m_sub, \
             patch.object(subprocess, "Popen") as m_popen:
            api.list_supported_models()
        m_sub.assert_not_called()
        m_popen.assert_not_called()


# ---------------------------------------------------------------------------
# 19: Operator manual routing rules
# ---------------------------------------------------------------------------


class TestOperatorManualRoutingRules:
    def test_routing_rules_for_config_tools_present(self):
        """Operator manual contains routing rules for new config introspection tools."""
        blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
        # Config routing keywords.
        assert "list_changeable_config_fields" in blob
        assert "list_available_features" in blob
        assert "list_supported_models" in blob
        assert "inspect_experiment_config" in blob
        assert "generate_config_change_draft" in blob
        # Anti-invention rule.
        assert "never invent" in blob

    def test_operator_manual_within_budget(self):
        """Operator manual JSON stays under the 5600-char ceiling."""
        assert len(json.dumps(zeto.get_zeto_operator_manual())) < 5600

    def test_operator_manual_contains_no_retry_rule(self):
        """Operator manual instructs the agent to stop on draft failure, not retry."""
        blob = json.dumps(zeto.get_zeto_operator_manual()).lower()
        assert "do not retry" in blob
        assert "proposed_value" in blob


# ---------------------------------------------------------------------------
# 20+: UX hardening — proposed_value alias, missing value errors, loop prevention
# ---------------------------------------------------------------------------


class TestGenerateConfigChangeDraftUxHardenings:
    def test_proposed_value_alias_accepted_for_add(self, tmp_path):
        """proposed_value is normalised to value for a feature add; draft succeeds."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                "proposed_value": {
                    "name": "ram_20",
                    "type": "risk_adjusted_momentum",
                    "params": {"mom_window": 20},
                },
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok", result.get("errors")
        add_change = next(c for c in result["draft"].changes if c.field == "entries.add")
        assert add_change.proposed_value["name"] == "ram_20"
        assert add_change.proposed_value["type"] == "risk_adjusted_momentum"

    def test_proposed_value_alias_accepted_for_remove(self, tmp_path):
        """proposed_value alias normalised to value for feature remove."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "remove",
                "proposed_value": "mom_60",
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok", result.get("errors")
        remove_change = next(c for c in result["draft"].changes if c.field == "entries.remove")
        assert remove_change.proposed_value == "mom_60"

    def test_proposed_value_alias_accepted_for_scalar_set(self, tmp_path):
        """proposed_value alias normalised to value for a scalar set."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "model.params.alpha",
                "operation": "set",
                "proposed_value": 3.0,
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok", result.get("errors")
        change = result["draft"].changes[0]
        assert change.proposed_value == 3.0

    def test_both_value_and_proposed_value_same_accepted(self, tmp_path):
        """Both keys with identical values is treated as canonical (no conflict)."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "model.params.alpha",
                "operation": "set",
                "value": 1.5,
                "proposed_value": 1.5,
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "ok", result.get("errors")

    def test_both_value_and_proposed_value_differ_rejected(self, tmp_path):
        """Both keys with different values returns invalid_changes with a clear message."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "model.params.alpha",
                "operation": "set",
                "value": 1.0,
                "proposed_value": 2.0,
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert any("differ" in e.lower() for e in result["errors"])

    def test_missing_value_for_add_returns_helpful_error(self, tmp_path):
        """Missing value key returns the guided error message with an example."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                # No 'value' or 'proposed_value' — truly missing.
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        err = " ".join(result["errors"])
        assert "Missing 'value'" in err
        assert "proposed_value" in err
        assert "risk_adjusted_momentum" in err  # example in the message

    def test_missing_value_for_remove_returns_helpful_error(self, tmp_path):
        """Missing value key for remove returns the guided error message."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "remove",
                # No 'value' or 'proposed_value'.
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert "Missing 'value'" in " ".join(result["errors"])

    def test_failure_suggests_stop_not_retry(self, tmp_path):
        """MCP tool always suggests stop_and_report_to_user on draft failure."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        with patch(f"{_MCP_MOD}.generate_config_change_draft") as m_gcd:
            m_gcd.return_value = {
                "status": "invalid_changes",
                "errors": ["Unsupported feature type 'invented_type'."],
            }
            out = zeto.generate_config_change_draft(
                "test_exp",
                changes=json.dumps([{
                    "field_path": "features.entries",
                    "operation": "add",
                    "value": {"name": "bad", "type": "invented_type", "params": {}},
                }]),
            )
        assert out["ok"] is False
        assert out["stage"] == "config_change_draft_failed"
        assert out["next_suggested_action"] == "stop_and_report_to_user"

    def test_failure_suggests_stop_for_field_path_errors(self, tmp_path):
        """Field-path errors also route to stop_and_report_to_user, not a retry tool."""
        with patch(f"{_MCP_MOD}.generate_config_change_draft") as m_gcd:
            m_gcd.return_value = {
                "status": "invalid_changes",
                "errors": ["Invalid or disallowed field_path 'model.foo'."],
            }
            out = zeto.generate_config_change_draft(
                "test_exp",
                changes=json.dumps([{
                    "field_path": "model.foo",
                    "operation": "set",
                    "value": 1,
                }]),
            )
        assert out["next_suggested_action"] == "stop_and_report_to_user"

    def test_invalid_feature_type_still_rejected_with_proposed_value_alias(self, tmp_path):
        """proposed_value alias is normalised but invalid feature type still fails."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        result = api.generate_config_change_draft(
            "test_exp",
            changes=[{
                "field_path": "features.entries",
                "operation": "add",
                "proposed_value": {
                    "name": "bad_feat",
                    "type": "completely_invented_type",
                    "params": {},
                },
            }],
            **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
        )
        assert result["status"] == "invalid_changes"
        assert any("completely_invented_type" in e for e in result["errors"])

    def test_no_side_effects_on_draft_failure(self, tmp_path):
        """A failed draft call never approves, renders, or executes anything."""
        _make_config(tmp_path)
        d = _draft_dirs(tmp_path)
        with (
            patch("src.orchestration.config_generation.draft_validator.approve_draft") as m_approve,
            patch("src.orchestration.config_generation.draft_generator.dump_json") as m_dump,
        ):
            result = api.generate_config_change_draft(
                "test_exp",
                changes=[{
                    "field_path": "features.entries",
                    "operation": "add",
                    "value": {"name": "x", "type": "nonexistent_type", "params": {}},
                }],
                **{k: d[k] for k in ("base", "llm_base", "configs_base", "reports_base")},
            )
        assert result["status"] == "invalid_changes"
        m_approve.assert_not_called()
        m_dump.assert_not_called()
