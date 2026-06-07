"""Tests for explicit parameter-change drafts and authoritative metrics.

Covers:
  * generate_parameter_change_draft — deterministic, LLM-free single-field draft
  * get_experiment_metrics / compare_experiment_metrics — real artefact metrics

All disk I/O uses tmp_path; no LLM, no quant engine, no LM Studio.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import yaml
from src.orchestration.api import research_api as api

_BASE_CONFIG = {
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

_DG = "src.orchestration.config_generation.draft_generator"


def _make_config(tmp_path: Path, name: str = "canonical_ml_showcase") -> Path:
    cfg_dir = tmp_path / "configs" / "experiments"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{name}.yaml").write_text(yaml.dump(_BASE_CONFIG), encoding="utf-8")
    return cfg_dir


def _dirs(tmp_path: Path) -> dict:
    return {
        "base": tmp_path / "results" / "experiments",
        "reports_base": tmp_path / "reports",
        "llm_base": tmp_path / "results" / "llm_reviews",
    }


# ---------------------------------------------------------------------------
# Part A — generate_parameter_change_draft
# ---------------------------------------------------------------------------


class TestParameterChangeDraft:
    def test_creates_exactly_requested_change(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.alpha", 2.0,
            reason="stronger ridge", configs_base=cfg, **d,
        )
        assert result["status"] == "ok"
        draft = result["draft"]
        assert len(draft.changes) == 1
        ch = draft.changes[0]
        assert (ch.section, ch.field) == ("model", "params.alpha")
        assert ch.proposed_value == 2.0
        assert ch.rationale == "stronger ridge"
        assert draft.approved is False

    def test_reads_current_value_from_config(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.alpha", 2.0, configs_base=cfg, **d,
        )
        # current value comes from the base config (0.5), not the caller.
        assert result["draft"].changes[0].current_value == 0.5

    def test_refuses_invalid_field_path(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.not_a_field", 2.0, configs_base=cfg, **d,
        )
        assert result["status"] == "invalid_field_path"
        assert "draft" not in result
        assert any("Invalid or disallowed field_path" in e for e in result["errors"])

    def test_refuses_schema_incompatible_value(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        # model.type is an allowed path, but "BananaRegression" is not a real model.
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.type", "BananaRegression", configs_base=cfg, **d,
        )
        assert result["status"] == "schema_incompatible"
        assert "draft" not in result
        assert result["errors"]

    def test_refuses_when_config_missing(self, tmp_path):
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "no_such_exp", "model.params.alpha", 2.0,
            configs_base=tmp_path / "configs" / "experiments", **d,
        )
        assert result["status"] == "config_not_found"

    def test_does_not_approve_render_or_execute(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.alpha", 2.0, configs_base=cfg, **d,
        )
        draft = result["draft"]
        assert draft.approved is False
        assert draft.approved_at is None
        # No YAML rendered: no config file for the proposed name was written.
        assert not (cfg / f"{draft.proposed_name}.yaml").exists()
        # No results directory created for the proposed name (nothing executed).
        assert not (d["base"] / draft.proposed_name).exists()

    def test_does_not_call_an_llm(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        with patch(f"{_DG}.call_llm") as m_llm:
            api.generate_parameter_change_draft(
                "canonical_ml_showcase", "model.params.alpha", 2.0, configs_base=cfg, **d,
            )
        m_llm.assert_not_called()

    def test_uses_unique_proposed_name(self, tmp_path):
        cfg = _make_config(tmp_path)
        # Pre-existing v2 config forces the next free suffix (v3).
        (cfg / "canonical_ml_showcase_v2.yaml").write_text(
            yaml.dump(_BASE_CONFIG), encoding="utf-8"
        )
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.alpha", 2.0, configs_base=cfg, **d,
        )
        assert result["draft"].proposed_name == "canonical_ml_showcase_v3"

    def test_persists_draft_artifact(self, tmp_path):
        cfg = _make_config(tmp_path)
        d = _dirs(tmp_path)
        result = api.generate_parameter_change_draft(
            "canonical_ml_showcase", "model.params.alpha", 2.0, configs_base=cfg, **d,
        )
        draft = result["draft"]
        path = d["llm_base"] / "canonical_ml_showcase" / f"draft_{draft.draft_id}.json"
        assert path.exists()
        # Reloads with the exact requested change.
        saved = json.loads(path.read_text())
        assert saved["changes"][0]["proposed_value"] == 2.0


# ---------------------------------------------------------------------------
# Part B — get_experiment_metrics / compare_experiment_metrics
# ---------------------------------------------------------------------------


def _write_experiment(
    base: Path,
    name: str,
    *,
    sharpe=0.61,
    mean_oos=-0.22,
    std_oos=1.11,
    max_dd=-0.312,
    with_validation=True,
) -> None:
    exp = base / name
    (exp / "diagnostics").mkdir(parents=True, exist_ok=True)
    (exp / "metadata.json").write_text(json.dumps({
        "experiment_name": name, "strategy_name": "MLStrategy(Ridge)",
        "created_at": "2026-06-06T00:00:00+00:00",
    }), encoding="utf-8")
    (exp / "metrics.json").write_text(json.dumps({
        "sharpe_ratio": sharpe,
        "annualized_return": 0.08,
        "annualized_volatility": 0.131,
        "max_drawdown": max_dd,
        "calmar_ratio": 0.26,
        "hit_rate": 0.51,
    }), encoding="utf-8")
    if with_validation:
        (exp / "diagnostics" / "split_metrics.json").write_text(json.dumps({
            "summary": {
                "n_splits": 6,
                "mean_sharpe": mean_oos,
                "std_sharpe": std_oos,
                "hit_rate_positive_sharpe": 0.33,
                "mean_annualized_return": 0.02,
                "worst_max_drawdown": -0.40,
            },
            "splits": [
                {"sharpe_ratio": 0.4}, {"sharpe_ratio": -0.5}, {"sharpe_ratio": -0.2},
                {"sharpe_ratio": 0.1}, {"sharpe_ratio": -0.6}, {"sharpe_ratio": -0.3},
            ],
        }), encoding="utf-8")


class TestExperimentMetrics:
    def test_loads_metrics_from_artefacts(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v2")
        result = api.get_experiment_metrics("exp_v2", base=base)
        assert result["status"] == "ok"
        m = result["metrics"]
        assert m["sharpe_ratio"] == 0.61
        assert m["mean_oos_sharpe"] == -0.22
        assert m["std_oos_sharpe"] == 1.11
        assert m["n_splits"] == 6
        assert m["max_drawdown_pct"] == "-31.20%"
        assert m["consistency_tier"] in {"weak", "moderate", "strong", "unknown"}
        assert result["missing_metrics"] == []
        assert result["metrics_path"].endswith("metrics.json")

    def test_failure_modes_present(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v2")  # negative mean OOS sharpe -> failure modes
        result = api.get_experiment_metrics("exp_v2", base=base)
        assert result["failure_modes"]  # detected from real diagnostics

    def test_missing_validation_metrics_handled_cleanly(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v2", with_validation=False)
        result = api.get_experiment_metrics("exp_v2", base=base)
        assert result["status"] == "ok"
        # Validation metrics are reported as missing, never invented.
        assert "mean_oos_sharpe" in result["missing_metrics"]
        assert result["metrics"]["mean_oos_sharpe"] is None
        assert result["metrics"]["sharpe_ratio"] == 0.61  # perf still present

    def test_not_found_experiment(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        result = api.get_experiment_metrics("nope", base=base)
        assert result["status"] == "not_found"
        assert result["metrics"]["sharpe_ratio"] is None
        assert "sharpe_ratio" in result["missing_metrics"]

    def test_does_not_use_rag_memory(self):
        # The metrics functions must not read the research-memory index.
        import inspect
        src = inspect.getsource(api.get_experiment_metrics)
        src += inspect.getsource(api.compare_experiment_metrics)
        assert "retrieve_memory" not in src
        assert "semantic" not in src
        assert "memory_index" not in src


class TestCompareMetrics:
    def test_computes_deltas(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v1", sharpe=0.40, mean_oos=-0.30, max_dd=-0.34)
        _write_experiment(base, "exp_v2", sharpe=0.60, mean_oos=-0.20, max_dd=-0.312)
        result = api.compare_experiment_metrics("exp_v1", "exp_v2", base=base)
        assert result["status"] == "ok"
        assert result["base_sharpe"] == 0.40
        assert result["candidate_sharpe"] == 0.60
        assert result["delta_sharpe"] == 0.2
        assert result["delta_mean_oos_sharpe"] == 0.1
        # max drawdown: -31.20% - (-34.00%) = +2.8 percentage points
        assert result["delta_max_drawdown_pct"] == 2.8
        assert "conclusion" in result and result["conclusion"]

    def test_missing_candidate_handled_cleanly(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v1")
        result = api.compare_experiment_metrics("exp_v1", "missing_v2", base=base)
        assert result["status"] == "not_found"
        assert "missing_v2" in result["missing_experiments"]

    def test_delta_none_when_metric_missing(self, tmp_path):
        base = tmp_path / "results" / "experiments"
        _write_experiment(base, "exp_v1", with_validation=False)
        _write_experiment(base, "exp_v2", with_validation=False)
        result = api.compare_experiment_metrics("exp_v1", "exp_v2", base=base)
        assert result["status"] == "ok"
        # No validation artefacts -> mean OOS sharpe missing -> delta is None.
        assert result["delta_mean_oos_sharpe"] is None
        # Sharpe still compares.
        assert result["delta_sharpe"] is not None
