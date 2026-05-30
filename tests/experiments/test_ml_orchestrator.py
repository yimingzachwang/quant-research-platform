"""Integration tests for the F3 ML experiment pipeline in orchestrator.py.

Data loading is patched so tests run offline and deterministically.
All tests exercise the version "2" config code path via run_experiment_from_config().
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from src.experiments.orchestrator import ExperimentRun, run_experiment_from_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(tickers: list[str] | None = None, n: int = 600) -> pd.DataFrame:
    tickers = tickers or ["SPY"]
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    data = rng.lognormal(0.0002, 0.01, size=(n, len(tickers))).cumprod(axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_raw_df(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return pd.DataFrame({"close": prices[ticker]})


def _make_load_universe_patch(prices: pd.DataFrame):
    def _mock(symbols, **kwargs):
        return {sym: _make_raw_df(prices, sym) for sym in symbols if sym in prices.columns}
    return _mock


def _write_v2_cfg(tmp_path: Path, cfg: dict | None = None, name: str = "ml_cfg.yaml") -> Path:
    base = {
        "version": "2",
        "name": "ml_test_run",
        "universe": {"tickers": ["SPY"]},
        "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
        "features": {
            "ticker": "SPY",
            "entries": [{"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}],
        },
        "labels": {"type": "forward_returns", "params": {"horizon": 5}},
        "model": {"type": "RidgeRegression", "params": {}},
        "signal": {"type": "sign", "params": {}},
        "output": {
            "base_dir": str(tmp_path / "results"),
            "registry_path": str(tmp_path / "results" / "registry.json"),
            "register": True,
            "save_plots": False,
        },
    }
    if cfg:
        base.update(cfg)
    p = tmp_path / name
    p.write_text(yaml.dump(base), encoding="utf-8")
    return p


_PRICES = _make_prices(n=600)
_PATCH = "src.portfolio.alignment.load_universe"


# ---------------------------------------------------------------------------
# Version routing
# ---------------------------------------------------------------------------


class TestVersionRouting:
    def test_v2_config_routes_to_ml_pipeline(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run = run_experiment_from_config(p)
        assert isinstance(run, ExperimentRun)

    def test_v1_config_unaffected(self, tmp_path: Path):
        v1_cfg = {
            "name": "v1_orch_test",
            "universe": {"tickers": ["SPY"]},
            "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
            "strategy": {"type": "EqualWeight", "parameters": {"rebalance_freq": "ME"}},
            "output": {
                "base_dir": str(tmp_path / "results_v1"),
                "registry_path": str(tmp_path / "results_v1" / "registry.json"),
                "register": False,
                "save_plots": False,
            },
        }
        p = tmp_path / "v1.yaml"
        p.write_text(yaml.dump(v1_cfg), encoding="utf-8")
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run = run_experiment_from_config(p)
        assert isinstance(run, ExperimentRun)


# ---------------------------------------------------------------------------
# ExperimentRun structure
# ---------------------------------------------------------------------------


class TestMLExperimentRunStructure:
    def _run(self, tmp_path: Path) -> ExperimentRun:
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            return run_experiment_from_config(p)

    def test_run_has_spec(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run.spec is not None
        assert run.spec.experiment_name == "ml_test_run"

    def test_run_has_strategy_result(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run.strategy_result is not None

    def test_run_has_experiment_result(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run.experiment_result is not None
        assert run.experiment_result.experiment_name == "ml_test_run"

    def test_run_no_walk_forward_when_none(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run.walk_forward is None

    def test_output_path_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run.output_path.is_dir()


# ---------------------------------------------------------------------------
# Artefacts written
# ---------------------------------------------------------------------------


class TestMLArtefacts:
    def _run(self, tmp_path: Path) -> ExperimentRun:
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            return run_experiment_from_config(p)

    def test_metadata_json_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert (run.output_path / "metadata.json").exists()

    def test_metrics_json_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert (run.output_path / "metrics.json").exists()

    def test_ml_provenance_json_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert (run.output_path / "ml_provenance.json").exists()

    def test_ml_provenance_contents(self, tmp_path: Path):
        run = self._run(tmp_path)
        with (run.output_path / "ml_provenance.json").open() as f:
            prov = json.load(f)
        assert prov["name"] == "ml_test_run"
        assert "ml_hash" in prov
        assert len(prov["ml_hash"]) == 12
        assert "features" in prov
        assert "labels" in prov
        assert "model" in prov
        assert "signal" in prov

    def test_raw_config_artefact_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert (run.output_path / "raw_config.yaml").exists()

    def test_normalized_config_artefact_exists(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert (run.output_path / "normalized_config.json").exists()

    def test_normalized_config_has_version_2(self, tmp_path: Path):
        run = self._run(tmp_path)
        with (run.output_path / "normalized_config.json").open() as f:
            norm = json.load(f)
        assert norm["version"] == "2"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestMLRegistry:
    def test_registry_entry_created(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run_experiment_from_config(p)
        registry_path = tmp_path / "results" / "registry.json"
        assert registry_path.exists()
        entries = json.loads(registry_path.read_text())
        assert len(entries) == 1
        assert entries[0]["experiment_name"] == "ml_test_run"

    def test_rerun_replaces_registry_entry(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run_experiment_from_config(p)
            run_experiment_from_config(p)
        registry_path = tmp_path / "results" / "registry.json"
        entries = json.loads(registry_path.read_text())
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Walk-forward in ML pipeline
# ---------------------------------------------------------------------------


class TestMLWalkForward:
    def test_walk_forward_runs(self, tmp_path: Path):
        extra = {
            "validation": {
                "type": "rolling",
                "parameters": {"train_months": 18, "test_months": 6},
            },
        }
        p = _write_v2_cfg(tmp_path, extra)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run = run_experiment_from_config(p)
        assert run.walk_forward is not None
        assert run.walk_forward.n_splits > 0


# ---------------------------------------------------------------------------
# Validation error propagation
# ---------------------------------------------------------------------------


class TestMLValidationErrors:
    def test_invalid_feature_type_raises(self, tmp_path: Path):
        extra = {
            "features": {
                "ticker": "SPY",
                "entries": [{"name": "bad", "type": "unknown_feat", "params": {}}],
            }
        }
        p = _write_v2_cfg(tmp_path, extra)
        with pytest.raises(ValueError, match="unknown"):
            with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
                run_experiment_from_config(p)

    def test_panel_signal_top_n_runs(self, tmp_path: Path):
        extra = {
            "signal": {"type": "top_n", "params": {"n": 1}},
            "labels": {"type": "ranking_target", "params": {"horizon": 5}},
        }
        p = _write_v2_cfg(tmp_path, extra)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES)):
            run = run_experiment_from_config(p)
        assert run.experiment_result is not None
