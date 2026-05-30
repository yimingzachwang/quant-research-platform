"""End-to-end workflow regression tests.

Covers the complete config → run → report pipeline for both v1 and v2 configs.
Data loading is patched so tests run offline and deterministically.
These tests verify that all expected artefacts are produced and that the report
pipeline can consume them without errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.contracts import check_artefact_dir, check_ml_artefacts
from src.experiments.orchestrator import run_experiment_from_config
from src.reporting.report_builder import generate_experiment_report, load_experiment_artefacts
from src.reporting.report_spec import COMPACT_REPORT, DIAGNOSTICS_REPORT


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_prices(tickers: list[str], n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    data = rng.lognormal(0.0002, 0.01, size=(n, len(tickers))).cumprod(axis=0) * 100
    return pd.DataFrame(data, index=idx, columns=tickers)


def _make_load_universe_patch(prices: pd.DataFrame):
    def _mock(symbols, **kwargs):
        return {sym: pd.DataFrame({"close": prices[sym]}) for sym in symbols if sym in prices.columns}
    return _mock


_PRICES_V1 = _make_prices(["SPY", "QQQ", "TLT"])
_PRICES_V2 = _make_prices(["SPY"])
_PATCH = "src.portfolio.alignment.load_universe"


def _write_v1_cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "1",
        "name": "wf_v1_test",
        "universe": {"tickers": ["SPY", "QQQ", "TLT"]},
        "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
        "strategy": {"type": "EqualWeight", "parameters": {"rebalance_freq": "ME"}},
        "output": {
            "base_dir": str(tmp_path / "results"),
            "registry_path": str(tmp_path / "results" / "registry.json"),
            "register": True,
            "save_plots": False,
        },
    }
    p = tmp_path / "v1.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def _write_v2_cfg(tmp_path: Path) -> Path:
    cfg = {
        "version": "2",
        "name": "wf_v2_test",
        "universe": {"tickers": ["SPY"]},
        "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
        "features": {
            "ticker": "SPY",
            "entries": [
                {"name": "mom_20", "type": "momentum", "params": {"lookback": 20}},
                {"name": "vol_21", "type": "rolling_volatility", "params": {"window": 21}},
            ],
        },
        "labels": {"type": "forward_returns", "params": {"horizon": 5}},
        "model": {"type": "RidgeRegression", "params": {"alpha": 1.0}},
        "signal": {"type": "sign", "params": {}},
        "output": {
            "base_dir": str(tmp_path / "results"),
            "registry_path": str(tmp_path / "results" / "registry.json"),
            "register": True,
            "save_plots": False,
        },
    }
    p = tmp_path / "v2.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# V1 workflow
# ---------------------------------------------------------------------------


class TestV1Workflow:
    """v1 strategy-based config runs to completion and produces required artefacts."""

    def _run(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            return run_experiment_from_config(p)

    def test_run_completes(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run is not None

    def test_required_artefacts_present(self, tmp_path: Path):
        run = self._run(tmp_path)
        violations = check_artefact_dir(run.output_path)
        assert violations == [], violations

    def test_metadata_json_correct(self, tmp_path: Path):
        run = self._run(tmp_path)
        meta = json.loads((run.output_path / "metadata.json").read_text())
        assert meta["experiment_name"] == "wf_v1_test"

    def test_metrics_json_has_sharpe(self, tmp_path: Path):
        run = self._run(tmp_path)
        metrics = json.loads((run.output_path / "metrics.json").read_text())
        assert "sharpe_ratio" in metrics

    def test_no_ml_provenance_for_v1(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert not (run.output_path / "ml_provenance.json").exists()

    def test_registry_entry_created(self, tmp_path: Path):
        run = self._run(tmp_path)
        registry_path = tmp_path / "results" / "registry.json"
        assert registry_path.exists()
        entries = json.loads(registry_path.read_text())
        names = [e["experiment_name"] for e in entries]
        assert "wf_v1_test" in names

    def test_registry_rerun_no_duplicate(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            run_experiment_from_config(p)
            run_experiment_from_config(p)
        entries = json.loads((tmp_path / "results" / "registry.json").read_text())
        names = [e["experiment_name"] for e in entries if e["experiment_name"] == "wf_v1_test"]
        assert len(names) == 1


# ---------------------------------------------------------------------------
# V2 workflow
# ---------------------------------------------------------------------------


class TestV2Workflow:
    """v2 ML config runs to completion and produces all expected artefacts."""

    def _run(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            return run_experiment_from_config(p)

    def test_run_completes(self, tmp_path: Path):
        run = self._run(tmp_path)
        assert run is not None

    def test_required_artefacts_present(self, tmp_path: Path):
        run = self._run(tmp_path)
        violations = check_artefact_dir(run.output_path)
        assert violations == [], violations

    def test_ml_artefacts_present(self, tmp_path: Path):
        run = self._run(tmp_path)
        violations = check_ml_artefacts(run.output_path)
        assert violations == [], violations

    def test_ml_provenance_fields(self, tmp_path: Path):
        run = self._run(tmp_path)
        prov = json.loads((run.output_path / "ml_provenance.json").read_text())
        assert prov["name"] == "wf_v2_test"
        assert "spec_version" in prov
        assert len(prov["ml_hash"]) == 12
        assert len(prov["features"]) == 2
        assert prov["labels"]["type"] == "forward_returns"
        assert prov["model"]["type"] == "RidgeRegression"
        assert prov["signal"]["type"] == "sign"

    def test_normalized_config_version_2(self, tmp_path: Path):
        run = self._run(tmp_path)
        norm = json.loads((run.output_path / "normalized_config.json").read_text())
        assert norm["version"] == "2"

    def test_registry_entry_created(self, tmp_path: Path):
        run = self._run(tmp_path)
        entries = json.loads((tmp_path / "results" / "registry.json").read_text())
        names = [e["experiment_name"] for e in entries]
        assert "wf_v2_test" in names

    def test_registry_rerun_no_duplicate(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run_experiment_from_config(p)
            run_experiment_from_config(p)
        entries = json.loads((tmp_path / "results" / "registry.json").read_text())
        names = [e["experiment_name"] for e in entries if e["experiment_name"] == "wf_v2_test"]
        assert len(names) == 1


# ---------------------------------------------------------------------------
# Report pipeline — v1 artefacts
# ---------------------------------------------------------------------------


class TestReportFromV1:
    """Report generation from v1 experiment artefacts."""

    def _setup(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            run = run_experiment_from_config(p)
        report_dir = tmp_path / "reports"
        paths = generate_experiment_report(run.output_path, output_dir=report_dir)
        return run, paths

    def test_markdown_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.markdown.exists()

    def test_html_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.html is not None and paths.html.exists()

    def test_provenance_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.provenance.exists()

    def test_provenance_fields(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        prov = json.loads(paths.provenance.read_text())
        assert prov["source_experiment"] == "wf_v1_test"
        assert "generated_at" in prov
        assert "report_version" in prov
        assert "artefact_version" in prov

    def test_markdown_contains_experiment_name(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "wf_v1_test" in content

    def test_markdown_no_ml_section(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "Model & Features" not in content


# ---------------------------------------------------------------------------
# Report pipeline — v2 artefacts
# ---------------------------------------------------------------------------


class TestReportFromV2:
    """Report generation from v2 (ML) experiment artefacts."""

    def _setup(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        report_dir = tmp_path / "reports"
        paths = generate_experiment_report(run.output_path, output_dir=report_dir)
        return run, paths

    def test_markdown_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.markdown.exists()

    def test_html_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.html is not None and paths.html.exists()

    def test_provenance_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.provenance.exists()

    def test_markdown_has_ml_behavior_section(self, tmp_path: Path):
        # Model & Features is now gated on include_ml_provenance_detail (off by default).
        # ML Model Behaviour is always rendered for v2 experiments (include_ml_analysis).
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "## ML Model Behaviour" in content

    def test_markdown_shows_model_type(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "RidgeRegression" in content

    def test_markdown_shows_feature_names(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "mom_20" in content
        assert "vol_21" in content

    def test_markdown_has_summary_section(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        content = paths.markdown.read_text()
        assert "**Period:**" in content

    def test_artefacts_loaded_correctly(self, tmp_path: Path):
        run, _ = self._setup(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.ml_provenance, dict)
        assert artefacts.ml_provenance["name"] == "wf_v2_test"


# ---------------------------------------------------------------------------
# Diagnostics persistence — v2 with walk-forward
# ---------------------------------------------------------------------------


def _write_v2_wf_cfg(tmp_path: Path) -> Path:
    """v2 config with walk-forward validation enabled."""
    cfg = {
        "version": "2",
        "name": "wf_v2_diag_test",
        "universe": {"tickers": ["SPY"]},
        "date_range": {"start": "2018-01-01", "end": "2020-12-31"},
        "features": {
            "ticker": "SPY",
            "entries": [{"name": "mom_20", "type": "momentum", "params": {"lookback": 20}}],
        },
        "labels": {"type": "forward_returns", "params": {"horizon": 5}},
        "model": {"type": "RidgeRegression", "params": {"alpha": 1.0}},
        "signal": {"type": "sign", "params": {}},
        "validation": {
            "type": "rolling",
            "parameters": {"train_months": 18, "test_months": 6},
        },
        "output": {
            "base_dir": str(tmp_path / "results"),
            "registry_path": str(tmp_path / "results" / "registry.json"),
            "register": False,
            "save_plots": False,
        },
    }
    p = tmp_path / "v2_wf.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


class TestDiagnosticsPersistence:
    """Diagnostics artefacts are written correctly during experiment runs."""

    def _run_v2_wf(self, tmp_path: Path):
        p = _write_v2_wf_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            return run_experiment_from_config(p)

    def test_ml_diagnostics_json_written(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        assert (run.output_path / "diagnostics" / "ml_diagnostics.json").exists()

    def test_split_metrics_json_written_when_wf_ran(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        assert (run.output_path / "diagnostics" / "split_metrics.json").exists()

    def test_split_metrics_structure(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        data = json.loads(
            (run.output_path / "diagnostics" / "split_metrics.json").read_text()
        )
        assert "n_splits" in data
        assert data["n_splits"] > 0
        assert "summary" in data
        assert "splits" in data
        assert isinstance(data["splits"], list)
        assert len(data["splits"]) == data["n_splits"]

    def test_split_metrics_splits_have_required_keys(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        data = json.loads(
            (run.output_path / "diagnostics" / "split_metrics.json").read_text()
        )
        for split in data["splits"]:
            assert "sharpe_ratio" in split
            assert "annualized_return" in split
            assert "test_start" in split

    def test_ml_diagnostics_structure(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        data = json.loads(
            (run.output_path / "diagnostics" / "ml_diagnostics.json").read_text()
        )
        assert data["model_type"] == "RidgeRegression"
        assert "average_turnover" in data
        assert "signal_activity" in data
        assert "n_weight_periods" in data

    def test_ml_diagnostics_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        raw = (run.output_path / "diagnostics" / "ml_diagnostics.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_split_metrics_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        raw = (run.output_path / "diagnostics" / "split_metrics.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_no_split_metrics_without_wf(self, tmp_path: Path):
        """v2 without walk-forward: split_metrics.json should not be written."""
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        assert not (run.output_path / "diagnostics" / "split_metrics.json").exists()

    def test_ml_diagnostics_loaded_into_artefacts(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.ml_diagnostics, dict)
        assert artefacts.ml_diagnostics["model_type"] == "RidgeRegression"

    def test_split_metrics_loaded_into_artefacts(self, tmp_path: Path):
        run = self._run_v2_wf(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.split_metrics, dict)
        assert artefacts.split_metrics["n_splits"] > 0


# ---------------------------------------------------------------------------
# Frontend manifests
# ---------------------------------------------------------------------------


class TestReportManifest:
    """report_manifest.json is written correctly by generate_experiment_report()."""

    def _setup(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        paths = generate_experiment_report(run.output_path, tmp_path / "reports")
        return run, paths

    def test_manifest_written(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        assert paths.manifest is not None
        assert paths.manifest.exists()

    def test_manifest_is_valid_json(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        assert isinstance(data, dict)

    def test_manifest_has_required_fields(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        for field in ("experiment_name", "generated_at", "report_version",
                      "artefact_version", "files", "metrics_summary",
                      "has_ml", "has_validation", "has_diagnostics"):
            assert field in data, f"Missing field: {field}"

    def test_manifest_experiment_name_correct(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        assert data["experiment_name"] == "wf_v2_test"

    def test_manifest_files_contain_markdown(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        assert "markdown" in data["files"]
        assert data["files"]["markdown"].endswith(".md")

    def test_manifest_metrics_summary_has_sharpe(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        assert "sharpe_ratio" in data["metrics_summary"]

    def test_manifest_has_ml_true_for_v2(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        assert data["has_ml"] is True

    def test_manifest_has_ml_false_for_v1(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            run = run_experiment_from_config(p)
        paths = generate_experiment_report(run.output_path, tmp_path / "reports")
        data = json.loads(paths.manifest.read_text())
        assert data["has_ml"] is False

    def test_manifest_files_use_relative_paths(self, tmp_path: Path):
        _, paths = self._setup(tmp_path)
        data = json.loads(paths.manifest.read_text())
        for path_str in data["files"].values():
            assert not path_str.startswith("/"), f"Absolute path in manifest: {path_str}"


# ---------------------------------------------------------------------------
# ResearchReportSpec end-to-end
# ---------------------------------------------------------------------------


class TestResearchArtefacts:
    """research/ artefacts are written during experiment runs."""

    def _run_v1(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            return run_experiment_from_config(p)

    def test_data_summary_json_written(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        assert (run.output_path / "research" / "data_summary.json").exists()

    def test_signal_transitions_json_written(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        assert (run.output_path / "research" / "signal_transitions.json").exists()

    def test_data_summary_structure(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "data_summary.json").read_text()
        )
        assert "n_days" in data
        assert "n_assets" in data
        assert data["n_assets"] == 3  # SPY, QQQ, TLT
        assert "assets" in data
        assert "nan_counts" in data
        assert "return_stats" in data

    def test_data_summary_nan_counts_present(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "data_summary.json").read_text()
        )
        for ticker in ["SPY", "QQQ", "TLT"]:
            assert ticker in data["nan_counts"]

    def test_signal_transitions_structure(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "signal_transitions.json").read_text()
        )
        assert "n_rebalances" in data
        assert "transitions" in data
        assert isinstance(data["transitions"], list)

    def test_signal_transitions_events_have_holdings(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "signal_transitions.json").read_text()
        )
        for ev in data["transitions"][:5]:
            assert "date" in ev
            assert "holdings" in ev
            assert "entered" in ev
            assert "exited" in ev

    def test_research_artefacts_loaded_into_loader(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.research_artefacts, dict)
        assert "data_summary" in artefacts.research_artefacts
        assert "signal_transitions" in artefacts.research_artefacts

    def test_data_summary_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        raw = (run.output_path / "research" / "data_summary.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw


class TestBacktestDiagnostics:
    """diagnostics/backtest_diagnostics.json is written for all experiment types."""

    def _run_v1(self, tmp_path: Path):
        p = _write_v1_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V1)):
            return run_experiment_from_config(p)

    def test_backtest_diagnostics_json_written(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        assert (run.output_path / "diagnostics" / "backtest_diagnostics.json").exists()

    def test_backtest_diagnostics_structure(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "diagnostics" / "backtest_diagnostics.json").read_text()
        )
        assert "rolling_sharpe_252d" in data
        assert "rolling_vol_63d" in data
        assert "monthly_avg_turnover" in data
        assert "drawdown_windows" in data
        assert "n_drawdown_windows_gt5pct" in data

    def test_rolling_sharpe_is_list_of_records(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        data = json.loads(
            (run.output_path / "diagnostics" / "backtest_diagnostics.json").read_text()
        )
        for rec in data["rolling_sharpe_252d"]:
            assert "date" in rec
            assert "value" in rec

    def test_backtest_diagnostics_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        raw = (run.output_path / "diagnostics" / "backtest_diagnostics.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    def test_backtest_diagnostics_loaded_into_artefacts(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.backtest_diagnostics, dict)

    def test_manifest_has_research_artefacts_flag(self, tmp_path: Path):
        run = self._run_v1(tmp_path)
        from src.reporting.report_builder import generate_experiment_report
        paths = generate_experiment_report(run.output_path, tmp_path / "reports")
        data = json.loads(paths.manifest.read_text())
        assert "has_research_artefacts" in data
        assert data["has_research_artefacts"] is True


class TestReportSpecEndToEnd:
    """ResearchReportSpec correctly controls output through the full pipeline."""

    def _run_and_report(self, tmp_path: Path, spec=None):
        p = _write_v2_wf_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        paths = generate_experiment_report(
            run.output_path, tmp_path / "reports", report_spec=spec
        )
        return run, paths

    def test_diagnostics_report_includes_diagnostics_section(self, tmp_path: Path):
        _, paths = self._run_and_report(tmp_path, DIAGNOSTICS_REPORT)
        content = paths.markdown.read_text()
        assert "## Diagnostics Appendix" in content

    def test_diagnostics_section_has_split_data(self, tmp_path: Path):
        _, paths = self._run_and_report(tmp_path, DIAGNOSTICS_REPORT)
        content = paths.markdown.read_text()
        assert "Walk-Forward Stability" in content
        assert "ML Signal Diagnostics" in content

    def test_compact_report_omits_validation_section(self, tmp_path: Path):
        _, paths = self._run_and_report(tmp_path, COMPACT_REPORT)
        content = paths.markdown.read_text()
        assert "## Walk-Forward Validation" not in content

    def test_compact_report_omits_figures(self, tmp_path: Path):
        _, paths = self._run_and_report(tmp_path, COMPACT_REPORT)
        content = paths.markdown.read_text()
        assert "## Figures" not in content

    def test_default_spec_has_no_diagnostics_section(self, tmp_path: Path):
        _, paths = self._run_and_report(tmp_path, spec=None)
        content = paths.markdown.read_text()
        assert "## Diagnostics Appendix" not in content

    def test_manifest_written_regardless_of_spec(self, tmp_path: Path):
        for spec in [None, COMPACT_REPORT, DIAGNOSTICS_REPORT]:
            _, paths = self._run_and_report(tmp_path, spec=spec)
            assert paths.manifest is not None and paths.manifest.exists()


# ---------------------------------------------------------------------------
# Feature engineering artefacts (v2 / ML pipeline)
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    """research/ feature artefacts are written during v2 ML experiment runs."""

    def _run_v2(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            return run_experiment_from_config(p)

    # --- file existence ---

    def test_feature_summary_json_written(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        assert (run.output_path / "research" / "feature_summary.json").exists()

    def test_feature_registry_json_written(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        assert (run.output_path / "research" / "feature_registry.json").exists()

    def test_alignment_diagnostics_json_written(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        assert (run.output_path / "research" / "alignment_diagnostics.json").exists()

    def test_feature_correlations_json_written(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        assert (run.output_path / "research" / "feature_correlations.json").exists()

    # --- feature_summary structure ---

    def test_feature_summary_top_level_keys(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_summary.json").read_text()
        )
        assert "n_rows_before_alignment" in data
        assert "n_rows_after_alignment" in data
        assert "features" in data

    def test_feature_summary_per_feature_stats(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_summary.json").read_text()
        )
        assert "mom_20" in data["features"]
        assert "vol_21" in data["features"]
        stats = data["features"]["mom_20"]
        for key in ("mean", "std", "skew", "kurtosis", "ar1", "sample_coverage"):
            assert key in stats, f"missing key {key!r} in feature stats"

    def test_feature_summary_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        raw = (run.output_path / "research" / "feature_summary.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    # --- feature_registry structure ---

    def test_feature_registry_top_level_keys(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_registry.json").read_text()
        )
        assert "ticker" in data
        assert "n_features" in data
        assert "features" in data
        assert "label_type" in data
        assert "label_horizon" in data

    def test_feature_registry_n_features_correct(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_registry.json").read_text()
        )
        assert data["n_features"] == 2
        assert len(data["features"]) == 2

    def test_feature_registry_entry_fields(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_registry.json").read_text()
        )
        for entry in data["features"]:
            assert "name" in entry
            assert "type" in entry
            assert "category" in entry
            assert "transform" in entry

    def test_feature_registry_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        raw = (run.output_path / "research" / "feature_registry.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    # --- alignment_diagnostics structure ---

    def test_alignment_diagnostics_top_level_keys(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "alignment_diagnostics.json").read_text()
        )
        assert "n_rows_raw" in data
        assert "warmup_rows_removed" in data
        assert "label_rows_removed" in data
        assert "n_rows_after_alignment" in data

    def test_alignment_rows_are_consistent(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "alignment_diagnostics.json").read_text()
        )
        # aligned <= raw
        assert data["n_rows_after_alignment"] <= data["n_rows_raw"]
        # rows_removed >= 0
        assert data["warmup_rows_removed"] >= 0
        assert data["label_rows_removed"] >= 0

    def test_alignment_diagnostics_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        raw = (run.output_path / "research" / "alignment_diagnostics.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    # --- feature_correlations structure ---

    def test_feature_correlations_top_level_keys(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_correlations.json").read_text()
        )
        assert "features" in data
        assert "matrix" in data
        assert "n_features" in data

    def test_feature_correlations_matrix_shape(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        data = json.loads(
            (run.output_path / "research" / "feature_correlations.json").read_text()
        )
        n = data["n_features"]
        assert len(data["matrix"]) == n
        assert all(len(row) == n for row in data["matrix"])

    def test_feature_correlations_no_nan_in_json(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        raw = (run.output_path / "research" / "feature_correlations.json").read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw

    # --- loader integration ---

    def test_artefacts_loader_populates_feature_fields(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        artefacts = load_experiment_artefacts(run.output_path, tmp_path / "reports")
        assert isinstance(artefacts.feature_summary, dict)
        assert isinstance(artefacts.feature_registry, dict)
        assert isinstance(artefacts.alignment_diagnostics, dict)
        assert isinstance(artefacts.feature_correlations, dict)

    # --- manifest flags ---

    def test_manifest_has_feature_engineering_flags(self, tmp_path: Path):
        run = self._run_v2(tmp_path)
        paths = generate_experiment_report(run.output_path, tmp_path / "reports")
        data = json.loads(paths.manifest.read_text())
        assert data.get("has_feature_summary") is True
        assert data.get("has_feature_registry") is True
        assert data.get("has_alignment_diagnostics") is True
        assert data.get("has_feature_correlations") is True
        assert data.get("has_feature_engineering") is True


# ---------------------------------------------------------------------------
# Phase E1 — signal column naming fix + inline figures + run_and_report preset
# ---------------------------------------------------------------------------


class TestSignalColumnNaming:
    """build_signal_fn uses asset_name so weights columns match returns columns."""

    def test_sign_signal_weights_use_ticker_column(self, tmp_path: Path):
        """Weights produced by sign signal must be keyed by ticker, not signal name."""
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        import pandas as pd
        weights = pd.read_parquet(run.output_path / "weights.parquet")
        assert list(weights.columns) == ["SPY"], (
            f"Expected ['SPY'], got {list(weights.columns)}"
        )

    def test_sign_signal_produces_nonzero_returns(self, tmp_path: Path):
        """With correctly named columns, backtest returns must not be all-zero."""
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        import pandas as pd
        returns = pd.read_parquet(run.output_path / "returns.parquet")
        assert returns["net_return"].abs().sum() > 0, "All returns are zero — signal column mismatch"

    def test_metrics_non_nan_after_fix(self, tmp_path: Path):
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        import json
        metrics = json.loads((run.output_path / "metrics.json").read_text())
        # Sharpe should be a real number, not NaN
        assert metrics["sharpe_ratio"] == metrics["sharpe_ratio"], (
            "Sharpe is NaN — signal column mismatch not fixed"
        )


class TestInlineFigurePlacement:
    """Primary figures appear inline in their section, not only in the Figures appendix."""

    def _setup(self, tmp_path: Path, spec=None):
        p = _write_v2_wf_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run = run_experiment_from_config(p)
        paths = generate_experiment_report(
            run.output_path, tmp_path / "reports",
            report_spec=spec,
        )
        return run, paths

    def test_equity_drawdown_inline_before_figures_appendix(self, tmp_path: Path):
        # equity_and_drawdown is claimed by _metrics() — when present it must
        # appear inline (before ## Figures), not only in the appendix.
        # When save_plots=False there are no figures; the assertion is skipped.
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        _, paths = self._setup(tmp_path, CANONICAL_SHOWCASE)
        content = paths.markdown.read_text()
        eq_pos = content.find("equity_and_drawdown.png")
        figs_pos = content.find("## Figures")
        if eq_pos != -1 and figs_pos != -1:
            assert eq_pos < figs_pos, "equity_and_drawdown.png should appear inline before ## Figures"

    def test_rolling_volatility_falls_to_appendix_when_present(self, tmp_path: Path):
        # rolling_volatility is unclaimed — when present it must appear only in the appendix.
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        _, paths = self._setup(tmp_path, CANONICAL_SHOWCASE)
        content = paths.markdown.read_text()
        rv_pos = content.find("rolling_volatility.png")
        figs_pos = content.find("## Figures")
        if rv_pos != -1 and figs_pos != -1:
            assert rv_pos > figs_pos, "rolling_volatility.png should be in the Figures appendix"

    def test_compact_report_has_no_figures_section(self, tmp_path: Path):
        from src.reporting.report_spec import COMPACT_REPORT
        _, paths = self._setup(tmp_path, COMPACT_REPORT)
        content = paths.markdown.read_text()
        assert "## Figures" not in content

    def test_manifest_sections_rendered_populated(self, tmp_path: Path):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        import json
        _, paths = self._setup(tmp_path, CANONICAL_SHOWCASE)
        manifest = json.loads(paths.manifest.read_text())
        sections = manifest.get("sections_rendered", [])
        assert "Performance Metrics" in sections
        assert "Walk-Forward Validation" in sections
        assert "Diagnostics Appendix" in sections


class TestRunAndReportPreset:
    """run_and_report() forwards report_spec correctly."""

    def test_run_and_report_canonical_generates_diagnostics(self, tmp_path: Path):
        from src.experiments.orchestrator import run_and_report
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        p = _write_v2_wf_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run, paths = run_and_report(
                p,
                report_output_dir=tmp_path / "reports",
                report_spec=CANONICAL_SHOWCASE,
            )
        content = paths.markdown.read_text()
        assert "## Diagnostics Appendix" in content

    def test_run_and_report_standard_omits_diagnostics(self, tmp_path: Path):
        from src.experiments.orchestrator import run_and_report
        from src.reporting.report_spec import STANDARD_REPORT
        p = _write_v2_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run, paths = run_and_report(
                p,
                report_output_dir=tmp_path / "reports",
                report_spec=STANDARD_REPORT,
            )
        content = paths.markdown.read_text()
        assert "## Diagnostics Appendix" not in content

    def test_run_and_report_manifest_has_report_spec_field(self, tmp_path: Path):
        from src.experiments.orchestrator import run_and_report
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        import json
        p = _write_v2_wf_cfg(tmp_path)
        with patch(_PATCH, side_effect=_make_load_universe_patch(_PRICES_V2)):
            run, paths = run_and_report(
                p,
                report_output_dir=tmp_path / "reports",
                report_spec=CANONICAL_SHOWCASE,
            )
        manifest = json.loads(paths.manifest.read_text())
        assert manifest.get("report_spec") == "CANONICAL_SHOWCASE"
