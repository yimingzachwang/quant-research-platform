"""Tests for src/reporting/markdown.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.reporting.markdown import render_report, _pipe_table, _label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artefacts(
    *,
    exp_name: str = "test_exp",
    strategy: str = "EqualWeight(freq=ME)",
    metrics: dict | None = None,
    config: dict | None = None,
) -> MagicMock:
    """Build a minimal mock ExperimentArtefacts."""
    a = MagicMock()
    a.metadata = {
        "experiment_name": exp_name,
        "strategy_name": strategy,
        "created_at": "2026-05-23T12:00:00+00:00",
    }
    a.metrics = metrics if metrics is not None else {
        "annualized_return": 0.08,
        "sharpe_ratio": 0.65,
        "max_drawdown": -0.15,
    }
    a.config = config
    return a


_FULL_CONFIG = {
    "universe": {"tickers": ["SPY", "QQQ", "IWM"]},
    "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
    "strategy": {
        "type": "MomentumRotation",
        "parameters": {"lookback": 252, "top_n": 3},
    },
    "execution": {"transaction_cost_bps": 5.0},
    "validation": {"type": "rolling", "parameters": {"train_months": 36, "test_months": 12}},
}

_NO_VALIDATION_CONFIG = {
    **_FULL_CONFIG,
    "validation": {"type": "none", "parameters": {}},
}


# ---------------------------------------------------------------------------
# render_report structure
# ---------------------------------------------------------------------------


def test_render_returns_string() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert isinstance(out, str)


def test_render_contains_experiment_name() -> None:
    a = _make_artefacts(exp_name="my_experiment")
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "my_experiment" in out


def test_render_contains_h1_title() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert out.startswith("# Experiment Report:")


def test_render_has_metadata_section() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Metadata" in out


def test_render_metadata_contains_strategy() -> None:
    a = _make_artefacts(strategy="MomentumRotation")
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "MomentumRotation" in out


def test_render_has_configuration_section_when_config_present() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Configuration" in out


def test_render_configuration_unavailable_when_no_config() -> None:
    a = _make_artefacts(config=None)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Configuration" in out
    assert "not available" in out


def test_render_configuration_shows_tickers() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "SPY" in out
    assert "QQQ" in out


def test_render_configuration_shows_date_range() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "2020-01-01" in out
    assert "2023-12-31" in out


def test_render_configuration_shows_strategy_params() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "lookback" in out
    assert "252" in out


def test_render_has_metrics_section() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Performance Metrics" in out


def test_render_metrics_shows_sharpe() -> None:
    a = _make_artefacts(metrics={"sharpe_ratio": 0.75})
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "0.7500" in out


def test_render_metrics_empty_graceful() -> None:
    a = _make_artefacts(metrics={})
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Performance Metrics" in out
    assert "No metrics" in out


def test_render_no_walk_forward_when_validation_none() -> None:
    a = _make_artefacts(config=_NO_VALIDATION_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Walk-Forward Validation" not in out


def test_render_walk_forward_section_when_configured() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Walk-Forward Validation" in out


def test_render_walk_forward_shows_params() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "train_months" in out
    assert "36" in out


def test_render_no_walk_forward_when_no_config() -> None:
    a = _make_artefacts(config=None)
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Walk-Forward Validation" not in out


def test_render_no_figures_section_when_empty() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "## Figures" not in out


def test_render_figures_inline_when_claimed() -> None:
    # equity_and_drawdown is claimed inline by _metrics() — it appears in the
    # report body but not in a ## Figures appendix section.
    a = _make_artefacts()
    figs = [("Equity And Drawdown", Path("../figures/exp/equity_and_drawdown.png"))]
    out = render_report(a, figs, "2026-05-23T00:00:00+00:00", "1")
    assert "equity_and_drawdown.png" in out   # present inline
    assert "## Figures" not in out            # not in appendix (all claimed)


def test_render_figures_appendix_for_unclaimed() -> None:
    # A figure not claimed by any section falls to the appendix.
    a = _make_artefacts()
    figs = [("Custom Research Plot", Path("../figures/exp/custom_research.png"))]
    out = render_report(a, figs, "2026-05-23T00:00:00+00:00", "1")
    assert "## Figures" in out
    assert "custom_research.png" in out


def test_rolling_volatility_claimed_inline_by_data_infrastructure() -> None:
    # rolling_volatility is claimed by _data_infrastructure → present inline, not in appendix.
    a = _make_artefacts()
    figs = [("Rolling Volatility", Path("../figures/exp/rolling_volatility.png"))]
    out = render_report(a, figs, "2026-05-23T00:00:00+00:00", "1")
    assert "rolling_volatility.png" in out
    assert "## Figures" not in out  # claimed inline, appendix empty


def test_render_figures_uses_provided_paths() -> None:
    a = _make_artefacts()
    custom_path = Path("custom/path/plot.png")
    figs = [("My Plot", custom_path)]
    out = render_report(a, figs, "2026-05-23T00:00:00+00:00", "1")
    assert "custom/path/plot.png" in out


def test_render_has_footer_separator() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "---" in out


def test_render_footer_contains_report_version() -> None:
    a = _make_artefacts()
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "Report version: 1" in out


def test_render_footer_contains_generated_at() -> None:
    ts = "2026-05-23T12:34:56+00:00"
    a = _make_artefacts()
    out = render_report(a, [], ts, "1")
    assert ts in out


def test_render_footer_contains_source_experiment() -> None:
    a = _make_artefacts(exp_name="special_exp")
    out = render_report(a, [], "2026-05-23T00:00:00+00:00", "1")
    assert "Source experiment: special_exp" in out


def test_render_deterministic_structure() -> None:
    a = _make_artefacts(config=_FULL_CONFIG)
    ts = "2026-05-23T00:00:00+00:00"
    out1 = render_report(a, [], ts, "1")
    out2 = render_report(a, [], ts, "1")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_pipe_table_headers() -> None:
    t = _pipe_table(["A", "B"], [("x", "1"), ("y", "2")])
    assert "| A | B |" in t
    assert "---" in t


def test_pipe_table_data_rows() -> None:
    t = _pipe_table(["Metric", "Value"], [("Sharpe", "0.65")])
    assert "| Sharpe | 0.65 |" in t


def test_label_converts_snake_case() -> None:
    assert _label("sharpe_ratio") == "Sharpe Ratio"
    assert _label("max_drawdown") == "Max Drawdown"
    assert _label("hit_rate") == "Hit Rate"
