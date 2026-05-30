"""Tests for src/reporting/report_spec.py and render_report() spec integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.reporting.report_spec import (
    AUDIT_REPORT,
    COMPACT_REPORT,
    DIAGNOSTICS_REPORT,
    FULL_DEMO_REPORT,
    ResearchReportSpec,
)


# ---------------------------------------------------------------------------
# ResearchReportSpec — dataclass properties
# ---------------------------------------------------------------------------


class TestResearchReportSpec:
    def test_defaults_match_full_demo(self):
        spec = ResearchReportSpec()
        assert spec == FULL_DEMO_REPORT

    def test_frozen(self):
        spec = ResearchReportSpec()
        with pytest.raises((AttributeError, TypeError)):
            spec.include_summary = False  # type: ignore[misc]

    def test_hashable(self):
        s = {ResearchReportSpec(), ResearchReportSpec()}
        assert len(s) == 1

    def test_equality(self):
        assert ResearchReportSpec() == ResearchReportSpec()
        a = ResearchReportSpec(include_diagnostics=True)
        b = ResearchReportSpec(include_diagnostics=False)
        assert a != b

    def test_all_flags_present(self):
        spec = ResearchReportSpec()
        for flag in (
            "include_summary", "include_metadata", "include_configuration",
            "include_metrics", "include_ml_analysis", "include_validation",
            "include_diagnostics", "include_figures", "include_provenance",
            "include_thesis", "include_methodology",
            "include_data_infrastructure", "include_portfolio_process",
            "include_failure_analysis", "include_feature_engineering",
        ):
            assert hasattr(spec, flag)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_full_demo_all_on_except_diagnostics(self):
        assert FULL_DEMO_REPORT.include_summary is True
        assert FULL_DEMO_REPORT.include_metrics is True
        assert FULL_DEMO_REPORT.include_figures is True
        assert FULL_DEMO_REPORT.include_validation is True
        assert FULL_DEMO_REPORT.include_diagnostics is False

    def test_compact_no_figures_no_validation(self):
        assert COMPACT_REPORT.include_figures is False
        assert COMPACT_REPORT.include_validation is False
        assert COMPACT_REPORT.include_diagnostics is False
        # Core sections still on
        assert COMPACT_REPORT.include_summary is True
        assert COMPACT_REPORT.include_metrics is True

    def test_diagnostics_report_includes_diagnostics(self):
        assert DIAGNOSTICS_REPORT.include_diagnostics is True
        assert DIAGNOSTICS_REPORT.include_metrics is True
        assert DIAGNOSTICS_REPORT.include_figures is True

    def test_audit_report_has_diagnostics_and_provenance(self):
        assert AUDIT_REPORT.include_diagnostics is True
        assert AUDIT_REPORT.include_provenance is True

    def test_full_demo_includes_new_sections(self):
        assert FULL_DEMO_REPORT.include_thesis is True
        assert FULL_DEMO_REPORT.include_methodology is True
        assert FULL_DEMO_REPORT.include_data_infrastructure is True
        assert FULL_DEMO_REPORT.include_failure_analysis is True

    def test_compact_excludes_new_sections(self):
        assert COMPACT_REPORT.include_thesis is False
        assert COMPACT_REPORT.include_methodology is False
        assert COMPACT_REPORT.include_data_infrastructure is False
        assert COMPACT_REPORT.include_portfolio_process is False
        assert COMPACT_REPORT.include_failure_analysis is False
        assert COMPACT_REPORT.include_feature_engineering is False

    def test_audit_excludes_thesis(self):
        assert AUDIT_REPORT.include_thesis is False

    def test_all_presets_are_distinct(self):
        presets = [FULL_DEMO_REPORT, COMPACT_REPORT, DIAGNOSTICS_REPORT, AUDIT_REPORT]
        assert len(set(presets)) == len(presets)

    def test_standard_report_differs_from_full_demo(self):
        from src.reporting.report_spec import STANDARD_REPORT
        assert STANDARD_REPORT != FULL_DEMO_REPORT
        assert STANDARD_REPORT.include_provenance is False
        assert STANDARD_REPORT.include_validation is True
        assert STANDARD_REPORT.include_thesis is True

    def test_canonical_showcase_differs_from_full_demo(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        assert CANONICAL_SHOWCASE != FULL_DEMO_REPORT
        assert CANONICAL_SHOWCASE.include_diagnostics is True
        assert CANONICAL_SHOWCASE.include_thesis is True
        assert CANONICAL_SHOWCASE.include_provenance is True

    def test_canonical_showcase_differs_from_diagnostics_report(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        assert CANONICAL_SHOWCASE != DIAGNOSTICS_REPORT
        # CANONICAL adds provenance; DIAGNOSTICS omits it for colleague sharing
        assert CANONICAL_SHOWCASE.include_provenance is True
        assert DIAGNOSTICS_REPORT.include_provenance is False
        # Both include the thesis (full narrative depth)
        assert CANONICAL_SHOWCASE.include_thesis is True
        assert DIAGNOSTICS_REPORT.include_thesis is True

    def test_all_six_presets_are_distinct(self):
        from src.reporting.report_spec import (
            AUDIT_REPORT, CANONICAL_SHOWCASE, COMPACT_REPORT,
            DIAGNOSTICS_REPORT, FULL_DEMO_REPORT, STANDARD_REPORT,
        )
        presets = [FULL_DEMO_REPORT, COMPACT_REPORT, DIAGNOSTICS_REPORT,
                   AUDIT_REPORT, STANDARD_REPORT, CANONICAL_SHOWCASE]
        assert len(set(presets)) == len(presets)


# ---------------------------------------------------------------------------
# render_report() spec integration
# ---------------------------------------------------------------------------


def _make_minimal_artefacts(**overrides):
    """Build a MagicMock ExperimentArtefacts for markdown rendering tests."""
    from pathlib import Path

    m = MagicMock()
    m.metadata = {
        "experiment_name": "spec_test",
        "strategy_name": "TestStrategy",
        "created_at": "2026-05-23T00:00:00+00:00",
    }
    m.metrics = {"sharpe_ratio": 0.85, "annualized_return": 0.10,
                 "annualized_volatility": 0.12, "max_drawdown": -0.15,
                 "calmar_ratio": 0.67, "hit_rate": 0.54}
    m.config = {
        "universe": {"tickers": ["SPY"]},
        "date_range": {"start": "2020-01-01", "end": "2024-12-31"},
        "strategy": {"type": "EqualWeight", "parameters": {}},
        "validation": {"type": "none", "parameters": {}},
        "execution": {"transaction_cost_bps": 5.0},
        "tags": ["test"],
    }
    m.ml_provenance = None
    m.split_metrics = None
    m.ml_diagnostics = None
    m.research_artefacts = None
    m.backtest_diagnostics = None
    m.feature_summary = None
    m.feature_registry = None
    m.alignment_diagnostics = None
    m.feature_correlations = None
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


class TestRenderReportSpec:
    """render_report() honours ResearchReportSpec flags."""

    def _render(self, spec=None, **overrides):
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts(**overrides)
        return render_report(arts, [], "2026-05-23T00:00:00Z", "1", report_spec=spec)

    def test_no_spec_includes_all_default_sections(self):
        md = self._render()
        assert "## Performance Metrics" in md
        assert "## Configuration" in md
        assert "## Metadata" in md

    def test_none_spec_equals_standard(self):
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import STANDARD_REPORT
        arts = _make_minimal_artefacts()
        md_none = render_report(arts, [], "ts", "1", report_spec=None)
        md_standard = render_report(arts, [], "ts", "1", report_spec=STANDARD_REPORT)
        assert md_none == md_standard

    def test_none_spec_does_not_equal_full_demo(self):
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts()
        # Provenance section is content-gated on hash presence; set one to trigger it
        arts.metadata["config_hash"] = "abc123deadbeef"
        md_none = render_report(arts, [], "ts", "1", report_spec=None)
        md_full = render_report(arts, [], "ts", "1", report_spec=FULL_DEMO_REPORT)
        # FULL_DEMO renders Provenance section; STANDARD (None default) does not
        assert "## Provenance" not in md_none
        assert "## Provenance" in md_full
        assert md_none != md_full

    def test_compact_omits_validation_and_figures(self):
        # Config has validation type 'rolling' to confirm section would normally appear
        arts = _make_minimal_artefacts()
        arts.config["validation"]["type"] = "rolling"
        from src.reporting.markdown import render_report
        md = render_report(arts, [("fig", __import__("pathlib").Path("x.png"))],
                           "ts", "1", report_spec=COMPACT_REPORT)
        assert "## Walk-Forward Validation" not in md
        assert "## Figures" not in md
        assert "## Performance Metrics" in md

    def test_compact_omits_provenance(self):
        arts = _make_minimal_artefacts()
        arts.metadata["config_hash"] = "abc123"
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=COMPACT_REPORT)
        assert "## Provenance" not in md

    def test_diagnostics_report_renders_diagnostics_note_when_absent(self):
        md = self._render(spec=DIAGNOSTICS_REPORT)
        assert "## Diagnostics Appendix" in md
        assert "not available" in md.lower()

    def test_diagnostics_report_renders_split_table_when_present(self):
        split_metrics = {
            "n_splits": 2,
            "summary": {
                "mean_sharpe": 0.72,
                "std_sharpe": 0.31,
                "hit_rate_positive_sharpe": 0.5,
                "mean_annualized_return": 0.08,
                "mean_max_drawdown": -0.12,
                "worst_max_drawdown": -0.18,
            },
            "splits": [
                {"split": 0, "test_start": "2020-01-01", "test_end": "2020-12-31",
                 "sharpe_ratio": 0.90, "annualized_return": 0.09, "max_drawdown": -0.10},
                {"split": 1, "test_start": "2021-01-01", "test_end": "2021-12-31",
                 "sharpe_ratio": 0.54, "annualized_return": 0.06, "max_drawdown": -0.14},
            ],
        }
        md = self._render(spec=DIAGNOSTICS_REPORT, split_metrics=split_metrics)
        assert "## Diagnostics Appendix" in md
        assert "Walk-Forward Stability" in md
        # Per-split table no longer duplicated in appendix — summary stats are shown instead
        assert "0.7200" in md  # mean_sharpe from summary table

    def test_diagnostics_report_renders_ml_diagnostics_when_present(self):
        ml_diag = {
            "model_type": "RidgeRegression",
            "average_turnover": 0.183,
            "signal_activity": 0.65,
            "n_weight_periods": 1200,
        }
        md = self._render(spec=DIAGNOSTICS_REPORT, ml_diagnostics=ml_diag)
        assert "ML Signal Diagnostics" in md
        assert "0.1830" in md  # average_turnover
        assert "65.0%" in md  # signal_activity

    def test_ml_analysis_suppressed_when_flag_false(self):
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_ml_analysis=False)
        ml_prov = {
            "model": {"type": "Ridge", "params": {}},
            "features": {"ticker": "SPY", "entries": []},
            "labels": {"type": "forward_returns", "params": {"horizon": 5}},
            "signal": {"type": "sign", "params": {}},
            "ml_hash": "abc123456789",
        }
        md = self._render(spec=spec, ml_provenance=ml_prov)
        assert "## Model & Features" not in md

    def test_summary_suppressed_when_flag_false(self):
        spec = ResearchReportSpec(include_summary=False)
        md = self._render(spec=spec)
        assert "**Period:**" not in md

    def test_footer_always_present(self):
        for spec in [FULL_DEMO_REPORT, COMPACT_REPORT, DIAGNOSTICS_REPORT, AUDIT_REPORT]:
            md = self._render(spec=spec)
            assert "Report version:" in md

    def test_thesis_included_when_flag_true(self):
        spec = ResearchReportSpec(include_thesis=True)
        md = self._render(spec=spec)
        assert "## Research Thesis" in md

    def test_thesis_excluded_when_flag_false(self):
        spec = ResearchReportSpec(include_thesis=False)
        md = self._render(spec=spec)
        assert "## Research Thesis" not in md

    def test_methodology_included_when_flag_true(self):
        spec = ResearchReportSpec(include_methodology=True)
        md = self._render(spec=spec)
        assert "## Backtesting Methodology" in md

    def test_methodology_excluded_when_flag_false(self):
        spec = ResearchReportSpec(include_methodology=False)
        md = self._render(spec=spec)
        assert "## Backtesting Methodology" not in md

    def test_failure_analysis_excluded_when_flag_false(self):
        spec = ResearchReportSpec(include_failure_analysis=False)
        md = self._render(spec=spec)
        assert "## Failure Analysis" not in md

    def test_compact_has_no_narrative_sections(self):
        md = self._render(spec=COMPACT_REPORT)
        assert "## Research Thesis" not in md
        assert "## Backtesting Methodology" not in md
        assert "## Failure Analysis" not in md

    def test_momentum_thesis_renders_hypothesis(self):
        arts = _make_minimal_artefacts()
        arts.config["strategy"] = {
            "type": "MomentumRotation",
            "parameters": {"lookback": 252, "top_n": 3, "rebalance_freq": "ME"},
        }
        arts.config["execution"] = {"transaction_cost_bps": 5.0}
        arts.config["universe"] = {"tickers": ["SPY", "QQQ", "TLT", "GLD"]}
        spec = ResearchReportSpec(include_thesis=True)
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=spec)
        assert "## Research Thesis" in md
        assert "Hypothesis" in md
        assert "momentum" in md.lower()

    def test_data_infrastructure_renders_without_research_artefacts(self):
        spec = ResearchReportSpec(include_data_infrastructure=True)
        md = self._render(spec=spec)
        assert "## Data Infrastructure" in md

    def test_feature_engineering_excluded_when_flag_false(self):
        spec = ResearchReportSpec(include_feature_engineering=False)
        md = self._render(spec=spec)
        assert "## Feature Engineering" not in md

    def test_compact_omits_feature_engineering(self):
        md = self._render(spec=COMPACT_REPORT)
        assert "## Feature Engineering" not in md

    def test_feature_engineering_renders_registry_when_present(self):
        registry = {
            "ticker": "SPY",
            "n_features": 2,
            "label_type": "forward_returns",
            "label_horizon": 5,
            "features": [
                {"name": "mom_20", "type": "momentum", "category": "momentum",
                 "transform": "trailing_return", "window": 20, "normalization_type": "none"},
                {"name": "vol_21", "type": "rolling_volatility", "category": "volatility",
                 "transform": "rolling_std_annualised", "window": 21, "normalization_type": "none"},
            ],
        }
        md = self._render(spec=FULL_DEMO_REPORT, feature_registry=registry)
        assert "## Feature Engineering" in md
        assert "20D Momentum" in md
        assert "21D Realized Volatility" in md

    def test_feature_engineering_renders_alignment_when_present(self):
        alignment = {
            "n_rows_raw": 600,
            "n_rows_features_clean": 579,
            "warmup_rows_removed": 21,
            "label_rows_removed": 5,
            "n_rows_after_alignment": 574,
            "alignment_loss_pct": 4.33,
        }
        registry = {
            "ticker": "SPY", "n_features": 1, "label_type": "forward_returns",
            "label_horizon": 5, "features": [
                {"name": "mom_20", "type": "momentum", "category": "momentum",
                 "transform": "trailing_return", "window": 20, "normalization_type": "none"},
            ],
        }
        md = self._render(
            spec=FULL_DEMO_REPORT,
            alignment_diagnostics=alignment,
            feature_registry=registry,
        )
        assert "## Feature Engineering" in md
        assert "Sample construction" in md

    def test_feature_engineering_renders_feature_stats_when_present(self):
        summary = {
            "n_rows_before_alignment": 600,
            "n_rows_after_alignment": 574,
            "features": {
                "mom_20": {
                    "mean": 0.0012, "std": 0.0231, "skew": -0.14,
                    "kurtosis": 3.1, "ar1": 0.87, "sample_coverage": 1.0,
                    "first_valid_date": "2018-02-01",
                },
                "vol_21": {
                    "mean": 0.0180, "std": 0.0045, "skew": 0.62,
                    "kurtosis": 4.2, "ar1": 0.92, "sample_coverage": 1.0,
                    "first_valid_date": "2018-02-05",
                },
            },
        }
        md = self._render(spec=FULL_DEMO_REPORT, feature_summary=summary)
        assert "## Feature Engineering" in md
        assert "Per-feature statistics" in md
        assert "20D Momentum" in md

    def test_feature_engineering_renders_correlations_when_present(self):
        correlations = {
            "features": ["mom_20", "vol_21"],
            "n_features": 2,
            "matrix": [[1.0, -0.12], [-0.12, 1.0]],
        }
        registry = {
            "ticker": "SPY", "n_features": 2, "label_type": "forward_returns",
            "label_horizon": 5, "features": [
                {"name": "mom_20", "type": "momentum", "category": "momentum",
                 "transform": "trailing_return", "window": 20, "normalization_type": "none"},
                {"name": "vol_21", "type": "rolling_volatility", "category": "volatility",
                 "transform": "rolling_std_annualised", "window": 21, "normalization_type": "none"},
            ],
        }
        md = self._render(
            spec=FULL_DEMO_REPORT,
            feature_correlations=correlations,
            feature_registry=registry,
        )
        # Feature Engineering section renders; correlation heatmap appears only when figure paths provided
        assert "## Feature Engineering" in md


# ---------------------------------------------------------------------------
# Phase D tests — section ordering, captions, ML thesis, manifest enrichment
# ---------------------------------------------------------------------------


class TestSectionOrdering:
    """D4 — Metadata and Configuration appear after research analysis sections."""

    def _render(self, spec=None, **overrides):
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts(**overrides)
        return render_report(arts, [], "ts", "1", report_spec=spec)

    def test_metrics_before_metadata(self):
        md = self._render()
        idx_metrics = md.index("## Performance Metrics")
        idx_meta = md.index("## Metadata")
        assert idx_metrics < idx_meta

    def test_configuration_after_metrics(self):
        md = self._render()
        idx_cfg = md.index("## Configuration")
        idx_metrics = md.index("## Performance Metrics")
        assert idx_metrics < idx_cfg

    def test_thesis_before_metrics(self):
        arts = _make_minimal_artefacts()
        arts.config["strategy"] = {
            "type": "MomentumRotation",
            "parameters": {"lookback": 252, "top_n": 3, "rebalance_freq": "ME"},
        }
        arts.config["execution"] = {"transaction_cost_bps": 5.0}
        arts.config["universe"] = {"tickers": ["SPY", "QQQ"]}
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_thesis=True, include_metrics=True)
        md = render_report(arts, [], "ts", "1", report_spec=spec)
        idx_thesis = md.index("## Research Thesis")
        idx_metrics = md.index("## Performance Metrics")
        assert idx_thesis < idx_metrics

    def test_validation_before_metadata(self):
        arts = _make_minimal_artefacts()
        arts.config["validation"]["type"] = "rolling"
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        idx_val = md.index("## Walk-Forward Validation")
        idx_meta = md.index("## Metadata")
        assert idx_val < idx_meta

    def test_failure_analysis_before_metadata(self):
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_failure_analysis=True)
        md = self._render(spec=spec)
        idx_fa = md.index("## Failure Analysis")
        idx_meta = md.index("## Metadata")
        assert idx_fa < idx_meta

    def test_ordering_applies_for_all_presets(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE, COMPACT_REPORT, STANDARD_REPORT
        for spec in [COMPACT_REPORT, STANDARD_REPORT, CANONICAL_SHOWCASE]:
            md = self._render(spec=spec)
            assert md.index("## Performance Metrics") < md.index("## Metadata")
            assert md.index("## Metadata") < md.index("## Configuration")


class TestFigureCaptions:
    """D5 — _build_figure_captions and _figures caption rendering."""

    def test_build_figure_captions_empty_without_plot_index(self):
        from src.reporting.markdown import _build_figure_captions
        arts = _make_minimal_artefacts()
        arts.plot_index = None
        result = _build_figure_captions(arts)
        assert result == {}

    def test_build_figure_captions_maps_display_names(self):
        from src.reporting.markdown import _build_figure_captions
        arts = _make_minimal_artefacts()
        arts.plot_index = [
            {"name": "equity_and_drawdown", "caption": "Equity curve and drawdown."},
            {"name": "rolling_sharpe", "caption": "252-day rolling Sharpe."},
        ]
        caps = _build_figure_captions(arts)
        assert caps["Equity And Drawdown"] == "Equity curve and drawdown."
        assert caps["Rolling Sharpe"] == "252-day rolling Sharpe."

    def test_figures_renders_caption_below_image(self):
        from pathlib import Path
        from src.reporting.markdown import _figures
        caps = {"My Plot": "This is the caption for the plot."}
        md = _figures([("My Plot", Path("figures/my_plot.png"))], captions=caps)
        assert "## Figures" in md
        assert "![My Plot]" in md
        assert "*This is the caption for the plot.*" in md

    def test_figures_no_caption_when_not_in_dict(self):
        from pathlib import Path
        from src.reporting.markdown import _figures
        caps = {"Other Plot": "Some caption."}
        md = _figures([("My Plot", Path("figures/my_plot.png"))], captions=caps)
        assert "## Figures" in md
        assert "*Some caption.*" not in md

    def test_figures_no_caption_when_captions_none(self):
        from pathlib import Path
        from src.reporting.markdown import _figures
        md = _figures([("My Plot", Path("figures/my_plot.png"))], captions=None)
        assert "## Figures" in md
        assert "*" not in md.replace("*[^*]*\\*", "")  # no italic text


class TestMLThesisSection:
    """D2 — ML thesis body is rendered for v2 experiments."""

    def _make_ml_artefacts(self):
        arts = _make_minimal_artefacts()
        arts.ml_provenance = {
            "model": {"type": "RidgeRegression", "params": {"alpha": 1.0}},
            "features": {"ticker": "SPY", "entries": [
                {"name": "mom_20", "type": "momentum",
                 "params": {"window": 20}, "normalization_type": "none"},
                {"name": "vol_21", "type": "rolling_volatility",
                 "params": {"window": 21}, "normalization_type": "none"},
            ]},
            "labels": {"type": "forward_returns", "params": {"horizon": 5}},
            "signal": {"type": "sign", "params": {}},
            "ml_hash": "abc123",
        }
        return arts

    def test_ml_thesis_renders_for_v2(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import ResearchReportSpec
        md = render_report(arts, [], "ts", "1",
                           report_spec=ResearchReportSpec(include_thesis=True))
        assert "## Research Thesis" in md

    def test_ml_thesis_contains_hypothesis(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "Hypothesis" in md
        assert "RidgeRegression" in md

    def test_ml_thesis_contains_feature_rationale(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "Feature rationale" in md
        assert "mom_20" in md

    def test_ml_thesis_contains_label_rationale(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "Label construction" in md
        assert "forward_returns" in md

    def test_ml_thesis_contains_key_risks(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "Key risks" in md

    def test_ml_thesis_excluded_when_flag_false(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import ResearchReportSpec
        md = render_report(arts, [], "ts", "1",
                           report_spec=ResearchReportSpec(include_thesis=False))
        assert "## Research Thesis" not in md


class TestMLPortfolioProcess:
    """D3 — ML portfolio construction narrative for v2 experiments."""

    def _make_ml_artefacts(self):
        arts = _make_minimal_artefacts()
        arts.ml_provenance = {
            "model": {"type": "RidgeRegression", "params": {}},
            "features": {"ticker": "SPY", "entries": [
                {"name": "mom_20", "type": "momentum", "params": {"window": 20}},
            ]},
            "labels": {"type": "forward_returns", "params": {"horizon": 5}},
            "signal": {"type": "sign", "params": {}},
            "ml_hash": "abc",
        }
        return arts

    def test_ml_portfolio_process_renders(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "## Portfolio Construction Process" in md

    def test_ml_portfolio_process_contains_pipeline(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "shift(1)" in md
        assert "look-ahead" in md.lower()

    def test_ml_portfolio_process_shows_signal_type(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=None)
        assert "sign" in md

    def test_ml_portfolio_process_excluded_when_portfolio_process_false(self):
        arts = self._make_ml_artefacts()
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_portfolio_process=False)
        md = render_report(arts, [], "ts", "1", report_spec=spec)
        assert "## Portfolio Construction Process" not in md


class TestManifestEnrichment:
    """D7 — _write_report_manifest enrichment fields."""

    def test_derive_validation_verdict_no_validation(self):
        from src.reporting.report_builder import _derive_validation_verdict
        assert _derive_validation_verdict(None) == "no_validation"
        assert _derive_validation_verdict({}) == "no_validation"

    def test_derive_validation_verdict_pass(self):
        from src.reporting.report_builder import _derive_validation_verdict
        sm = {"summary": {"hit_rate_positive_sharpe": 0.75, "mean_sharpe": 0.8}}
        assert _derive_validation_verdict(sm) == "pass"

    def test_derive_validation_verdict_marginal(self):
        from src.reporting.report_builder import _derive_validation_verdict
        sm = {"summary": {"hit_rate_positive_sharpe": 0.5, "mean_sharpe": 0.1}}
        assert _derive_validation_verdict(sm) == "marginal"

    def test_derive_validation_verdict_fail(self):
        from src.reporting.report_builder import _derive_validation_verdict
        sm = {"summary": {"hit_rate_positive_sharpe": 0.25, "mean_sharpe": -0.3}}
        assert _derive_validation_verdict(sm) == "fail"

    def test_diagnostics_appendix_heading(self):
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        arts = _make_minimal_artefacts()
        md = render_report(arts, [], "ts", "1", report_spec=DIAGNOSTICS_REPORT)
        assert "## Diagnostics Appendix" in md
        assert "## Diagnostics\n" not in md

    def test_standard_report_omits_provenance_section(self):
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import STANDARD_REPORT
        arts = _make_minimal_artefacts()
        arts.metadata["config_hash"] = "deadbeef"
        md = render_report(arts, [], "ts", "1", report_spec=STANDARD_REPORT)
        assert "## Provenance" not in md

    def test_canonical_showcase_includes_all_narrative(self):
        from src.reporting.markdown import render_report
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        arts = _make_minimal_artefacts()
        arts.config["strategy"] = {
            "type": "MomentumRotation",
            "parameters": {"lookback": 252, "top_n": 3, "rebalance_freq": "ME"},
        }
        arts.config["execution"] = {"transaction_cost_bps": 5.0}
        arts.config["universe"] = {"tickers": ["SPY", "QQQ"]}
        arts.config["validation"]["type"] = "rolling"
        md = render_report(arts, [], "ts", "1", report_spec=CANONICAL_SHOWCASE)
        assert "## Research Thesis" in md
        assert "## Backtesting Methodology" in md
        assert "## Diagnostics Appendix" in md


# ---------------------------------------------------------------------------
# Phase E0 tests — default preset, portfolio_process flag, spec name, frontend contract
# ---------------------------------------------------------------------------


class TestDefaultPreset:
    """E0.1 — STANDARD_REPORT is the new default for None spec."""

    def _render(self, spec=None, **overrides):
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts(**overrides)
        return render_report(arts, [], "ts", "1", report_spec=spec)

    def test_none_resolves_to_standard_not_full_demo(self):
        from src.reporting.report_spec import FULL_DEMO_REPORT, STANDARD_REPORT
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts()
        arts.metadata["config_hash"] = "abc123"
        md_none = render_report(arts, [], "ts", "1", report_spec=None)
        md_std = render_report(arts, [], "ts", "1", report_spec=STANDARD_REPORT)
        md_full = render_report(arts, [], "ts", "1", report_spec=FULL_DEMO_REPORT)
        # None == STANDARD_REPORT
        assert md_none == md_std
        # None != FULL_DEMO_REPORT (FULL_DEMO has provenance; STANDARD does not)
        assert md_none != md_full

    def test_none_spec_omits_provenance(self):
        arts = _make_minimal_artefacts()
        arts.metadata["config_hash"] = "deadbeef"
        md = self._render()
        assert "## Provenance" not in md

    def test_none_spec_includes_full_narrative(self):
        md = self._render()
        assert "## Performance Metrics" in md
        assert "## Configuration" in md
        assert "## Metadata" in md

    def test_none_spec_includes_backtesting_methodology(self):
        md = self._render()
        assert "## Backtesting Methodology" in md

    def test_full_demo_is_not_default_but_still_works(self):
        md = self._render(spec=FULL_DEMO_REPORT)
        assert "## Performance Metrics" in md

    def test_standard_report_is_default(self):
        # The module docstring and code both document STANDARD_REPORT as default
        import inspect
        from src.reporting import markdown
        src = inspect.getsource(markdown.render_report)
        assert "STANDARD_REPORT" in src


class TestPortfolioProcessFlag:
    """E0.3 — include_portfolio_process is semantically independent of include_data_infrastructure."""

    def _render(self, spec, **overrides):
        from src.reporting.markdown import render_report
        arts = _make_minimal_artefacts(**overrides)
        return render_report(arts, [], "ts", "1", report_spec=spec)

    def test_portfolio_process_flag_exists(self):
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec()
        assert hasattr(spec, "include_portfolio_process")
        assert spec.include_portfolio_process is True

    def test_portfolio_process_on_data_infra_off(self):
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_data_infrastructure=False, include_portfolio_process=True)
        arts = _make_minimal_artefacts()
        arts.config["strategy"] = {
            "type": "MomentumRotation",
            "parameters": {"lookback": 252, "top_n": 3, "rebalance_freq": "ME"},
        }
        arts.config["execution"] = {"transaction_cost_bps": 5.0}
        arts.config["universe"] = {"tickers": ["SPY", "QQQ"]}
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=spec)
        assert "## Data Infrastructure" not in md
        assert "## Portfolio Construction Process" in md

    def test_data_infra_on_portfolio_process_off(self):
        from src.reporting.report_spec import ResearchReportSpec
        spec = ResearchReportSpec(include_data_infrastructure=True, include_portfolio_process=False)
        arts = _make_minimal_artefacts()
        arts.config["strategy"] = {
            "type": "MomentumRotation",
            "parameters": {"lookback": 252, "top_n": 3, "rebalance_freq": "ME"},
        }
        arts.config["execution"] = {"transaction_cost_bps": 5.0}
        arts.config["universe"] = {"tickers": ["SPY", "QQQ"]}
        from src.reporting.markdown import render_report
        md = render_report(arts, [], "ts", "1", report_spec=spec)
        assert "## Data Infrastructure" in md
        assert "## Portfolio Construction Process" not in md

    def test_compact_has_portfolio_process_false(self):
        from src.reporting.report_spec import COMPACT_REPORT
        assert COMPACT_REPORT.include_portfolio_process is False

    def test_standard_has_portfolio_process_true(self):
        from src.reporting.report_spec import STANDARD_REPORT
        assert STANDARD_REPORT.include_portfolio_process is True

    def test_canonical_has_portfolio_process_true(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        assert CANONICAL_SHOWCASE.include_portfolio_process is True

    def test_both_flags_independent_in_all_presets(self):
        from src.reporting.report_spec import (
            AUDIT_REPORT, CANONICAL_SHOWCASE, COMPACT_REPORT,
            DIAGNOSTICS_REPORT, FULL_DEMO_REPORT, STANDARD_REPORT,
        )
        for preset in [STANDARD_REPORT, CANONICAL_SHOWCASE, DIAGNOSTICS_REPORT,
                       AUDIT_REPORT, FULL_DEMO_REPORT]:
            # All non-compact presets have both True
            assert preset.include_data_infrastructure is True
            assert preset.include_portfolio_process is True
        # COMPACT has both False
        assert COMPACT_REPORT.include_data_infrastructure is False
        assert COMPACT_REPORT.include_portfolio_process is False


class TestSpecNameResolution:
    """E0.2 — _resolve_spec_name correctly identifies all presets."""

    def test_resolves_standard_report(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import STANDARD_REPORT
        assert _resolve_spec_name(STANDARD_REPORT) == "STANDARD_REPORT"

    def test_resolves_canonical_showcase(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        assert _resolve_spec_name(CANONICAL_SHOWCASE) == "CANONICAL_SHOWCASE"

    def test_resolves_compact_report(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import COMPACT_REPORT
        assert _resolve_spec_name(COMPACT_REPORT) == "COMPACT_REPORT"

    def test_resolves_diagnostics_report(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        assert _resolve_spec_name(DIAGNOSTICS_REPORT) == "DIAGNOSTICS_REPORT"

    def test_resolves_audit_report(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import AUDIT_REPORT
        assert _resolve_spec_name(AUDIT_REPORT) == "AUDIT_REPORT"

    def test_resolves_full_demo_report(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import FULL_DEMO_REPORT
        assert _resolve_spec_name(FULL_DEMO_REPORT) == "FULL_DEMO_REPORT"

    def test_custom_spec_returns_custom(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import ResearchReportSpec
        custom = ResearchReportSpec(include_summary=False, include_figures=False)
        assert _resolve_spec_name(custom) == "custom"

    def test_all_six_named_presets_resolve(self):
        from src.reporting.report_builder import _resolve_spec_name
        from src.reporting.report_spec import (
            AUDIT_REPORT, CANONICAL_SHOWCASE, COMPACT_REPORT,
            DIAGNOSTICS_REPORT, FULL_DEMO_REPORT, STANDARD_REPORT,
        )
        expected = {
            STANDARD_REPORT: "STANDARD_REPORT",
            CANONICAL_SHOWCASE: "CANONICAL_SHOWCASE",
            DIAGNOSTICS_REPORT: "DIAGNOSTICS_REPORT",
            COMPACT_REPORT: "COMPACT_REPORT",
            AUDIT_REPORT: "AUDIT_REPORT",
            FULL_DEMO_REPORT: "FULL_DEMO_REPORT",
        }
        for spec, name in expected.items():
            assert _resolve_spec_name(spec) == name


class TestFrontendContract:
    """E0.5 — manifest canonical primitive fields are always present and deterministic."""

    def _write_manifest(self, spec=None, **meta_overrides):
        """Write a manifest via _write_report_manifest and return the parsed JSON."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock
        from src.reporting.report_builder import _write_report_manifest, ReportPaths
        from src.reporting.report_spec import STANDARD_REPORT

        effective_spec = spec if spec is not None else STANDARD_REPORT

        arts = MagicMock()
        arts.metadata = {"experiment_name": "contract_test"}
        arts.metrics = {"sharpe_ratio": 0.9, "annualized_return": 0.12,
                        "annualized_volatility": 0.15, "max_drawdown": -0.10,
                        "calmar_ratio": 1.2, "hit_rate": 0.55}
        arts.config = {"tags": [], "strategy": {"type": "EqualWeight"}}
        arts.ml_provenance = None
        arts.split_metrics = None
        arts.ml_diagnostics = None
        arts.research_artefacts = None
        arts.ml_model_diagnostics = None
        arts.wf_equity_curves = None
        arts.feature_summary = None
        arts.feature_registry = None
        arts.feature_correlations = None
        arts.alignment_diagnostics = None
        arts.plot_index = None
        for k, v in meta_overrides.items():
            setattr(arts, k, v)

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "contract_test.md"
            prov_path = Path(tmp) / "contract_test_provenance.json"
            paths = ReportPaths(markdown=md_path, html=None, provenance=prov_path)
            manifest_path = _write_report_manifest(
                arts, paths, "2026-05-25T00:00:00Z", "1",
                copied_figures=[], sections_rendered=["Performance Metrics"],
                report_spec=effective_spec,
            )
            with manifest_path.open() as f:
                return json.load(f)

    def test_canonical_primitive_fields_always_present(self):
        manifest = self._write_manifest()
        required = [
            "experiment_name", "experiment_type", "strategy_type", "report_spec",
            "generated_at", "report_version", "artefact_version",
            "tags", "files", "figures", "metrics_summary",
            "sections_rendered", "validation_verdict", "plot_index",
        ]
        for field in required:
            assert field in manifest, f"Missing canonical field: {field}"

    def test_report_spec_field_is_preset_name(self):
        from src.reporting.report_spec import CANONICAL_SHOWCASE
        manifest = self._write_manifest(spec=CANONICAL_SHOWCASE)
        assert manifest["report_spec"] == "CANONICAL_SHOWCASE"

    def test_report_spec_defaults_to_standard(self):
        manifest = self._write_manifest(spec=None)
        assert manifest["report_spec"] == "STANDARD_REPORT"

    def test_experiment_type_is_v1_without_ml(self):
        manifest = self._write_manifest()
        assert manifest["experiment_type"] == "v1_strategy"

    def test_sections_rendered_is_list(self):
        manifest = self._write_manifest()
        assert isinstance(manifest["sections_rendered"], list)
        assert "Performance Metrics" in manifest["sections_rendered"]

    def test_validation_verdict_is_valid_value(self):
        manifest = self._write_manifest()
        assert manifest["validation_verdict"] in ("pass", "marginal", "fail", "no_validation")

    def test_plot_index_is_list(self):
        manifest = self._write_manifest()
        assert isinstance(manifest["plot_index"], list)

    def test_metrics_summary_contains_core_keys(self):
        manifest = self._write_manifest()
        assert "sharpe_ratio" in manifest["metrics_summary"]
        assert "max_drawdown" in manifest["metrics_summary"]

    def test_report_spec_custom_for_unknown_spec(self):
        from src.reporting.report_spec import ResearchReportSpec
        custom = ResearchReportSpec(include_summary=False)
        manifest = self._write_manifest(spec=custom)
        assert manifest["report_spec"] == "custom"


class TestPresetIdentityCleanup:
    """E0.4 — Preset identity checks after DIAGNOSTICS_REPORT and FULL_DEMO_REPORT cleanup."""

    def test_diagnostics_report_has_thesis(self):
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        assert DIAGNOSTICS_REPORT.include_thesis is True

    def test_diagnostics_report_no_provenance(self):
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        assert DIAGNOSTICS_REPORT.include_provenance is False

    def test_diagnostics_report_has_diagnostics(self):
        from src.reporting.report_spec import DIAGNOSTICS_REPORT
        assert DIAGNOSTICS_REPORT.include_diagnostics is True

    def test_full_demo_is_not_standard(self):
        from src.reporting.report_spec import FULL_DEMO_REPORT, STANDARD_REPORT
        assert FULL_DEMO_REPORT != STANDARD_REPORT
        # Legacy preset has provenance; standard does not
        assert FULL_DEMO_REPORT.include_provenance is True
        assert STANDARD_REPORT.include_provenance is False

    def test_three_tier_narrative_progression(self):
        from src.reporting.report_spec import (
            CANONICAL_SHOWCASE, DIAGNOSTICS_REPORT, STANDARD_REPORT,
        )
        # STANDARD: no diagnostics, no provenance
        assert STANDARD_REPORT.include_diagnostics is False
        assert STANDARD_REPORT.include_provenance is False
        # DIAGNOSTICS: diagnostics added, still no provenance
        assert DIAGNOSTICS_REPORT.include_diagnostics is True
        assert DIAGNOSTICS_REPORT.include_provenance is False
        # CANONICAL: diagnostics + provenance
        assert CANONICAL_SHOWCASE.include_diagnostics is True
        assert CANONICAL_SHOWCASE.include_provenance is True

    def test_audit_report_flags(self):
        from src.reporting.report_spec import AUDIT_REPORT
        assert AUDIT_REPORT.include_figures is True
        assert AUDIT_REPORT.include_diagnostics is True
        assert AUDIT_REPORT.include_provenance is True
