"""Phase D1 / F3 orchestrator: run a full experiment from a config file.

Single public entrypoint:
    result = run_experiment_from_config(path)

Version "1" sequence (D1 pipeline):
    1. load_config  → raw dict (file I/O only)
    2. validate_config → raises on schema errors
    3. normalize_config → fills defaults
    4. factory layer → Strategy, UniverseSpec, ValidationConfig, ExperimentSpec
    5. Data loading → price DataFrame (I/O lives here, not in the factory)
    6. run_strategy → StrategyResult
    7. run_walk_forward_validation (if validation != "none")
    8. Build ExperimentResult
    9. Generate plots
    10. save_run → persists artefacts (core files + plots)
    11. _write_raw_config, _write_normalized_config → dual config artefacts
    12. ExperimentRegistry.register → registry.json

Version "2" sequence (F3 ML pipeline):
    Same 12 steps, with strategy replaced by ML components (features, labels,
    model, signal) built via ml_factory. A fresh model instance is used for
    walk-forward validation to prevent state contamination from the full run.
    An additional ml_provenance.json sidecar is written alongside the artefacts.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.config import ExperimentSpec
from src.experiments.config_io import load_config, normalize_config, validate_config
from src.experiments.factory import (
    build_experiment_spec,
    build_strategy,
    build_universe_spec,
    build_validation_config,
    build_validation_splits,
)
from src.experiments.registry import ExperimentRegistry
from src.experiments.results import ExperimentResult
from src.experiments.tracking import save_run
from src.portfolio.alignment import align_prices, load_universe
from src.strategies.runner import StrategyResult, run_strategy
from src.validation.walk_forward import WalkForwardResult, run_walk_forward_validation

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRun:
    """All outputs from a single orchestrated experiment."""

    spec: ExperimentSpec
    strategy_result: StrategyResult
    experiment_result: ExperimentResult
    walk_forward: WalkForwardResult | None
    output_path: Path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_plots(
    backtest_df: pd.DataFrame,
    wf: WalkForwardResult | None,
    strategy_name: str,
    weights: pd.DataFrame | None = None,
    ml_data: dict | None = None,
    universe_data: dict | None = None,
    stress_mask: pd.Series | None = None,
    profile: str = "report",
) -> dict:
    """Generate matplotlib figures; return name → Figure dict.

    Standard set (always):
        equity_and_drawdown  — full-period equity curve + drawdown panel
        rolling_sharpe       — 252-day rolling Sharpe
        rolling_volatility   — 63-day rolling annualised vol
        allocation_history   — stacked weight history (when weights provided)
        portfolio_turnover   — daily turnover + 21d rolling average (when weights provided)

    Walk-forward set (when wf ran):
        walk_forward_stitched  — chained out-of-sample equity curve
        split_sharpes          — per-split Sharpe bar chart
        split_equity_curves    — per-split equity curve overlay
        train_vs_test_sharpe   — train vs test Sharpe comparison

    ML set (when ml_data provided):
        ml_information_coefficient  — rolling IC / prediction-correlation series
        ml_prediction_distribution  — histogram of raw predictions
        ml_coefficient_stability    — mean±std bar chart per feature
        ml_signal_turnover          — per-period signal turnover
    """
    from src.visualization.backtest_plots import (
        plot_equity_and_drawdown,
        plot_rolling_sharpe,
        plot_rolling_volatility,
    )
    from src.visualization.portfolio_plots import plot_turnover, plot_weights
    from src.visualization.styles import apply_research_style
    from src.visualization.validation_plots import (
        plot_split_sharpes,
        plot_walk_forward_stitched,
    )

    apply_research_style(profile=profile)

    plots: dict = {}

    # --- Standard set ---
    plots["equity_and_drawdown"] = plot_equity_and_drawdown(
        backtest_df, title=f"{strategy_name} — Equity & Drawdown"
    )
    plots["rolling_sharpe"] = plot_rolling_sharpe(
        backtest_df, title=f"{strategy_name} — Rolling Sharpe Ratio (252d)"
    )
    plots["rolling_volatility"] = plot_rolling_volatility(
        backtest_df, title=f"{strategy_name} — Rolling Volatility (63d)"
    )

    if weights is not None and not weights.empty:
        plots["allocation_history"] = plot_weights(
            weights, title=f"{strategy_name} — Allocation History"
        )
        plots["portfolio_turnover"] = plot_turnover(
            weights, title=f"{strategy_name} — Portfolio Turnover"
        )

    # --- Walk-forward set ---
    if wf is not None and wf.n_splits > 0:
        from src.visualization.validation_plots import (
            plot_train_vs_test,
            plot_walk_forward_equity,
            plot_walk_forward_timeline,
        )
        plots["walk_forward_stitched"] = plot_walk_forward_stitched(
            wf, title=f"{strategy_name} — Walk-Forward (stitched)"
        )
        plots["split_sharpes"] = plot_split_sharpes(
            wf, title=f"{strategy_name} — Split Sharpe Ratios"
        )
        plots["split_equity_curves"] = plot_walk_forward_equity(
            wf, title=f"{strategy_name} — Walk-Forward Equity Curves"
        )
        plots["train_vs_test_sharpe"] = plot_train_vs_test(
            wf, title=f"{strategy_name} — Train vs Test Sharpe"
        )
        plots["walk_forward_timeline"] = plot_walk_forward_timeline(
            wf, title=f"{strategy_name} — Walk-Forward Window Timeline"
        )

    # --- ML diagnostic set ---
    if ml_data:
        from src.visualization.ml_plots import (
            plot_coefficient_stability,
            plot_feature_correlation_heatmap,
            plot_information_coefficient,
            plot_prediction_distribution,
            plot_signal_turnover,
        )
        ic_series = ml_data.get("ic_series")
        predictions = ml_data.get("predictions")
        coeff_stability = ml_data.get("coeff_stability")
        signal_turnover_series = ml_data.get("signal_turnover")
        feature_corr_df = ml_data.get("corr_df")

        if ic_series is not None and len(ic_series) >= 2:
            plots["ml_information_coefficient"] = plot_information_coefficient(
                ic_series, title=f"{strategy_name} — Prediction Correlation (monthly)"
            )
        # Cross-sectional IC (panel mode primary inline figure)
        cs_ic_daily = ml_data.get("cs_ic_daily")
        if cs_ic_daily is not None and len(cs_ic_daily) >= 10:
            from src.visualization.ml_plots import plot_ic_regime
            plots["cross_sectional_ic"] = plot_ic_regime(
                cs_ic_daily,
                title=f"{strategy_name} — Daily Cross-Sectional IC",
                stress_mask=stress_mask,
            )

        ic_daily_rolling = ml_data.get("ic_daily_rolling")
        if ic_daily_rolling is not None and len(ic_daily_rolling) >= 21:
            from src.visualization.ml_plots import plot_ic_regime
            plots["ml_ic_regime"] = plot_ic_regime(
                ic_daily_rolling,
                title=f"{strategy_name} — Rolling IC Through Time",
                stress_mask=stress_mask,
            )
        rolling_da = ml_data.get("rolling_da")
        if rolling_da is not None and len(rolling_da) >= 21:
            from src.visualization.ml_plots import plot_rolling_directional_accuracy
            is_panel_da = ml_data.get("is_panel", False)
            da_title = (
                f"{strategy_name} — Rolling IC Consistency (126d)"
                if is_panel_da
                else f"{strategy_name} — Rolling Directional Accuracy (126d)"
            )
            plots["ml_rolling_da"] = plot_rolling_directional_accuracy(
                rolling_da,
                title=da_title,
                stress_mask=stress_mask,
                is_panel=is_panel_da,
            )
        if predictions is not None and len(predictions) >= 10:
            plots["ml_prediction_distribution"] = plot_prediction_distribution(
                predictions, title=f"{strategy_name} — Prediction Distribution"
            )
        if coeff_stability is not None and not coeff_stability.empty:
            plots["ml_coefficient_stability"] = plot_coefficient_stability(
                coeff_stability, title=f"{strategy_name} — Coefficient Stability"
            )
        # Temporal split labels shared across heatmaps
        wf_split_labels: list[str] | None = None
        if wf is not None and wf.n_splits > 0:
            wf_split_labels = [
                sr.split.test_start.strftime("%Y-%m") for sr in wf.splits
            ]

        coeff_stability_df = ml_data.get("coeff_stability_df")
        if coeff_stability_df is not None and not coeff_stability_df.empty:
            from src.visualization.ml_plots import (
                plot_coefficient_evolution,
                plot_coefficient_sign_heatmap,
            )
            plots["ml_coefficient_evolution"] = plot_coefficient_evolution(
                coeff_stability_df,
                title=f"{strategy_name} — Coefficient Evolution",
                split_labels=wf_split_labels,
            )
            plots["ml_coefficient_sign_heatmap"] = plot_coefficient_sign_heatmap(
                coeff_stability_df,
                title=f"{strategy_name} — Coefficient Sign Stability",
                split_labels=wf_split_labels,
            )

        feature_ic_df = ml_data.get("feature_ic_splits")
        if feature_ic_df is not None and not feature_ic_df.empty:
            from src.visualization.ml_plots import plot_feature_ic_heatmap
            plots["feature_ic_heatmap"] = plot_feature_ic_heatmap(
                feature_ic_df,
                title=f"{strategy_name} — Per-Feature IC by Walk-Forward Split",
                split_labels=wf_split_labels,
            )

            feature_families = ml_data.get("feature_families")
            if feature_families:
                from src.visualization.ml_plots import plot_feature_family_ic
                plots["feature_family_ic"] = plot_feature_family_ic(
                    feature_ic_df,
                    feature_families,
                    title=f"{strategy_name} — Feature Family IC by Walk-Forward Split",
                    split_labels=wf_split_labels,
                )

        # Regime-conditional IC plot (Step 2 — vol regime interpretation)
        regime_stats = ml_data.get("regime_stats")
        if regime_stats and regime_stats.get("family_ic_by_regime"):
            try:
                from src.visualization.ml_plots import plot_ic_by_vol_regime
                plots["ic_by_vol_regime"] = plot_ic_by_vol_regime(
                    regime_stats["family_ic_by_regime"],
                    ml_data.get("feature_families"),
                    title=f"{strategy_name} — Feature Family IC by Volatility Regime",
                )
            except Exception:
                pass

        # Prediction-strength plot (Step 3 — confidence & outcome monotonicity)
        ps_data = ml_data.get("prediction_strength")
        if ps_data and ps_data.get("group_monthly") is not None:
            try:
                from src.visualization.ml_plots import plot_prediction_strength
                plots["prediction_strength"] = plot_prediction_strength(
                    ps_data,
                    title=f"{strategy_name} — Prediction Strength by Score Group",
                )
            except Exception:
                pass

        # Ranking geometry plot (Phase I — S1-S5 statistical state observability)
        rg_data = ml_data.get("ranking_geometry") or {}
        if rg_data:
            try:
                from src.visualization.ml_plots import plot_ranking_geometry
                plots["ranking_geometry"] = plot_ranking_geometry(
                    rg_data,
                    title=f"{strategy_name} — Cross-Sectional Ranking Geometry",
                )
            except Exception:
                pass

        # Feature contribution plots (Phase II — C1 & C2)
        fc_data = ml_data.get("feature_contributions") or {}
        if fc_data:
            try:
                from src.visualization.ml_plots import (
                    plot_family_contribution_timeline,
                    plot_feature_contribution_heatmap,
                )
                contrib_df = fc_data.get("contribution_df")
                if contrib_df is not None and not contrib_df.empty:
                    plots["feature_contribution_heatmap"] = plot_feature_contribution_heatmap(
                        contrib_df,
                        feature_families=ml_data.get("feature_families"),
                        title=f"{strategy_name} — Feature Contribution Through Time",
                    )
            except Exception:
                pass
            try:
                from src.visualization.ml_plots import plot_family_contribution_timeline
                fam_contrib = fc_data.get("family_contrib_df")
                fam_share = fc_data.get("family_share_df")
                if fam_contrib is not None or fam_share is not None:
                    plots["family_contribution_timeline"] = plot_family_contribution_timeline(
                        fam_contrib,
                        fam_share,
                        title=f"{strategy_name} — Feature Family Contribution Timeline",
                    )
            except Exception:
                pass

        # Allocation research plots (Phase 2 — panel mode)
        alloc_research = ml_data.get("alloc_research") or {}
        if alloc_research:
            try:
                from src.visualization.allocation_plots import (
                    plot_concentration_evolution,
                    plot_confidence_calibration,
                    plot_prediction_dispersion,
                )
                ar_weights = alloc_research.get("weights")
                if ar_weights is not None and not ar_weights.empty:
                    plots["allocation_concentration_evolution"] = plot_concentration_evolution(
                        ar_weights,
                        title=f"{strategy_name} — Allocation Concentration Through Time",
                    )
                ar_scores = alloc_research.get("score_wide")
                if ar_scores is not None and not ar_scores.empty:
                    plots["prediction_dispersion"] = plot_prediction_dispersion(
                        ar_scores,
                        stress_mask=stress_mask,
                        title=f"{strategy_name} — Cross-Sectional Prediction Dispersion",
                    )
                ar_calib = alloc_research.get("calibration_data")
                if ar_calib:
                    plots["confidence_calibration"] = plot_confidence_calibration(
                        ar_calib,
                        title=f"{strategy_name} — Confidence Calibration",
                    )
            except Exception:
                pass

        if signal_turnover_series is not None and len(signal_turnover_series) >= 2:
            plots["ml_signal_turnover"] = plot_signal_turnover(
                signal_turnover_series, title=f"{strategy_name} — Signal Turnover"
            )
        if feature_corr_df is not None and len(feature_corr_df) >= 2:
            plots["feature_correlation_heatmap"] = plot_feature_correlation_heatmap(
                feature_corr_df,
                title=f"{strategy_name} — Feature Correlation Matrix",
                feature_families=ml_data.get("feature_families"),
            )
        feature_matrix = ml_data.get("feature_matrix")
        if feature_matrix is not None and not feature_matrix.empty:
            from src.visualization.ml_plots import plot_feature_heatmap
            plots["ml_feature_regimes"] = plot_feature_heatmap(
                feature_matrix,
                title=f"{strategy_name} — Feature Regime Behaviour",
            )

        # D6: prediction vs actual scatter + overlay
        actual_s = ml_data.get("actual")
        pred_aligned_s = ml_data.get("predictions_aligned")
        if actual_s is not None and pred_aligned_s is not None and len(actual_s) >= 10:
            from src.visualization.ml_plots import plot_prediction_vs_actual, plot_residuals
            plots["ml_prediction_vs_actual"] = plot_prediction_vs_actual(
                actual_s, pred_aligned_s,
                title=f"{strategy_name} — Prediction vs Actual",
            )
            plots["ml_residuals"] = plot_residuals(
                actual_s, pred_aligned_s,
                title=f"{strategy_name} — Residual Diagnostics",
            )

    # --- Universe diagnostic set ---
    if universe_data:
        from src.visualization.universe_plots import (
            plot_asset_availability_timeline,
            plot_cross_asset_volatility,
            plot_universe_correlation_heatmap,
            plot_universe_coverage_heatmap,
        )
        monthly_cov = universe_data.get("monthly_coverage_df")
        if monthly_cov is not None and not monthly_cov.empty:
            plots["universe_coverage_heatmap"] = plot_universe_coverage_heatmap(
                monthly_cov,
                title=f"{strategy_name} — Universe Coverage by Asset & Month",
            )

        if (universe_data.get("n_assets", 0) >= 2
                and "monthly_coverage_df" in universe_data):
            # Use the raw prices implied by the monthly coverage index + asset count
            # Build a proxy prices frame from coverage (availability = notna proxy)
            universe_data["monthly_coverage_df"]
            # Reconstruct daily availability from asset_coverage for the timeline
            asset_cov = universe_data.get("asset_coverage") or []
            if asset_cov and "rolling_vol_df" in universe_data:
                # Build a prices-like frame with NaN for unavailable dates
                vol_df_ref = universe_data["rolling_vol_df"]
                avail_proxy = vol_df_ref.notna().astype(float)
                avail_proxy[avail_proxy == 0] = float("nan")
                plots["asset_availability_timeline"] = plot_asset_availability_timeline(
                    avail_proxy,
                    title=f"{strategy_name} — Rolling Asset Availability",
                )

        vol_df = universe_data.get("rolling_vol_df")
        if vol_df is not None and not vol_df.empty:
            plots["cross_asset_volatility"] = plot_cross_asset_volatility(
                vol_df,
                title=f"{strategy_name} — Cross-Asset Realised Volatility (63d)",
            )

        corr_df = universe_data.get("corr_df")
        if corr_df is not None and len(corr_df) >= 2:
            plots["universe_correlation_heatmap"] = plot_universe_correlation_heatmap(
                corr_df,
                title=f"{strategy_name} — Cross-Asset Correlation Structure",
            )

    return plots


_PLOT_METADATA: list[dict[str, str]] = [
    # Core performance — always present
    {"name": "equity_and_drawdown", "group": "performance", "importance": "primary",
     "caption": "Full-period equity curve (top) and drawdown (bottom). Cumulative net-of-cost growth anchored at 1.0; the drawdown panel shows peak-to-trough loss at each date."},
    {"name": "rolling_sharpe", "group": "performance", "importance": "primary",
     "caption": "252-trading-day rolling Sharpe ratio. Values consistently above zero confirm the strategy's risk-adjusted edge is not concentrated in a single sub-period."},
    {"name": "rolling_volatility", "group": "performance", "importance": "secondary",
     "caption": "63-day rolling annualised volatility. Persistent elevation indicates regime shifts; spikes mark acute stress events that may explain drawdown windows."},
    # Portfolio construction
    {"name": "allocation_history", "group": "portfolio", "importance": "secondary",
     "caption": "Stacked allocation history — fractional weight per asset at each rebalance date. Concentration corresponds to momentum leaders in the cross-sectional ranking."},
    {"name": "portfolio_turnover", "group": "portfolio", "importance": "secondary",
     "caption": "Daily portfolio turnover (bars) and 21-day rolling average (line). Turnover directly scales transaction-cost drag; spikes at regime transitions are expected."},
    # Walk-forward validation
    {"name": "walk_forward_stitched", "group": "validation", "importance": "primary",
     "caption": "Stitched out-of-sample equity curve: each segment is one walk-forward test window concatenated in chronological order. Upward drift confirms structural alpha."},
    {"name": "split_sharpes", "group": "validation", "importance": "primary",
     "caption": "Per-split out-of-sample Sharpe ratios. Consistent positive values across splits indicate a regime-independent signal; negative outliers highlight stress periods."},
    {"name": "split_equity_curves", "group": "validation", "importance": "secondary",
     "caption": "Overlay of individual out-of-sample equity curves by split. Similar trajectories confirm cross-period stability; divergent curves indicate regime sensitivity."},
    {"name": "train_vs_test_sharpe", "group": "validation", "importance": "secondary",
     "caption": "In-sample vs out-of-sample Sharpe comparison per split. Large train/test gaps suggest in-sample overfitting; modest gaps are consistent with genuine generalisation."},
    # ML signal
    {"name": "ml_information_coefficient", "group": "ml_signal", "importance": "primary",
     "caption": "Monthly rolling information coefficient (Pearson correlation of predicted vs actual returns). Persistent positive IC confirms the model adds directional information beyond chance."},
    {"name": "ml_prediction_vs_actual", "group": "ml_signal", "importance": "secondary",
     "caption": "Prediction vs actual overlay (top panel) and scatter (bottom panel). Appendix-level supplement to the rolling IC diagnostics. Concentration along the positive diagonal confirms directional alignment."},
    {"name": "ml_prediction_distribution", "group": "ml_signal", "importance": "secondary",
     "caption": "Distribution of raw model predictions. Near-symmetric distributions centred near zero indicate a well-calibrated model without directional drift or label leakage."},
    {"name": "ml_residuals", "group": "ml_signal", "importance": "primary",
     "caption": "Residual diagnostics: distribution of prediction errors (top) and rolling residual mean (bottom). A rolling mean persistently away from zero indicates systematic regime-specific bias."},
    {"name": "ml_coefficient_stability", "group": "ml_model", "importance": "primary",
     "caption": "Mean ± std of model coefficients across walk-forward splits. Bars with consistent sign and similar magnitude confirm a stable, replicable learned relationship."},
    {"name": "ml_coefficient_evolution", "group": "ml_model", "importance": "secondary",
     "caption": "Coefficient trajectory across walk-forward splits (chronological). Stable features maintain consistent sign and magnitude; unstable features cross zero, indicating regime-dependent learning."},
    {"name": "ml_signal_turnover", "group": "ml_signal", "importance": "secondary",
     "caption": "Per-period signal turnover (absolute position change). High turnover inflates transaction costs; the model's net alpha must comfortably exceed the implied cost drag."},
    {"name": "feature_correlation_heatmap", "group": "ml_features", "importance": "secondary",
     "caption": "Pairwise Pearson correlation matrix of the feature space. Low off-diagonal values indicate orthogonal information dimensions; high correlations flag potential multicollinearity."},
    {"name": "ml_feature_regimes", "group": "ml_features", "importance": "secondary",
     "caption": "Feature z-score heatmap (±3σ). Red marks periods of extreme positive feature values; blue marks extreme negative. Reveals feature regime transitions and co-movement across the backtest period."},
    {"name": "ml_ic_regime", "group": "ml_signal", "importance": "primary",
     "caption": "63-day rolling mean cross-sectional IC through time. Green fill marks sustained periods of positive IC; red marks signal breakdown. Width of each regime reveals persistence vs transience."},
    {"name": "ml_rolling_da", "group": "ml_signal", "importance": "primary",
     "caption": "126-day rolling IC consistency: fraction of days with positive cross-sectional IC. Values above 0.50 indicate the model ranks assets correctly more often than not; below 0.50 marks signal degradation."},
    {"name": "ml_coefficient_sign_heatmap", "group": "ml_model", "importance": "primary",
     "caption": "Coefficient values across walk-forward splits (x-axis) and features (y-axis). Blue = negative, red = positive; saturation encodes magnitude. Colour changes across splits reveal sign reversals and regime-specific learning."},
    {"name": "walk_forward_timeline", "group": "validation", "importance": "primary",
     "caption": "Gantt-style walk-forward window timeline. Each row is a split; grey = train window, coloured = test window (green positive OOS Sharpe, red negative). OOS Sharpe annotated on each test bar."},
    {"name": "feature_ic_heatmap", "group": "ml_features", "importance": "primary",
     "caption": "Per-feature IC against test-period labels by walk-forward split. Green = positive predictive IC; red = negative. Reveals which features drove signal quality in which regimes and where individual features broke down."},
    {"name": "feature_family_ic", "group": "ml_features", "importance": "primary",
     "caption": "Mean IC aggregated by feature family across walk-forward splits. Grouped bars reveal which hypothesis families (Trend, Volatility, Mean-Reversion, Market Structure) provided net positive signal in each regime and which degraded."},
    {"name": "ic_by_vol_regime", "group": "ml_features", "importance": "primary",
     "caption": "Feature family mean IC disaggregated by volatility regime. Each family shows two bars: high-volatility test splits (solid) vs low-volatility test splits (faded). Reveals which feature families provided signal preferentially under stressed vs calm market conditions. The vol regime classification uses the median cross-asset 21D realised vol across walk-forward test windows as the threshold."},
    {"name": "prediction_strength", "group": "ml_signal", "importance": "primary",
     "caption": "Prediction-strength bucket analysis. Top panel: mean realized N-day forward return by prediction score group (top, mid, bottom thirds). Monotonic ordering left-to-right confirms that score magnitude — not merely sign — carries economically meaningful cross-sectional information. Bottom panel: cumulative return of each prediction group over time; persistent separation between top and bottom groups confirms durable signal strength."},
    # Cross-sectional ranking geometry (Phase I)
    {"name": "ranking_geometry", "group": "ml_signal", "importance": "primary",
     "caption": "Cross-sectional ranking geometry — four panels covering signal geometry, discrimination, and temporal stability. Panel 1: rolling score IQR (left) and rolling IC standard deviation (right, dashed) — low IQR with high IC std identifies compressed-but-erratic regimes. Panel 2: rolling top-vs-bottom score spread — near-zero marks periods of ranking indifference. Panel 3: rolling realized forward-return spread between top and bottom ranked groups (pre-cost gross diagnostic). Panel 4: monthly Spearman rank autocorrelation — high values indicate stable model convictions; near-zero marks arbitrary rank flips each rebalance."},
    # Feature contribution diagnostics (Phase II)
    {"name": "feature_contribution_heatmap", "group": "ml_model", "importance": "primary",
     "caption": "Feature contribution heatmap: realised predictive influence (coefficient × standardised feature value) for each feature through time, grouped by family. Red = positive contribution (model predicts above-average return for this feature state); blue = negative. Regime shifts appear as horizontal band colour transitions; simultaneous sign changes across a family reveal coordinated hypothesis activation or suppression."},
    {"name": "family_contribution_timeline", "group": "ml_model", "importance": "primary",
     "caption": "Feature family contribution timeline. Top panel: signed rolling family contributions — shows which hypothesis family drove predictions and in which direction. Bottom panel: normalised absolute contribution share — shows which family dominated regardless of sign. Regime shifts appear as share transitions between families; sustained dominance by one family indicates the model operated in a stable predictive regime."},
    {"name": "cross_sectional_ic", "group": "ml_signal", "importance": "primary",
     "caption": "Daily cross-sectional Spearman IC: Spearman rank correlation between predicted scores and realized returns across all assets per date. Persistent positive values confirm the model correctly ranks assets in the cross-section."},
    # Allocation research (Phase 2 — panel mode only)
    {"name": "allocation_concentration_evolution", "group": "portfolio", "importance": "secondary",
     "caption": "Rolling 63-day concentration dynamics: HHI (top), effective breadth 1/HHI (middle), and entropy-based effective N (bottom). Equal-weight across k assets yields HHI = 1/k and effective N = k; higher HHI and lower breadth/N indicate concentrated bets. Persistent elevation in the HHI panel identifies regimes where the model systematically concentrates into a narrow set of assets."},
    {"name": "prediction_dispersion", "group": "ml_signal", "importance": "secondary",
     "caption": "Rolling 63-day cross-sectional prediction dispersion: cross-sectional standard deviation of raw scores (top) and top-minus-bottom score spread (bottom). Low dispersion indicates score compression — the model assigns near-identical scores across all assets, making rank differences economically arbitrary. Near-zero spread identifies ranking indifference regimes where allocation becomes effectively random within the cross-section."},
    {"name": "confidence_calibration", "group": "ml_signal", "importance": "primary",
     "caption": "Confidence calibration: mean realized forward return by prediction-score quintile (Q1 = lowest scores to Q5 = highest). A monotonically increasing pattern — higher-scored assets realizing higher forward returns — confirms that prediction magnitude carries economic information beyond directional sign alone. Non-monotonic patterns reveal calibration failure and flag the need for equal-weight or threshold-gated allocation rather than confidence-weighted schemes."},
    # Universe diagnostics (G1)
    {"name": "universe_coverage_heatmap", "group": "universe", "importance": "primary",
     "caption": "Monthly price coverage fraction by asset. Green = full availability; red = data gaps. Identifies structurally incomplete assets that reduce effective universe breadth."},
    {"name": "asset_availability_timeline", "group": "universe", "importance": "primary",
     "caption": "Rolling count of assets with valid prices. Structural drops reveal asset additions, delistings, or persistent data gaps affecting cross-sectional breadth."},
    {"name": "cross_asset_volatility", "group": "universe", "importance": "primary",
     "caption": "63-day rolling annualised volatility per asset. Persistent divergence between risk-on and risk-off assets confirms macro regime heterogeneity. Synchronised spikes identify systemic stress."},
    {"name": "universe_correlation_heatmap", "group": "universe", "importance": "primary",
     "caption": "Full-period pairwise return correlation matrix. Correlated clusters reduce effective cross-sectional breadth; orthogonal pairs (e.g. TLT vs equities) confirm regime diversification."},
]


def _write_plot_index(plots: dict, out_path: Path) -> None:
    """Write plots/plot_index.json — semantic ordering and captions for figures.

    Only entries for plots that were actually generated are included.
    """
    plots_dir = out_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    index = [entry for entry in _PLOT_METADATA if entry["name"] in plots]
    idx_path = plots_dir / "plot_index.json"
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _prepare_universe_diagnostics(prices: pd.DataFrame) -> dict:
    """Compute universe coverage and structure diagnostics from aligned price panel.

    Returns a dict consumed by both _build_plots() and _write_universe_artefacts().
    Non-figure data (monthly_coverage_df, rolling_vol_df, corr_df) is kept in-memory
    only; JSON-serialisable data is persisted by _write_universe_artefacts().

    Keys populated:
        tickers             list[str]
        n_assets            int
        n_trading_days      int
        date_range          dict with start/end
        asset_coverage      list[dict] — per-asset missingness + first/last date
        monthly_coverage_df pd.DataFrame (months × assets) — figure-only
        rolling_vol_df      pd.DataFrame (dates × assets) — figure-only
        corr_df             pd.DataFrame (assets × assets) — figure-only
        correlation_matrix  dict[ticker→dict[ticker→float]] — persisted
        vol_summary         dict[ticker→{mean_vol, min_vol, max_vol}] — persisted
    """
    import numpy as np

    result: dict = {}
    tickers = list(prices.columns)
    result["tickers"] = tickers
    result["n_assets"] = len(tickers)
    result["n_trading_days"] = len(prices)

    if len(prices.index) > 0:
        result["date_range"] = {
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
        }

    # Per-asset coverage statistics
    asset_coverage: list[dict] = []
    for ticker in tickers:
        s = prices[ticker]
        valid = s.dropna()
        asset_coverage.append({
            "ticker": ticker,
            "n_days": int(len(valid)),
            "first_date": str(valid.index.min().date()) if len(valid) > 0 else None,
            "last_date": str(valid.index.max().date()) if len(valid) > 0 else None,
            "missingness_pct": float(s.isna().mean()),
        })
    result["asset_coverage"] = asset_coverage

    # Monthly coverage fraction (figure-only)
    try:
        monthly = prices.resample("ME").apply(lambda x: float(x.notna().mean()))
        result["monthly_coverage_df"] = monthly
    except Exception:
        pass

    # Cross-asset returns (used for correlation + volatility)
    returns = prices.pct_change().dropna(how="all")

    # Per-asset rolling volatility (63d annualised, figure-only)
    if len(returns) >= 21:
        vol = returns.rolling(63, min_periods=21).std() * np.sqrt(252)
        result["rolling_vol_df"] = vol

        vol_summary: dict[str, dict] = {}
        for ticker in tickers:
            v = vol[ticker].dropna() if ticker in vol.columns else pd.Series(dtype=float)
            if len(v) > 0:
                vol_summary[ticker] = {
                    "mean_vol": _sanitize_for_json(float(v.mean())),
                    "min_vol": _sanitize_for_json(float(v.min())),
                    "max_vol": _sanitize_for_json(float(v.max())),
                }
        result["vol_summary"] = vol_summary

    # Cross-asset correlation matrix
    if len(returns) >= 21 and len(tickers) >= 2:
        try:
            corr = returns.corr()
            result["corr_df"] = corr
            result["correlation_matrix"] = {
                t1: {
                    t2: _sanitize_for_json(float(corr.loc[t1, t2]))
                    for t2 in tickers if t2 in corr.columns
                }
                for t1 in tickers if t1 in corr.index
            }
        except Exception:
            pass

    return result


def _write_universe_artefacts(universe_data: dict, out_path: Path) -> None:
    """Persist diagnostics/universe_coverage.json from pre-computed universe_data.

    Serialises only JSON-safe data — DataFrames (monthly_coverage_df,
    rolling_vol_df, corr_df) are excluded and remain in-memory only.
    """
    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    payload: dict[str, Any] = {
        "tickers": universe_data.get("tickers", []),
        "n_assets": universe_data.get("n_assets", 0),
        "n_trading_days": universe_data.get("n_trading_days", 0),
        "date_range": universe_data.get("date_range", {}),
        "asset_coverage": universe_data.get("asset_coverage", []),
        "vol_summary": universe_data.get("vol_summary", {}),
    }
    if "correlation_matrix" in universe_data:
        payload["correlation_matrix"] = universe_data["correlation_matrix"]

    with (diag_dir / "universe_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_raw_config(raw_cfg: dict[str, Any], out_dir: Path, source_path: Path) -> None:
    """Copy the original config file verbatim into out_dir/raw_config.<ext>."""
    suffix = source_path.suffix  # .yaml, .yml, or .json
    dest = out_dir / f"raw_config{suffix}"
    shutil.copy2(source_path, dest)


def _write_normalized_config(norm_cfg: dict[str, Any], out_dir: Path) -> None:
    """Write the normalized config as JSON with sorted keys."""
    dest = out_dir / "normalized_config.json"
    with dest.open("w", encoding="utf-8") as f:
        json.dump(norm_cfg, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_experiment_from_config(path: str | Path, profile: str = "report") -> ExperimentRun:
    """Run a complete experiment from a YAML or JSON config file.

    Supports version "1" configs (D1 strategy-based pipeline) and version "2"
    configs (F3 ML pipeline).  Version is detected from the config's 'version'
    field; configs without a version field default to "1".

    Args:
        path: Path to the config file (.yaml, .yml, or .json).
        profile: Rendering/export profile for canonical figures.
                 "report" (default) — compact publication density.
                 "frontend" — browser-inspectable typography and spacing.

    Returns:
        ExperimentRun containing all produced artefacts and output path.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config fails validation.
    """
    import matplotlib
    matplotlib.use("Agg")

    source_path = Path(path).resolve()

    # ------------------------------------------------------------------
    # 1. Load raw config — version routing happens here
    # ------------------------------------------------------------------
    raw_cfg = load_config(source_path)
    version = str(raw_cfg.get("version", "1"))
    if version == "2":
        return _run_ml_experiment(raw_cfg, source_path, profile=profile)

    # ------------------------------------------------------------------
    # 2-3. Validate, normalize (version "1" pipeline)
    # ------------------------------------------------------------------
    validate_config(raw_cfg)
    norm_cfg = normalize_config(raw_cfg)

    # ------------------------------------------------------------------
    # 4. Factory layer — pure construction, no I/O
    # ------------------------------------------------------------------
    strategy = build_strategy(norm_cfg["strategy"])
    uni_spec = build_universe_spec(norm_cfg["universe"], norm_cfg["date_range"])
    val_config = build_validation_config(norm_cfg["validation"])
    spec = build_experiment_spec(norm_cfg)

    # ------------------------------------------------------------------
    # 5. Data loading (I/O lives here, not in the factory)
    # ------------------------------------------------------------------
    universe = load_universe(list(uni_spec.tickers))
    prices = align_prices(universe)

    # Slice to configured date range
    prices = prices.loc[uni_spec.start_date : uni_spec.end_date]

    # ------------------------------------------------------------------
    # 6. Run strategy
    # ------------------------------------------------------------------
    cost_bps = norm_cfg["execution"]["transaction_cost_bps"]
    sr = run_strategy(prices, strategy, transaction_cost_bps=cost_bps)

    # ------------------------------------------------------------------
    # 7. Walk-forward validation
    # ------------------------------------------------------------------
    wf: WalkForwardResult | None = None
    if val_config.type != "none":
        splits = build_validation_splits(val_config, prices.index)
        if splits:
            wf = run_walk_forward_validation(
                prices=prices,
                strategy=strategy,
                splits=splits,
                transaction_cost_bps=cost_bps,
            )

    # ------------------------------------------------------------------
    # 8. Build ExperimentResult
    # ------------------------------------------------------------------
    experiment_result = ExperimentResult(
        experiment_name=spec.experiment_name,
        strategy_name=strategy.name,
        parameters=spec.parameters,
        metrics=sr.metrics,
        weights=sr.weights,
        equity_curve=sr.backtest["equity_curve"],
        returns=sr.backtest["net_return"],
        created_at=datetime.now(UTC),
    )

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    universe_data_v1 = _prepare_universe_diagnostics(prices)
    plots = _build_plots(
        sr.backtest, wf, strategy.name,
        weights=sr.weights,
        universe_data=universe_data_v1,
        profile=profile,
    )

    # ------------------------------------------------------------------
    # 10. Save artefacts
    # ------------------------------------------------------------------
    output_cfg = norm_cfg["output"]
    output_dir = Path(output_cfg["base_dir"])
    out_path = save_run(
        experiment_result,
        spec=spec,
        output_dir=output_dir,
        plots=plots if output_cfg.get("save_plots", True) else None,
    )
    if output_cfg.get("save_plots", True):
        _write_plot_index(plots, out_path)

    import matplotlib.pyplot as plt
    for fig in plots.values():
        plt.close(fig)

    # ------------------------------------------------------------------
    # 11. Write dual config artefacts
    # ------------------------------------------------------------------
    _write_raw_config(raw_cfg, out_path, source_path)
    _write_normalized_config(norm_cfg, out_path)

    # ------------------------------------------------------------------
    # 11b. Persist diagnostics (consumers of already-computed results)
    # ------------------------------------------------------------------
    if wf is not None and wf.n_splits > 0:
        _write_split_metrics(wf, out_path)
        _write_wf_equity_curves(wf, out_path)
    _write_backtest_diagnostics(sr, out_path)
    _write_research_artefacts(prices, strategy, sr, out_path)
    _write_universe_artefacts(universe_data_v1, out_path)

    # ------------------------------------------------------------------
    # 12. Register
    # ------------------------------------------------------------------
    if output_cfg.get("register", True):
        registry_path = Path(output_cfg["registry_path"])
        registry = ExperimentRegistry(registry_path)
        registry.register(experiment_result, spec=spec, path=out_path)

    return ExperimentRun(
        spec=spec,
        strategy_result=sr,
        experiment_result=experiment_result,
        walk_forward=wf,
        output_path=out_path,
    )


# ---------------------------------------------------------------------------
# Diagnostics persistence helpers
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None and coerce date-like objects."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return str(obj)[:10]  # date only — Timestamp, date, datetime
    if hasattr(obj, "item"):
        return _sanitize_for_json(obj.item())  # numpy scalar → Python scalar
    return obj


def _write_split_metrics(wf: WalkForwardResult, out_path: Path) -> None:
    """Persist per-split metrics to diagnostics/split_metrics.json.

    Written for every experiment where walk-forward validation ran.
    Consumes the already-computed WalkForwardResult — no recomputation.
    """
    from src.validation.stability import split_metrics_table, summarize_stability

    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    summary = _sanitize_for_json(summarize_stability(wf))
    table = split_metrics_table(wf)

    splits_data: list[dict[str, Any]] = []
    if not table.empty:
        for row in table.reset_index().to_dict("records"):
            splits_data.append(_sanitize_for_json(row))

    payload: dict[str, Any] = {
        "n_splits": wf.n_splits,
        "summary": summary,
        "splits": splits_data,
    }
    with (diag_dir / "split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _detect_drawdown_windows(
    equity_curve: pd.Series,
    threshold: float = -0.05,
) -> list[dict]:
    """Return list of drawdown windows where peak-to-trough exceeds threshold."""
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    in_dd = dd < threshold

    windows: list[dict] = []
    in_window = False
    start_date = None

    for date, val in in_dd.items():
        if val and not in_window:
            in_window = True
            start_date = date
        elif not val and in_window:
            in_window = False
            seg = dd.loc[start_date:date]
            trough = seg.idxmin()
            windows.append({
                "start": str(start_date.date()),
                "trough": str(trough.date()),
                "recovery": str(date.date()),
                "max_dd": float(seg.min()),
                "duration_days": int((date - start_date).days),
            })

    # Open drawdown at end of series
    if in_window and start_date is not None:
        seg = dd.loc[start_date:]
        trough = seg.idxmin()
        windows.append({
            "start": str(start_date.date()),
            "trough": str(trough.date()),
            "recovery": None,
            "max_dd": float(seg.min()),
            "duration_days": int((equity_curve.index[-1] - start_date).days),
        })

    return windows


def _write_research_artefacts(
    prices: pd.DataFrame,
    strategy: Any,
    sr: StrategyResult,
    out_path: Path,
) -> None:
    """Persist intermediate research artefacts to research/ subdirectory.

    Always written:
        research/data_summary.json      — universe coverage, NaN counts, return stats
        research/signal_transitions.json — rebalance events with holdings/entries/exits

    Written for MomentumRotation strategies:
        momentum scores embedded in each transition event

    These are recomputed from already-available objects (prices, sr.weights) — no
    additional data loading or strategy re-execution occurs.
    """
    research_dir = out_path / "research"
    research_dir.mkdir(exist_ok=True)

    returns = prices.pct_change()

    # ------------------------------------------------------------------ #
    # data_summary.json
    # ------------------------------------------------------------------ #
    nan_counts: dict[str, int] = {
        col: int(prices[col].isna().sum()) for col in prices.columns
    }

    return_stats: dict[str, Any] = {}
    for col in prices.columns:
        r = returns[col].dropna()
        if len(r) >= 2:
            return_stats[col] = _sanitize_for_json({
                "mean_annual": float(r.mean() * 252),
                "vol_annual": float(r.std() * math.sqrt(252)),
                "skew": float(r.skew()),
            })
        else:
            return_stats[col] = {"mean_annual": None, "vol_annual": None, "skew": None}

    data_summary: dict[str, Any] = {
        "start_date": str(prices.index[0].date()) if len(prices) else None,
        "end_date": str(prices.index[-1].date()) if len(prices) else None,
        "n_days": len(prices),
        "n_assets": len(prices.columns),
        "assets": list(prices.columns),
        "nan_counts": nan_counts,
        "join_policy": "inner",
        "return_stats": return_stats,
    }
    with (research_dir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data_summary, f, indent=2)

    # ------------------------------------------------------------------ #
    # signal_transitions.json
    # ------------------------------------------------------------------ #
    weights = sr.weights  # already lagged (as applied)

    # Momentum scores per rebalance date — strategy-specific
    momentum_by_date: dict[str, dict[str, float]] = {}
    try:
        from src.portfolio.panel import universe_momentum
        from src.strategies.momentum_rotation import MomentumRotationStrategy

        if isinstance(strategy, MomentumRotationStrategy):
            mom = universe_momentum(prices, window=strategy.lookback)
            periodic_mom = mom.resample(strategy.rebalance_freq).last()
            for date in periodic_mom.index:
                row = {col: _sanitize_for_json(float(v))
                       for col, v in periodic_mom.loc[date].items()
                       if not (isinstance(v, float) and math.isnan(v))}
                if row:
                    momentum_by_date[str(date.date())] = row
    except Exception:
        pass  # momentum artefacts are best-effort

    # Detect rebalance events from weight changes
    prev = weights.shift(1).fillna(0.0)
    changed = (weights.abs() - prev.abs()).abs().sum(axis=1) > 1e-6
    active = weights.abs().sum(axis=1) > 1e-8
    rebalance_mask = changed & active

    rebalance_dates = weights.index[rebalance_mask]
    rebalance_freq = ""
    try:
        from src.strategies.momentum_rotation import MomentumRotationStrategy
        if isinstance(strategy, MomentumRotationStrategy):
            rebalance_freq = strategy.rebalance_freq
    except Exception:
        pass

    transitions: list[dict[str, Any]] = []
    prev_holdings: set[str] = set()

    for date in rebalance_dates:
        current_holdings: set[str] = set(
            col for col in weights.columns if weights.loc[date, col] > 1e-8
        )
        entered = sorted(current_holdings - prev_holdings)
        exited = sorted(prev_holdings - current_holdings)

        event: dict[str, Any] = {
            "date": str(date.date()),
            "holdings": sorted(current_holdings),
            "entered": entered,
            "exited": exited,
        }

        date_str = str(date.date())
        if date_str in momentum_by_date:
            event["momentum_scores"] = momentum_by_date[date_str]

        transitions.append(event)
        prev_holdings = current_holdings

    signal_transitions: dict[str, Any] = {
        "n_rebalances": len(transitions),
        "rebalance_frequency": rebalance_freq,
        "transitions": transitions,
    }
    with (research_dir / "signal_transitions.json").open("w", encoding="utf-8") as f:
        json.dump(signal_transitions, f, indent=2)


def _write_backtest_diagnostics(
    sr: StrategyResult,
    out_path: Path,
) -> None:
    """Persist rolling backtest diagnostics to diagnostics/backtest_diagnostics.json.

    Derived entirely from the already-computed StrategyResult — no recomputation.
    All time series are sub-sampled to monthly frequency to keep file size small.
    """
    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    net_return = sr.backtest["net_return"]
    equity_curve = sr.backtest["equity_curve"]
    turnover = sr.backtest["turnover"]

    # Rolling Sharpe (252d) — monthly sub-sample
    roll_mean = net_return.rolling(252).mean()
    roll_std = net_return.rolling(252).std().replace(0.0, float("nan"))
    rolling_sharpe = (roll_mean / roll_std) * math.sqrt(252)
    monthly_sharpe = rolling_sharpe.resample("ME").last().dropna()

    # Rolling vol (63d, annualised) — monthly sub-sample
    rolling_vol = net_return.rolling(63).std() * math.sqrt(252)
    monthly_vol = rolling_vol.resample("ME").last().dropna()

    # Monthly average turnover
    monthly_turnover = turnover.resample("ME").mean().dropna()

    # Drawdown windows > 5%
    dd_windows = _detect_drawdown_windows(equity_curve, threshold=-0.05)

    payload: dict[str, Any] = {
        "rolling_sharpe_252d": [
            {"date": str(d.date()), "value": _sanitize_for_json(float(v))}
            for d, v in monthly_sharpe.items()
        ],
        "rolling_vol_63d": [
            {"date": str(d.date()), "value": _sanitize_for_json(float(v))}
            for d, v in monthly_vol.items()
        ],
        "monthly_avg_turnover": [
            {"date": str(d.date()), "value": _sanitize_for_json(float(v))}
            for d, v in monthly_turnover.items()
        ],
        "drawdown_windows": dd_windows,
        "n_drawdown_windows_gt5pct": len(dd_windows),
    }

    with (diag_dir / "backtest_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_ml_diagnostics(
    sr: StrategyResult,
    model_type: str,
    wf: WalkForwardResult | None,
    out_path: Path,
) -> None:
    """Persist ML-specific diagnostics to diagnostics/ml_diagnostics.json.

    Written only for version "2" (ML) experiments.  Computes turnover and
    signal activity from the already-available StrategyResult — no refitting.
    """
    from src.ml.diagnostics.turnover import average_turnover, turnover_by_split

    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    weights = sr.weights
    avg_to = average_turnover(weights)
    n_periods = len(weights)

    # Fraction of days with any non-zero position
    if not weights.empty:
        signal_activity = float((weights.abs().sum(axis=1) > 1e-8).mean())
    else:
        signal_activity = 0.0

    payload: dict[str, Any] = {
        "model_type": model_type,
        "average_turnover": _sanitize_for_json(avg_to),
        "signal_activity": signal_activity,
        "n_weight_periods": n_periods,
    }

    # Per-split turnover summary when walk-forward ran
    if wf is not None and wf.n_splits > 0:
        split_weights = [s.weights for s in wf.splits]
        to_df = turnover_by_split(split_weights)
        if not to_df.empty:
            payload["turnover_by_split"] = _sanitize_for_json(
                to_df.reset_index().to_dict("records")
            )

    with (diag_dir / "ml_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Allocation diagnostics sidecar
# ---------------------------------------------------------------------------


def _write_allocation_diagnostics(
    weights: pd.DataFrame,
    ml_spec: Any,
    out_path: Path,
    alloc_research: dict | None = None,
) -> None:
    """Write diagnostics/allocation_diagnostics.json for portfolio construction research.

    Computes concentration, breadth, and entropy metrics from the weight DataFrame
    generated by the full-period strategy run.  Metrics are row-wise; no cross-date
    normalization is applied.  When alloc_research is provided, dispersion and
    calibration summary stats are merged into the JSON payload.

    Args:
        weights: Date × Asset weight DataFrame from the strategy run.
        ml_spec: MLExperimentSpec (used for weighting policy metadata).
        out_path: Experiment output directory.
        alloc_research: Optional dict from _prepare_allocation_research_diagnostics.
    """
    import math as _math

    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    if weights is None or weights.empty:
        payload: dict[str, Any] = {"available": False, "reason": "no weight data"}
        with (diag_dir / "allocation_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return

    w = weights.fillna(0.0)
    abs_w = w.abs()

    # Active rows only (at least one non-zero weight)
    active_mask = (abs_w > 1e-10).any(axis=1)
    w.loc[active_mask]
    abs_active = abs_w.loc[active_mask]

    n_periods = int(active_mask.sum())

    if n_periods == 0:
        payload = {"available": False, "reason": "all weights are zero"}
        with (diag_dir / "allocation_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return

    # HHI (Herfindahl-Hirschman Index) — row-wise sum of squared weights
    hhi_series = (abs_active ** 2).sum(axis=1)
    mean_hhi = float(hhi_series.mean())

    # Effective breadth: 1 / HHI (number of equally-effective bets implied by concentration)
    effective_breadth = float((1.0 / hhi_series.replace(0.0, float("nan"))).mean())

    # Shannon entropy of weight distribution → effective N = exp(H)
    def _row_entropy(row: pd.Series) -> float:
        pos = row[row > 1e-10]
        if pos.empty:
            return 0.0
        total = pos.sum()
        if total <= 0:
            return 0.0
        p = pos / total
        return float(-float((p * p.apply(_math.log)).sum()))

    entropy_series = abs_active.apply(_row_entropy, axis=1)
    mean_entropy = float(entropy_series.mean())
    effective_n_entropy = float(_math.exp(mean_entropy)) if mean_entropy > 0 else 1.0

    # Held count per period
    held_count = (abs_active > 1e-10).sum(axis=1)
    mean_held = float(held_count.mean())
    max_held = int(held_count.max())
    min_held = int(held_count.min())

    # Max weight per period
    max_weight_series = abs_active.max(axis=1)
    mean_max_weight = float(max_weight_series.mean())
    overall_max_weight = float(max_weight_series.max())

    payload = {
        "available": True,
        "n_active_periods": n_periods,
        "weighting_scheme": ml_spec.portfolio_construction.weighting.scheme,
        "prediction_normalization": ml_spec.portfolio_construction.weighting.prediction_normalization,
        "temperature": ml_spec.portfolio_construction.weighting.temperature,
        "concentration": {
            "mean_hhi": _sanitize_for_json(mean_hhi),
            "mean_effective_breadth": _sanitize_for_json(effective_breadth),
            "mean_entropy": _sanitize_for_json(mean_entropy),
            "effective_n_entropy": _sanitize_for_json(effective_n_entropy),
        },
        "holdings": {
            "mean_held_count": _sanitize_for_json(mean_held),
            "max_held_count": max_held,
            "min_held_count": min_held,
        },
        "weights": {
            "mean_max_weight": _sanitize_for_json(mean_max_weight),
            "overall_max_weight": _sanitize_for_json(overall_max_weight),
        },
    }

    # Merge dispersion + calibration summaries from allocation research diagnostics
    if alloc_research:
        if "dispersion_summary" in alloc_research:
            payload["prediction_dispersion"] = alloc_research["dispersion_summary"]
        if "calibration_summary" in alloc_research:
            payload["confidence_calibration"] = alloc_research["calibration_summary"]

    with (diag_dir / "allocation_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Feature engineering diagnostics helpers (Phase C)
# ---------------------------------------------------------------------------

# Feature type → semantic category
_FEATURE_CATEGORY: dict[str, str] = {
    "momentum": "momentum",
    "rolling_volatility": "volatility",
    "rolling_zscore": "normalization",
    "sma": "trend",
    "ema": "trend",
    "compute_returns": "returns",
    "trend_strength": "trend",
    "downside_volatility": "volatility",
    "vol_of_vol": "volatility",
    "vol_percentile": "volatility",
    "bollinger_distance": "mean_reversion",
    "rolling_skewness": "market_structure",
    "rolling_autocorrelation": "market_structure",
    # Phase H-1 additions
    "trend_persistence": "trend",
    "breakout_strength": "trend",
    "drawdown_distance": "mean_reversion",
    "vol_compression": "volatility",
    "rolling_beta": "market_structure",
    "risk_adjusted_momentum": "momentum",
}

# Feature type → transform label
_FEATURE_TRANSFORM: dict[str, str] = {
    "momentum": "trailing_return",
    "rolling_volatility": "rolling_std_annualised",
    "rolling_zscore": "rolling_zscore",
    "sma": "simple_moving_average",
    "ema": "exp_moving_average",
    "compute_returns": "log_return",
    "trend_strength": "slope_r2",
    "downside_volatility": "rolling_semi_std",
    "vol_of_vol": "rolling_std_of_vol",
    "vol_percentile": "rolling_pct_rank",
    "bollinger_distance": "band_distance",
    "rolling_skewness": "rolling_skewness",
    "rolling_autocorrelation": "rolling_autocorr",
    # Phase H-1 additions
    "trend_persistence": "rolling_pos_day_fraction",
    "breakout_strength": "price_vs_rolling_high",
    "drawdown_distance": "price_vs_rolling_peak",
    "vol_compression": "short_long_vol_ratio",
    "rolling_beta": "rolling_cov_over_var",
    "risk_adjusted_momentum": "momentum_over_vol",
}


def _prepare_feature_diagnostics(
    strategy: Any,
    prices: pd.DataFrame,
    ml_spec: Any,
    is_panel: bool = False,
) -> dict:
    """Compute feature-engineering diagnostics from the already-fitted strategy.

    Returns a dict consumed by both _build_plots() and
    _write_feature_engineering_artefacts().  All operations best-effort.

    Keys populated:
        feature_names          list[str]
        feature_stats          dict[feature → stats_dict]
        alignment              dict with row-count diagnostics
        corr_df                pd.DataFrame (feature × feature Pearson corr)
        feature_registry       list[dict] from ml_spec.features.entries
        ticker                 str
    """
    from src.ml.feature_matrix import align_features_and_labels, build_feature_matrix

    result: dict = {}

    # For panel mode, access feature functions via the builder for the first ticker
    try:
        if is_panel and hasattr(strategy, "_feature_fn_builder") and strategy._tickers:
            ref_ticker = strategy._tickers[0]
            feature_fns = strategy._feature_fn_builder(ref_ticker)
        else:
            feature_fns = strategy._feature_fns
    except Exception:
        feature_fns = getattr(strategy, "_feature_fns", {})

    try:
        X = build_feature_matrix(prices, feature_fns)
        feature_names = list(X.columns)
        result["feature_names"] = feature_names

        # Per-feature statistics
        feature_stats: dict[str, Any] = {}
        for col in X.columns:
            s = X[col]
            valid = s.dropna()
            n_valid = len(valid)
            stats: dict[str, Any] = {
                "n_valid": n_valid,
                "n_nan": int(s.isna().sum()),
                "first_valid_date": str(valid.index[0].date()) if n_valid > 0 else None,
                "sample_coverage": round(float(s.notna().mean()), 4),
                "mean": _sanitize_for_json(float(valid.mean())) if n_valid >= 2 else None,
                "std": _sanitize_for_json(float(valid.std())) if n_valid >= 2 else None,
                "skew": _sanitize_for_json(float(valid.skew())) if n_valid >= 3 else None,
                "kurtosis": _sanitize_for_json(float(valid.kurtosis())) if n_valid >= 4 else None,
                "ar1": _sanitize_for_json(float(valid.autocorr(lag=1))) if n_valid >= 3 else None,
                "min": _sanitize_for_json(float(valid.min())) if n_valid >= 1 else None,
                "max": _sanitize_for_json(float(valid.max())) if n_valid >= 1 else None,
            }
            feature_stats[col] = stats
        result["feature_stats"] = feature_stats
        result["feature_stats_pooled"] = False  # default; overridden in panel path below

        # Panel mode: replace single-ref stats with pooled panel statistics
        if is_panel and hasattr(strategy, "_tickers") and strategy._tickers:
            try:
                from src.ml.feature_matrix import build_feature_matrix as _bfm
                all_cols: dict[str, list] = {col: [] for col in feature_names}
                for tk in strategy._tickers:
                    if tk not in prices.columns:
                        continue
                    fns = strategy._feature_fn_builder(tk)
                    Xk = _bfm(prices, fns)
                    for col in feature_names:
                        if col in Xk.columns:
                            all_cols[col].append(Xk[col].dropna())
                pooled_stats: dict[str, Any] = {}
                for col, series_list in all_cols.items():
                    if not series_list:
                        pooled_stats[col] = feature_stats.get(col, {})
                        continue
                    pooled = pd.concat(series_list, ignore_index=True)
                    n_valid = len(pooled)
                    pooled_stats[col] = {
                        "n_valid": n_valid,
                        "n_nan": 0,
                        "first_valid_date": None,
                        "sample_coverage": round(float(n_valid / (n_valid + 1)), 4),
                        "mean": _sanitize_for_json(float(pooled.mean())) if n_valid >= 2 else None,
                        "std": _sanitize_for_json(float(pooled.std())) if n_valid >= 2 else None,
                        "skew": _sanitize_for_json(float(pooled.skew())) if n_valid >= 3 else None,
                        "kurtosis": _sanitize_for_json(float(pooled.kurtosis())) if n_valid >= 4 else None,
                        "ar1": _sanitize_for_json(float(pooled.autocorr(lag=1))) if n_valid >= 3 else None,
                        "min": _sanitize_for_json(float(pooled.min())) if n_valid >= 1 else None,
                        "max": _sanitize_for_json(float(pooled.max())) if n_valid >= 1 else None,
                    }
                result["feature_stats"] = pooled_stats
                result["feature_stats_pooled"] = True
            except Exception:
                pass  # fall back to ref-asset stats already stored

        # Alignment diagnostics
        X_clean = X.dropna()
        if is_panel and hasattr(strategy, "_tickers") and strategy._tickers:
            # For panel mode, use single-asset forward returns for alignment diagnostics
            from src.ml.labels import forward_returns as _fwd_ret
            ref_ticker = strategy._tickers[0]
            horizon = getattr(strategy, "_horizon", 21)
            labels = _fwd_ret(prices[ref_ticker], horizon) if ref_ticker in prices.columns else None
        else:
            labels = strategy._label_fn(prices)
        if labels is not None and isinstance(labels, pd.Series):
            X_aligned, _y_aligned = align_features_and_labels(X, labels)
        else:
            X_aligned = X_clean

        n_raw = len(X)
        n_feature_clean = len(X_clean)
        n_aligned = len(X_aligned)

        n_universe_assets = len(strategy._tickers) if (
            is_panel and hasattr(strategy, "_tickers")
        ) else None
        result["alignment"] = {
            "n_rows_raw": n_raw,
            "n_rows_features_clean": n_feature_clean,
            "n_rows_after_alignment": n_aligned,
            "warmup_rows_removed": n_raw - n_feature_clean,
            "label_rows_removed": n_feature_clean - n_aligned,
            "alignment_loss_pct": round((1 - n_aligned / n_raw) * 100, 2) if n_raw > 0 else None,
            "sample_start": str(X_aligned.index[0].date()) if n_aligned > 0 else None,
            "sample_end": str(X_aligned.index[-1].date()) if n_aligned > 0 else None,
            # Panel mode: these counts are per-asset (temporal). Pooled panel
            # observations = n_rows_after_alignment × n_universe_assets.
            "n_universe_assets": n_universe_assets,
            "is_panel": is_panel,
        }

        # Pairwise correlation matrix (on aligned subset for consistency)
        if n_aligned > 1 and len(feature_names) >= 2:
            corr = X_aligned.corr()
            result["corr_df"] = corr

        # Full feature matrix — used for feature regime heatmap
        result["feature_matrix"] = X
    except Exception:
        pass

    # Feature registry from ml_spec
    try:
        from src.features.families import get_family_for_type, group_by_family

        registry: list[dict[str, Any]] = []
        feature_type_map: dict[str, str] = {}
        for entry in ml_spec.features.entries:
            params = entry.params or {}
            window = (
                params.get("window")
                or params.get("lookback")
                or params.get("span")
            )
            family = get_family_for_type(entry.type)
            registry.append({
                "name": entry.name,
                "type": entry.type,
                "family": family,
                "category": _FEATURE_CATEGORY.get(entry.type, "other"),
                "transform": _FEATURE_TRANSFORM.get(entry.type, entry.type),
                "params": params,
                "window": window,
                "normalization_type": (
                    "zscore" if entry.type == "rolling_zscore" else "raw"
                ),
            })
            feature_type_map[entry.name] = entry.type
        result["feature_registry"] = registry
        if is_panel and hasattr(strategy, "_tickers") and strategy._tickers:
            result["ticker"] = strategy._tickers[0]
            result["panel_tickers"] = strategy._tickers
        else:
            result["ticker"] = ml_spec.features.ticker

        # Group features by family (using type-based lookup for precision)
        feature_names_reg = [e["name"] for e in registry]
        result["feature_families"] = group_by_family(
            feature_names_reg, feature_types=feature_type_map
        )
    except Exception:
        pass

    return result


def _write_feature_engineering_artefacts(
    feature_data: dict,
    ml_spec: Any,
    out_path: Path,
) -> None:
    """Persist feature engineering artefacts to the research/ subdirectory.

    Written only for v2 (ML) experiments.  Consumes the dict returned by
    _prepare_feature_diagnostics() — no refitting, no price reloading.

    Writes:
        research/feature_summary.json      — per-feature stats + alignment counts
        research/feature_registry.json     — structured feature + label metadata
        research/alignment_diagnostics.json — detailed sample construction report
        research/feature_correlations.json  — pairwise Pearson correlation matrix
    """
    research_dir = out_path / "research"
    research_dir.mkdir(exist_ok=True)

    # feature_summary.json
    if "feature_stats" in feature_data:
        alignment = feature_data.get("alignment") or {}
        payload: dict[str, Any] = {
            "n_rows_before_alignment": alignment.get("n_rows_raw"),
            "n_rows_after_alignment": alignment.get("n_rows_after_alignment"),
            "alignment_loss_pct": alignment.get("alignment_loss_pct"),
            "sample_start": alignment.get("sample_start"),
            "sample_end": alignment.get("sample_end"),
            "features": feature_data["feature_stats"],
            "pooled_panel_stats": feature_data.get("feature_stats_pooled", False),
        }
        with (research_dir / "feature_summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # feature_registry.json
    if "feature_registry" in feature_data:
        label_type = getattr(getattr(ml_spec, "labels", None), "type", None)
        label_params = getattr(getattr(ml_spec, "labels", None), "params", {}) or {}
        reg_payload: dict[str, Any] = {
            "ticker": feature_data.get("ticker"),
            "n_features": len(feature_data["feature_registry"]),
            "features": feature_data["feature_registry"],
            "label_type": label_type,
            "label_horizon": label_params.get("horizon"),
            "label_params": label_params,
        }
        with (research_dir / "feature_registry.json").open("w", encoding="utf-8") as f:
            json.dump(reg_payload, f, indent=2)

    # alignment_diagnostics.json
    if "alignment" in feature_data:
        al = dict(feature_data["alignment"])
        if "feature_stats" in feature_data:
            al["per_feature_first_valid_date"] = {
                name: stats.get("first_valid_date")
                for name, stats in feature_data["feature_stats"].items()
            }
        with (research_dir / "alignment_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(al, f, indent=2)

    # feature_correlations.json
    if "corr_df" in feature_data:
        corr = feature_data["corr_df"]
        corr_payload: dict[str, Any] = {
            "features": list(corr.columns),
            "n_features": len(corr.columns),
            "sample_size": (feature_data.get("alignment") or {}).get(
                "n_rows_after_alignment"
            ),
            "matrix": _sanitize_for_json(corr.values.tolist()),
        }
        with (research_dir / "feature_correlations.json").open("w", encoding="utf-8") as f:
            json.dump(corr_payload, f, indent=2)

    # feature_families.json
    if "feature_families" in feature_data:
        fam_payload: dict[str, Any] = {
            "families": feature_data["feature_families"],
            "n_families": len(feature_data["feature_families"]),
        }
        with (research_dir / "feature_families.json").open("w", encoding="utf-8") as f:
            json.dump(fam_payload, f, indent=2)


# ---------------------------------------------------------------------------
# ML diagnostics helpers (Phase A4 + Phase B)
# ---------------------------------------------------------------------------


def _extract_linear_coef(
    model_wrapper: Any,
    feature_names: list[str],
) -> dict[str, float] | None:
    """Extract coefficients from a linear sklearn model wrapper (e.g. RidgeRegressionModel).

    model_wrapper is strategy._model (the wrapper class), which holds the actual
    sklearn estimator at model_wrapper._model.  Returns None for models without
    a coef_ attribute (e.g. tree-based) or on any error.
    """
    try:
        sk = model_wrapper._model
        if not hasattr(sk, "coef_"):
            return None
        coef = sk.coef_
        if coef.ndim != 1 or len(coef) != len(feature_names):
            return None
        return {name: float(v) for name, v in zip(feature_names, coef, strict=False)}
    except Exception:
        return None


def _collect_wf_coefficients(
    prices: pd.DataFrame,
    wf: WalkForwardResult,
    ml_spec: Any,
    is_panel: bool = False,
    tickers: list[str] | None = None,
) -> pd.DataFrame | None:
    """Re-fit on each split's training window to collect per-split coefficients.

    Lightweight second pass: only fitting, no backtest.  Returns a
    (n_splits × n_features) DataFrame for use with coefficient_stability(),
    or None when coefficients are unavailable (non-linear models, errors).
    """
    from src.ml.feature_matrix import build_feature_matrix

    rows: dict[int, dict[str, float]] = {}

    for sr in wf.splits:
        split = sr.split
        try:
            train_prices = prices.loc[split.train_start : split.train_end]
            if is_panel and tickers:
                from src.experiments.ml_factory import build_panel_ml_strategy
                strategy_tmp = build_panel_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal,
                    tickers,
                    portfolio_construction=ml_spec.portfolio_construction,
                )
                strategy_tmp.fit(train_prices)
                # Build features for first ticker to get feature names
                feature_fns = strategy_tmp._feature_fn_builder(tickers[0])
                X = build_feature_matrix(train_prices, feature_fns)
            else:
                from src.experiments.ml_factory import build_ml_strategy
                strategy_tmp = build_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal
                )
                strategy_tmp.fit(train_prices)
                X = build_feature_matrix(train_prices, strategy_tmp._feature_fns)
            feature_names = list(X.columns)
            coef = _extract_linear_coef(strategy_tmp._model, feature_names)
            if coef is not None:
                rows[split.split_index] = coef
        except Exception:
            pass

    if not rows:
        return None
    df = pd.DataFrame(rows).T
    df.index.name = "split"
    return df


def _prepare_prediction_strength(
    prices: pd.DataFrame,
    ml_data: dict,
    horizon: int = 21,
) -> dict:
    """Prediction-strength bucket analysis: do stronger cross-sectional scores → stronger returns?

    Ranks assets by predicted score on each monthly rebalance date, assigns to
    top/mid/bottom thirds, then computes realized forward returns per group.
    Used for the prediction confidence & outcome monotonicity diagnostic.

    Returns dict with keys (in-memory DataFrame excluded from JSON persistence):
        group_mean_returns:  {"top": float, "mid": float, "bottom": float}
        group_monthly:       pd.DataFrame (Month × group) — plot only, not JSON
        ls_spread:           float — top mean return minus bottom mean return
        is_monotonic:        bool — top > mid > bottom?
        is_ordered:          bool — top > bottom?
        n_obs:               int
        n_assets_per_group:  int
        horizon:             int
    """
    import numpy as _np

    score_wide = ml_data.get("score_wide")
    if score_wide is None or score_wide.empty:
        return {}

    try:
        from src.ml.labels import forward_returns as _fwd_ret
    except Exception:
        return {}

    tickers = [t for t in score_wide.columns if t in prices.columns]
    if not tickers:
        return {}

    fwd_panel = pd.DataFrame({t: _fwd_ret(prices[t], horizon) for t in tickers})
    common_idx = score_wide.index.intersection(fwd_panel.index)
    if len(common_idx) < 12:
        return {}

    scores_aligned = score_wide.loc[common_idx, tickers]
    fwd_aligned = fwd_panel.loc[common_idx, tickers]

    # Monthly sampling: last date of each calendar month
    month_periods = scores_aligned.index.to_period("M")
    monthly_dates = [
        scores_aligned.index[month_periods == m][-1]
        for m in month_periods.unique()
    ]

    n_assets = len(tickers)
    n_per_group = max(1, n_assets // 3)

    top_rets: list[float] = []
    mid_rets: list[float] = []
    bot_rets: list[float] = []
    date_index: list = []

    for date in monthly_dates:
        scores = scores_aligned.loc[date].dropna()
        fwd = fwd_aligned.loc[date].dropna()
        common = scores.index.intersection(fwd.index)
        if len(common) < 3:
            continue
        ranked = scores[common].sort_values(ascending=False)
        n = len(ranked)
        per_g = max(1, n // 3)
        top_assets = ranked.index[:per_g]
        bot_assets = ranked.index[-per_g:]
        mid_assets = ranked.index[per_g : n - per_g]
        top_ret = float(fwd[top_assets].mean())
        bot_ret = float(fwd[bot_assets].mean())
        mid_ret = float(fwd[mid_assets].mean()) if len(mid_assets) > 0 else float("nan")
        if _np.isnan(top_ret) or _np.isnan(bot_ret):
            continue
        top_rets.append(top_ret)
        bot_rets.append(bot_ret)
        mid_rets.append(mid_ret)
        date_index.append(date)

    if not top_rets:
        return {}

    top_mean = float(_np.nanmean(top_rets))
    mid_mean = float(_np.nanmean(mid_rets))
    bot_mean = float(_np.nanmean(bot_rets))

    result: dict = {
        "group_mean_returns": {"top": top_mean, "mid": mid_mean, "bottom": bot_mean},
        "ls_spread": top_mean - bot_mean,
        "is_ordered": bool(top_mean > bot_mean),
        "is_monotonic": bool(top_mean > mid_mean > bot_mean),
        "n_obs": len(date_index),
        "n_assets_per_group": n_per_group,
        "horizon": horizon,
    }

    if date_index:
        result["group_monthly"] = pd.DataFrame(
            {"top": top_rets, "mid": mid_rets, "bottom": bot_rets},
            index=pd.DatetimeIndex(date_index),
        )

    return result


def _prepare_allocation_research_diagnostics(
    ml_data: dict,
    weights: pd.DataFrame | None,
    prices: pd.DataFrame,
    ml_spec: Any,
    horizon: int = 21,
) -> dict:
    """Allocation research diagnostics for panel ML experiments (Phase 2).

    Computes three diagnostic streams from existing in-memory data:
    1. Prediction dispersion — rolling cross-sectional std + top-minus-bottom
       spread from score_wide, revealing signal compression regimes.
    2. Confidence calibration — quintile mean forward returns from score_wide
       + prices, confirming whether score magnitude carries economic information.
    3. Concentration time-series — per-period HHI, breadth, entropy from
       weights; used by plot_concentration_evolution.

    All computation is strictly read-only on in-memory DataFrames; no model
    refitting, no data loading.

    Returns:
        Dict with in-memory DataFrames for plots and scalar summaries for
        JSON persistence. Empty dict if score_wide is unavailable.
    """
    import math as _math

    import numpy as _np

    score_wide = ml_data.get("score_wide")
    if score_wide is None or score_wide.empty:
        return {}

    result: dict = {}

    # ── 1. Prediction dispersion from score_wide ──────────────────────────────
    sw = score_wide.dropna(how="all")
    if len(sw) >= 5:
        cs_std = sw.std(axis=1)
        cs_spread = sw.max(axis=1) - sw.min(axis=1)
        result["score_wide"] = sw  # pass-through for plot_prediction_dispersion
        result["dispersion_summary"] = {
            "mean_cs_std": _sanitize_for_json(float(cs_std.mean())),
            "mean_cs_spread": _sanitize_for_json(float(cs_spread.mean())),
            "min_cs_std": _sanitize_for_json(float(cs_std.min())),
            "max_cs_std": _sanitize_for_json(float(cs_std.max())),
        }

    # ── 2. Confidence calibration — quintile forward returns ──────────────────
    tickers = [t for t in sw.columns if t in prices.columns]
    if tickers and len(sw) >= 12:
        try:
            from src.ml.labels import forward_returns as _fwd_ret

            fwd_panel = pd.DataFrame({t: _fwd_ret(prices[t], horizon) for t in tickers})
            common_idx = sw.index.intersection(fwd_panel.index)
            if len(common_idx) >= 12:
                scores_al = sw.loc[common_idx, tickers]
                fwd_al = fwd_panel.loc[common_idx, tickers]

                # Monthly sampling — last date per calendar month
                month_periods = scores_al.index.to_period("M")
                monthly_dates = [
                    scores_al.index[month_periods == m][-1]
                    for m in month_periods.unique()
                ]

                n_quintiles = 5
                quintile_buckets: list[list[float]] = [[] for _ in range(n_quintiles)]

                for date in monthly_dates:
                    s = scores_al.loc[date].dropna()
                    f = fwd_al.loc[date].dropna()
                    common = s.index.intersection(f.index)
                    if len(common) < n_quintiles:
                        continue
                    ranked = s[common].rank(pct=True)
                    for i in range(n_quintiles):
                        lo = i / n_quintiles
                        hi = (i + 1) / n_quintiles
                        bucket_assets = ranked[(ranked > lo) & (ranked <= hi)].index
                        if len(bucket_assets) == 0 and i == 0:
                            bucket_assets = ranked[ranked <= hi].index
                        if len(bucket_assets) > 0:
                            ret_val = float(f[bucket_assets].mean())
                            if not _np.isnan(ret_val):
                                quintile_buckets[i].append(ret_val)

                if all(len(b) > 0 for b in quintile_buckets):
                    q_means = [float(_np.mean(b)) for b in quintile_buckets]
                    q_counts = [len(b) for b in quintile_buckets]
                    q_labels = [f"Q{i+1}" for i in range(n_quintiles)]
                    quintile_returns = pd.Series(q_means, index=q_labels)
                    quintile_counts = pd.Series(q_counts, index=q_labels)
                    spread = q_means[-1] - q_means[0]
                    monotonic_up = all(
                        q_means[i] <= q_means[i + 1] for i in range(len(q_means) - 1)
                    )
                    result["calibration_data"] = {
                        "quintile_returns": quintile_returns,
                        "quintile_counts": quintile_counts,
                        "monotonic_up": monotonic_up,
                        "top_minus_bottom_spread": spread,
                    }
                    result["calibration_summary"] = {
                        "quintile_mean_returns": {q_labels[i]: _sanitize_for_json(q_means[i])
                                                  for i in range(n_quintiles)},
                        "top_minus_bottom_spread": _sanitize_for_json(spread),
                        "monotonic_up": monotonic_up,
                    }
        except Exception:
            pass

    # ── 3. Concentration time-series from weights ──────────────────────────────
    if weights is not None and not weights.empty:
        w = weights.fillna(0.0)
        abs_w = w.abs()
        active_mask = (abs_w > 1e-10).any(axis=1)
        if active_mask.any():
            abs_active = abs_w.loc[active_mask]

            def _row_eff_n(row: pd.Series) -> float:
                pos = row[row > 1e-10]
                if pos.empty:
                    return float("nan")
                total = pos.sum()
                if total <= 0:
                    return float("nan")
                p = pos / total
                h = float(-(p * p.apply(_math.log)).sum())
                return float(_math.exp(h))

            hhi_series = (abs_active ** 2).sum(axis=1)
            eff_breadth = (1.0 / hhi_series.replace(0.0, float("nan"))).fillna(0.0)
            eff_n = abs_active.apply(_row_eff_n, axis=1)

            result["weights"] = w  # pass-through for plot_concentration_evolution
            result["concentration_summary"] = {
                "mean_hhi": _sanitize_for_json(float(hhi_series.mean())),
                "mean_eff_breadth": _sanitize_for_json(float(eff_breadth.mean())),
                "mean_eff_n": _sanitize_for_json(float(eff_n.dropna().mean()))
                if not eff_n.dropna().empty else None,
            }

    return result


def _prepare_feature_contributions(
    ml_data: dict,
    smooth_window: int = 21,
    family_window: int = 63,
) -> dict:
    """Temporal feature contribution diagnostics (Phase II, C1–C3).

    Realised predictive influence: contribution(t, feature) ≈ coef[feature] × z(feature(t)).
    For panel mode, the reference-ticker feature matrix is used as a temporal proxy
    (the panel model shares one coefficient set across all assets).

    Keys populated (time-series, in-memory only):
        contribution_df     Date × Feature — 21d rolling mean signed contribution
        family_contrib_df   Date × Family  — 63d rolling mean signed family sum
        family_share_df     Date × Family  — 63d rolling normalised absolute share

    Scalar summaries (persisted to JSON):
        dominant_family            str   — family with most leadership periods
        dominant_family_pct        float — fraction of dates that family leads
        n_family_transitions       int   — number of family leadership changes
        mean_hhi                   float — mean rolling Herfindahl concentration
        contribution_volatility    dict  — per-feature contribution std
        most_volatile_feature      str   — feature with highest contribution std
    """

    result: dict = {}

    feature_matrix = ml_data.get("feature_matrix")
    coefficients = ml_data.get("coefficients")
    feature_families = ml_data.get("feature_families")

    if feature_matrix is None or not isinstance(feature_matrix, pd.DataFrame) or feature_matrix.empty:
        return result
    if not coefficients:
        return result

    # Align features present in both matrix and coefficients
    feature_names = [c for c in feature_matrix.columns if c in coefficients]
    if len(feature_names) < 2:
        return result

    X = feature_matrix[feature_names].dropna(how="all")
    if X.empty:
        return result

    # Standardise features column-wise (fill with column mean where NaN)
    x_mean = X.mean()
    x_std = X.std().replace(0, 1.0)
    z = (X - x_mean) / x_std

    # Coefficient vector aligned to feature_names
    coef_vec = pd.Series({fn: float(coefficients[fn]) for fn in feature_names})

    # Raw daily contribution
    contrib_raw = z.multiply(coef_vec, axis=1)

    # Smoothed daily contribution (C1 heatmap data)
    contrib_smooth = (
        contrib_raw.rolling(smooth_window, min_periods=smooth_window // 3)
        .mean()
        .dropna(how="all")
    )
    if len(contrib_smooth) >= 30:
        result["contribution_df"] = contrib_smooth

    # Per-feature contribution volatility (C3)
    try:
        vol = contrib_raw.std()
        result["contribution_volatility"] = {fn: float(v) for fn, v in vol.items()}
        if not vol.empty:
            result["most_volatile_feature"] = str(vol.idxmax())
    except Exception:
        pass

    # Family aggregation
    if feature_families:
        try:
            fam_cols: dict[str, pd.Series] = {}
            for fam, members in feature_families.items():
                present = [m for m in members if m in feature_names]
                if present:
                    fam_cols[fam] = contrib_raw[present].sum(axis=1)

            if len(fam_cols) >= 2:
                fam_raw = pd.DataFrame(fam_cols)

                # Signed rolling mean (C2 top panel)
                fam_smooth = (
                    fam_raw.rolling(family_window, min_periods=family_window // 3)
                    .mean()
                    .dropna(how="all")
                )
                if len(fam_smooth) >= 30:
                    result["family_contrib_df"] = fam_smooth

                # Normalised absolute share (C2 bottom panel)
                abs_smooth = (
                    fam_raw.abs()
                    .rolling(family_window, min_periods=family_window // 3)
                    .mean()
                    .dropna(how="all")
                )
                row_sum = abs_smooth.sum(axis=1).replace(0, 1.0)
                share_df = abs_smooth.div(row_sum, axis=0)
                if len(share_df) >= 30:
                    result["family_share_df"] = share_df

                # C3 scalars: dominant family, transitions, concentration
                if len(share_df) >= 30:
                    dominant_per_day = share_df.idxmax(axis=1)
                    dom = dominant_per_day.value_counts()
                    if not dom.empty:
                        dom_family = str(dom.idxmax())
                        result["dominant_family"] = dom_family
                        result["dominant_family_pct"] = float(
                            (dominant_per_day == dom_family).mean()
                        )

                    # Count leadership changes (day-over-day)
                    transitions = int(
                        max(0, (dominant_per_day != dominant_per_day.shift()).sum() - 1)
                    )
                    result["n_family_transitions"] = transitions

                    # Herfindahl concentration (rolling)
                    hhi = (share_df ** 2).sum(axis=1)
                    result["mean_hhi"] = float(hhi.mean())
        except Exception:
            pass

    return result


def _prepare_ranking_geometry(
    prices: pd.DataFrame,
    ml_data: dict,
    horizon: int = 21,
    window: int = 63,
) -> dict:
    """Lightweight cross-sectional ranking-geometry diagnostics (Phase I, S1–S5).

    Computes five complementary diagnostics from score_wide and cs_ic_daily.
    All operations are best-effort; failures leave the key absent.

    Keys populated (time-series, in-memory only):
        rolling_score_iqr        pd.Series — rolling cross-sectional IQR of scores (S1)
        rolling_ic_std           pd.Series — rolling std of daily cross-sectional IC (S5)
        rolling_score_spread     pd.Series — rolling top-group minus bottom-group score (S2)
        rolling_realized_spread  pd.Series — rolling realized return spread top vs bottom (S3)
        rank_persistence         pd.Series — monthly Spearman rank autocorrelation (S4)

    Scalar summaries (persisted to JSON):
        mean_score_iqr           float
        min_score_iqr            float
        mean_score_spread        float
        mean_realized_spread     float
        pct_positive_realized    float
        mean_rank_persistence    float
        pct_positive_persistence float
    """
    import numpy as _np

    result: dict = {}

    score_wide = ml_data.get("score_wide")
    if score_wide is None or score_wide.empty or score_wide.shape[1] < 2:
        return result

    sw = score_wide.dropna(how="all")
    if sw.empty:
        return result

    # S1: Rolling cross-sectional IQR of predicted scores
    try:
        cross_iqr = sw.quantile(0.75, axis=1) - sw.quantile(0.25, axis=1)
        rolling_iqr = cross_iqr.rolling(window, min_periods=window // 3).mean().dropna()
        if len(rolling_iqr) >= 21:
            result["rolling_score_iqr"] = rolling_iqr
            result["mean_score_iqr"] = float(rolling_iqr.mean())
            result["min_score_iqr"] = float(rolling_iqr.min())
    except Exception:
        pass

    # S5: Rolling std of daily cross-sectional IC
    try:
        cs_ic = ml_data.get("cs_ic_daily")
        if cs_ic is not None and len(cs_ic) >= 21:
            rolling_ic_std = cs_ic.rolling(window, min_periods=window // 3).std().dropna()
            if len(rolling_ic_std) >= 21:
                result["rolling_ic_std"] = rolling_ic_std
    except Exception:
        pass

    # S2: Rolling top-vs-bottom score spread
    try:
        n_assets = sw.shape[1]
        n_top = max(1, n_assets // 3)

        def _top_bot_spread(row: pd.Series) -> float:
            vals = row.dropna().sort_values(ascending=False)
            if len(vals) < 2 * n_top:
                return float("nan")
            return float(vals.iloc[:n_top].mean() - vals.iloc[-n_top:].mean())

        cross_spread = sw.apply(_top_bot_spread, axis=1)
        rolling_spread = cross_spread.rolling(window, min_periods=window // 3).mean().dropna()
        if len(rolling_spread) >= 21:
            result["rolling_score_spread"] = rolling_spread
            result["mean_score_spread"] = float(rolling_spread.mean())
    except Exception:
        pass

    # S3: Rolling realized top-vs-bottom forward-return spread
    try:
        from src.ml.labels import forward_returns as _fwd_ret

        tickers = [t for t in sw.columns if t in prices.columns]
        if tickers:
            fwd_panel = pd.DataFrame({t: _fwd_ret(prices[t], horizon) for t in tickers})
            common_idx = sw.index.intersection(fwd_panel.index)
            if len(common_idx) >= 21:
                scores_al = sw.loc[common_idx, tickers]
                fwd_al = fwd_panel.loc[common_idx, tickers]
                max(1, len(tickers) // 3)

                def _realized_spread(i: int) -> float:
                    row_s = scores_al.iloc[i].dropna().sort_values(ascending=False)
                    n = max(1, len(row_s) // 3)
                    top_ret = float(fwd_al.iloc[i][row_s.index[:n]].mean())
                    bot_ret = float(fwd_al.iloc[i][row_s.index[-n:]].mean())
                    if _np.isnan(top_ret) or _np.isnan(bot_ret):
                        return float("nan")
                    return top_ret - bot_ret

                raw_spread = pd.Series(
                    [_realized_spread(i) for i in range(len(common_idx))],
                    index=common_idx,
                ).dropna()
                rolling_real = raw_spread.rolling(window, min_periods=window // 3).mean().dropna()
                if len(rolling_real) >= 21:
                    result["rolling_realized_spread"] = rolling_real
                    result["mean_realized_spread"] = float(rolling_real.mean())
                    result["pct_positive_realized"] = float((rolling_real > 0).mean())
    except Exception:
        pass

    # S4: Monthly Spearman rank autocorrelation of asset score vectors
    try:
        # Identify last date in each calendar month where sw has data
        monthly_dates: list = []
        for _, grp_idx in sw.groupby(sw.index.to_period("M")).groups.items():
            if len(grp_idx) > 0:
                monthly_dates.append(max(grp_idx))
        monthly_dates = sorted(monthly_dates)

        rank_corrs: list[float] = []
        rank_dates: list = []
        for i in range(1, len(monthly_dates)):
            prev_row = sw.loc[monthly_dates[i - 1]].dropna()
            curr_row = sw.loc[monthly_dates[i]].dropna()
            common = prev_row.index.intersection(curr_row.index)
            if len(common) < 4:
                continue
            prev_ranks = prev_row[common].rank()
            curr_ranks = curr_row[common].rank()
            corr = float(prev_ranks.corr(curr_ranks))
            if not _np.isnan(corr):
                rank_corrs.append(corr)
                rank_dates.append(monthly_dates[i])

        if len(rank_corrs) >= 10:
            rank_pers = pd.Series(
                rank_corrs,
                index=pd.DatetimeIndex(rank_dates),
            )
            result["rank_persistence"] = rank_pers
            result["mean_rank_persistence"] = float(rank_pers.mean())
            result["pct_positive_persistence"] = float((rank_pers > 0).mean())
    except Exception:
        pass

    return result


def _collect_wf_feature_ic(
    prices: pd.DataFrame,
    wf: WalkForwardResult,
    ml_spec: Any,
    is_panel: bool = False,
    tickers: list[str] | None = None,
) -> pd.DataFrame | None:
    """Compute per-feature IC against test-period labels for each split.

    Single-asset mode: Pearson IC between each feature and forward returns.
    Panel mode: time-averaged cross-sectional Spearman IC per feature.

    Returns a (n_splits × n_features) DataFrame, or None when insufficient data.
    """
    from src.ml.feature_matrix import build_feature_matrix

    rows: dict[int, dict[str, float]] = {}

    for sr in wf.splits:
        split = sr.split
        try:
            train_prices = prices.loc[split.train_start : split.train_end]
            test_prices = prices.loc[split.test_start : split.test_end]

            if is_panel and tickers:
                from src.experiments.ml_factory import build_panel_ml_strategy
                from src.ml.labels import ranking_target
                from src.ml.panel import build_panel_feature_matrix

                strategy_tmp = build_panel_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal,
                    tickers,
                    portfolio_construction=ml_spec.portfolio_construction,
                )
                strategy_tmp.fit(train_prices)

                horizon = int(ml_spec.labels.params.get("horizon", 21))
                avail_tickers = [t for t in tickers if t in prices.columns]

                # Build panel features using FULL history (not just test window)
                # so long-lookback features (e.g. mom_252) have valid warmup values.
                # Restrict evaluation to test-window dates after building.
                test_dates = test_prices.index
                X_panel_full = build_panel_feature_matrix(
                    prices, strategy_tmp._feature_fn_builder, avail_tickers
                )
                # Filter MultiIndex to test-window date level
                X_panel_test = X_panel_full.loc[
                    X_panel_full.index.get_level_values("date").isin(test_dates)
                ].dropna()

                # Actual cross-sectional return ranks for test window
                actual_ranks = ranking_target(test_prices[avail_tickers], horizon)
                y_stacked = actual_ranks.stack(future_stack=True)
                y_stacked.index.names = ["date", "asset"]

                common_idx = X_panel_test.index.intersection(y_stacked.dropna().index)
                if len(common_idx) >= 5:
                    X_al = X_panel_test.loc[common_idx]
                    y_al = y_stacked.loc[common_idx]
                    rows_feat: dict[str, float] = {}
                    for col in X_al.columns:
                        ic = float(X_al[col].corr(y_al))
                        if not math.isnan(ic):
                            rows_feat[col] = ic
                    if rows_feat:
                        rows[split.split_index] = rows_feat
            else:
                from src.experiments.ml_factory import build_ml_strategy
                strategy_tmp = build_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal
                )
                strategy_tmp.fit(train_prices)
                # Build features using full history so long-lookback features
                # have valid warmup values; restrict to test-window dates.
                X_full = build_feature_matrix(prices, strategy_tmp._feature_fns)
                X_test = X_full.loc[
                    X_full.index.isin(test_prices.index)
                ].dropna()
                labels = strategy_tmp._label_fn(test_prices)
                if isinstance(labels, pd.Series):
                    idx = X_test.index.intersection(labels.dropna().index)
                    if len(idx) >= 5:
                        X_al = X_test.loc[idx]
                        y_al = labels.loc[idx]
                        rows[split.split_index] = {
                            col: float(X_al[col].corr(y_al)) for col in X_al.columns
                        }
        except Exception:
            pass

    if not rows:
        return None
    df = pd.DataFrame(rows).T
    df.index.name = "split"
    return df


def _prepare_ml_diagnostics(
    strategy: Any,
    prices: pd.DataFrame,
    wf: WalkForwardResult | None,
    ml_spec: Any,
    is_panel: bool = False,
) -> dict:
    """Compute ML diagnostics from the already-fitted full-period strategy.

    Returns a dict consumed by both _build_plots() and _write_ml_model_diagnostics_json().
    All operations are best-effort: failures leave the key absent from the dict.

    Keys populated:
        feature_names          list[str]
        coefficients           dict[feature→float]
        predictions            pd.Series (aligned to X_clean index)
        directional_accuracy   float
        ic_series              pd.Series (monthly rolling Pearson corr, IC proxy)
        prediction_stats       dict with mean/std/min/max/pct_positive
        n_aligned_samples      int
        signal_turnover        pd.Series (per-period turnover from strategy weights)
        coeff_stability_df     pd.DataFrame (n_splits × n_features)
        coeff_stability        pd.DataFrame (feature stats from coefficient_stability())
    """
    import numpy as np

    from src.ml.feature_matrix import build_feature_matrix

    result: dict = {}

    # Panel mode: cross-sectional diagnostics using pooled panel predictions
    if is_panel and hasattr(strategy, "_feature_fn_builder") and strategy._tickers:
        try:
            from src.ml.labels import forward_returns as _fwd_ret
            from src.ml.panel import build_panel_feature_matrix, compute_cross_sectional_ic

            tickers_avail = [t for t in strategy._tickers if t in prices.columns]
            X_panel = build_panel_feature_matrix(
                prices, strategy._feature_fn_builder, tickers_avail
            )
            X_panel_clean = X_panel.dropna()

            if not X_panel_clean.empty:
                # Panel coefficients — linear models have one coefficient per feature
                ref_ticker = tickers_avail[0]
                feature_fns_ref = strategy._feature_fn_builder(ref_ticker)
                X_ref = build_feature_matrix(prices, feature_fns_ref)
                feature_names = list(X_ref.columns)
                coef = _extract_linear_coef(strategy._model, feature_names)
                if coef is not None:
                    result["coefficients"] = coef
                    result["feature_names"] = feature_names

                preds = strategy._model.predict(X_panel_clean)
                pred_series = preds.values  # pd.Series with MultiIndex(date, asset)
                result["predictions"] = pred_series if isinstance(pred_series, pd.Series) else None

                # Reshape to Date × Asset for cross-sectional IC and score dispersion
                try:
                    pred_df = pred_series.unstack(level="asset")
                    result["score_wide"] = pred_df
                except Exception:
                    pred_df = None

                horizon = getattr(strategy, "_horizon", 21)
                _fwd_ret(prices[tickers_avail[0]], horizon)  # for single-asset ref
                # Full panel actual returns
                actual_panel = pd.DataFrame({
                    t: _fwd_ret(prices[t], horizon)
                    for t in tickers_avail
                })

                if pred_df is not None and not actual_panel.empty:
                    cs_ic = compute_cross_sectional_ic(pred_df, actual_panel)
                    if len(cs_ic) >= 10:
                        result["cs_ic_daily"] = cs_ic
                        result["ic_daily_rolling"] = (
                            cs_ic.rolling(63, min_periods=21).mean().dropna()
                        )
                        monthly_ic = cs_ic.resample("ME").mean().dropna()
                        if len(monthly_ic) >= 2:
                            result["ic_series"] = monthly_ic

                        # Rolling cross-sectional IC (63d)
                        if len(cs_ic) >= 21:
                            result["ic_daily_rolling"] = cs_ic.rolling(
                                63, min_periods=21
                            ).mean().dropna()

                    # Directional accuracy: fraction of positive CS-IC days
                    if len(cs_ic) >= 10:
                        result["directional_accuracy"] = float((cs_ic > 0).mean())

                    # Rolling IC consistency: 126d rolling fraction of positive-IC days
                    # Analogous to rolling directional accuracy in single-asset mode.
                    # Baseline is 0.5 (random cross-sectional ranking).
                    if len(cs_ic) >= 63:
                        result["rolling_da"] = (
                            cs_ic.rolling(126, min_periods=63)
                            .apply(lambda x: float((x > 0).mean()))
                            .dropna()
                        )

                result["n_aligned_samples"] = len(X_panel_clean)
        except Exception:
            pass

        # Per-split WF diagnostics in panel mode
        if wf is not None and wf.n_splits > 0:
            try:
                coeff_df = _collect_wf_coefficients(
                    prices, wf, ml_spec, is_panel=True, tickers=strategy._tickers
                )
                if coeff_df is not None and not coeff_df.empty:
                    result["coeff_stability_df"] = coeff_df
                    from src.ml.diagnostics.stability import coefficient_stability
                    stab = coefficient_stability(coeff_df)
                    if not stab.empty:
                        result["coeff_stability"] = stab
            except Exception:
                pass

            try:
                feature_ic_df = _collect_wf_feature_ic(
                    prices, wf, ml_spec, is_panel=True, tickers=strategy._tickers
                )
                if feature_ic_df is not None and not feature_ic_df.empty:
                    result["feature_ic_splits"] = feature_ic_df
            except Exception:
                pass

        return result

    try:
        X = build_feature_matrix(prices, strategy._feature_fns)
        feature_names = list(X.columns)
        X_clean = X.dropna()

        if not X_clean.empty:
            # Coefficients from fitted model
            coef = _extract_linear_coef(strategy._model, feature_names)
            if coef is not None:
                result["coefficients"] = coef
                result["feature_names"] = feature_names

            # Full-period predictions
            preds = strategy._model.predict(X_clean)
            pred_series: pd.Series = preds.values  # type: ignore[assignment]
            result["predictions"] = pred_series

            # Align with labels (forward returns)
            labels = strategy._label_fn(prices)
            if isinstance(labels, pd.Series):
                aligned = pd.DataFrame({
                    "actual": labels,
                    "predicted": pred_series,
                }).dropna()

                if len(aligned) >= 10:
                    result["n_aligned_samples"] = len(aligned)
                    # Expose aligned series for prediction_vs_actual plot (D6)
                    result["actual"] = aligned["actual"]
                    result["predictions_aligned"] = aligned["predicted"]

                    # Directional accuracy
                    valid_mask = (aligned["actual"] != 0) & (aligned["predicted"] != 0)
                    if valid_mask.sum() > 0:
                        result["directional_accuracy"] = float((
                            np.sign(aligned.loc[valid_mask, "actual"].to_numpy()) ==
                            np.sign(aligned.loc[valid_mask, "predicted"].to_numpy())
                        ).mean())

                    # Monthly IC proxy: rolling 21d Pearson corr resampled monthly
                    roll_corr = (
                        aligned["predicted"]
                        .rolling(21, min_periods=10)
                        .corr(aligned["actual"])
                    )
                    monthly_ic = roll_corr.resample("ME").last().dropna()
                    if len(monthly_ic) >= 2:
                        result["ic_series"] = monthly_ic

                    # Continuous 63-day rolling IC for regime visibility
                    ic_daily = (
                        aligned["predicted"]
                        .rolling(63, min_periods=21)
                        .corr(aligned["actual"])
                        .dropna()
                    )
                    if len(ic_daily) >= 21:
                        result["ic_daily_rolling"] = ic_daily

                    # 126-day rolling directional accuracy
                    try:
                        from src.ml.diagnostics.prediction import (
                            rolling_directional_accuracy,
                        )
                        rolling_da = rolling_directional_accuracy(
                            aligned["actual"], aligned["predicted"], window=126
                        ).dropna()
                        if len(rolling_da) >= 21:
                            result["rolling_da"] = rolling_da
                    except Exception:
                        pass

                    # Prediction confidence calibration (quintile analysis)
                    try:
                        from src.ml.diagnostics.prediction import prediction_quantiles
                        q_labels = prediction_quantiles(
                            aligned["predicted"], n_quantiles=5
                        )
                        aligned_cal = pd.DataFrame({
                            "actual": aligned["actual"],
                            "predicted": aligned["predicted"],
                            "quintile": q_labels,
                        }).dropna()
                        calibration: list[dict] = []
                        for q in sorted(aligned_cal["quintile"].dropna().unique()):
                            q_int = int(q)
                            sub = aligned_cal[aligned_cal["quintile"] == q]
                            if len(sub) < 5:
                                continue
                            valid = (sub["actual"] != 0) & (sub["predicted"] != 0)
                            da_q = float((
                                np.sign(sub.loc[valid, "predicted"].to_numpy()) ==
                                np.sign(sub.loc[valid, "actual"].to_numpy())
                            ).mean()) if valid.sum() > 0 else float("nan")
                            calibration.append({
                                "quintile": q_int,
                                "n_obs": int(len(sub)),
                                "mean_predicted": _sanitize_for_json(
                                    float(sub["predicted"].mean())
                                ),
                                "mean_actual": _sanitize_for_json(
                                    float(sub["actual"].mean())
                                ),
                                "directional_accuracy": _sanitize_for_json(da_q),
                            })
                        if len(calibration) >= 3:
                            result["calibration_by_quintile"] = calibration
                    except Exception:
                        pass

                    # Volatility-regime-conditioned performance
                    try:
                        vol = aligned["actual"].rolling(63, min_periods=21).std()
                        vol_clean = vol.dropna()
                        if len(vol_clean) >= 9:
                            low_thr = float(vol_clean.quantile(1 / 3))
                            high_thr = float(vol_clean.quantile(2 / 3))
                            regime_definitions = [
                                ("Low", vol <= low_thr),
                                ("Medium", (vol > low_thr) & (vol <= high_thr)),
                                ("High", vol > high_thr),
                            ]
                            regime_stats: list[dict] = []
                            for regime_label, vol_mask in regime_definitions:
                                sub_r = aligned[vol_mask.reindex(
                                    aligned.index, fill_value=False
                                )].dropna()
                                if len(sub_r) < 10:
                                    continue
                                ic_r = float(
                                    sub_r["predicted"].corr(sub_r["actual"])
                                )
                                valid_r = (
                                    (sub_r["actual"] != 0) &
                                    (sub_r["predicted"] != 0)
                                )
                                da_r = float((
                                    np.sign(sub_r.loc[valid_r, "predicted"].to_numpy()) ==
                                    np.sign(sub_r.loc[valid_r, "actual"].to_numpy())
                                ).mean()) if valid_r.sum() > 0 else float("nan")
                                ret_std = float(sub_r["actual"].std())
                                sharpe_r = (
                                    float(sub_r["actual"].mean()) * 252
                                    / (ret_std * np.sqrt(252))
                                ) if ret_std > 1e-12 else float("nan")
                                regime_stats.append({
                                    "regime": regime_label,
                                    "n_obs": int(len(sub_r)),
                                    "ic": _sanitize_for_json(ic_r),
                                    "directional_accuracy": _sanitize_for_json(da_r),
                                    "realized_sharpe": _sanitize_for_json(sharpe_r),
                                })
                            if len(regime_stats) >= 2:
                                result["regime_conditioned_stats"] = regime_stats
                    except Exception:
                        pass

                    # Prediction statistics
                    p_clean = aligned["predicted"].dropna()
                    result["prediction_stats"] = {
                        "mean": _sanitize_for_json(float(p_clean.mean())),
                        "std": _sanitize_for_json(float(p_clean.std())),
                        "min": _sanitize_for_json(float(p_clean.min())),
                        "max": _sanitize_for_json(float(p_clean.max())),
                        "pct_positive": float((p_clean > 0).mean()),
                    }
    except Exception:
        pass

    # Signal turnover from strategy weights (no re-computation needed)
    try:
        from src.ml.diagnostics.turnover import signal_turnover as _signal_turnover
        to_series = _signal_turnover(strategy._signal_fn(
            strategy._model.predict(
                build_feature_matrix(prices, strategy._feature_fns).dropna()
            )
        )).dropna()
        if len(to_series) >= 2:
            result["signal_turnover"] = to_series
    except Exception:
        pass

    # Per-split coefficient stability (re-fit only, no backtest)
    if wf is not None and wf.n_splits > 0:
        try:
            coeff_df = _collect_wf_coefficients(prices, wf, ml_spec)
            if coeff_df is not None and not coeff_df.empty:
                result["coeff_stability_df"] = coeff_df
                from src.ml.diagnostics.stability import coefficient_stability
                stab = coefficient_stability(coeff_df)
                if not stab.empty:
                    result["coeff_stability"] = stab
        except Exception:
            pass

        try:
            feature_ic_df = _collect_wf_feature_ic(prices, wf, ml_spec)
            if feature_ic_df is not None and not feature_ic_df.empty:
                result["feature_ic_splits"] = feature_ic_df
        except Exception:
            pass

    return result


def _write_ml_model_diagnostics_json(
    ml_data: dict,
    ml_spec: Any,
    out_path: Path,
) -> None:
    """Persist diagnostics/ml_model_diagnostics.json from pre-computed ml_data.

    Consumes the dict returned by _prepare_ml_diagnostics() — no refitting.
    """
    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    payload: dict[str, Any] = {
        "model_type": ml_spec.model.type,
        "hyperparams": ml_spec.model.params or {},
    }

    for key in ("feature_names", "coefficients", "directional_accuracy",
                "prediction_stats", "n_aligned_samples"):
        if key in ml_data:
            payload[key] = ml_data[key]

    if "ic_series" in ml_data:
        ic = ml_data["ic_series"]
        payload["monthly_ic_series"] = [
            {"date": str(d.date()), "value": _sanitize_for_json(float(v))}
            for d, v in ic.items()
        ]
        ic_clean = ic.dropna()
        if len(ic_clean) > 0:
            payload["ic_summary"] = {
                "mean_ic": _sanitize_for_json(float(ic_clean.mean())),
                "std_ic": _sanitize_for_json(float(ic_clean.std())),
                "pct_positive_ic": float((ic_clean > 0).mean()),
            }

    if "coeff_stability_df" in ml_data:
        coeff_df = ml_data["coeff_stability_df"]
        coeff_evolution: dict[str, dict[str, Any]] = {}
        for idx in coeff_df.index:
            row = coeff_df.loc[idx]
            coeff_evolution[str(idx)] = _sanitize_for_json(row.to_dict())
        payload["coefficient_evolution"] = coeff_evolution

    if "coeff_stability" in ml_data:
        stab = ml_data["coeff_stability"]
        payload["coefficient_stability_summary"] = _sanitize_for_json(
            stab.reset_index().to_dict("records")
        )

    for key in ("calibration_by_quintile", "regime_conditioned_stats"):
        if key in ml_data:
            payload[key] = _sanitize_for_json(ml_data[key])

    # Prediction-strength bucket analysis (Step 3 — confidence observability)
    if "prediction_strength" in ml_data:
        ps = ml_data["prediction_strength"]
        payload["prediction_strength"] = _sanitize_for_json({
            "group_mean_returns": ps.get("group_mean_returns"),
            "ls_spread": ps.get("ls_spread"),
            "is_ordered": ps.get("is_ordered"),
            "is_monotonic": ps.get("is_monotonic"),
            "n_obs": ps.get("n_obs"),
            "n_assets_per_group": ps.get("n_assets_per_group"),
            "horizon": ps.get("horizon"),
        })

    # Vol-regime conditional IC stats (Step 2 — regime interpretation)
    if "regime_stats" in ml_data:
        rs = ml_data["regime_stats"]
        payload["regime_stats"] = _sanitize_for_json({
            "high_vol_frac": rs.get("high_vol_frac"),
            "n_high_vol_splits": rs.get("n_high_vol_splits"),
            "n_low_vol_splits": rs.get("n_low_vol_splits"),
            "family_ic_by_regime": rs.get("family_ic_by_regime"),
            "dominant_family": rs.get("dominant_family"),
        })

    if "feature_ic_splits" in ml_data:
        feat_ic_df = ml_data["feature_ic_splits"]
        feat_ic_records: dict[str, dict[str, Any]] = {}
        for idx in feat_ic_df.index:
            feat_ic_records[str(idx)] = _sanitize_for_json(feat_ic_df.loc[idx].to_dict())
        payload["feature_ic_by_split"] = feat_ic_records

    # Ranking geometry scalar summaries (Phase I — time-series excluded from JSON)
    if "ranking_geometry" in ml_data:
        rg = ml_data["ranking_geometry"]
        _rg_scalars = {
            k: rg[k]
            for k in (
                "mean_score_iqr", "min_score_iqr",
                "mean_score_spread",
                "mean_realized_spread", "pct_positive_realized",
                "mean_rank_persistence", "pct_positive_persistence",
            )
            if k in rg
        }
        if _rg_scalars:
            payload["ranking_geometry"] = _sanitize_for_json(_rg_scalars)

    # Feature contribution scalar summaries (Phase II — time-series excluded from JSON)
    if "feature_contributions" in ml_data:
        fc = ml_data["feature_contributions"]
        _fc_scalars: dict[str, Any] = {}
        for k in (
            "dominant_family", "dominant_family_pct",
            "n_family_transitions", "mean_hhi",
            "most_volatile_feature",
        ):
            if k in fc:
                _fc_scalars[k] = fc[k]
        if "contribution_volatility" in fc:
            _fc_scalars["contribution_volatility"] = fc["contribution_volatility"]
        if _fc_scalars:
            payload["feature_contributions"] = _sanitize_for_json(_fc_scalars)

    with (diag_dir / "ml_model_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_wf_equity_curves(wf: WalkForwardResult, out_path: Path) -> None:
    """Persist per-split equity curves to diagnostics/wf_equity_curves.json.

    Equity curves are monthly-subsampled for compact storage.
    Written for any experiment where walk-forward validation ran.
    """
    diag_dir = out_path / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    splits_data: list[dict[str, Any]] = []
    for sr in wf.splits:
        monthly_eq = sr.equity_curve.resample("ME").last().dropna()
        splits_data.append({
            "split": sr.split.split_index,
            "test_start": str(sr.split.test_start.date()),
            "test_end": str(sr.split.test_end.date()),
            "equity_curve": [
                {"date": str(d.date()), "value": float(v)}
                for d, v in monthly_eq.items()
            ],
        })

    payload: dict[str, Any] = {
        "n_splits": wf.n_splits,
        "splits": splits_data,
    }
    with (diag_dir / "wf_equity_curves.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# F3: ML experiment pipeline (version "2" configs)
# ---------------------------------------------------------------------------


def _write_ml_provenance(
    ml_spec: Any,
    out_path: Path,
) -> None:
    """Write ml_provenance.json to the experiment output directory."""
    from src.experiments.ml_config import ml_experiment_hash

    provenance = {
        "spec_version": "2",
        "name": ml_spec.name,
        "ml_hash": ml_experiment_hash(ml_spec),
        "features": ml_spec.features.to_dict(),
        "labels": ml_spec.labels.to_dict(),
        "model": ml_spec.model.to_dict(),
        "signal": ml_spec.signal.to_dict(),
        "portfolio_construction": ml_spec.portfolio_construction.to_dict(),
    }
    with (out_path / "ml_provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


def _run_ml_experiment(
    raw_cfg: dict[str, Any],
    source_path: Path,
    profile: str = "report",
) -> ExperimentRun:
    """Execute a version "2" ML experiment config.

    Called by run_experiment_from_config() when version == "2".
    Two model instances are created when walk-forward validation is requested
    to prevent state contamination from the full-period run.

    Args:
        raw_cfg: Raw config dict (already loaded, version == "2").
        source_path: Resolved path to the config file (for dual-config artefacts).

    Returns:
        ExperimentRun identical in structure to the D1 pipeline output.
    """
    import matplotlib

    from src.experiments.ml_config import (
        PANEL_SIGNAL_TYPES,
        build_ml_experiment_spec,
        normalize_ml_config,
        validate_ml_config,
    )
    from src.experiments.ml_factory import build_ml_strategy, build_panel_ml_strategy
    matplotlib.use("Agg")

    # 1-3. Validate and normalize
    validate_ml_config(raw_cfg)
    norm_cfg = normalize_ml_config(raw_cfg)

    # 4. Build ML objects (pure, no I/O)
    ml_spec = build_ml_experiment_spec(norm_cfg)
    uni_spec = build_universe_spec(norm_cfg["universe"], norm_cfg["date_range"])
    is_panel = ml_spec.signal.type in PANEL_SIGNAL_TYPES

    if is_panel:
        strategy = build_panel_ml_strategy(
            ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal,
            list(uni_spec.tickers),
            portfolio_construction=ml_spec.portfolio_construction,
        )
    else:
        strategy = build_ml_strategy(
            ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal
        )
    val_config = build_validation_config(norm_cfg["validation"])

    # Build ExperimentSpec for registry compatibility
    spec = ExperimentSpec(
        experiment_name=ml_spec.name,
        strategy_name=strategy.name,
        universe=ml_spec.universe,
        start_date=ml_spec.start_date,
        end_date=ml_spec.end_date,
        rebalance_frequency="D",
        parameters=ml_spec.model.params,
        tags=ml_spec.tags,
        description=ml_spec.description,
    )

    # 5. Data loading
    universe = load_universe(list(uni_spec.tickers))
    prices = align_prices(universe)
    prices = prices.loc[uni_spec.start_date : uni_spec.end_date]

    # 6. Run strategy — fit on full prices first (in-sample full-period run)
    cost_bps = norm_cfg["execution"]["transaction_cost_bps"]
    strategy.fit(prices)
    sr = run_strategy(prices, strategy, transaction_cost_bps=cost_bps)

    # 7. Walk-forward validation — fresh model instance to avoid state contamination
    wf: WalkForwardResult | None = None
    if val_config.type != "none":
        splits = build_validation_splits(val_config, prices.index)
        if splits:
            if is_panel:
                strategy_wf = build_panel_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal,
                    list(uni_spec.tickers),
                    portfolio_construction=ml_spec.portfolio_construction,
                )
            else:
                strategy_wf = build_ml_strategy(
                    ml_spec.features, ml_spec.labels, ml_spec.model, ml_spec.signal
                )
            wf = run_walk_forward_validation(
                prices=prices,
                strategy=strategy_wf,
                splits=splits,
                transaction_cost_bps=cost_bps,
            )

    # 8. Build ExperimentResult
    experiment_result = ExperimentResult(
        experiment_name=ml_spec.name,
        strategy_name=strategy.name,
        parameters=ml_spec.model.params,
        metrics=sr.metrics,
        weights=sr.weights,
        equity_curve=sr.backtest["equity_curve"],
        returns=sr.backtest["net_return"],
        created_at=datetime.now(UTC),
    )

    # 9. Pre-compute all diagnostics data (used by both plots and JSON persistence)
    feature_data = _prepare_feature_diagnostics(strategy, prices, ml_spec, is_panel=is_panel)
    ml_data = _prepare_ml_diagnostics(strategy, prices, wf, ml_spec, is_panel=is_panel)
    universe_data = _prepare_universe_diagnostics(prices)
    # Merge feature diagnostics into ml_data for unified plot generation
    ml_data.update({k: v for k, v in feature_data.items() if k not in ml_data})
    ml_data["is_panel"] = is_panel

    # 9a-supplement. Lightweight vol-regime conditional IC statistics (Step 2)
    try:
        from src.reporting.regime import compute_regime_stats
        regime_stats = compute_regime_stats(prices, wf, ml_data)
        if regime_stats:
            ml_data["regime_stats"] = regime_stats
    except Exception:
        pass

    # 9a-supplement-2: Prediction-strength bucket analysis (Step 3)
    try:
        horizon_ps = int(ml_spec.labels.params.get("horizon", 21))
        ps_data = _prepare_prediction_strength(prices, ml_data, horizon=horizon_ps)
        if ps_data:
            ml_data["prediction_strength"] = ps_data
    except Exception:
        pass

    # 9a-supplement-3: Ranking geometry diagnostics (Phase I — S1-S5)
    try:
        horizon_rg = int(ml_spec.labels.params.get("horizon", 21))
        rg_data = _prepare_ranking_geometry(prices, ml_data, horizon=horizon_rg)
        if rg_data:
            ml_data["ranking_geometry"] = rg_data
    except Exception:
        pass

    # 9a-supplement-4: Feature contribution diagnostics (Phase II — C1–C3)
    try:
        fc_data = _prepare_feature_contributions(ml_data)
        if fc_data:
            ml_data["feature_contributions"] = fc_data
    except Exception:
        pass

    # 9a-supplement-5: Allocation research diagnostics (Phase 2 — panel only)
    if is_panel:
        try:
            horizon_ar = int(ml_spec.labels.params.get("horizon", 21))
            alloc_research = _prepare_allocation_research_diagnostics(
                ml_data, sr.weights, prices, ml_spec, horizon=horizon_ar
            )
            if alloc_research:
                ml_data["alloc_research"] = alloc_research
        except Exception:
            pass

    # 9b. Compute high-volatility stress mask for regime shading on IC figures.
    # Stress = 21-day realised vol > 2σ of its own rolling 252-day distribution.
    stress_mask: pd.Series | None = None
    try:
        rets = prices.pct_change()
        vol_panel = rets.rolling(21).std() * (252 ** 0.5)
        avg_vol = vol_panel.mean(axis=1)
        threshold = avg_vol.rolling(252, min_periods=63).mean() + \
                    2.0 * avg_vol.rolling(252, min_periods=63).std()
        stress_mask = (avg_vol > threshold).reindex(prices.index).fillna(False)
    except Exception:
        pass

    # 9c. Plots (receives ml_data with both model + feature diagnostics)
    plots = _build_plots(
        sr.backtest, wf, strategy.name,
        weights=sr.weights,
        ml_data=ml_data,
        universe_data=universe_data,
        stress_mask=stress_mask,
        profile=profile,
    )

    # 10. Save artefacts
    output_cfg = norm_cfg["output"]
    output_dir = Path(output_cfg["base_dir"])
    out_path = save_run(
        experiment_result,
        spec=spec,
        output_dir=output_dir,
        plots=plots if output_cfg.get("save_plots", True) else None,
    )
    if output_cfg.get("save_plots", True):
        _write_plot_index(plots, out_path)

    import matplotlib.pyplot as plt
    for fig in plots.values():
        plt.close(fig)

    # Write ML provenance sidecar
    _write_ml_provenance(ml_spec, out_path)

    # 11. Dual config artefacts
    _write_raw_config(raw_cfg, out_path, source_path)
    _write_normalized_config(norm_cfg, out_path)

    # 11b. Persist diagnostics
    if wf is not None and wf.n_splits > 0:
        _write_split_metrics(wf, out_path)
        _write_wf_equity_curves(wf, out_path)
    _write_ml_diagnostics(sr, ml_spec.model.type, wf, out_path)
    _write_ml_model_diagnostics_json(ml_data, ml_spec, out_path)
    _write_feature_engineering_artefacts(feature_data, ml_spec, out_path)
    _write_backtest_diagnostics(sr, out_path)
    _write_research_artefacts(prices, strategy, sr, out_path)
    _write_universe_artefacts(universe_data, out_path)
    if is_panel:
        _write_allocation_diagnostics(
            sr.weights, ml_spec, out_path,
            alloc_research=ml_data.get("alloc_research"),
        )

    # 12. Register
    if output_cfg.get("register", True):
        registry_path = Path(output_cfg["registry_path"])
        registry = ExperimentRegistry(registry_path)
        registry.register(experiment_result, spec=spec, path=out_path)

    return ExperimentRun(
        spec=spec,
        strategy_result=sr,
        experiment_result=experiment_result,
        walk_forward=wf,
        output_path=out_path,
    )


# ---------------------------------------------------------------------------
# Summary and composite helpers
# ---------------------------------------------------------------------------


def format_run_summary(run: ExperimentRun) -> str:
    """Return a concise human-readable summary of a completed experiment run.

    Args:
        run: The ExperimentRun returned by run_experiment_from_config().

    Returns:
        Multi-line string suitable for printing to stdout.
    """
    lines: list[str] = [
        f"Experiment : {run.spec.experiment_name}",
        f"Strategy   : {run.strategy_result.strategy_name}",
        f"Saved to   : {run.output_path}",
    ]

    if run.walk_forward is not None:
        lines.append(f"WF splits  : {run.walk_forward.n_splits}")

    lines.append("\n─── Metrics ───")
    for k, v in run.experiment_result.metrics.items():
        lines.append(f"  {k:<28}: {v:.4f}")

    return "\n".join(lines)


def run_and_report(
    config_path: str | Path,
    report_output_dir: str | Path = Path("reports"),
    include_html: bool = True,
    report_spec: Any | None = None,
    profile: str = "report",
) -> tuple[ExperimentRun, Any]:
    """Run an experiment from config then immediately generate a report.

    Thin wrapper: delegates entirely to run_experiment_from_config() then
    generate_experiment_report().  No additional logic lives here.

    Args:
        config_path: Path to the YAML or JSON config file.
        report_output_dir: Root directory for generated reports.
        include_html: Whether to also produce an HTML report.
        report_spec: ResearchReportSpec preset.  None resolves to STANDARD_REPORT.
        profile: Rendering/export profile for canonical figures.
                 "report" (default) or "frontend".

    Returns:
        (ExperimentRun, ReportPaths) tuple.
    """
    from src.reporting.report_builder import generate_experiment_report

    run = run_experiment_from_config(config_path, profile=profile)
    paths = generate_experiment_report(
        run.output_path,
        output_dir=report_output_dir,
        include_html=include_html,
        report_spec=report_spec,
    )
    return run, paths
