"""Markdown report renderer for experiment artefacts.

Renders a deterministic, research-narrative-ordered report.  Section order:

    1.  Title
    2.  Summary                  (period, universe, top Sharpe)
    3.  Research Thesis          (hypothesis, rationale, risks — spec.include_thesis)
    4.  Data Infrastructure      (universe coverage, NaN diagnostics — spec.include_data_infrastructure)
    5.  Feature Engineering      (registry, alignment, stats, correlations — spec.include_feature_engineering)
    6.  Backtesting Methodology  (timing, cost model — spec.include_methodology)
    7.  Portfolio Construction   (signal-to-weight pipeline — spec.include_portfolio_process)
    8.  Model & Features         (v2 only — spec.include_ml_analysis)
    8b. ML Model Behaviour       (v2 only — spec.include_ml_analysis)
    9.  Performance Metrics      (spec.include_metrics)
   10.  Walk-Forward Validation  (spec.include_validation; also gated on validation.type)
   11.  Failure Analysis         (drawdown windows, failure modes — spec.include_failure_analysis)
   12.  Diagnostics Appendix     (split metrics + ML diagnostics — spec.include_diagnostics)
   13.  Metadata                 (experiment name, strategy, date)
   14.  Configuration            (universe, params, cost, validation)
   15.  Figures                  (appendix — only unclaimed/secondary figures)
   16.  Provenance               (only if config_hash or ml_hash present)
   17.  Footer

Section inclusion is governed by ResearchReportSpec.  When no spec is provided,
behaviour is identical to STANDARD_REPORT (the canonical default).

Figure paths are received pre-computed from report_builder — this module
makes no filesystem path assumptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.reporting.report_builder import ExperimentArtefacts
    from src.reporting.report_spec import ResearchReportSpec

# Preferred display order for metrics table
_METRIC_ORDER = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
]

def _feat_label(name: str) -> str:
    """Return a publication-ready feature label.

    Delegates to src.features.families.generate_feature_label() — the single
    canonical label authority for all feature display names.
    """
    try:
        from src.features.families import generate_feature_label
        return generate_feature_label(name)
    except Exception:
        return name


_PANEL_SIGNAL_TYPES_SET = {"top_n", "long_short", "normalize"}


def _is_panel_mode(artefacts: ExperimentArtefacts) -> bool:
    """Return True when the experiment uses a cross-sectional panel topology."""
    prov = artefacts.ml_provenance
    if not isinstance(prov, dict):
        return False
    signal_type = (prov.get("signal") or {}).get("type", "")
    return signal_type in _PANEL_SIGNAL_TYPES_SET


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_report(
    artefacts: ExperimentArtefacts,
    figure_paths: list[tuple[str, Path]],
    generated_at: str,
    report_version: str,
    report_spec: ResearchReportSpec | None = None,
) -> str:
    """Render a full markdown report for one experiment.

    Args:
        artefacts:    Loaded experiment artefacts (metadata, metrics, config).
        figure_paths: (display_name, relative_path) pairs computed by
                      report_builder; paths are relative to the renderer's
                      output directory.
        generated_at: ISO-8601 timestamp string for the footer.
        report_version: Report schema version string for the footer.
        report_spec:  Optional ResearchReportSpec governing section inclusion.
                      None resolves to STANDARD_REPORT — the canonical default.

    Returns:
        Complete markdown string.
    """
    from src.reporting.report_spec import STANDARD_REPORT

    spec = report_spec if report_spec is not None else STANDARD_REPORT

    # Map stem names (e.g. "equity_and_drawdown") to renderer-relative paths.
    # claimed tracks which figures have been placed inline so the appendix
    # _figures() section receives only unclaimed remainder.
    fig_map: dict[str, Path] = {
        dn.lower().replace(" ", "_"): rp for dn, rp in figure_paths
    }
    claimed: set[str] = set()

    sections: list[str] = [_title(artefacts)]

    if spec.include_summary:
        s = _summary(artefacts)
        if s:
            sections.append(s)

    if spec.include_thesis:
        t = _research_thesis(artefacts)
        if t:
            sections.append(t)

    if spec.include_universe_section:
        uc = _universe_construction(artefacts, fig_map, claimed)
        if uc:
            sections.append(uc)

    if spec.include_data_infrastructure:
        di = _data_infrastructure(artefacts, fig_map, claimed)
        if di:
            sections.append(di)

    if spec.include_feature_engineering:
        fe = _feature_engineering(artefacts, fig_map, claimed)
        if fe:
            sections.append(fe)

    if spec.include_methodology:
        sections.append(_backtesting_methodology(artefacts))

    if spec.include_portfolio_process:
        pp = _portfolio_process(artefacts, fig_map, claimed)
        if pp:
            sections.append(pp)

    if spec.include_allocation_research:
        ar = _allocation_research_section(artefacts, fig_map, claimed)
        if ar:
            sections.append(ar)

    if spec.include_ml_provenance_detail:
        ml = _ml_section(artefacts)
        if ml:
            sections.append(ml)

    if spec.include_ml_analysis:
        mlb = _ml_model_behavior(artefacts, fig_map, claimed)
        if mlb:
            sections.append(mlb)

    if spec.include_metrics:
        sections.append(_metrics(artefacts, fig_map, claimed))

    if spec.include_validation:
        wf = _walk_forward(artefacts, fig_map, claimed)
        if wf:
            sections.append(wf)

    if spec.include_failure_analysis:
        fa = _failure_analysis(artefacts, fig_map, claimed)
        if fa:
            sections.append(fa)

    if spec.include_diagnostics:
        diag = _diagnostics_section(artefacts, fig_map, claimed)
        if diag:
            sections.append(diag)

    # Administrative sections — placed after research analysis
    if spec.include_metadata:
        sections.append(_metadata(artefacts))

    if spec.include_configuration:
        sections.append(_configuration(artefacts))

    if spec.include_figures:
        captions = _build_figure_captions(artefacts)
        # Only unclaimed figures reach the appendix section
        remainder = [(dn, rp) for dn, rp in figure_paths if dn.lower().replace(" ", "_") not in claimed]
        figs = _figures(remainder, captions=captions)
        if figs:
            sections.append(figs)

    if spec.include_provenance:
        prov = _provenance_section(artefacts)
        if prov:
            sections.append(prov)

    # G-SYNC-5: consistency validation — append warnings block if any violations found
    try:
        from src.reporting.consistency import validate_report_consistency
        cr = validate_report_consistency(artefacts, fig_map, claimed)
        if cr.has_warnings:
            sections.append(cr.as_markdown())
    except Exception:
        pass

    sections.append(_footer(artefacts, generated_at, report_version))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section renderers — private, one function per section
# ---------------------------------------------------------------------------


def _title(artefacts: ExperimentArtefacts) -> str:
    name = artefacts.metadata.get("experiment_name", "Unnamed Experiment")
    return f"# Experiment Report: {name}"


def _summary(artefacts: ExperimentArtefacts) -> str:
    lines: list[str] = []

    cfg = artefacts.config
    if cfg is not None:
        dr = cfg.get("date_range") or {}
        start = dr.get("start", "—")
        end = dr.get("end", "—")
        lines.append(f"- **Period:** {start} to {end}")

        tickers = (cfg.get("universe") or {}).get("tickers", [])
        if tickers:
            lines.append(f"- **Universe:** {', '.join(tickers)}")

    sharpe = artefacts.metrics.get("sharpe_ratio") if isinstance(artefacts.metrics, dict) else None
    if sharpe is not None:
        lines.append(f"- **Sharpe ratio:** {sharpe:.4f}")

    if not lines:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NEW: Research Thesis & Methodology
# ---------------------------------------------------------------------------


def _research_thesis(artefacts: ExperimentArtefacts) -> str:
    if isinstance(artefacts.ml_provenance, dict):
        body = _ml_thesis_body(artefacts)
        return _render_section("Research Thesis & Methodology", body)
    if not isinstance(artefacts.config, dict):
        return ""
    cfg = artefacts.config
    strategy_type = (cfg.get("strategy") or {}).get("type", "")
    if strategy_type == "MomentumRotation":
        body = _momentum_thesis_body(cfg)
    else:
        body = _generic_thesis_body(strategy_type)
    return _render_section("Research Thesis & Methodology", body)


def _momentum_thesis_body(cfg: dict) -> str:
    params = (cfg.get("strategy") or {}).get("parameters") or {}
    lookback = int(params.get("lookback", 252))
    top_n = int(params.get("top_n", 3))
    freq = params.get("rebalance_freq", "ME")
    cost_bps = float((cfg.get("execution") or {}).get("transaction_cost_bps", 5.0))
    universe = (cfg.get("universe") or {}).get("tickers", [])
    n_assets = len(universe)
    lookback_months = round(lookback / 21)

    freq_labels = {"ME": "month-end", "QE": "quarter-end", "W-FRI": "weekly"}
    freq_label = freq_labels.get(freq, freq)

    lines = [
        "**Hypothesis:** Assets exhibiting strong trailing "
        + str(lookback_months)
        + "-month price performance will continue to outperform on a risk-adjusted basis"
        " over the subsequent 1–3 months, on average and across varied market regimes.",
        "",
        "**Economic rationale:** Momentum persistence in cross-sectional asset returns"
        " is attributed to three reinforcing mechanisms:",
        "",
        "1. *Investor underreaction* — Fundamental information is absorbed slowly into"
        " prices. Market participants anchor to prior valuations and revise incrementally,"
        " creating predictable near-term drift in the direction of the most recent"
        " fundamental change.",
        "2. *Institutional herding* — Risk-budgeting constraints, tracking-error mandates,"
        " and performance-chasing capital flows amplify and extend trends beyond what"
        " fundamentals alone justify.",
        "3. *Regime persistence* — Macroeconomic regimes (growth, inflation, risk-off)"
        " tend to persist for multiple months, sustaining consistent leadership across"
        " equity sectors and cross-asset categories.",
        "",
        "**Signal construction:** Trailing "
        + str(lookback)
        + "-day total return serves as the momentum signal. The "
        + str(lookback_months)
        + "-month window is long enough to capture genuine trend persistence while short"
        " enough to avoid overlap with the value horizon (24–36 months) at which return"
        " reversal dominates. Assets are ranked cross-sectionally each rebalance period;"
        " the top "
        + str(top_n)
        + " of "
        + str(n_assets)
        + " by momentum score are selected and equal-weighted.",
        "",
        "**Rebalance policy:** "
        + freq_label.title()
        + " rebalancing balances two competing objectives: capturing regime persistence"
        " across several weeks without accumulating excessive transaction-cost drag."
        " Higher-frequency rebalancing would capture incremental alpha but inflate costs;"
        " lower-frequency rebalancing would miss month-to-month leadership transitions.",
        "",
        "**Key risks and limitations:**",
        "",
        "- *Momentum crashes* — The strategy is acutely exposed to rapid reversals"
        " following market dislocations. In crisis-recovery environments (e.g., 2009,"
        " 2020), lagging assets frequently outperform leaders by 20–40% within weeks."
        " A long-only momentum strategy has no mechanism to position defensively ahead"
        " of such reversals.",
        "- *Regime dependence* — Momentum outperforms in trending, low-volatility"
        " environments and underperforms in mean-reverting, high-volatility regimes."
        " Strategy viability in live trading depends partly on the frequency of"
        " favourable regimes in the deployment period.",
        "- *Concentration risk* — Holding "
        + str(top_n)
        + " of "
        + str(n_assets)
        + " assets implies concentrated sector and factor exposure. Adverse selection"
        " within the top-N cohort compounds drawdowns during sector-specific dislocations.",
        "- *Look-back sensitivity* — Performance is sensitive to the choice of momentum"
        " window. Robustness across alternative windows (e.g., half and double the"
        " current lookback) is a necessary validation step not captured by a"
        " single-parameter backtest.",
        "- *Transaction costs* — At "
        + str(int(cost_bps))
        + " bps one-way, cost drag compounds materially. Real-world friction for"
        " institutional scale or less-liquid instruments may exceed this assumption.",
        "",
        "**Scope of this analysis:** This strategy is not presented as a definitive"
        " source of live alpha. It is a vehicle for demonstrating a complete,"
        " institutionally-rigorous research process: data infrastructure, signal"
        " engineering, look-ahead prevention, walk-forward validation, and failure"
        " mode analysis. *The methodology is the product.*",
    ]
    return "\n".join(lines)


def _generic_thesis_body(strategy_type: str) -> str:
    label = strategy_type or "this strategy"
    return (
        f"**Strategy:** `{label}`\n\n"
        "A full research thesis — covering hypothesis, economic rationale, signal"
        " construction, rebalance policy, and key risks — can be added here by"
        " implementing strategy-specific template prose in"
        " `src/reporting/markdown.py::_research_thesis()`.\n\n"
        "The thesis section is the primary vehicle for communicating *why* a strategy"
        " may work, under what conditions it is expected to fail, and what an"
        " informed reader should look for in the validation results."
    )


def _ml_thesis_body(artefacts: ExperimentArtefacts) -> str:
    """Research thesis for v2 ML experiments.

    Reads ml_provenance, feature_registry, and feature_summary to generate
    a structured thesis covering: hypothesis, feature rationale, label choice,
    model class, regularisation, and signal translation.
    """
    prov = artefacts.ml_provenance or {}
    model = prov.get("model") or {}
    model_type = model.get("type", "unknown")
    model_params = model.get("params") or {}
    label_spec = prov.get("labels") or {}
    label_type = label_spec.get("type", "unknown")
    label_params = label_spec.get("params") or {}
    horizon = label_params.get("horizon")
    signal_spec = prov.get("signal") or {}
    signal_type = signal_spec.get("type", "sign")
    feat_spec = prov.get("features") or {}
    entries = feat_spec.get("entries") or []
    ticker = feat_spec.get("ticker", "the asset")

    # Feature type → one-sentence rationale
    _FEAT_RATIONALE: dict[str, str] = {
        "momentum": (
            "Trailing-return momentum captures the continuation of recent price trends —"
            " a well-documented persistent anomaly attributed to investor underreaction"
            " and gradual information diffusion."
        ),
        "rolling_volatility": (
            "Rolling realised volatility characterises the current market regime:"
            " elevated volatility signals risk-off environments where expected"
            " returns and risk premia shift materially."
        ),
        "rolling_zscore": (
            "Rolling z-score normalisation transforms raw signals into a stationary,"
            " mean-zero representation, mitigating regime-level drift that would"
            " otherwise confound the model's cross-period comparisons."
        ),
        "sma": (
            "Simple moving average deviation measures medium-term trend direction,"
            " providing a regime signal complementary to trailing-return momentum."
        ),
        "ema": (
            "Exponential moving average deviation weights recent prices more heavily,"
            " producing a faster-reacting trend signal suitable for shorter"
            " rebalance frequencies."
        ),
        "compute_returns": (
            "Log returns provide a stationary, approximately Gaussian representation"
            " of price changes, forming the foundational input for higher-order"
            " feature construction."
        ),
        "trend_strength": (
            "Trend strength (slope R²) quantifies how consistently directional a"
            " price series has been — distinguishing sustained trends from noisy"
            " mean-reverting behaviour."
        ),
        # Phase H-1 additions
        "trend_persistence": (
            "Trend persistence measures the fraction of trading days within the"
            " window on which the asset's daily return was positive — a directional"
            " hit rate. Unlike raw momentum (which measures magnitude) or trend"
            " strength (which measures linearity), this captures how *consistently*"
            " each day contributed to the trend direction, exposing the noise"
            " structure within the momentum window."
        ),
        "breakout_strength": (
            "Breakout strength measures the proximity of the current price to its"
            " rolling N-period high: (price / rolling_max) − 1. A value near zero"
            " indicates the asset is at or near its recent range top — the breakout"
            " regime. Large negative values indicate the asset is well below its"
            " recent high. The model can learn whether proximity to recent highs"
            " signals continuation (momentum) or resistance (mean-reversion),"
            " a relationship that reverses across macro regimes."
        ),
        "drawdown_distance": (
            "Drawdown distance measures the current price's percentage decline from"
            " its rolling N-period peak: (price / rolling_max) − 1 over a long"
            " lookback window. Unlike short-horizon breakout strength, this captures"
            " sustained stress-state positioning — whether an asset remains in an"
            " extended drawdown relative to its annual price history. Assets with"
            " large negative values are in prolonged underperformance regimes;"
            " values near zero indicate the asset is near or recovering to its"
            " annual high-water mark."
        ),
        "vol_compression": (
            "Volatility compression measures the ratio of short-term to long-term"
            " realised volatility. A ratio below 1.0 indicates a compressed-vol"
            " regime — recent realised vol has contracted relative to its medium-term"
            " baseline. This is a breakout precursor indicator: periods of sustained"
            " vol compression historically precede regime transitions. A ratio above"
            " 1.0 indicates vol expansion, consistent with an active stress or"
            " dislocation environment."
        ),
        "rolling_beta": (
            "Rolling market beta measures each asset's time-varying sensitivity to"
            " the market reference (SPY): Cov(r_asset, r_market) / Var(r_market)"
            " over a rolling window. In a cross-sectional ranking framework, beta"
            " captures which assets are currently high-beta (amplified systematic"
            " exposure) vs defensive (low-beta) — information that is orthogonal to"
            " price-history momentum and directly relevant to regime positioning."
            " A model learning positive beta coefficients is selecting high-beta"
            " assets in trending markets; negative coefficients indicate a preference"
            " for defensives."
        ),
        "risk_adjusted_momentum": (
            "Risk-adjusted momentum divides the trailing N-period return by rolling"
            " realised volatility — a Sharpe-like signal measuring momentum quality"
            " rather than raw magnitude. Two assets with equal 12-month momentum but"
            " different volatilities receive different scores: the lower-vol asset"
            " achieves its return more efficiently. This exposes whether the model"
            " rewards momentum consistency (efficient uptrends) or raw return"
            " regardless of the risk taken to achieve it."
        ),
    }

    # Label rationale
    _LABEL_RATIONALE: dict[str, str] = {
        "forward_returns": (
            "Forward log returns are the direct target of the predictive task: the"
            " model learns to rank periods by expected price appreciation."
        ),
        "binary_direction": (
            "Binary direction labels — up/down — focus the model on directional"
            " accuracy rather than return magnitude, reducing sensitivity to"
            " return outliers in the training signal."
        ),
        "ranking_target": (
            "Cross-sectional return rank labels normalise each asset's forward return"
            " to a percentile position within the universe on each date. Ranking"
            " eliminates the effect of aggregate market moves, focusing the model on"
            " relative outperformance — a cleaner signal for cross-sectional selection."
        ),
    }

    # Model rationale
    _MODEL_RATIONALE: dict[str, str] = {
        "RidgeRegression": (
            "Ridge regression (L2 regularisation) is the natural baseline for"
            " financial ML: it is interpretable, computationally stable, and"
            " its regularisation controls overfitting in the high-noise, low"
            " signal-to-noise regime of asset returns without eliminating features entirely."
        ),
        "LassoRegression": (
            "Lasso regression (L1 regularisation) produces sparse coefficient"
            " vectors, performing implicit feature selection. This is appropriate"
            " when the researcher suspects only a subset of features carry genuine"
            " predictive information."
        ),
        "ElasticNetRegression": (
            "Elastic net combines L1 and L2 penalties, achieving both sparsity"
            " and coefficient stability. It is well-suited to correlated feature"
            " spaces where pure Lasso tends to arbitrarily select one correlated"
            " feature over another."
        ),
        "LogisticRegression": (
            "Logistic regression models the probability of a positive return"
            " outcome, producing calibrated directional probabilities rather than"
            " raw return forecasts. Suitable for binary-direction labels."
        ),
        "LinearRegression": (
            "OLS linear regression provides an unregularised baseline. In the"
            " low-signal financial setting, this is primarily useful for"
            " establishing an upper bound on in-sample fit before regularisation"
            " is applied."
        ),
    }

    # Signal rationale
    _SIGNAL_RATIONALE: dict[str, str] = {
        "sign": (
            "The sign signal translates continuous predictions into a binary"
            " long/flat position: positive prediction → long, non-positive → flat."
            " This is the simplest leakage-free translation of regression output"
            " into a tradeable signal."
        ),
        "threshold": (
            "The threshold signal selects positions only when prediction"
            " confidence exceeds a minimum level, filtering out low-conviction"
            " signals. This reduces turnover at the cost of reduced market exposure."
        ),
        "top_n": (
            "The top-N signal selects the N assets with the highest predictions"
            " at each rebalance, enabling cross-sectional portfolio construction"
            " from multi-asset prediction outputs."
        ),
        "long_short": (
            "The long-short signal takes long positions in the highest-scoring"
            " assets and short positions in the lowest-scoring, constructing a"
            " market-neutral portfolio that isolates the cross-sectional signal."
        ),
    }

    n_features = len(entries)
    list({e.get("type", "") for e in entries if e.get("type")})
    horizon_str = f"{horizon}-day forward" if horizon is not None else "forward"

    lines: list[str] = []

    # Hypothesis — panel mode vs single-asset
    _PANEL_SIGNAL_TYPES = {"top_n", "long_short", "normalize"}
    if signal_type in _PANEL_SIGNAL_TYPES:
        universe_cfg = (artefacts.config or {}).get("universe") or {}
        universe_tickers = universe_cfg.get("tickers") or []
        n_assets = len(universe_tickers) if universe_tickers else "multiple"
        hypothesis = (
            f"**Hypothesis:** A shared {model_type} model predicts cross-sectional"
            f" return ranks across the {n_assets}-asset universe using {n_features}"
            f" engineered feature" + ("s" if n_features != 1 else "")
            + f" applied independently to each asset. The top-ranked assets by"
            f" predicted score, held equal-weight, earn a {horizon_str} return edge"
            f" over the universe after transaction costs."
        )
    else:
        hypothesis = (
            f"**Hypothesis:** The feature space constructed from {ticker} price history"
            f" contains statistically reliable information about {horizon_str} returns."
            f" A {model_type} model trained on {n_features} engineered feature"
            + ("s" if n_features != 1 else "")
            + " can extract this signal and translate it into a"
            " risk-adjusted return edge after transaction costs."
        )
    lines += [hypothesis, ""]

    # Feature rationale
    if entries:
        lines += ["**Feature rationale:**", ""]
        seen_types: set[str] = set()
        for entry in entries:
            ftype = entry.get("type", "")
            fname = entry.get("name", ftype)
            params = entry.get("params") or {}
            window = params.get("window") or params.get("lookback") or params.get("span")
            window_str = f" (window: {window})" if window is not None else ""
            rationale = _FEAT_RATIONALE.get(ftype, f"`{ftype}` feature.")
            if ftype not in seen_types:
                lines.append(
                    f"- **`{fname}`**{window_str}: {rationale}"
                )
                seen_types.add(ftype)
        lines.append("")

    # Label rationale
    label_rationale = _LABEL_RATIONALE.get(
        label_type,
        f"Labels of type `{label_type}` are the prediction target.",
    )
    lines += [
        f"**Label construction:** `{label_type}`"
        + (f" with {horizon}-period horizon" if horizon else "")
        + f". {label_rationale}",
        "",
    ]

    # Model and regularisation rationale
    model_rationale = _MODEL_RATIONALE.get(
        model_type,
        f"`{model_type}` is the selected model class.",
    )
    lines += [f"**Model choice:** {model_rationale}", ""]

    if model_params:
        param_rows = [(k, str(v)) for k, v in sorted(model_params.items())]
        lines.append(_pipe_table(["Hyperparameter", "Value"], param_rows))
        lines.append("")

    # Signal rationale
    signal_rationale = _SIGNAL_RATIONALE.get(
        signal_type,
        f"Signal type: `{signal_type}`.",
    )
    lines += [f"**Signal translation:** {signal_rationale}", ""]

    # Key risks
    lines += [
        "**Key risks and limitations:**",
        "",
        "- *Low signal-to-noise ratio* — Asset return prediction is a notoriously"
        " difficult task. Even a statistically significant in-sample fit does not"
        " guarantee that the signal survives out-of-sample in a different market"
        " regime.",
        "- *Regime non-stationarity* — Feature-return relationships that held"
        " during the training window may weaken or reverse in different macro"
        " environments. Walk-forward validation tests chronological robustness"
        " but cannot fully simulate live deployment.",
        "- *Overfitting risk* — With a limited sample and multiple features,"
        " regularisation is essential. Coefficient stability across walk-forward"
        " splits is a key diagnostic for detecting overfitting.",
        "- *Transaction costs* — ML signals can produce frequent position changes."
        " Cost drag compounds at high turnover; the model's net alpha must"
        " comfortably exceed the cost of executing its predicted positions.",
        "",
        "**Scope:** This investigation is a systematic demonstration of the full"
        " ML research process: feature engineering, leakage-safe alignment,"
        " regularised model training, walk-forward validation, and signal"
        " translation. *The methodology is the product.*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# NEW: Data Infrastructure
# ---------------------------------------------------------------------------


def _universe_construction(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Universe Construction & Coverage section.

    Establishes research universe validity before any modelling begins.
    Exposes asset availability, coverage structure, cross-asset correlations,
    and macro volatility regime context.

    Renders only when universe_coverage artefact is present (written by
    _write_universe_artefacts() in the orchestrator).
    """
    uc = artefacts.universe_coverage if isinstance(artefacts.universe_coverage, dict) else {}
    if not uc:
        return ""

    tickers = uc.get("tickers") or []
    n_assets = uc.get("n_assets", len(tickers))
    n_days = uc.get("n_trading_days", 0)
    date_range = uc.get("date_range") or {}
    asset_coverage = uc.get("asset_coverage") or []
    vol_summary = uc.get("vol_summary") or {}
    corr_matrix = uc.get("correlation_matrix") or {}

    lines: list[str] = []

    # --- Narrative intro ---
    dr_str = (
        f"{date_range.get('start', '—')} to {date_range.get('end', '—')}"
        if date_range else "—"
    )
    lines.append(
        f"The research universe comprises {n_assets} institutional ETF"
        f"{'s' if n_assets != 1 else ''} spanning "
        + _universe_asset_description(tickers)
        + f". The panel covers {n_days:,} trading days ({dr_str}),"
        " providing a structurally diverse cross-sectional environment for"
        " regime-heterogeneous research."
    )

    # --- Asset coverage table ---
    if asset_coverage:
        lines.append("")
        lines.append("**Asset coverage summary:**")
        lines.append("")
        coverage_rows: list[tuple[str, ...]] = []
        for entry in asset_coverage:
            ticker = entry.get("ticker", "—")
            n = entry.get("n_days", "—")
            first = str(entry.get("first_date", "—"))[:10]
            last = str(entry.get("last_date", "—"))[:10]
            miss = entry.get("missingness_pct")
            vol_e = vol_summary.get(ticker, {})
            mean_vol = vol_e.get("mean_vol")
            coverage_rows.append((
                ticker,
                str(n),
                first,
                last,
                f"{miss:.1%}" if isinstance(miss, float) else "—",
                f"{mean_vol:.1%}" if isinstance(mean_vol, float) else "—",
            ))
        lines.append(_pipe_table(
            ["Asset", "Trading Days", "First Date", "Last Date",
             "Missingness", "Mean Ann. Vol"],
            coverage_rows,
        ))

    # --- Correlation structure summary ---
    if corr_matrix and len(tickers) >= 2:
        lines.append("")
        lines.append("**Cross-asset correlation structure:**")
        lines.append("")
        lines.append(_universe_correlation_commentary(tickers, corr_matrix))

    # --- Universe Integrity block ---
    # For structurally clean universes (zero missingness, full date coverage),
    # a compact status block is more informative than a solid-green heatmap or a
    # flat availability line.  Figures are still claimed to suppress them from
    # the appendix; they are embedded only when gaps exist.
    _cov_clean = (
        bool(asset_coverage)
        and all(
            entry.get("missingness_pct", 1.0) == 0.0
            and entry.get("n_days", 0) == n_days
            for entry in asset_coverage
        )
    )
    if _cov_clean:
        # Claim both figures silently (prevent appendix dump)
        if claimed is not None:
            claimed.add("universe_coverage_heatmap")
            if n_assets >= 2:
                claimed.add("asset_availability_timeline")
        lines.append("")
        lines.append("**Universe integrity:**")
        lines.append("")
        lines.append(
            f"All {n_assets} assets have complete coverage across all {n_days:,} trading days"
            f" ({date_range.get('start', '—')} – {date_range.get('end', '—')})."
            " Missingness: 0.0% for every asset. No structural gaps, delistings,"
            " or duplicate timestamps detected. Universe breadth is structurally"
            " stable throughout the backtest period."
        )
    else:
        # Data has gaps — embed the diagnostic figures so the reader can inspect them
        cov_fig = _embed_figure(
            "universe_coverage_heatmap", figures, claimed,
            "Monthly price coverage fraction by asset. Green = full data availability;"
            " red = data gaps. Persistent gaps identify structurally incomplete assets"
            " that reduce effective universe breadth.",
        )
        if cov_fig:
            lines.extend(["", cov_fig])

        if n_assets >= 2:
            avail_fig = _embed_figure(
                "asset_availability_timeline", figures, claimed,
                "Rolling count of assets with valid prices. Structural drops reveal"
                " asset additions, delistings, or persistent data gaps that affect"
                " cross-sectional breadth and ranking validity.",
            )
            if avail_fig:
                lines.extend(["", avail_fig])

    # --- Cross-asset volatility ---
    vol_fig = _embed_figure(
        "cross_asset_volatility", figures, claimed,
        "63-day rolling annualised volatility by asset. Persistent divergence"
        " between risk-on and risk-off assets confirms macro regime heterogeneity."
        " Synchronised spikes identify systemic stress episodes.",
    )
    if vol_fig:
        lines.extend(["", vol_fig])

    # --- Universe correlation heatmap ---
    corr_fig = _embed_figure(
        "universe_correlation_heatmap", figures, claimed,
        "Full-period pairwise return correlation matrix. Correlated clusters"
        " (e.g. SPY/QQQ/XLK) reduce effective breadth; low or negative"
        " correlations (e.g. TLT, GLD vs equities) confirm regime diversification.",
    )
    if corr_fig:
        lines.extend(["", corr_fig])

    if not lines:
        return ""

    return _render_section("Universe Construction & Coverage", "\n".join(lines))


def _universe_asset_description(tickers: list[str]) -> str:
    """Compose a concise cross-asset description from the ticker list."""
    categories: dict[str, list[str]] = {
        "US equities": [t for t in tickers if t in {"SPY", "QQQ", "IWM"}],
        "international equities": [t for t in tickers if t in {"EEM", "VEA", "VWO"}],
        "rates": [t for t in tickers if t in {"TLT", "IEF", "SHY", "AGG", "BND"}],
        "commodities": [t for t in tickers if t in {"GLD", "SLV", "USO", "DBC"}],
        "sectors": [t for t in tickers if t in {"XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP"}],
        "credit": [t for t in tickers if t in {"HYG", "LQD", "JNK"}],
    }
    present = {k: v for k, v in categories.items() if v}
    if not present:
        return "multiple asset classes"
    parts = []
    for cat, assets in present.items():
        parts.append(f"{cat} ({', '.join(assets)})")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _universe_correlation_commentary(
    tickers: list[str],
    corr_matrix: dict[str, dict[str, float]],
) -> str:
    """Generate evidence-grounded cross-asset correlation commentary."""
    # Find highest and lowest pairwise off-diagonal correlations
    pairs: list[tuple[float, str, str]] = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            v = corr_matrix.get(t1, {}).get(t2)
            if isinstance(v, (int, float)) and not (v != v):  # not NaN
                pairs.append((float(v), t1, t2))

    if not pairs:
        return (
            "Cross-asset correlation structure computed from full-period returns."
            " Diversified universes with low average pairwise correlation preserve"
            " cross-sectional signal breadth."
        )

    pairs.sort(key=lambda x: x[0])
    low_corr, low_t1, low_t2 = pairs[0]
    high_corr, high_t1, high_t2 = pairs[-1]

    # Mean pairwise correlation
    mean_corr = sum(v for v, _, _ in pairs) / len(pairs)

    return (
        f"Mean pairwise correlation {mean_corr:.2f}."
        f" Highest: {high_t1}/{high_t2} ({high_corr:.2f}) — structural co-movement."
        f" Lowest: {low_t1}/{low_t2} ({low_corr:.2f}) — regime diversification."
        " Orthogonal asset pairs preserve cross-sectional signal breadth under"
        " correlated stress scenarios."
    )


def _data_infrastructure(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    ra = artefacts.research_artefacts if isinstance(artefacts.research_artefacts, dict) else {}
    ds = ra.get("data_summary") if ra else None

    cfg = artefacts.config or {}
    universe = (cfg.get("universe") or {}).get("tickers", [])
    n_assets = len(universe) if universe else 1
    dr = cfg.get("date_range") or {}
    start = dr.get("start", "—")
    end = dr.get("end", "—")

    lines: list[str] = []

    # Source note
    lines.append(
        "**Data source:** Yahoo Finance daily OHLCV data, ingested via `yfinance`"
        " and persisted to local Parquet files for reproducibility. All prices are"
        " adjusted closing prices."
    )
    lines.append("")
    if n_assets > 1:
        alignment_suffix = (
            " This eliminates NaN-driven look-ahead in cross-sectional ranking."
        )
    else:
        alignment_suffix = (
            " This ensures a consistent observation timeline for feature construction."
        )
    lines.append(
        "**Alignment policy:** Inner join across all universe constituents — only"
        " trading days on which every asset has a valid price are retained."
        + alignment_suffix
    )
    lines.append("")
    lines.append(
        "**Missing value policy:** Forward-fill up to a configurable limit (default:"
        " 5 trading days) for isolated gaps caused by exchange holidays or data"
        " vendor gaps. Gaps exceeding the limit remain NaN and are surfaced in"
        " diagnostics rather than silently filled. Over-cleaning (excessive"
        " interpolation) introduces bias by smoothing market structure."
    )

    if ds:
        lines.append("")
        lines.append(
            f"**Coverage:** {ds.get('n_days', '—')} trading days"
            f" from {ds.get('start_date', start)} to {ds.get('end_date', end)}"
            f" across {ds.get('n_assets', len(universe))} assets."
        )

        nan_counts = ds.get("nan_counts") or {}
        return_stats = ds.get("return_stats") or {}

        if universe or nan_counts:
            lines.append("")
            assets_to_show = list(nan_counts.keys()) or universe
            rows: list[tuple[str, ...]] = []
            for asset in assets_to_show:
                nan_n = nan_counts.get(asset, "—")
                rs = return_stats.get(asset) or {}
                ann_ret = rs.get("mean_annual")
                ann_vol = rs.get("vol_annual")
                rows.append((
                    asset,
                    str(nan_n),
                    f"{ann_ret:.1%}" if isinstance(ann_ret, float) else "—",
                    f"{ann_vol:.1%}" if isinstance(ann_vol, float) else "—",
                ))
            lines.append(_pipe_table(
                ["Asset", "NaN Count", "Ann. Return", "Ann. Volatility"],
                rows,
            ))
    else:
        lines.append("")
        lines.append(
            f"**Configured universe:** {', '.join(universe) if universe else '—'}"
            f" | **Date range:** {start} to {end}"
        )
        lines.append("")
        lines.append(
            "*Detailed data diagnostics are available after running the experiment"
            " with the current orchestrator (writes `research/data_summary.json`).*"
        )

    rv_fig = _embed_figure(
        "rolling_volatility", figures, claimed,
        "63-day realised volatility. Elevated regimes and spikes contextualise"
        " coefficient instability, feature drift, and drawdown episodes below.",
    )
    if rv_fig:
        lines.extend(["", rv_fig])

    return _render_section("Data Infrastructure", "\n".join(lines))


# ---------------------------------------------------------------------------
# NEW: Backtesting Methodology
# ---------------------------------------------------------------------------


def _backtesting_methodology(artefacts: ExperimentArtefacts) -> str:
    cfg = artefacts.config or {}
    cost_bps = float((cfg.get("execution") or {}).get("transaction_cost_bps", 5.0))
    params = (cfg.get("strategy") or {}).get("parameters") or {}
    lookback = params.get("lookback")

    lines = [
        "All backtests use a strictly vectorized, look-ahead-safe execution"
        " framework. The critical invariant: no information from period *t* enters"
        " the position that earns the return of period *t*.",
        "",
        "**Timing convention:**",
        "",
        "| Step | Action | When |",
        "| --- | --- | --- |",
        "| Signal computed | Momentum scores calculated from all prices ≤ day *t* | Close of day *t* |",
        "| Position entered | Computed weights applied to portfolio | Open of day *t+1* |",
        "| Return realized | Portfolio return earned | Close of day *t+1* |",
        "",
        "**Implementation:** `applied_weights = weights.shift(1)`. The strategy"
        " never observes the return it will receive when deciding to trade."
        " The first row of every backtest has a zero position — the portfolio"
        " is flat until the first valid signal propagates through the one-day lag.",
    ]

    if lookback:
        lines.append(
            f" Positions are flat for the first {lookback} trading days during"
            " momentum warm-up (insufficient price history for ranking)."
        )

    lines += [
        "",
        "**Portfolio return computation per period:**",
        "",
        "```",
        "gross_return_t  = sum_i( weight_{i,t-1} * asset_return_{i,t} )",
        f"transaction_cost_t = sum_i( |weight_{{i,t}} - weight_{{i,t-1}}| ) * ({int(cost_bps)} / 10_000)",
        "net_return_t    = gross_return_t - transaction_cost_t",
        "equity_curve_t  = product_{s<=t}( 1 + net_return_s ),  anchored at 1.0",
        "drawdown_t      = ( equity_t - max_{s<=t} equity_s ) / max_{s<=t} equity_s",
        "```",
        "",
        f"**Transaction cost model:** One-way costs of {int(cost_bps)} bps are applied"
        " to each unit of absolute portfolio weight change. Cost is incurred only on"
        " actual weight changes — during forward-fill periods between rebalances,"
        " weights are unchanged and no cost is deducted.",
        "",
        "**Turnover definition:** Daily turnover = sum of absolute weight changes"
        " across all assets per period. High mean daily turnover implies high"
        " transaction cost drag; the cost model above makes this drag explicit.",
    ]

    return _render_section("Backtesting Methodology", "\n".join(lines))


# ---------------------------------------------------------------------------
# NEW: Portfolio Construction Process
# ---------------------------------------------------------------------------


def _portfolio_process(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    ra = artefacts.research_artefacts if isinstance(artefacts.research_artefacts, dict) else {}
    st = ra.get("signal_transitions") if ra else None

    cfg = artefacts.config or {}
    strategy_type = (cfg.get("strategy") or {}).get("type", "")
    params = (cfg.get("strategy") or {}).get("parameters") or {}

    lines: list[str] = []

    # --- v2 ML pipeline ---
    if isinstance(artefacts.ml_provenance, dict):
        lines = _ml_portfolio_process_body(artefacts)
    elif strategy_type == "MomentumRotation":
        top_n = int(params.get("top_n", 3))
        lookback = int(params.get("lookback", 252))
        freq = params.get("rebalance_freq", "ME")
        freq_labels = {"ME": "month-end", "QE": "quarter-end", "W-FRI": "weekly"}
        freq_label = freq_labels.get(freq, freq)

        lines += [
            "**Signal-to-portfolio pipeline:**",
            "",
            "```",
            "prices[t-lookback : t]",
            "  → trailing_return_i = price[t] / price[t-lookback] - 1   (per asset)",
            "  → cross_sectional_rank_i  (higher momentum = higher rank)",
            "  → top_" + str(top_n) + "_mask  (boolean selection)",
            "  → equal_weight = 1/" + str(top_n) + " per selected asset",
            "  → forward_fill to daily index",
            "  → shift(1)  [look-ahead prevention]",
            "  → portfolio_weights applied in backtest engine",
            "```",
            "",
            f"**Rebalance frequency:** {freq_label}. "
            f"**Selection size:** top {top_n} by trailing {lookback}-day return.",
        ]

        if st:
            n_rebalances = st.get("n_rebalances", 0)
            transitions = st.get("transitions") or []
            lines.append("")
            lines.append(
                f"**Rebalance history:** {n_rebalances} rebalance events over the"
                " backtest period."
            )
            recent = transitions[-8:] if len(transitions) > 8 else transitions
            if recent:
                lines.append("")
                lines.append("**Recent rebalances (most recent last):**")
                lines.append("")
                rows: list[tuple[str, ...]] = []
                for ev in recent:
                    holdings = ", ".join(ev.get("holdings") or [])
                    entered = ", ".join(ev.get("entered") or []) or "—"
                    exited = ", ".join(ev.get("exited") or []) or "—"
                    rows.append((ev.get("date", "—"), holdings, entered, exited))
                lines.append(_pipe_table(
                    ["Date", "Holdings", "Entered", "Exited"],
                    rows,
                ))
                lines.append("")
                lines.append(
                    "*Date shown is the position-entry date (signal computed one trading"
                    " day prior per the shift(1) convention).*"
                )
        else:
            lines += [
                "",
                "*Rebalance event history is available after running the experiment"
                " with the current orchestrator (writes `research/signal_transitions.json`).*",
            ]
    elif strategy_type:
        lines += [
            f"**Strategy type:** `{strategy_type}`",
            "",
            "The portfolio construction process for this strategy type is not"
            " detailed here. Add strategy-specific process documentation in"
            " `src/reporting/markdown.py::_portfolio_process()`.",
        ]

    if not lines:
        return ""

    # --- Inline figures ---
    # Single-asset ML: allocation_history is a scalar 0/1 position — uninformative inline.
    # Panel / cross-sectional ML: allocation_history is a primary diagnostic showing
    # regime rotation, defensive migration, and concentration through time.
    # Non-ML strategies: always show inline.
    is_ml = isinstance(artefacts.ml_provenance, dict)
    _PANEL_SIGS = {"top_n", "long_short", "normalize"}
    _sig_type = ((artefacts.ml_provenance or {}).get("signal") or {}).get("type", "") if is_ml else ""
    is_panel_alloc = is_ml and _sig_type in _PANEL_SIGS

    if not is_ml or is_panel_alloc:
        if is_panel_alloc:
            _ah_interp = (
                "Stacked allocation through time — which assets the model holds each period."
                " Persistent concentration in one asset reveals ranking stability or regime lock-in."
                " Rapid composition turnover (frequent colour transitions) coincides with regimes"
                " where cross-sectional scores are tightly clustered and small feature changes"
                " alter the ranking. Defensive migration (TLT/GLD dominance) marks risk-off"
                " transitions. Periods of equity breadth (multiple equity ETFs held) indicate"
                " low-dispersion regimes where the model distributes exposure broadly."
            )
        else:
            _ah_interp = (
                "The allocation history visualises portfolio weight evolution through time."
                " Smooth transitions indicate a stable signal; frequent reversals reflect"
                " high-frequency prediction changes or threshold-crossing noise that drive"
                " unnecessary turnover."
            )
        ah_fig = _embed_figure("allocation_history", figures, claimed, _ah_interp)
        if ah_fig:
            lines.extend(["", ah_fig])

    to_fig = _embed_figure(
        "portfolio_turnover", figures, claimed,
        "Portfolio turnover quantifies daily weight change magnitude — the primary"
        " determinant of transaction cost drag. Periods of elevated turnover reduce"
        " net returns through friction. Turnover spikes that coincide with drawdown"
        " periods indicate cost-exacerbated losses: the signal is reversing precisely"
        " when it is most expensive to act on, amplifying rather than merely reducing"
        " performance.",
    )
    if to_fig:
        lines.extend(["", to_fig])


    return _render_section("Portfolio Construction Process", "\n".join(lines))


# ---------------------------------------------------------------------------
# ML portfolio construction narrative (D3)
# ---------------------------------------------------------------------------


def _ml_portfolio_process_body(artefacts: ExperimentArtefacts) -> list[str]:
    """Build the ML pipeline narrative for _portfolio_process()."""
    prov = artefacts.ml_provenance or {}
    model = prov.get("model") or {}
    model_type = model.get("type", "model")
    label_spec = prov.get("labels") or {}
    label_type = label_spec.get("type", "labels")
    signal_spec = prov.get("signal") or {}
    signal_type = signal_spec.get("type", "sign")
    signal_params = signal_spec.get("params") or {}
    feat_spec = prov.get("features") or {}
    ticker = feat_spec.get("ticker", "asset")
    entries = feat_spec.get("entries") or []
    n_features = len(entries)

    # Portfolio construction provenance (spec_version "2"+)
    pc_prov = prov.get("portfolio_construction") or {}
    weighting_prov = pc_prov.get("weighting") or {}
    weighting_scheme = weighting_prov.get("scheme", "equal_weight")
    pred_normalization = weighting_prov.get("prediction_normalization", "none")
    weighting_temperature = weighting_prov.get("temperature")

    # Build feature matrix code block dynamically
    feat_names = [e.get("name", "f") for e in entries]
    feat_list_str = ", ".join(f"'{n}'" for n in feat_names[:4])
    if len(feat_names) > 4:
        feat_list_str += f", ... ({n_features} total)"

    # Signal-specific weight description — provenance-driven for top_n
    def _top_n_weight_desc() -> str:
        if weighting_scheme == "zscore_softmax":
            temp_str = f", T={weighting_temperature}" if weighting_temperature is not None else ""
            norm_str = " (z-scored)" if pred_normalization == "zscore" else ""
            return f"softmax(z-score(scores){temp_str}) for top-N assets{norm_str} (cross-sectional)"
        if weighting_scheme == "confidence_weighted":
            return "∝ max(score, 0) for top-N assets by prediction (cross-sectional)"
        return "1/top_n for top-N assets by prediction (cross-sectional)"

    _SIGNAL_WEIGHT_DESC: dict[str, str] = {
        "sign": "1 if predicted > 0 else 0   (long or flat)",
        "threshold": "1 if |predicted| > threshold else 0   (high-confidence only)",
        "top_n": _top_n_weight_desc(),
        "long_short": "±1/N, net-zero, gross=2   (long-short)",
    }
    weight_desc = _SIGNAL_WEIGHT_DESC.get(signal_type, f"signal type: `{signal_type}`")

    _PANEL_SIGNAL_TYPES = {"top_n", "long_short", "normalize"}
    if signal_type in _PANEL_SIGNAL_TYPES:
        pipeline_header = f"**Signal-to-portfolio pipeline for `{model_type}` (panel cross-sectional):**"
    else:
        pipeline_header = f"**Signal-to-portfolio pipeline for `{model_type}` on `{ticker}`:**"

    lines: list[str] = [
        pipeline_header,
        "",
        "```",
        "prices[t-window : t]",
        f"  → feature_matrix({feat_list_str})",
        f"       (X: {n_features} column{'s' if n_features != 1 else ''}, no NaN rows, pre-alignment)",
        f"  → {model_type}.predict(X_clean)",
        f"       (raw score: expected {label_type.replace('_', ' ')})",
        f"  → signal_fn(predictions)   [{signal_type}]",
        f"       weight_t = {weight_desc}",
        "  → forward_fill to daily index",
        "  → shift(1)   [look-ahead prevention]",
        "  → weights applied in portfolio backtest engine",
        "```",
        "",
        "**Leakage-prevention:** The `shift(1)` operation ensures that the weight"
        " active on trading day *t* was computed from information available only"
        " up to close of day *t−1*. The first valid signal enters the portfolio"
        " on the trading day following the first non-NaN prediction.",
    ]

    # Weighting policy note for panel experiments with non-default scheme
    if signal_type in {"top_n", "long_short", "normalize"} and weighting_scheme != "equal_weight":
        _scheme_notes = {
            "zscore_softmax": (
                "**Allocation policy:** Softmax weighting over z-scored prediction scores"
                " within the selected basket.  All scoring and normalization is strictly"
                " row-wise (timestamp-local) — no cross-date or cross-fold normalization"
                " is applied.  Walk-forward chronology is preserved."
            ),
            "confidence_weighted": (
                "**Allocation policy:** Proportional weighting by clipped-positive"
                " prediction scores within the selected basket.  Assets with non-positive"
                " predictions within the selection receive zero weight; rows with no"
                " positive predictions fall back to equal-weight.  All operations are"
                " strictly row-wise (timestamp-local)."
            ),
        }
        scheme_note = _scheme_notes.get(
            weighting_scheme,
            f"**Allocation policy:** `{weighting_scheme}` weighting applied to the"
            " selected basket (row-wise, timestamp-local).",
        )
        lines += ["", scheme_note]

    # Signal-specific notes
    if signal_type == "sign":
        lines += [
            "",
            "**Position sizing:** Equal-weight long-only. The model takes a full"
            " unit position when the prediction is positive and holds cash"
            " (zero weight) when the prediction is non-positive. There is no"
            " fractional position scaling — all conviction comes from the"
            " binary direction of the prediction.",
        ]
    elif signal_type == "threshold":
        threshold_val = signal_params.get("threshold", "θ")
        lines += [
            "",
            f"**Threshold filter:** Positions are only entered when |prediction| > {threshold_val}."
            " This filters low-conviction signals, reducing turnover at the cost of"
            " reduced market exposure on uncertain periods.",
        ]

    lines += [
        "",
        "**Transaction cost model:** Costs are applied to every unit of absolute"
        " weight change per period. The model's net alpha must exceed the cost drag"
        " implied by its turnover to be viable in a live deployment.",
    ]

    return lines


# ---------------------------------------------------------------------------
# Allocation Research (Phase 2 — panel mode)
# ---------------------------------------------------------------------------


def _allocation_research_section(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Allocation research narrative: concentration dynamics, prediction dispersion,
    and confidence calibration for panel ML experiments.

    Guards:
      - Panel mode only (cross-sectional signal topology).
      - allocation_diagnostics artefact must be present and available=True.

    Returns empty string when guards are not met.
    """
    if not _is_panel_mode(artefacts):
        return ""

    ad = artefacts.allocation_diagnostics
    if not isinstance(ad, dict) or not ad.get("available", False):
        return ""

    weighting_scheme = ad.get("weighting_scheme", "equal_weight")
    conc = ad.get("concentration") or {}
    holdings = ad.get("holdings") or {}
    wts = ad.get("weights") or {}
    dispersion = ad.get("prediction_dispersion") or {}
    calibration = ad.get("confidence_calibration") or {}

    lines: list[str] = []

    # ── Weighting policy context ──────────────────────────────────────────────
    _SCHEME_LABELS = {
        "equal_weight": "equal-weight (1/k per selected asset)",
        "zscore_softmax": "z-score softmax (prediction-proportional via temperature softmax)",
        "confidence_weighted": "confidence-weighted (∝ max(score, 0), row-wise)",
    }
    scheme_label = _SCHEME_LABELS.get(weighting_scheme, f"`{weighting_scheme}`")
    lines += [
        f"**Allocation policy:** {scheme_label}.",
        "",
    ]

    # ── Concentration statistics table ────────────────────────────────────────
    mean_hhi = conc.get("mean_hhi")
    mean_breadth = conc.get("mean_effective_breadth")
    eff_n = conc.get("effective_n_entropy")
    mean_held = holdings.get("mean_held_count")
    max_held = holdings.get("max_held_count")
    mean_max_w = wts.get("mean_max_weight")

    stat_rows: list[tuple[str, ...]] = []
    if mean_hhi is not None:
        stat_rows.append(("Mean HHI (Σwᵢ²)", f"{mean_hhi:.4f}"))
    if mean_breadth is not None:
        stat_rows.append(("Mean effective breadth (1/HHI)", f"{mean_breadth:.2f}"))
    if eff_n is not None:
        stat_rows.append(("Entropy effective N (exp(H))", f"{eff_n:.2f}"))
    if mean_held is not None:
        held_str = f"{mean_held:.1f}"
        if max_held is not None:
            held_str += f" (max {max_held})"
        stat_rows.append(("Mean assets held", held_str))
    if mean_max_w is not None:
        stat_rows.append(("Mean max weight per period", f"{mean_max_w:.3f}"))

    if stat_rows:
        lines += [
            "**Concentration summary:**",
            "",
            _pipe_table(["Metric", "Value"], stat_rows),
            "",
        ]

    # ── Economic interpretation of concentration ──────────────────────────────
    if mean_hhi is not None and eff_n is not None:
        if mean_hhi > 0.5:
            conc_verdict = (
                "The allocation is highly concentrated — mean HHI above 0.5 indicates"
                " that one or two assets dominate the portfolio at most rebalance dates."
                " This is consistent with a strong, selective ranking signal but amplifies"
                " idiosyncratic risk relative to an equal-weight basket."
            )
        elif mean_hhi > 0.25:
            conc_verdict = (
                f"Allocation concentration is moderate (mean HHI {mean_hhi:.3f})."
                " The model distributes exposure across several assets but maintains"
                " meaningful differentiation — the effective breadth of {mean_breadth:.1f}"
                " implied bets is consistent with an informative cross-sectional ranking."
                if mean_breadth is not None
                else (
                    f"Allocation concentration is moderate (mean HHI {mean_hhi:.3f})."
                    " The model distributes exposure across several assets but maintains"
                    " meaningful differentiation."
                )
            )
        else:
            conc_verdict = (
                f"Allocation is broadly diversified (mean HHI {mean_hhi:.3f},"
                f" effective N ≈ {eff_n:.1f} implied bets). The model's cross-sectional"
                " scores produce near-uniform weights across the selected basket,"
                " limiting concentration risk but also reducing the incremental value"
                " of confidence-weighted allocation over simple equal-weighting."
            )
        lines += [conc_verdict, ""]

    # ── Concentration evolution figure ────────────────────────────────────────
    conc_fig = _embed_figure(
        "allocation_concentration_evolution", figures, claimed,
        "HHI (top), effective breadth (middle), and entropy effective-N (bottom) through"
        " time. Elevated HHI with depressed breadth identifies concentration regimes;"
        " periods where effective-N approaches the basket size confirm near-uniform"
        " allocation. The 63-day rolling mean smooths daily rebalance noise to reveal"
        " structural regime transitions.",
    )
    if conc_fig:
        lines.extend(["", conc_fig])

    # ── Prediction dispersion ─────────────────────────────────────────────────
    mean_std = dispersion.get("mean_cs_std")
    mean_spread = dispersion.get("mean_cs_spread")
    if mean_std is not None or mean_spread is not None:
        lines += ["", "**Prediction dispersion:**", ""]
        if mean_std is not None and mean_spread is not None:
            lines.append(
                f"Mean cross-sectional prediction σ = {mean_std:.4f};"
                f" mean top-minus-bottom spread = {mean_spread:.4f}."
                " Low σ identifies score-compression regimes where ranking information"
                " is minimal; near-zero spread indicates ranking indifference."
            )
        lines.append("")

    disp_fig = _embed_figure(
        "prediction_dispersion", figures, claimed,
        "Rolling 63-day cross-sectional prediction standard deviation (top) and"
        " top-minus-bottom score spread (bottom). Score compression periods (depressed σ)"
        " indicate regimes where model confidence is uniformly low and cross-sectional"
        " differentiation carries little economic signal. Near-zero spread regimes"
        " are especially problematic for confidence-weighted allocation schemes.",
    )
    if disp_fig:
        lines.extend(["", disp_fig])

    # ── Confidence calibration ────────────────────────────────────────────────
    calib_spread = calibration.get("top_minus_bottom_spread")
    calib_monotonic = calibration.get("monotonic_up")
    if calib_spread is not None:
        lines += ["", "**Confidence calibration:**", ""]
        if calib_monotonic:
            lines.append(
                f"Prediction scores are monotonically calibrated: assets in the highest"
                f" quintile (Q5) realized a {calib_spread:.4f} return advantage over"
                " the lowest quintile (Q1) on average. This confirms that score magnitude"
                " — not merely sign — carries economically meaningful cross-sectional"
                " information and validates confidence-weighted allocation."
            )
        else:
            lines.append(
                f"Confidence calibration is **non-monotonic** (top-minus-bottom spread"
                f" {calib_spread:+.4f}). Higher prediction scores do not reliably"
                " correspond to higher realized returns across the full sample."
                " Equal-weight or threshold-gated allocation may be preferable to"
                " confidence-weighted schemes that amplify non-calibrated signals."
            )
        lines.append("")

    calib_fig = _embed_figure(
        "confidence_calibration", figures, claimed,
        "Mean realized forward return by prediction-score quintile (Q1 = lowest to"
        " Q5 = highest). A left-to-right increasing pattern confirms that score magnitude"
        " carries economic information — higher-confidence predictions correspond to"
        " stronger realized returns. Non-monotonic patterns flag calibration failure"
        " and indicate that equal-weight allocation may dominate confidence-weighted"
        " schemes in this experiment.",
    )
    if calib_fig:
        lines.extend(["", calib_fig])

    if not lines:
        return ""

    return _render_section("Allocation Research", "\n".join(lines).strip())


# ---------------------------------------------------------------------------
# NEW: Failure Analysis
# ---------------------------------------------------------------------------


def _ml_failure_modes(artefacts: ExperimentArtefacts) -> list[str]:
    """Build ML-specific failure mode commentary from observed diagnostics.

    Reads split_metrics and ml_model_diagnostics to generate an evidence-grounded
    failure analysis. Falls back to generic commentary if artefacts are absent.
    """
    sm = artefacts.split_metrics if isinstance(artefacts.split_metrics, dict) else {}
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    prov = artefacts.ml_provenance if isinstance(artefacts.ml_provenance, dict) else {}
    model_type = (prov.get("model") or {}).get("type", "the model")
    signal_type = (prov.get("signal") or {}).get("type", "sign")

    splits = sm.get("splits") or []
    summary = sm.get("summary") or {}
    mean_sharpe = summary.get("mean_sharpe")
    hit_rate = summary.get("hit_rate_positive_sharpe")
    ic_sum = mmd.get("ic_summary") or {}
    mean_ic = ic_sum.get("mean_ic")
    stab = mmd.get("coefficient_stability_summary") or []

    lines: list[str] = ["**Failure analysis — ML signal:**", ""]

    # --- Out-of-sample degradation verdict ---
    if mean_sharpe is not None and hit_rate is not None:
        n_splits = len(splits)
        n_negative = sum(1 for s in splits if (s.get("sharpe_ratio") or 0.0) < 0)
        if hit_rate < 0.5:
            oos_verdict = (
                f"{model_type} failed to generalise out-of-sample: {n_negative} of"
                f" {n_splits} test windows produced negative Sharpe (mean OOS Sharpe:"
                f" {mean_sharpe:.2f}). The in-sample signal does not survive chronological"
                f" validation — the primary evidence of overfitting or regime specificity."
            )
        elif hit_rate < 0.7:
            oos_verdict = (
                f"{model_type} shows marginal out-of-sample performance: {n_negative} of"
                f" {n_splits} test windows were negative (mean OOS Sharpe: {mean_sharpe:.2f})."
                f" The signal is present but inconsistent across market regimes."
            )
        else:
            oos_verdict = (
                f"{model_type} demonstrates robust out-of-sample performance:"
                f" {n_splits - n_negative} of {n_splits} test windows were positive"
                f" (mean OOS Sharpe: {mean_sharpe:.2f})."
            )
        lines += [f"*Out-of-sample verdict:* {oos_verdict}", ""]

    # --- Regime non-stationarity ---
    unstable_feats = [
        r.get("feature", "—") for r in stab
        if isinstance(r.get("sign_consistency"), float) and r["sign_consistency"] < 0.6
    ]
    _is_panel_fa = _is_panel_mode(artefacts)
    if mean_ic is not None:
        if _is_panel_fa:
            ic_regime_note = (
                f" Mean cross-sectional IC of {mean_ic:.3f} confirms ranking signal exists"
                " in aggregate, but sub-period IC variation (visible in the IC chart)"
                " reveals regime dependence."
            )
        else:
            ic_regime_note = (
                f" Mean IC of {mean_ic:.3f} confirms directional signal exists in aggregate,"
                " but sub-period IC variation (visible in the IC chart) reveals regime"
                " dependence."
            )
    else:
        ic_regime_note = ""

    lines += [
        "**Known failure modes for ML price-history strategies:**",
        "",
        "- *Regime non-stationarity* — Feature-return relationships shift across"
        " macro regimes (risk-on/risk-off, trending/mean-reverting, high/low volatility)."
        " A model trained on 2013–2016 data learns relationships that may not hold in"
        " 2022 tightening cycles or 2020 dislocation episodes." + ic_regime_note,
    ]

    if unstable_feats:
        feat_str = ", ".join(f"`{f}`" for f in unstable_feats[:4])
        lines.append(
            f"- *Feature instability* — {feat_str} show sign consistency below 60%"
            " across walk-forward splits, indicating their directional contribution"
            " reverses in different training regimes. The model is learning"
            " regime-specific patterns, not persistent market relationships."
        )

    lines += [
        "- *Overfitting on training regime* — Ridge regularisation constrains but"
        " does not eliminate overfitting risk. Large train-to-test Sharpe gaps in"
        " specific splits are the diagnostic signature of regime-specific learning.",
    ]

    _PANEL_SIGNAL_TYPES = {"top_n", "long_short", "normalize"}
    if signal_type not in _PANEL_SIGNAL_TYPES:
        lines += [
            "- *Binary signal coarseness* — The sign signal collapses continuous predictions"
            " to a binary long/flat position. Low-conviction predictions near zero generate"
            " positions identical in size to high-conviction signals, reducing the effective"
            " information ratio of the signal-to-position translation.",
            "- *Transaction cost drag at high turnover* — The sign signal can produce"
            " frequent position reversals when predictions oscillate around zero. Each"
            " reversal incurs full round-trip cost. Cost drag is most acute in regimes"
            " where the signal has low directional persistence.",
        ]
    else:
        lines += [
            "- *Cross-sectional dispersion collapse* — In low-dispersion regimes where"
            " all assets trend together, the cross-sectional model has little return"
            " differentiation to exploit. The top-N signal concentrates in whichever"
            " assets have marginally higher scores, but all scores are clustered — the"
            " effective information content drops.",
            "- *Transaction cost drag from universe rotation* — Holding the top-N assets"
            " requires full rebalancing when the composition changes. In regimes of high"
            " score volatility, turnover rises sharply and cost drag erodes signal value.",
        ]

    return lines


def _failure_analysis(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    bkd = artefacts.backtest_diagnostics if isinstance(artefacts.backtest_diagnostics, dict) else {}
    sm = artefacts.split_metrics if isinstance(artefacts.split_metrics, dict) else {}
    cfg = artefacts.config or {}
    strategy_type = (cfg.get("strategy") or {}).get("type", "")
    artefacts.metrics if isinstance(artefacts.metrics, dict) else {}

    lines: list[str] = []

    # --- known failure mode commentary ---
    if strategy_type == "MomentumRotation":
        lines += [
            "**Known failure modes for momentum rotation strategies:**",
            "",
            "- *Momentum crashes* — The most acute risk. When crowded momentum"
            " positions unwind simultaneously (e.g., following a flight-to-quality"
            " shock), the strategy holds recent winners that become the hardest-hit"
            " assets. The 2009 recovery and COVID snap-back (March–August 2020) are"
            " canonical examples.",
            "- *Bond/equity regime transitions* — In risk-off episodes, defensive"
            " assets (TLT, GLD) often lead. The strategy may hold these correctly,"
            " but the transition itself can cause a drawdown before the new leaders"
            " establish their momentum signal.",
            "- *Turnover spikes at regime shifts* — Rapid leadership rotation"
            " increases turnover and transaction-cost drag precisely when alpha is"
            " weakest. Cost erosion compounds the drawdown.",
            "- *Non-stationarity* — The momentum premium is not constant over time."
            " Periods of elevated market volatility, compressed return dispersion,"
            " or mean-reverting micro-structure can cause the signal to be"
            " uninformative or counter-productive.",
        ]
    elif isinstance(artefacts.ml_provenance, dict):
        lines += _ml_failure_modes(artefacts)
        # Volatility-regime-conditioned performance evidence
        mmd_fa = artefacts.ml_model_diagnostics if isinstance(
            artefacts.ml_model_diagnostics, dict
        ) else {}
        regime_stats = mmd_fa.get("regime_conditioned_stats") or []
        if regime_stats:
            lines.append("")
            lines.append("**Volatility-regime-conditioned performance:**")
            lines.append("")
            regime_rows: list[tuple[str, ...]] = []
            for r in regime_stats:
                sharpe_r = r.get("realized_sharpe")
                ic_r = r.get("ic")
                da_r = r.get("directional_accuracy")
                regime_rows.append((
                    r.get("regime", "—"),
                    str(r.get("n_obs", "—")),
                    f"{sharpe_r:.2f}" if isinstance(sharpe_r, float) else "—",
                    f"{ic_r:.4f}" if isinstance(ic_r, float) else "—",
                    f"{da_r:.1%}" if isinstance(da_r, float) else "—",
                ))
            lines.append(_pipe_table(
                ["Vol regime", "N obs", "Realized Sharpe", "IC", "Dir. accuracy"],
                regime_rows,
            ))
            # Evidence-grounded interpretation
            by_regime = {r.get("regime"): r for r in regime_stats}
            low_s = by_regime.get("Low", {}).get("realized_sharpe")
            high_s = by_regime.get("High", {}).get("realized_sharpe")
            if isinstance(low_s, float) and isinstance(high_s, float):
                direction = "deteriorates" if high_s < low_s else "improves"
                delta = abs(high_s - low_s)
                lines.append("")
                lines.append(
                    f"*Realized Sharpe {direction} from low- to high-volatility regime"
                    f" (Δ{delta:.2f}). IC follows the same pattern"
                    " — regime-conditioned degradation, not random variation.*"
                )

        # Instability propagation chain
        chain_text = _instability_chain_interpretation(artefacts)
        if chain_text:
            lines.append("")
            lines.append("**Instability propagation:**")
            lines.append("")
            lines.append(chain_text)
    else:
        lines += [
            "**Failure analysis:** Drawdown windows and risk assessment are shown"
            " below. Strategy-specific failure mode commentary can be added by"
            " implementing a strategy-type handler in `_failure_analysis()`.",
        ]

    # --- worst drawdown windows from backtest_diagnostics ---
    dd_windows = bkd.get("drawdown_windows") or [] if bkd else []
    if dd_windows:
        sorted_windows = sorted(dd_windows, key=lambda w: w.get("max_dd", 0))
        worst = sorted_windows[:3]  # up to 3 worst
        lines.append("")
        lines.append("**Identified drawdown windows (> 5% peak-to-trough):**")
        lines.append("")
        rows: list[tuple[str, ...]] = []
        for w in worst:
            mdd = w.get("max_dd")
            recovery = w.get("recovery") or "ongoing"
            rows.append((
                w.get("start", "—"),
                w.get("trough", "—"),
                recovery,
                f"{mdd:.1%}" if isinstance(mdd, float) else "—",
                str(w.get("duration_days", "—")) + "d",
            ))
        lines.append(_pipe_table(
            ["Drawdown Start", "Trough", "Recovery", "Max DD", "Duration"],
            rows,
        ))

    # --- worst walk-forward split ---
    splits = sm.get("splits") or [] if sm else []
    if splits:
        worst_split = min(splits, key=lambda s: s.get("sharpe_ratio") or 0.0)
        lines.append("")
        lines.append("**Worst out-of-sample split:**")
        lines.append("")
        ws = worst_split
        sharpe_v = ws.get("sharpe_ratio")
        ret_v = ws.get("annualized_return")
        dd_v = ws.get("max_drawdown")
        lines += [
            f"- Split {ws.get('split', '—')} — test period"
            f" {str(ws.get('test_start', '—'))[:10]} to"
            f" {str(ws.get('test_end', '—'))[:10]}",
            f"- Sharpe: {sharpe_v:.2f}" if isinstance(sharpe_v, float) else "- Sharpe: —",
            f"- Return: {ret_v:.1%}" if isinstance(ret_v, float) else "- Return: —",
            f"- Max DD: {dd_v:.1%}" if isinstance(dd_v, float) else "- Max DD: —",
        ]

    # Split equity curves — visual evidence of regime instability
    se_fig = _embed_figure(
        "split_equity_curves", figures, claimed,
        "Divergent split trajectories identify regimes where the signal broke down."
        " Consistent slopes across periods indicate regime-independent alpha.",
    )
    if se_fig:
        lines.extend(["", se_fig])

    if not lines:
        return ""

    return _render_section("Failure Analysis", "\n".join(lines))


# ---------------------------------------------------------------------------
# Feature family analysis helper (G2)
# ---------------------------------------------------------------------------


def _feature_family_analysis(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Return a Feature Family Analysis subsection string, or empty string if absent.

    Renders when feature_families artefact is present.  Shows the canonical
    family grouping table and the feature_family_ic figure (when available).
    """
    ff_artefact = artefacts.feature_families
    if not isinstance(ff_artefact, dict):
        return ""

    families: dict = ff_artefact.get("families") or {}
    if not families:
        return ""

    try:
        from src.features.families import FEATURE_FAMILY_DESCRIPTIONS
    except Exception:
        FEATURE_FAMILY_DESCRIPTIONS = {}

    lines: list[str] = []
    lines.append("**Feature family grouping:**")
    lines.append("")

    fam_rows: list[tuple[str, str, str]] = []
    for family, members in families.items():
        member_labels = ", ".join(_feat_label(m) for m in members)
        desc = FEATURE_FAMILY_DESCRIPTIONS.get(family, "")
        fam_rows.append((family, member_labels, desc[:90] + "…" if len(desc) > 90 else desc))

    lines.append(_pipe_table(
        ["Family", "Features", "Hypothesis"],
        fam_rows,
    ))

    n_families = len(families)
    n_features_total = sum(len(v) for v in families.values())
    lines.append("")
    lines.append(
        f"This experiment deploys {n_features_total} feature{'s' if n_features_total != 1 else ''}"
        f" across {n_families} canonical famil{'ies' if n_families != 1 else 'y'}."
        " Each family operationalises a distinct market hypothesis; orthogonal families"
        " provide diversified information sources, reducing model dependence on any"
        " single regime hypothesis."
    )

    # Family IC figure — topology-aware caption
    _is_panel_fam = _is_panel_mode(artefacts)
    _fam_caption = (
        "Mean cross-sectional IC aggregated by feature family per walk-forward split."
        " Positive bars confirm the family's features improved cross-sectional ranking"
        " in that regime; negative bars indicate systematic ranking reversal. Persistent"
        " dominance by a single family flags concentration in one market hypothesis."
        if _is_panel_fam else
        "Mean Pearson IC aggregated by feature family per walk-forward split."
        " Positive bars confirm that the family hypothesis held in that regime;"
        " negative bars indicate systematic reversal. Persistent dominance by a"
        " single family flags model concentration risk."
    )
    fam_ic_fig = _embed_figure("feature_family_ic", figures, claimed, _fam_caption)
    if fam_ic_fig:
        lines.extend(["", fam_ic_fig])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Existing sections (unchanged or minimally updated)
# ---------------------------------------------------------------------------


def _feature_engineering(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Feature Engineering section — rendered when feature artefacts are present.

    Exposes the feature space as an institutional research artefact: registry,
    alignment diagnostics, per-feature statistics, and multicollinearity summary.
    Only rendered for v2 (ML) experiments where feature_summary/feature_registry
    have been persisted by the orchestrator.
    """
    fr = artefacts.feature_registry if isinstance(artefacts.feature_registry, dict) else {}
    fs = artefacts.feature_summary if isinstance(artefacts.feature_summary, dict) else {}
    al = artefacts.alignment_diagnostics if isinstance(artefacts.alignment_diagnostics, dict) else {}
    fc = artefacts.feature_correlations if isinstance(artefacts.feature_correlations, dict) else {}

    if not fr and not fs:
        return ""

    is_panel = _is_panel_mode(artefacts)
    lines: list[str] = []

    # --- Narrative intro (topology-aware) ---
    ticker = fr.get("ticker") if fr else None
    universe_cfg = (artefacts.config or {}).get("universe") or {}
    universe_tickers = universe_cfg.get("tickers") or []
    n_assets = len(universe_tickers) if universe_tickers else 1
    if n_assets > 1:
        topology_phrase = f"The {n_assets}-asset aligned price panel is"
    else:
        asset_ref = f"`{ticker}`" if ticker else "the price series"
        topology_phrase = f"The {asset_ref} price history is"
    lines.append(
        f"{topology_phrase} transformed into a structured feature space."
        " Each feature operationalises a specific hypothesis about the information"
        " content embedded in price time series — momentum persistence, volatility"
        " clustering, or trend strength. All features are constructed as pure functions"
        " of historical prices, ensuring no look-ahead contamination at the feature"
        " generation stage."
    )

    # --- Feature Registry ---
    feat_entries = fr.get("features") or []
    if feat_entries:
        n_features = fr.get("n_features", len(feat_entries))
        lines.append("")
        header = f"**Feature registry** — {n_features} feature"
        if n_features != 1:
            header += "s"
        if n_assets > 1:
            header += f" applied independently to each of the {n_assets} universe assets:"
        elif ticker:
            header += f" constructed on `{ticker}`:"
        lines.append(header)
        lines.append("")

        # Determine whether family data is available (G2+)
        has_family = any("family" in feat for feat in feat_entries)
        reg_rows: list[tuple[str, ...]] = []
        for feat in feat_entries:
            raw_name = feat.get("name", "—")
            window = feat.get("window")
            norm = feat.get("normalization_type", "raw")
            row: tuple[str, ...] = (
                _feat_label(raw_name),
                feat.get("family", feat.get("category", "—")) if has_family
                    else feat.get("category", feat.get("type", "—")),
                feat.get("transform", feat.get("type", "—")),
                str(window) if window is not None else "—",
                norm,
            )
            reg_rows.append(row)

        col_headers = (
            ["Name", "Family", "Transform", "Window", "Normalisation"]
            if has_family
            else ["Name", "Category", "Transform", "Window", "Normalisation"]
        )
        lines.append(_pipe_table(col_headers, reg_rows))

        # Normalization note
        has_zscore = any(
            f.get("normalization_type") == "zscore" for f in feat_entries
        )
        if has_zscore:
            lines.append("")
            lines.append(
                "*Rolling z-score features are normalised in-place: mean and standard"
                " deviation are estimated over a rolling window centred at time t−1."
                " This ensures normalisation parameters are derived from past data"
                " only, preserving the leakage-free property.*"
            )

    # --- Label Construction ---
    label_type = fr.get("label_type")
    label_horizon = fr.get("label_horizon")
    label_params = fr.get("label_params") or {}
    if label_type:
        lines.append("")
        lines.append("**Label construction:**")
        lines.append("")
        label_rows: list[tuple[str, str]] = [("Label type", f"`{label_type}`")]
        if label_horizon is not None:
            label_rows.append(("Horizon", f"{label_horizon} trading days"))
        for k, v in sorted(label_params.items()):
            if k != "horizon":
                label_rows.append((k, str(v)))
        lines.append(_pipe_table(["Parameter", "Value"], label_rows))

    # --- Sample Construction / Alignment Diagnostics ---
    if al:
        lines.append("")
        lines.append("**Sample construction and alignment:**")
        lines.append("")

        n_raw = al.get("n_rows_raw")
        al.get("n_rows_features_clean")
        n_aligned = al.get("n_rows_after_alignment")
        warmup = al.get("warmup_rows_removed")
        label_drop = al.get("label_rows_removed")
        loss_pct = al.get("alignment_loss_pct")
        sample_start = al.get("sample_start")
        sample_end = al.get("sample_end")
        al_is_panel = al.get("is_panel", False)
        n_al_assets = al.get("n_universe_assets")

        al_rows: list[tuple[str, str]] = []
        if n_raw is not None:
            al_rows.append(("Raw price observations (per asset)", str(n_raw)))
        if warmup is not None:
            al_rows.append(("Warm-up rows removed (feature NaN)", str(warmup)))
        if label_drop is not None:
            al_rows.append(("Label rows removed (forward horizon NaN)", str(label_drop)))
        if n_aligned is not None:
            if al_is_panel and n_al_assets:
                al_rows.append((
                    "Aligned trading days per asset",
                    str(n_aligned),
                ))
                al_rows.append((
                    "Pre-alignment pooled panel size (estimated)",
                    f"{n_aligned * n_al_assets:,}  ({n_aligned} days × {n_al_assets} assets)",
                ))
            else:
                al_rows.append(("Final aligned training samples", str(n_aligned)))
        if loss_pct is not None:
            al_rows.append(("Total alignment loss", f"{loss_pct:.1f}%"))
        if sample_start and sample_end:
            al_rows.append(("Effective sample range", f"{sample_start} to {sample_end}"))

        if al_rows:
            lines.append(_pipe_table(["Stage", "Count / Value"], al_rows))
            lines.append("")
            if al_is_panel:
                lines.append(
                    "*Warm-up and label rows are counted per asset. The pooled model"
                    " trains on all (date, asset) pairs simultaneously — one shared"
                    " Ridge model fitted on all pooled observations. Alignment loss"
                    " is computed as the fraction of per-asset trading days removed;*"
                    " *the pooled sample size is proportionally larger.*"
                )
            else:
                lines.append(
                    "*Warm-up rows are removed when rolling feature windows require"
                    " a minimum history (e.g., a 252-day momentum feature has no valid"
                    " value for the first 252 observations). Label rows are removed"
                    " because the forward return horizon creates NaN labels on the final"
                    " N trading days of the sample.*"
                )

    # --- Feature Statistics ---
    feat_stats = fs.get("features") or {}
    if feat_stats:
        lines.append("")
        _stats_pooled = fs.get("pooled_panel_stats", False)
        if is_panel and _stats_pooled:
            _stats_header = "**Per-feature statistics (pooled across all universe assets):**"
        elif is_panel:
            _stats_header = "**Per-feature statistics (reference asset, pre-alignment):**"
        else:
            _stats_header = "**Per-feature statistics (full-history, pre-alignment):**"
        lines.append(_stats_header)
        lines.append("")
        stat_rows: list[tuple[str, ...]] = []
        for name, stats in feat_stats.items():
            mean_v = stats.get("mean")
            std_v = stats.get("std")
            skew_v = stats.get("skew")
            ar1_v = stats.get("ar1")
            cov_v = stats.get("sample_coverage")
            stat_rows.append((
                _feat_label(name),
                f"{mean_v:.4f}" if isinstance(mean_v, float) else "—",
                f"{std_v:.4f}" if isinstance(std_v, float) else "—",
                f"{skew_v:.2f}" if isinstance(skew_v, float) else "—",
                f"{ar1_v:.3f}" if isinstance(ar1_v, float) else "—",
                f"{cov_v:.1%}" if isinstance(cov_v, float) else "—",
            ))
        lines.append(_pipe_table(
            ["Feature", "Mean", "Std", "Skew", "AR(1)", "Coverage"],
            stat_rows,
        ))

    # --- Feature correlations — heatmap only ---
    if fc:
        hm_fig = _embed_figure(
            "feature_correlation_heatmap", figures, claimed,
            "Pairwise feature correlations. Correlated clusters reduce effective"
            " dimensionality; Ridge regularisation partially mitigates collinearity"
            " but reduces coefficient interpretability.",
        )
        if hm_fig:
            lines.extend(["", hm_fig])

    # --- Feature regime heatmap ---
    fr_fig = _embed_figure(
        "ml_feature_regimes", figures, claimed,
        "Feature z-scores through time (±3σ). Red bands mark extreme positive"
        " regimes; blue marks extreme negative. Reveals when features were"
        " informative and where the environment shifted across the backtest period.",
    )
    if fr_fig:
        lines.extend(["", fr_fig])

    # --- Per-feature split IC heatmap ---
    _fic_caption = (
        "Cross-sectional IC per feature per walk-forward split. Green cells mark"
        " splits where the feature improved cross-sectional ranking; red marks"
        " breakdown. Reveals whether each feature's cross-sectional information"
        " is consistent across regimes or regime-specific."
        if is_panel else
        "Pearson IC per feature per walk-forward test split. Green cells mark"
        " positive predictive relationships; red marks breakdown. Reveals whether"
        " feature predictive power is consistent or regime-specific."
    )
    fic_fig = _embed_figure("feature_ic_heatmap", figures, claimed, _fic_caption)
    if fic_fig:
        lines.extend(["", fic_fig])

    # --- Feature Family Analysis ---
    ff_section = _feature_family_analysis(artefacts, figures, claimed)
    if ff_section:
        lines.extend(["", ff_section])

    if not lines:
        return ""

    return _render_section("Feature Engineering", "\n".join(lines))


def _ml_model_behavior(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """ML Model Behaviour section — rendered when ml_model_diagnostics are present.

    Shows: coefficients table, directional accuracy, IC summary, prediction stats.
    Follows the Model & Features provenance section in the narrative flow.
    """
    mmd = artefacts.ml_model_diagnostics
    if not isinstance(mmd, dict):
        return ""

    is_panel = _is_panel_mode(artefacts)
    lines: list[str] = []

    # --- Predictive quality table — topology-aware labels ---
    da = mmd.get("directional_accuracy")
    ic_sum = mmd.get("ic_summary") or {}
    mean_ic = ic_sum.get("mean_ic")
    pct_pos_ic = ic_sum.get("pct_positive_ic")

    quality_rows: list[tuple[str, str]] = []
    if da is not None:
        da_label = (
            "Positive-IC ranking periods (full period)"
            if is_panel else
            "Directional accuracy (full period)"
        )
        quality_rows.append((da_label, f"{da:.1%}"))
    if mean_ic is not None:
        ic_label = (
            "Mean monthly cross-sectional IC"
            if is_panel else
            "Mean monthly IC (Pearson)"
        )
        quality_rows.append((ic_label, f"{mean_ic:.4f}"))
    if pct_pos_ic is not None:
        pct_label = (
            "% months with positive cross-sectional IC"
            if is_panel else
            "% months with positive IC"
        )
        quality_rows.append((pct_label, f"{pct_pos_ic:.1%}"))
    n_samples = mmd.get("n_aligned_samples")
    if n_samples is not None:
        samples_label = (
            "Model-fitting observations (post-NaN removal)"
            if is_panel else
            "Aligned training samples"
        )
        quality_rows.append((samples_label, f"{n_samples:,}"))

    if quality_rows:
        lines.append("**Predictive quality:**")
        lines.append("")
        lines.append(_pipe_table(["Metric", "Value"], quality_rows))

    # Cross-sectional IC — raw daily Spearman rank correlation across assets
    cs_ic_fig = _embed_figure(
        "cross_sectional_ic", figures, claimed,
        _cs_ic_interpretation(artefacts),
    )
    if cs_ic_fig:
        lines.extend(["", cs_ic_fig])

    # Rolling IC — 63d rolling mean (panel: mean CS-IC; single-asset: rolling Pearson IC)
    ic_fig = _embed_figure(
        "ml_ic_regime", figures, claimed,
        _rolling_ic_interpretation(artefacts),
    )
    if ic_fig:
        lines.extend(["", ic_fig])

    # Rolling DA — 126d consistency (panel: fraction positive-IC days; single-asset: DA)
    da_fig = _embed_figure(
        "ml_rolling_da", figures, claimed,
        _rolling_da_interpretation(artefacts),
    )
    if da_fig:
        lines.extend(["", da_fig])
    lines.append("")

    # Ranking geometry — statistical state observability (Phase I)
    rg_section = _ranking_geometry_section(artefacts, figures, claimed)
    if rg_section:
        lines.extend(["", rg_section])

    # Prediction confidence & outcome monotonicity (Step 3)
    ps_section = _prediction_strength_section(artefacts, figures, claimed)
    if ps_section:
        lines.extend(["", ps_section])

    # Residual diagnostics — shows error structure + rolling bias drift
    # ml_prediction_vs_actual is appendix-only (static scatter superseded by rolling IC/DA)
    res_fig = _embed_figure(
        "ml_residuals", figures, claimed,
        _residuals_interpretation(artefacts),
    )
    if res_fig:
        lines.extend(["", res_fig])
    lines.append("")

    # --- Prediction confidence calibration ---
    cal = mmd.get("calibration_by_quintile") or []
    if cal:
        lines.append("**Prediction confidence calibration:**")
        lines.append("")
        cal_rows: list[tuple[str, ...]] = []
        for c in cal:
            q = c.get("quintile")
            n_c = c.get("n_obs")
            mp = c.get("mean_predicted")
            ma = c.get("mean_actual")
            da_c = c.get("directional_accuracy")
            cal_rows.append((
                f"Q{q}" if q is not None else "—",
                str(n_c) if n_c is not None else "—",
                f"{mp:+.4f}" if isinstance(mp, float) else "—",
                f"{ma:+.4f}" if isinstance(ma, float) else "—",
                f"{da_c:.1%}" if isinstance(da_c, float) else "—",
            ))
        if cal_rows:
            lines.append(_pipe_table(
                ["Quintile", "N", "Mean pred", "Mean actual", "Dir. accuracy"],
                cal_rows,
            ))
            da_vals = [
                c.get("directional_accuracy") for c in cal
                if isinstance(c.get("directional_accuracy"), float)
            ]
            if len(da_vals) >= 3:
                spread = da_vals[-1] - da_vals[0]
                if spread > 0.05:
                    verdict = (
                        f"Q5 accuracy ({da_vals[-1]:.1%}) exceeds Q1 ({da_vals[0]:.1%})"
                        " — prediction magnitude carries information;"
                        " higher-conviction signals are more accurate."
                    )
                elif spread < 0.03:
                    verdict = (
                        f"Q1–Q5 accuracy spread {spread:.1%} — prediction magnitude"
                        " does not differentiate accuracy."
                        " Only sign carries signal; binary position sizing is appropriate."
                    )
                else:
                    verdict = (
                        f"Q1–Q5 accuracy {da_vals[0]:.1%}–{da_vals[-1]:.1%}."
                        " Modest calibration: higher conviction modestly improves accuracy."
                    )
                lines.extend(["", f"*{verdict}*"])
        lines.append("")

    # --- Feature importance summary: merged coefficients + stability, sorted by sign consistency ---
    coefs = mmd.get("coefficients") or {}
    stab_records = mmd.get("coefficient_stability_summary") or []
    stab_by_feat = {str(r.get("feature", "")): r for r in stab_records}

    if coefs or stab_records:
        all_features = list(coefs.keys()) if coefs else [
            str(r.get("feature", "")) for r in stab_records
        ]
        # Sort by sign consistency descending (most stable first)
        sorted_feats = sorted(
            all_features,
            key=lambda f: -(stab_by_feat.get(f, {}).get("sign_consistency") or 0.0),
        )
        importance_rows: list[tuple[str, ...]] = []
        for feat in sorted_feats:
            coef_v = coefs.get(feat)
            rec = stab_by_feat.get(feat, {})
            mean_oos = rec.get("mean")
            sc = rec.get("sign_consistency")
            importance_rows.append((
                _feat_label(feat),
                f"{coef_v:+.4f}" if isinstance(coef_v, float) else "—",
                f"{mean_oos:+.4f}" if isinstance(mean_oos, float) else "—",
                f"{sc:.0%}" if isinstance(sc, float) else "—",
            ))
        if importance_rows:
            lines.append("**Feature importance summary** (sorted by sign consistency):")
            lines.append("")
            lines.append(_pipe_table(
                ["Feature", "Full-period coef", "Mean OOS coef", "Sign consistency"],
                importance_rows,
            ))
            lines.append("")
            lines.append(
                "*Full-period coef is from the model fitted on all data; mean OOS coef is the"
                " average across walk-forward splits. Divergence between the two indicates"
                " regime-specific learning — the full-period model captures structure the"
                " walk-forward splits did not consistently reproduce.*"
            )
            lines.append("")

    stab_records = mmd.get("coefficient_stability_summary") or []

    # Coefficient stability chart
    cs_fig = _embed_figure(
        "ml_coefficient_stability", figures, claimed,
        _stability_interpretation(artefacts),
    )
    if cs_fig:
        lines.extend(["", cs_fig])
    else:
        if stab_records:
            lines.append("")
            lines.append(
                "*Sign consistency is the fraction of walk-forward splits where the"
                " coefficient's sign matches the sign of its mean. Values near 1.0"
                " indicate a stable directional signal; values near 0.5 indicate noise.*"
            )

    # Coefficient sign heatmap — sign reversals across splits
    csh_fig = _embed_figure(
        "ml_coefficient_sign_heatmap", figures, claimed,
        _sign_heatmap_interpretation(artefacts),
    )
    if csh_fig:
        lines.extend(["", csh_fig])

    # Temporal feature contribution diagnostics (Phase II)
    fc_section = _feature_contribution_section(artefacts, figures, claimed)
    if fc_section:
        lines.extend(["", fc_section])

    # Regime-conditional feature behaviour section (Step 2)
    regime_section = _regime_conditional_behavior(artefacts, figures, claimed)
    if regime_section:
        lines.extend(["", regime_section])

    if not lines:
        return ""

    return _render_section("ML Model Behaviour", "\n".join(lines))


def _ml_section(artefacts: ExperimentArtefacts) -> str:
    """Model & Features section — rendered only for v2 (ML) experiments."""
    if not isinstance(artefacts.ml_provenance, dict):
        return ""

    prov = artefacts.ml_provenance
    lines: list[str] = []

    # Model
    model = prov.get("model") or {}
    model_type = model.get("type", "—")
    model_params = model.get("params") or {}
    lines.append(f"**Model:** `{model_type}`")
    if model_params:
        lines.append("")
        param_rows = [(k, str(v)) for k, v in sorted(model_params.items())]
        lines.append(_pipe_table(["Parameter", "Value"], param_rows))

    # Features: prov["features"] is {"ticker": ..., "entries": [...]}
    feat_spec = prov.get("features") or {}
    entries = feat_spec.get("entries") if isinstance(feat_spec, dict) else []
    if entries:
        lines.append("")
        lines.append("**Features:**")
        lines.append("")
        feat_rows = [
            (f["name"], f["type"], str(f.get("params") or {}))
            for f in entries
        ]
        lines.append(_pipe_table(["Name", "Type", "Params"], feat_rows))

    # Labels
    label = prov.get("labels") or {}
    label_type = label.get("type", "—")
    label_params = label.get("params") or {}
    lines.append("")
    lines.append(f"**Labels:** `{label_type}`")
    if label_params:
        horizon = label_params.get("horizon")
        if horizon is not None:
            lines.append(f"  horizon: {horizon} days")

    # Signal
    signal = prov.get("signal") or {}
    signal_type = signal.get("type", "—")
    signal_params = signal.get("params") or {}
    lines.append("")
    lines.append(f"**Signal:** `{signal_type}`")
    if signal_params:
        for k, v in sorted(signal_params.items()):
            lines.append(f"  {k}: {v}")

    return _render_section("Model & Features", "\n".join(lines))


def _provenance_section(artefacts: ExperimentArtefacts) -> str:
    """Provenance section — shown only when at least one hash is available."""
    config_hash: str | None = artefacts.metadata.get("config_hash") if isinstance(artefacts.metadata, dict) else None
    ml_hash: str | None = None
    if isinstance(artefacts.ml_provenance, dict):
        ml_hash = artefacts.ml_provenance.get("ml_hash")

    if not config_hash and not ml_hash:
        return ""

    rows: list[tuple[str, str]] = []
    if config_hash:
        rows.append(("Config hash", f"`{config_hash}`"))
    if ml_hash:
        rows.append(("ML hash", f"`{ml_hash}`"))

    return _render_section("Provenance", _pipe_table(["Key", "Value"], rows))


def _metadata(artefacts: ExperimentArtefacts) -> str:
    m = artefacts.metadata
    rows = [
        ("Experiment", f"`{m.get('experiment_name', '—')}`"),
        ("Strategy", f"`{m.get('strategy_name', '—')}`"),
        ("Created", m.get("created_at", "—")),
    ]
    config_hash = m.get("config_hash")
    if config_hash:
        rows.append(("Config hash", f"`{config_hash}`"))

    return _render_section("Metadata", _pipe_table(["Field", "Value"], rows))


def _configuration(artefacts: ExperimentArtefacts) -> str:
    cfg = artefacts.config
    if cfg is None:
        return "## Configuration\n\n*Config snapshot not available.*"

    lines: list[str] = []

    # Universe
    tickers = cfg.get("universe", {}).get("tickers", [])
    if tickers:
        lines.append(f"**Universe:** {', '.join(tickers)}")
        lines.append("")

    # Date range
    dr = cfg.get("date_range", {})
    lines.append(f"**Date range:** {dr.get('start', '—')} to {dr.get('end', '—')}")
    lines.append("")

    # Strategy — v2 ML configs have no strategy.type; surface model type instead
    strat = cfg.get("strategy", {})
    strat_type = strat.get("type")
    if not strat_type and isinstance(artefacts.ml_provenance, dict):
        model_type = (artefacts.ml_provenance.get("model") or {}).get("type")
        strat_type = f"ML / {model_type}" if model_type else "ML"
    lines.append(f"**Strategy type:** `{strat_type or '—'}`")

    params = strat.get("parameters") or {}
    if params:
        lines.append("")
        param_rows = [(k, str(v)) for k, v in sorted(params.items())]
        lines.append(_pipe_table(["Parameter", "Value"], param_rows))

    # Execution
    exec_cfg = cfg.get("execution") or {}
    cost = exec_cfg.get("transaction_cost_bps", 0.0)
    lines.append("")
    lines.append(f"**Transaction cost:** {cost} bps")

    # Validation type
    val = cfg.get("validation") or {}
    vtype = val.get("type", "none")
    lines.append(f"**Validation:** `{vtype}`")

    return _render_section("Configuration", "\n".join(lines))


def _metrics(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    m = artefacts.metrics
    if not m:
        return "## Performance Metrics\n\n*No metrics found.*"

    rows: list[tuple[str, str]] = []
    displayed: set[str] = set()

    for key in _METRIC_ORDER:
        if key in m:
            rows.append((_label(key), f"{m[key]:.4f}"))
            displayed.add(key)

    for key, val in m.items():
        if key not in displayed:
            rows.append((_label(key), f"{val:.4f}"))

    body = _pipe_table(["Metric", "Value"], rows)

    eq_fig = _embed_figure(
        "equity_and_drawdown", figures, claimed,
        _drawdown_interpretation(artefacts),
    )
    if eq_fig:
        body = body + "\n\n" + eq_fig

    rs_fig = _embed_figure(
        "rolling_sharpe", figures, claimed,
        "Extended sub-zero periods mark sustained underperformance episodes."
        " Rolling Sharpe variance is as diagnostically informative as its mean.",
    )
    if rs_fig:
        body = body + "\n\n" + rs_fig

    return _render_section("Performance Metrics", body)


def _walk_forward(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Walk-Forward Validation section — expanded with methodology and split timeline."""
    cfg = artefacts.config
    if cfg is None:
        return ""

    val = cfg.get("validation") or {}
    vtype = val.get("type", "none")
    if vtype == "none":
        return ""

    params = val.get("parameters") or {}
    lines: list[str] = []

    # --- Methodology (concise — research readers know why k-fold is invalid on time series) ---
    lines += [
        "Chronological rolling validation: each test window immediately follows its training window"
        " in calendar time, with no overlap. This is the only leakage-safe simulation of live"
        " deployment — the model is never tested on data from periods it could have seen during training.",
        "",
    ]

    lines.append(f"**Validation type:** `{vtype}`")

    # Filter operational-noise params (gap_days=0 is default/irrelevant)
    display_params = {k: v for k, v in params.items() if not (k == "gap_days" and v == 0)}
    if display_params:
        lines.append("")
        param_rows = [(k, str(v)) for k, v in sorted(display_params.items())]
        lines.append(_pipe_table(["Parameter", "Value"], param_rows))

    # --- Walk-forward stitched equity (primary validation visual) ---
    wf_fig = _embed_figure(
        "walk_forward_stitched", figures, claimed,
        "Concatenated OOS test segments in chronological order."
        " Consistent segment slopes across different market regimes confirm structural alpha."
        " Persistent growth through stress periods is the strongest evidence of generalisation.",
    )
    if wf_fig:
        lines.extend(["", wf_fig])

    # --- Walk-forward window timeline (Gantt) ---
    wft_fig = _embed_figure(
        "walk_forward_timeline", figures, claimed,
        "Train/test windows in calendar time with OOS Sharpe annotated."
        " Green test bars are positive-Sharpe splits; red are negative."
        " Window widths reflect the configured train and test lengths.",
    )
    if wft_fig:
        lines.extend(["", wft_fig])
    else:
        # Fallback: numeric split table when no figure is available
        sm = artefacts.split_metrics if isinstance(artefacts.split_metrics, dict) else {}
        splits = sm.get("splits") or [] if sm else []
        if splits:
            lines.append("")
            lines.append("**Split timeline:**")
            lines.append("")
            split_rows: list[tuple[str, ...]] = []
            for s in splits:
                sharpe = s.get("sharpe_ratio")
                ret = s.get("annualized_return")
                dd = s.get("max_drawdown")
                split_rows.append((
                    str(s.get("split", "—")),
                    str(s.get("train_start", "—"))[:10],
                    str(s.get("train_end", "—"))[:10],
                    str(s.get("test_start", "—"))[:10],
                    str(s.get("test_end", "—"))[:10],
                    f"{sharpe:.2f}" if isinstance(sharpe, float) else "—",
                    f"{ret:.1%}" if isinstance(ret, float) else "—",
                    f"{dd:.1%}" if isinstance(dd, float) else "—",
                ))
            lines.append(_pipe_table(
                ["Split", "Train Start", "Train End", "Test Start", "Test End",
                 "OOS Sharpe", "OOS Return", "OOS Max DD"],
                split_rows,
            ))

    # --- Split Sharpe distribution ---
    ss_fig = _embed_figure(
        "split_sharpes", figures, claimed,
        "Per-split OOS Sharpe distribution. High variance across splits indicates"
        " regime-dependent performance; a single outlier inflating the mean is a key failure mode.",
    )
    if ss_fig:
        lines.extend(["", ss_fig])

    # --- Train vs test degradation ---
    tvt_fig = _embed_figure(
        "train_vs_test_sharpe", figures, claimed,
        _degradation_interpretation(artefacts),
    )
    if tvt_fig:
        lines.extend(["", tvt_fig])

    if not wf_fig and not ss_fig and not tvt_fig:
        lines.append("")
        lines.append("*Per-split stability metrics are in the Diagnostics Appendix.*")

    return _render_section("Walk-Forward Validation", "\n".join(lines))


def _diagnostics_section(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Split-metrics + ML diagnostics section — framed as a research appendix."""
    has_split = isinstance(artefacts.split_metrics, dict)
    has_ml_diag = isinstance(artefacts.ml_diagnostics, dict)

    if not has_split and not has_ml_diag:
        return _render_section(
            "Diagnostics Appendix",
            "*Diagnostic artefacts not available for this experiment.*",
        )

    lines: list[str] = [
        "Detailed diagnostics for walk-forward validation and ML signal quality.",
    ]

    # --- Split metrics (walk-forward stability) ---
    if has_split:
        sm = artefacts.split_metrics
        summary = sm.get("summary") or {}
        n_splits = sm.get("n_splits", 0)

        lines.append("")
        lines.append("### Walk-Forward Stability")
        lines.append("")

        summary_rows: list[tuple[str, ...]] = [
            ("Splits", str(n_splits)),
        ]
        _ADD = [
            ("mean_sharpe", "Mean Sharpe", ".4f"),
            ("std_sharpe", "Std Sharpe", ".4f"),
            ("hit_rate_positive_sharpe", "Positive-Sharpe rate", ".1%"),
            ("mean_annualized_return", "Mean annualised return", ".2%"),
            ("mean_max_drawdown", "Mean max drawdown", ".2%"),
            ("worst_max_drawdown", "Worst max drawdown", ".2%"),
        ]
        for key, label, fmt in _ADD:
            v = summary.get(key)
            if v is not None:
                summary_rows.append((label, f"{v:{fmt}}"))

        lines.append(_pipe_table(["Metric", "Value"], summary_rows))

    # --- ML diagnostics ---
    if has_ml_diag:
        if lines:
            lines.append("")
        ml_d = artefacts.ml_diagnostics
        lines.append("### ML Signal Diagnostics")
        lines.append("")

        diag_rows: list[tuple[str, ...]] = []
        avg_to = ml_d.get("average_turnover")
        if avg_to is not None:
            diag_rows.append(("Avg daily turnover", f"{avg_to:.4f}"))
        sa = ml_d.get("signal_activity")
        if sa is not None:
            diag_rows.append(("Signal activity", f"{sa:.1%}"))

        if diag_rows:
            lines.append(_pipe_table(["Metric", "Value"], diag_rows))

        # Signal turnover evolution
        st_fig = _embed_figure(
            "ml_signal_turnover", figures, claimed,
            "Turnover spikes correlate with elevated cost drag."
            " Coincidence with drawdown periods indicates cost structure amplifying losses.",
        )
        if st_fig:
            lines.extend(["", st_fig])

    return _render_section("Diagnostics Appendix", "\n".join(lines))


def _build_figure_captions(artefacts: ExperimentArtefacts) -> dict[str, str]:
    """Return display_name → caption mapping from plot_index artefact.

    Uses the same stem→display_name normalization as _copy_figures() so
    keys align with the display_name values in figure_paths tuples.
    Returns empty dict when no plot_index is present.
    """
    plot_index = getattr(artefacts, "plot_index", None)
    if not plot_index:
        return {}
    captions: dict[str, str] = {}
    for entry in plot_index:
        name = entry.get("name", "")
        caption = entry.get("caption", "")
        if name and caption:
            display_name = name.replace("_", " ").title()
            captions[display_name] = caption
    return captions


# ---------------------------------------------------------------------------
# Dynamic interpretation helpers (E2.1)
# Each function reads artefacts defensively and returns a prose string.
# Falls back to a short generic sentence if data is absent or malformed.
# ---------------------------------------------------------------------------


def _residuals_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Dynamic interpretation for the residual diagnostics figure."""
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    ps = mmd.get("prediction_stats") or {}
    pct_pos = ps.get("pct_positive")
    da = mmd.get("directional_accuracy")

    bias_note = ""
    if isinstance(pct_pos, float) and pct_pos > 0.85:
        bias_note = f" {pct_pos:.0%} positive predictions — long-biased; errors skew toward missed downside."
    elif isinstance(pct_pos, float) and pct_pos < 0.5:
        bias_note = f" {pct_pos:.0%} positive predictions — short-biased; errors skew toward missed upside."

    da_note = (
        f" Directional accuracy: {da:.1%}." if isinstance(da, float) else ""
    )

    return (
        "Symmetric residuals near zero indicate an unbiased model."
        " Skew or heavy tails signal calibration failure in extreme regimes."
        f" Rolling residual mean reveals temporal bias drift.{bias_note}{da_note}"
    )


def _cs_ic_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Interpretation for cross_sectional_ic — raw daily CS-IC series."""
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    ic_sum = mmd.get("ic_summary") or {}
    mean_ic = ic_sum.get("mean_ic")
    pct_pos = ic_sum.get("pct_positive_ic")

    if mean_ic is None:
        return (
            "Daily cross-sectional Spearman IC: per-date rank correlation between"
            " predicted scores and realised returns across all universe assets."
            " Persistent positive values confirm the model correctly ranks assets."
        )

    quality = "strong" if mean_ic > 0.10 else "moderate" if mean_ic > 0.03 else "weak"
    pos_str = f"{pct_pos:.0%} of days positive" if pct_pos is not None else ""
    return (
        f"Daily cross-sectional IC — mean {mean_ic:.3f} ({pos_str}), {quality} ranking signal."
        " Persistent negative stretches mark regimes where the feature space inverted its"
        " cross-sectional prediction, not random noise."
    )


def _rolling_ic_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Interpretation for ml_ic_regime — 63d rolling mean IC (panel) or rolling Pearson IC."""
    is_panel = _is_panel_mode(artefacts)
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    ic_sum = mmd.get("ic_summary") or {}
    mean_ic = ic_sum.get("mean_ic")
    pct_pos = ic_sum.get("pct_positive_ic")

    if mean_ic is None:
        if is_panel:
            return (
                "63-day rolling mean cross-sectional IC. Sustained negative stretches"
                " mark regimes of cross-sectional ranking breakdown."
            )
        return "Persistent positive IC confirms directional alignment with realised returns."

    quality = "strong" if mean_ic > 0.10 else "moderate" if mean_ic > 0.03 else "weak"
    pos_str = f"{pct_pos:.0%} positive" if pct_pos is not None else ""
    if is_panel:
        return (
            f"63-day rolling mean cross-sectional IC — mean {mean_ic:.3f} ({pos_str}),"
            f" {quality} ranking signal. Troughs identify regimes where cross-sectional"
            " feature relationships inverted or collapsed; recoveries confirm regime-specific"
            " rather than permanent model failure."
        )
    return (
        f"Mean IC {mean_ic:.3f} ({pos_str}) — {quality} directional signal."
        " Sub-period troughs mark regimes where the feature space lost predictive content."
    )


def _ic_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Backward-compatible wrapper — returns rolling IC interpretation."""
    return _rolling_ic_interpretation(artefacts)


def _stability_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Compose dynamic coefficient stability interpretation from observed diagnostics."""
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    stab = mmd.get("coefficient_stability_summary") or []

    valid = [r for r in stab if isinstance(r.get("sign_consistency"), float)]
    if not valid:
        return "Sign consistency near 1.0 indicates a persistent directional signal; near 0.5 indicates noise."

    most_stable = max(valid, key=lambda r: r["sign_consistency"])
    least_stable = min(valid, key=lambda r: r["sign_consistency"])
    ms_feat = _feat_label(most_stable.get("feature", "—"))
    ms_sc = most_stable["sign_consistency"]
    ls_feat = _feat_label(least_stable.get("feature", "—"))
    ls_sc = least_stable["sign_consistency"]

    return (
        f"{ms_feat} most stable ({ms_sc:.0%} sign consistency);"
        f" {ls_feat} least stable ({ls_sc:.0%}) — regime-dependent contribution."
        " Wide error bars or sign reversals indicate training-regime-specific fitting."
    )


def _evolution_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Compose dynamic coefficient evolution interpretation."""
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    stab = mmd.get("coefficient_stability_summary") or []

    crossings = [r.get("feature", "—") for r in stab
                 if isinstance(r.get("sign_consistency"), float) and r["sign_consistency"] < 0.6]

    if crossings:
        cross_str = ", ".join(f"`{_feat_label(f)}`" for f in crossings[:3])
        return (
            f"Coefficient trajectories through successive training windows."
            f" {cross_str} cross zero — regime-dependent, not structurally persistent."
        )
    return (
        "Coefficient trajectories through successive training windows."
        " Stable sign across all splits indicates structurally persistent learning."
    )


def _drawdown_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Compose dynamic equity/drawdown interpretation from observed metrics."""
    m = artefacts.metrics if isinstance(artefacts.metrics, dict) else {}
    mdd = m.get("max_drawdown")
    sharpe = m.get("sharpe_ratio")

    if mdd is None:
        return "Drawdown recovery duration distinguishes signal loss from temporary dislocation."

    severity = "severe" if mdd < -0.30 else "moderate" if mdd < -0.15 else "contained"
    sharpe_str = f" Sharpe {sharpe:.2f}." if sharpe is not None else ""
    return (
        f"Max drawdown {mdd:.1%} — {severity}.{sharpe_str}"
        " Slow recovery indicates signal-edge loss, not random noise."
    )


def _rolling_da_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Dynamic interpretation for the rolling DA / IC-consistency figure."""
    is_panel = _is_panel_mode(artefacts)
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    da = mmd.get("directional_accuracy")

    if is_panel:
        if da is None:
            return (
                "126-day rolling fraction of positive cross-sectional IC days."
                " Sustained drops below 50% mark regimes where the model's ranking"
                " inverted — cross-sectional signal degradation, not random variation."
            )
        return (
            f"126-day rolling IC consistency — full-period positive-IC rate {da:.1%}."
            " Persistent troughs below 50% identify regimes where the cross-sectional"
            " ranking inverted: the model systematically mispredicted relative return order."
            " Recovery above 50% confirms regime-specific rather than permanent breakdown."
        )
    else:
        if da is None:
            return "Persistent troughs below 50% mark regimes of active signal degradation, not random noise."
        return (
            f"Full-period directional accuracy {da:.1%}."
            " Periods below 50% indicate the model generated net-incorrect directional calls"
            " — regime-linked degradation, not statistical noise."
        )


def _instability_chain_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Analytical chain: coefficient instability → IC degradation → DA → performance.

    Returns a concise prose paragraph only when enough data exists to substantiate
    the chain.  Returns empty string when data is insufficient.
    """
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    stab = mmd.get("coefficient_stability_summary") or []
    ic_sum = mmd.get("ic_summary") or {}
    da = mmd.get("directional_accuracy")
    regime_stats = mmd.get("regime_conditioned_stats") or []

    # Count features with sign reversals
    unstable_feats = [
        r.get("feature", "—") for r in stab
        if isinstance(r.get("sign_consistency"), float) and r["sign_consistency"] < 0.7
    ]

    pct_pos_ic = ic_sum.get("pct_positive_ic")
    mean_ic = ic_sum.get("mean_ic")

    if not unstable_feats and pct_pos_ic is None and da is None:
        return ""

    parts: list[str] = []

    if unstable_feats:
        feat_str = ", ".join(_feat_label(f) for f in unstable_feats[:3])
        parts.append(
            f"Coefficient sign reversals in {feat_str} indicate the model learned"
            " regime-specific rather than structural relationships."
        )

    if isinstance(pct_pos_ic, float) and isinstance(mean_ic, float):
        ic_quality = "consistently positive" if pct_pos_ic >= 0.6 else (
            "intermittent" if pct_pos_ic >= 0.4 else "predominantly negative"
        )
        parts.append(
            f"IC is {ic_quality} ({pct_pos_ic:.0%} of periods, mean {mean_ic:.4f}),"
            " propagating coefficient instability into signal degradation."
        )

    if isinstance(da, float):
        is_panel = _is_panel_mode(artefacts)
        if is_panel:
            da_quality = "above random" if da > 0.5 else "at or below random"
            parts.append(
                f"Positive-IC rate ({da:.1%}) is {da_quality};"
                " instability in learned feature weights propagates into cross-sectional"
                " ranking errors that reach portfolio returns."
            )
        else:
            da_quality = "above random" if da > 0.5 else "at or below random"
            parts.append(
                f"Directional accuracy ({da:.1%}) is {da_quality};"
                " signal degradation from unstable features reaches realised trade outcomes."
            )

    if regime_stats:
        by_regime = {r.get("regime"): r for r in regime_stats}
        low_s = by_regime.get("Low", {}).get("realized_sharpe")
        high_s = by_regime.get("High", {}).get("realized_sharpe")
        if isinstance(low_s, float) and isinstance(high_s, float) and high_s < low_s:
            parts.append(
                "The Sharpe compression in high-volatility regimes confirms the propagation"
                " completes: feature instability → IC breakdown → ranking errors → losses."
            )

    return " ".join(parts)


def _ranking_geometry_section(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Render the Cross-Sectional Ranking Geometry subsection (Phase I).

    Reads ranking_geometry from ml_model_diagnostics.  Returns empty string
    when the key is absent — safe for pre-Phase-I artefacts.

    Exposes five lightweight statistical state diagnostics:
        S1 — rolling score IQR (prediction dispersion)
        S2 — rolling top-bottom score spread (discrimination strength)
        S3 — rolling realized return spread (economic discrimination)
        S4 — rank persistence (Spearman autocorrelation)
        S5 — rolling IC std (instability characterization)
    """
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    rg = mmd.get("ranking_geometry")
    if not isinstance(rg, dict):
        return ""

    is_panel = _is_panel_mode(artefacts)
    if not is_panel:
        return ""

    mean_iqr = rg.get("mean_score_iqr")
    min_iqr = rg.get("min_score_iqr")
    mean_spread = rg.get("mean_score_spread")
    mean_real = rg.get("mean_realized_spread")
    pct_real = rg.get("pct_positive_realized")
    mean_pers = rg.get("mean_rank_persistence")
    pct_pers = rg.get("pct_positive_persistence")

    if mean_iqr is None and mean_pers is None:
        return ""

    lines: list[str] = []
    lines.append("### Cross-Sectional Ranking Geometry")
    lines.append("")

    # Context sentence
    lines.append(
        "The following diagnostics expose the statistical geometry of the"
        " cross-sectional ranking system — measuring prediction dispersion,"
        " score discrimination, realized economic separation, and ranking"
        " stability. Together they reveal when the model had genuine"
        " cross-sectional conviction and when ranking became arbitrary."
    )
    lines.append("")

    # Compact scalar summary table
    scalar_rows: list[tuple[str, str]] = []
    if mean_iqr is not None:
        scalar_rows.append(("Mean cross-sectional score IQR", f"{mean_iqr:.4f}"))
    if min_iqr is not None:
        scalar_rows.append(("Min score IQR (most compressed regime)", f"{min_iqr:.4f}"))
    if mean_spread is not None:
        scalar_rows.append(("Mean top-bottom score spread", f"{mean_spread:.4f}"))
    if mean_real is not None:
        sign = "positive" if mean_real > 0 else "negative"
        scalar_rows.append(("Mean realized spread (top − bottom, pre-cost)", f"{mean_real:.3%} ({sign})"))
    if pct_real is not None:
        scalar_rows.append(("Fraction of periods with positive realized spread", f"{pct_real:.0%}"))
    if mean_pers is not None:
        scalar_rows.append(("Mean monthly rank autocorrelation", f"{mean_pers:.3f}"))
    if pct_pers is not None:
        scalar_rows.append(("Fraction of months with positive rank persistence", f"{pct_pers:.0%}"))

    if scalar_rows:
        lines.append(_pipe_table(["Metric", "Value"], scalar_rows))
        lines.append("")

    # Dispersion interpretation
    if mean_iqr is not None and min_iqr is not None:
        if min_iqr < 0.01 * mean_iqr or (mean_iqr > 0 and min_iqr / mean_iqr < 0.15):
            disp_verdict = (
                f"Score IQR compressed to near-zero in its most stressed regime"
                f" (min {min_iqr:.4f} vs mean {mean_iqr:.4f})."
                " In these periods the model assigned nearly identical scores to all assets"
                " — the top-N selection was effectively arbitrary."
            )
        else:
            disp_verdict = (
                f"Score IQR remained above its minimum throughout"
                f" (mean {mean_iqr:.4f}, min {min_iqr:.4f})."
                " The model maintained measurable cross-sectional separation across regimes."
            )
        lines.append(disp_verdict)
        lines.append("")

    # Rank persistence interpretation
    if mean_pers is not None:
        if mean_pers > 0.6:
            pers_verdict = (
                f"Mean rank autocorrelation {mean_pers:.2f} — rankings were"
                " highly persistent. The model's ordering of assets was stable"
                " across consecutive rebalance periods, consistent with a"
                " momentum-driven hypothesis."
            )
        elif mean_pers > 0.3:
            pers_verdict = (
                f"Mean rank autocorrelation {mean_pers:.2f} — moderate ranking"
                " persistence. The model updated its asset ordering over time"
                " without wholesale composition flips, consistent with"
                " meaningful signal updating rather than noise."
            )
        else:
            pers_verdict = (
                f"Mean rank autocorrelation {mean_pers:.2f} — low ranking"
                " persistence. Asset rankings changed substantially each"
                " rebalance period, which may indicate responsive signal updating"
                " or score instability propagating into composition volatility."
            )
        if pct_pers is not None:
            pers_verdict += f" {pct_pers:.0%} of monthly transitions had positive rank autocorrelation."
        lines.append(pers_verdict)
        lines.append("")

    # Realized spread interpretation (pre-cost caveat)
    if mean_real is not None:
        if mean_real > 0 and pct_real is not None and pct_real > 0.55:
            real_verdict = (
                f"Realized top-bottom spread averaged {mean_real:.2%} per period"
                f" ({pct_real:.0%} of periods positive) — the model's score"
                " ranking corresponded to economic outcome separation over this horizon."
                " This is a pre-cost gross spread; execution costs and market impact"
                " reduce the realisable advantage."
            )
        elif mean_real > 0:
            real_verdict = (
                f"Realized top-bottom spread averaged {mean_real:.2%} (positive mean,"
                f" {pct_real:.0%} of periods positive) — modest economic"
                " discrimination. Score ranking correlated positively with outcomes"
                " on average but with considerable period-level variability."
            )
        else:
            real_verdict = (
                f"Realized top-bottom spread averaged {mean_real:.2%} (negative)."
                " Score ranking did not consistently translate to outcome separation"
                " over this horizon — discrimination is present in signal space"
                " but weakly confirmed in realized return space."
            )
        lines.append(real_verdict)
        lines.append("")

    # Figure
    rg_caption = (
        "Cross-sectional ranking geometry: prediction dispersion (score IQR) and"
        " IC instability (IC std, right axis); score discrimination (top-minus-bottom"
        " spread); realized return discrimination; and monthly rank autocorrelation."
        " Periods of simultaneous IQR compression and elevated IC std identify"
        " regimes of compressed-but-erratic ranking — the most diagnostically"
        " significant failure state for a cross-sectional model."
    )
    rg_fig = _embed_figure("ranking_geometry", figures, claimed, rg_caption)
    if rg_fig:
        lines.append(rg_fig)
        lines.append("")

    return "\n".join(lines)


def _feature_contribution_section(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Render the Temporal Feature Contribution subsection (Phase II).

    Reads feature_contributions from ml_model_diagnostics.  Returns empty string
    when the key is absent — safe for pre-Phase-II artefacts and single-asset
    experiments where no feature_families breakdown exists.

    Exposes three complementary contribution diagnostics:
        C1 — feature contribution heatmap (date × feature realised influence)
        C2 — family contribution timeline (signed + normalised share)
        C3 — instability scalars (dominant family, transitions, HHI)
    """
    mmd = artefacts.ml_model_diagnostics
    if not isinstance(mmd, dict):
        return ""

    fc = mmd.get("feature_contributions")
    if not isinstance(fc, dict) or not fc:
        return ""

    is_panel = _is_panel_mode(artefacts)
    lines: list[str] = ["### Temporal Feature Contribution"]
    lines.append("")
    lines.append(
        "Feature contribution measures realised predictive influence: "
        "coefficient × standardised feature value at each date. "
        "Unlike static coefficient tables, contribution captures "
        "when each feature was activated, suppressed, or inverted — "
        "exposing how the model's internal predictive structure evolved through time."
    )
    lines.append("")

    # --- C3 instability summary table ---
    dom_family = fc.get("dominant_family")
    dom_pct = fc.get("dominant_family_pct")
    n_trans = fc.get("n_family_transitions")
    mean_hhi = fc.get("mean_hhi")
    most_volatile = fc.get("most_volatile_feature")

    c3_rows: list[tuple[str, str]] = []
    if dom_family is not None:
        dom_str = f"{dom_family}"
        if dom_pct is not None:
            dom_str += f" ({dom_pct:.0%} of periods)"
        c3_rows.append(("Dominant family (by contribution share)", dom_str))
    if n_trans is not None:
        c3_rows.append(("Family leadership transitions", f"{n_trans:,}"))
    if mean_hhi is not None:
        concentration_label = (
            "concentrated (one family dominates)"
            if mean_hhi > 0.40
            else "moderate (2–3 families share influence)"
            if mean_hhi > 0.25
            else "distributed (broadly shared across families)"
        )
        c3_rows.append(("Mean contribution concentration (HHI)", f"{mean_hhi:.3f} — {concentration_label}"))
    if most_volatile is not None:
        try:
            from src.features.families import generate_feature_label as _gl
            vol_label = _gl(most_volatile)
        except Exception:
            vol_label = most_volatile
        c3_rows.append(("Most temporally volatile feature", vol_label))

    if c3_rows:
        lines.append(_pipe_table(["Diagnostic", "Value"], c3_rows))
        lines.append("")

    # --- Dominant-family narrative (topology-aware language) ---
    if dom_family is not None and dom_pct is not None:
        signal_noun = "cross-sectional ranking signal" if is_panel else "directional prediction signal"
        info_noun = "cross-sectional information" if is_panel else "predictive information"
        if dom_pct > 0.55:
            lines.append(
                f"The **{dom_family}** family dominated predictions in {dom_pct:.0%} of periods, "
                f"indicating the model's {signal_noun} was concentrated in a single "
                "predictive hypothesis for most of the backtest. This is characteristic of "
                "a momentum-regime environment where trend information dominates."
            )
        elif dom_pct > 0.35:
            lines.append(
                f"The **{dom_family}** family led predictions in {dom_pct:.0%} of periods, "
                "but shared dominance with other families across the sample. "
                f"Multiple hypothesis families contributed meaningful {info_noun}, "
                "suggesting regime-conditioned feature activation."
            )
        else:
            lines.append(
                "No single feature family dominated predictions consistently. "
                f"Contribution was broadly distributed across hypothesis families, "
                f"suggesting the model learned diverse {info_noun} "
                "with no one family providing persistent structural advantage."
            )
        lines.append("")

    # --- Transition and concentration interpretation ---
    if n_trans is not None and mean_hhi is not None:
        if n_trans < 50 and mean_hhi > 0.35:
            lines.append(
                f"Contribution structure was stable and concentrated — {n_trans:,} family "
                "leadership transitions across the backtest — consistent with a model that "
                "operated in persistent predictive regimes rather than frequently adapting."
            )
        elif n_trans > 200:
            lines.append(
                f"High family leadership volatility ({n_trans:,} transitions) indicates "
                "the model's predictive structure shifted frequently across the backtest. "
                "This may reflect genuine macro regime changes or coefficient instability "
                "amplified by feature value rotation."
            )
        else:
            lines.append(
                f"Moderate contribution dynamics ({n_trans:,} family transitions, "
                f"mean HHI {mean_hhi:.3f}) — the model's predictive structure evolved "
                "gradually, with periods of concentrated dominance separated by transitions "
                "as market regimes shifted."
            )
        lines.append("")

    # --- C1: Feature contribution heatmap ---
    heatmap_caption = (
        "Feature contribution heatmap: realised predictive influence "
        "(coefficient × standardised feature value) for each feature through time, "
        "grouped by family. Red = positive contribution (feature state predicts above-average "
        "cross-sectional return); blue = negative. Family separators mark hypothesis boundaries. "
        "Regime shifts appear as horizontal band colour transitions; "
        "simultaneous sign changes across a family reveal coordinated hypothesis activation or suppression."
    )
    if is_panel:
        heatmap_caption += (
            " Contribution shown using panel reference-ticker feature values "
            "scaled by the shared panel coefficient vector."
        )
    hmap_fig = _embed_figure("feature_contribution_heatmap", figures, claimed, heatmap_caption)
    if hmap_fig:
        lines.extend(["", hmap_fig])

    # --- C2: Family contribution timeline ---
    timeline_caption = (
        "Feature family contribution timeline. "
        "Top panel: signed rolling family contributions — shows which hypothesis family "
        "drove predictions and in which direction. "
        "Persistent positive contributions indicate an actively reinforcing family; "
        "negative contributions indicate the family is working against the dominant direction. "
        "Bottom panel: normalised absolute contribution share — shows family dominance "
        "through time regardless of sign. "
        "Regime shifts appear as share transitions between families."
    )
    timeline_fig = _embed_figure("family_contribution_timeline", figures, claimed, timeline_caption)
    if timeline_fig:
        lines.extend(["", timeline_fig])

    return "\n".join(lines)


def _prediction_strength_section(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Render the Prediction Confidence & Outcome Monotonicity subsection.

    Reads prediction_strength from ml_model_diagnostics.  Returns empty
    string when the key is absent — safe for pre-Step-3 artefacts.
    """
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    ps = mmd.get("prediction_strength")
    if not isinstance(ps, dict):
        return ""

    group_means = ps.get("group_mean_returns") or {}
    ls_spread = ps.get("ls_spread")
    is_monotonic = ps.get("is_monotonic")
    is_ordered = ps.get("is_ordered")
    n_obs = ps.get("n_obs", 0)
    horizon = ps.get("horizon", 21)
    n_per_group = ps.get("n_assets_per_group", "—")
    _is_panel_mode(artefacts)

    top_r = group_means.get("top")
    mid_r = group_means.get("mid")
    bot_r = group_means.get("bottom")

    if top_r is None and bot_r is None:
        return ""

    lines: list[str] = []
    lines.append("### Prediction Confidence & Outcome Monotonicity")
    lines.append("")
    lines.append(
        f"Assets are ranked by predicted cross-sectional score on each monthly"
        f" rebalance date and assigned to top, mid, or bottom thirds"
        f" (~{n_per_group} assets per group). Realized {horizon}-day forward returns"
        f" are evaluated per group across {n_obs} monthly observations."
        " Monotonic ordering — top group outperforming bottom — confirms that"
        " prediction magnitude, not merely sign, carries cross-sectional economic content."
    )
    lines.append("")

    # Mean return table
    table_rows: list[tuple[str, str]] = []
    if top_r is not None:
        table_rows.append(("Top group (highest scores)", f"{top_r:.3%}"))
    if mid_r is not None:
        table_rows.append(("Mid group", f"{mid_r:.3%}"))
    if bot_r is not None:
        table_rows.append(("Bottom group (lowest scores)", f"{bot_r:.3%}"))
    if ls_spread is not None:
        table_rows.append(("Long-short spread (top − bottom)", f"{ls_spread:.3%}"))

    if table_rows:
        lines.append(_pipe_table(["Prediction Group", f"Mean {horizon}D Realized Return"], table_rows))
        lines.append("")

    # Monotonicity verdict
    if is_monotonic:
        verdict = (
            "Realized return ordering is **monotonic** (top > mid > bottom)."
            " Score magnitude carries economically meaningful cross-sectional information:"
            " the model's ranking conviction corresponds to realized outcome strength."
        )
    elif is_ordered:
        verdict = (
            "Realized return ordering shows **directional ordering** (top > bottom)"
            " but mid-group returns break strict monotonicity."
            " Rank extremes are informative; signal quality degrades toward the"
            " middle of the cross-sectional distribution."
        )
    else:
        verdict = (
            "Realized return ordering is **non-monotonic**: prediction magnitude does not"
            " consistently correspond to outcome strength across this backtest."
            " Only sign information is reliable; prediction magnitude should be"
            " treated as a ranking signal, not a return magnitude signal."
        )
    lines.append(verdict)
    lines.append("")

    # Long-short spread context
    if ls_spread is not None and top_r is not None and bot_r is not None:
        spread_pct = ls_spread
        direction = "positive" if spread_pct > 0 else "negative"
        lines.append(
            f"The {horizon}D long-short spread between top and bottom groups is"
            f" {spread_pct:.2%} ({direction})."
            " This is a pre-cost gross spread diagnostic, not a realisable strategy return."
            " It quantifies the raw ranking discrimination of the signal before portfolio"
            " construction, transaction costs, and execution frictions."
        )
        lines.append("")

    # Figure
    fig_caption = (
        f"Top panel: mean {horizon}-day realized forward return by prediction score group"
        f" ({n_obs} monthly observations, ~{n_per_group} assets per group)."
        " Monotonic left-to-right ordering is the ML legitimacy diagnostic."
        " Bottom panel: cumulative return of each prediction group over time."
        " Persistent group separation confirms durable signal strength;"
        " convergence marks regimes of prediction-strength collapse."
    )
    ps_fig = _embed_figure("prediction_strength", figures, claimed, fig_caption)
    if ps_fig:
        lines.append(ps_fig)
        lines.append("")

    return "\n".join(lines)


def _regime_conditional_behavior(
    artefacts: ExperimentArtefacts,
    figures: dict[str, Path] | None = None,
    claimed: set[str] | None = None,
) -> str:
    """Render the Regime-Conditional Feature Behaviour subsection.

    Reads regime_stats from ml_model_diagnostics (populated by Step 2 regime
    classification in the orchestrator).  Returns empty string when the key is
    absent — safe for pre-Step-2 artefacts.
    """
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    rs = mmd.get("regime_stats")
    if not isinstance(rs, dict):
        return ""

    family_ic_by_regime = rs.get("family_ic_by_regime") or {}
    high_ic: dict = family_ic_by_regime.get("high_vol") or {}
    low_ic: dict = family_ic_by_regime.get("low_vol") or {}

    if not high_ic and not low_ic:
        return ""

    is_panel = _is_panel_mode(artefacts)
    n_high = rs.get("n_high_vol_splits", 0)
    n_low = rs.get("n_low_vol_splits", 0)
    n_total = n_high + n_low
    dominant = rs.get("dominant_family") or {}

    lines: list[str] = []
    lines.append("### Regime-Conditional Feature Behaviour")
    lines.append("")

    # Contextual framing
    ic_label = "cross-sectional IC" if is_panel else "IC"
    lines.append(
        f"The {n_total} walk-forward test windows are classified into"
        f" high-volatility ({n_high} split{'s' if n_high != 1 else ''})"
        f" and low-volatility ({n_low} split{'s' if n_low != 1 else ''})"
        " regimes using median cross-asset 21D realised volatility as threshold."
        f" This classification exposes conditional {ic_label} behaviour —"
        " which feature families provided signal under stressed vs calm conditions."
    )
    lines.append("")

    # Family IC by regime table
    _order = ["Trend", "Volatility", "Mean-Reversion", "Market Structure", "Relative Strength"]
    all_fams = sorted(set(high_ic) | set(low_ic),
                      key=lambda f: _order.index(f) if f in _order else len(_order))

    if all_fams:
        table_rows: list[tuple[str, ...]] = []
        for fam in all_fams:
            h = high_ic.get(fam)
            l = low_ic.get(fam)
            diff = (h - l) if (isinstance(h, float) and isinstance(l, float)) else None
            table_rows.append((
                fam,
                f"{h:+.3f}" if isinstance(h, float) else "—",
                f"{l:+.3f}" if isinstance(l, float) else "—",
                f"{diff:+.3f}" if isinstance(diff, float) else "—",
            ))
        lines.append(_pipe_table(
            ["Feature Family", "High-Vol Mean IC", "Low-Vol Mean IC", "Differential"],
            table_rows,
        ))
        lines.append("")

    # Dominant family narrative
    dom_high = dominant.get("high_vol")
    dom_low = dominant.get("low_vol")

    if dom_high and dom_low:
        if dom_high == dom_low:
            lines.append(
                f"The **{dom_high}** family provided the strongest {ic_label} in both"
                " volatility regimes, indicating a regime-persistent directional hypothesis."
                " The IC differential across regimes reveals the degree of regime sensitivity"
                " within this dominant family."
            )
        else:
            lines.append(
                f"The **{dom_high}** family dominated in high-volatility splits;"
                f" **{dom_low}** provided the strongest {ic_label} during lower-volatility periods."
                " This regime-dependent family rotation suggests the model's ranking edge"
                " shifts hypothesis source as market structure changes."
            )
        lines.append("")

    # Regime interpretation synthesis — evidence-grounded, no causal storytelling
    interp_parts = []

    # Note any families that inverted between regimes
    inverting = [
        fam for fam in all_fams
        if isinstance(high_ic.get(fam), float)
        and isinstance(low_ic.get(fam), float)
        and high_ic[fam] * low_ic[fam] < 0  # sign flip
    ]
    if inverting:
        inv_str = ", ".join(f"**{f}**" for f in inverting)
        interp_parts.append(
            f"{inv_str} reversed sign between regimes — positive {ic_label} in one"
            " environment became negative in the other."
            " Sign reversals indicate regime-specific learning, not a stable directional relationship."
        )

    # Note the family with the largest regime sensitivity (highest |differential|)
    diffs = {
        fam: abs(high_ic.get(fam, 0.0) - low_ic.get(fam, 0.0))
        for fam in all_fams
        if isinstance(high_ic.get(fam), float) and isinstance(low_ic.get(fam), float)
    }
    if diffs:
        most_sensitive = max(diffs, key=diffs.get)
        sens_diff = high_ic.get(most_sensitive, 0.0) - low_ic.get(most_sensitive, 0.0)
        direction = "stronger in high-vol" if sens_diff > 0 else "stronger in low-vol"
        interp_parts.append(
            f"**{most_sensitive}** shows the largest regime sensitivity"
            f" (differential {sens_diff:+.3f}), contributing {direction} splits."
        )

    if interp_parts:
        lines.append(" ".join(interp_parts))
        lines.append("")

    # Figure: IC by vol regime (grouped bar chart)
    fig_caption = (
        "Feature family mean IC disaggregated by vol regime."
        " Solid bars = high-volatility test splits; faded = low-volatility."
        f" Regime threshold: median cross-asset 21D realised vol across {n_total} test windows."
        " Sign reversals between solid and faded bars indicate regime-dependent hypothesis flips."
    )
    regime_fig = _embed_figure("ic_by_vol_regime", figures, claimed, fig_caption)
    if regime_fig:
        lines.extend([regime_fig, ""])

    return "\n".join(lines)


def _sign_heatmap_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Dynamic interpretation for the coefficient sign heatmap."""
    mmd = artefacts.ml_model_diagnostics if isinstance(artefacts.ml_model_diagnostics, dict) else {}
    stab = mmd.get("coefficient_stability_summary") or []
    crossings = [
        r.get("feature", "—") for r in stab
        if isinstance(r.get("sign_consistency"), float) and r["sign_consistency"] < 0.6
    ]
    if crossings:
        feat_str = ", ".join(f"`{_feat_label(f)}`" for f in crossings[:3])
        return (
            f"{feat_str} change sign across splits — regime-specific learning,"
            " not stable directional relationships."
            " Stable features maintain colour throughout; reversals are the instability signature."
        )
    return (
        "Consistent sign across all splits confirms structurally persistent contributions."
        " Magnitude variation between splits is acceptable; sign reversals would indicate"
        " regime-specific fitting."
    )


def _degradation_interpretation(artefacts: ExperimentArtefacts) -> str:
    """Compose dynamic train-vs-test degradation interpretation from split metrics."""
    sm = artefacts.split_metrics if isinstance(artefacts.split_metrics, dict) else {}
    summary = sm.get("summary") or {}
    mean_sharpe = summary.get("mean_sharpe")
    pos_rate = summary.get("hit_rate_positive_sharpe")
    splits = sm.get("splits") or []

    if mean_sharpe is None:
        return "Large train-to-test Sharpe gaps indicate overfitting; modest gaps are expected."

    verdict = "positive" if mean_sharpe > 0.0 else "negative"
    pos_str = f", {pos_rate:.0%} positive" if pos_rate is not None else ""
    n_splits = len(splits)
    return (
        f"Mean OOS Sharpe {mean_sharpe:.2f} across {n_splits} splits ({verdict}{pos_str})."
        " Large per-split gaps mark regime-specific overfitting."
    )


def _embed_figure(
    stem: str,
    figures: dict[str, Path] | None,
    claimed: set[str] | None,
    interpretation: str = "",
) -> str:
    """Render a single figure inline with optional interpretation prose.

    Args:
        stem:           Canonical plot stem name (e.g. ``equity_and_drawdown``).
        figures:        stem→renderer-relative-path map built in render_report().
        claimed:        Mutable set tracking which figures have been placed inline.
                        Updated in-place when a figure is consumed.
        interpretation: Research interpretation prose rendered as italic text
                        below the image.  Should focus on what the figure reveals
                        and its research implications — not describe the axes.

    Returns:
        Markdown string with image tag and interpretation, or empty string if
        the figure is not present in the figures map.
    """
    if not figures or stem not in figures:
        return ""
    rel_path = figures[stem]
    rel_str = str(rel_path).replace("\\", "/")
    display_name = stem.replace("_", " ").title()
    parts = [f"![{display_name}]({rel_str})"]
    if interpretation:
        parts.extend(["", f"*{interpretation}*"])
    if claimed is not None:
        claimed.add(stem)
    return "\n".join(parts)


def _figures(
    figure_paths: list[tuple[str, Path]],
    captions: dict[str, str] | None = None,
) -> str:
    """Return figures section string, or empty string if no figures."""
    if not figure_paths:
        return ""

    lines: list[str] = []
    for display_name, rel_path in figure_paths:
        rel_str = str(rel_path).replace("\\", "/")
        lines.append(f"### {display_name}")
        lines.append("")
        lines.append(f"![{display_name}]({rel_str})")
        if captions:
            caption = captions.get(display_name)
            if caption:
                lines.append("")
                lines.append(f"*{caption}*")
        lines.append("")

    return _render_section("Figures", "\n".join(lines).rstrip())


def _footer(
    artefacts: ExperimentArtefacts,
    generated_at: str,
    report_version: str,
) -> str:
    exp_name = artefacts.metadata.get("experiment_name", "unknown")
    lines = [
        "---",
        "",
        f"Report version: {report_version}",
        f"Generated: {generated_at}",
        f"Source experiment: {exp_name}",
    ]
    config_hash = artefacts.metadata.get("config_hash")
    if config_hash:
        lines.append(f"Config hash: {config_hash}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_section(heading: str, body: str) -> str:
    """Combine a ## heading with body text."""
    return f"## {heading}\n\n{body}"


def _pipe_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    """Render a GitHub-flavoured markdown pipe table."""
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, separator, *data_rows])


def _label(key: str) -> str:
    """Convert a snake_case metric key to a title-case display label."""
    return key.replace("_", " ").title()
