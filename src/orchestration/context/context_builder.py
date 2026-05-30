"""Assembles all summarizers into a single LLMContext.

This is the only module that LLM-facing code needs to call — it orchestrates
retrieval, summarization, and failure detection into one deterministic bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.orchestration.api.schemas import LLMContext
from src.orchestration.context.context_schema import CONTEXT_VERSION
from src.orchestration.context.failure_mode_detector import detect_failure_modes
from src.orchestration.context.metric_summarizer import summarize_metrics
from src.orchestration.context.ml_diagnostic_summarizer import (
    summarize_feature_context,
    summarize_ml_diagnostics,
)
from src.orchestration.context.validation_summarizer import summarize_validation
from src.orchestration.retrieval.diagnostics_retriever import load_all_diagnostics
from src.orchestration.retrieval.manifest_retriever import get_rendered_sections
from src.orchestration.retrieval.plot_retriever import get_plot_index
from src.orchestration.utils.filesystem import metadata_path, metrics_path
from src.orchestration.utils.serialization import dump_json, load_json


def build_context(
    experiment_name: str,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
) -> LLMContext:
    """Build a complete LLMContext from disk artefacts.

    All data is loaded from the filesystem — no computation, no quant engine
    calls.  Returns a fully self-contained context object suitable for
    serialization and LLM consumption.
    """
    meta = load_json(metadata_path(experiment_name, base)) or {}
    metrics = load_json(metrics_path(experiment_name, base)) or {}
    diags = load_all_diagnostics(experiment_name, base)

    performance = summarize_metrics(metrics)
    validation = summarize_validation(diags.get("split_metrics"))
    ml_diagnostics = summarize_ml_diagnostics(diags.get("ml_model_diagnostics"))
    feature_summary = summarize_feature_context(
        diags.get("alignment_diagnostics"),
        diags.get("feature_families"),
    )

    universe_summary = _summarize_universe(diags.get("universe_coverage"))

    failure_modes = detect_failure_modes(
        metrics=metrics,
        split_metrics=diags.get("split_metrics"),
        ml_model_diagnostics=diags.get("ml_model_diagnostics"),
        backtest_diagnostics=diags.get("backtest_diagnostics"),
        alignment_diagnostics=diags.get("alignment_diagnostics"),
    )
    failure_mode_dicts = [
        {
            "name": fm.name,
            "severity": fm.severity,
            "description": fm.description,
            "evidence": fm.evidence,
        }
        for fm in failure_modes
    ]

    plot_index = get_plot_index(experiment_name, base)
    available_plots = [
        {"name": p.name, "group": p.group, "importance": p.importance, "caption": p.caption}
        for p in plot_index
        if p.importance == "primary"
    ]

    report_sections = get_rendered_sections(experiment_name, reports_base)

    return LLMContext(
        experiment_name=experiment_name,
        strategy_name=meta.get("strategy_name", ""),
        tags=meta.get("tags", []),
        created_at=meta.get("created_at", ""),
        performance=performance,
        validation=validation,
        ml_diagnostics=ml_diagnostics,
        failure_modes=failure_mode_dicts,
        feature_summary=feature_summary,
        universe_summary=universe_summary,
        available_plots=available_plots,
        report_sections=report_sections,
        extra={"context_version": CONTEXT_VERSION},
    )


def build_and_persist_context(
    experiment_name: str,
    base: Path | str | None = None,
    reports_base: Path | str | None = None,
    llm_base: Path | str | None = None,
) -> LLMContext:
    """Build LLMContext and write it to llm_context.json on disk."""
    from src.orchestration.utils.filesystem import llm_context_path

    ctx = build_context(experiment_name, base, reports_base)
    out_path = llm_context_path(experiment_name, llm_base)
    dump_json(_context_to_dict(ctx), out_path)
    return ctx


def _context_to_dict(ctx: LLMContext) -> dict[str, Any]:
    raw = {
        "experiment_name": ctx.experiment_name,
        "strategy_name": ctx.strategy_name,
        "tags": ctx.tags,
        "created_at": ctx.created_at,
        "performance": ctx.performance,
        "validation": ctx.validation,
        "ml_diagnostics": ctx.ml_diagnostics,
        "failure_modes": ctx.failure_modes,
        "feature_summary": ctx.feature_summary,
        "universe_summary": ctx.universe_summary,
        "available_plots": ctx.available_plots,
        "report_sections": ctx.report_sections,
        **ctx.extra,
    }
    return _prune_nulls(raw)


def _prune_nulls(obj: Any) -> Any:
    """Recursively remove None, empty-dict, and empty-list values.

    Preserves 0, False, and empty strings (semantically meaningful).
    Top-level identity fields (experiment_name, strategy_name, created_at)
    are never None so they survive naturally.
    """
    if isinstance(obj, dict):
        pruned = {}
        for k, v in obj.items():
            cleaned = _prune_nulls(v)
            if cleaned is None or cleaned == {} or cleaned == []:
                continue
            pruned[k] = cleaned
        return pruned
    if isinstance(obj, list):
        return [_prune_nulls(item) for item in obj]
    return obj


def _summarize_universe(universe_coverage: dict[str, Any] | None) -> dict[str, Any]:
    if not universe_coverage:
        return {}
    # universe_coverage.json uses "tickers" (list of str) and "asset_coverage"
    # (list of {ticker, missingness_pct, ...}) — not "assets" or "coverage_matrix"
    tickers: list[str] = universe_coverage.get("tickers", [])
    n_assets: int = universe_coverage.get("n_assets", len(tickers))
    date_range: dict[str, Any] = universe_coverage.get("date_range", {})
    asset_coverage: list[dict[str, Any]] = universe_coverage.get("asset_coverage", [])

    missingness_vals = [
        a.get("missingness_pct", 0.0)
        for a in asset_coverage
        if isinstance(a, dict)
    ]
    mean_coverage_pct = (
        f"{(1.0 - sum(missingness_vals) / len(missingness_vals)) * 100:.1f}%"
        if missingness_vals
        else None
    )

    return {
        "n_assets": n_assets,
        "asset_tickers": tickers,
        "date_range": date_range,
        "mean_coverage_pct": mean_coverage_pct,
    }
