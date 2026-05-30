"""Phase D2 reporting entrypoint.

Canonical usage:
    paths = generate_experiment_report(experiment_dir, output_dir="reports")

Sequence:
    1. load_experiment_artefacts(path, output_dir)
           reads metadata.json, metrics.json, config (optional), discovers PNGs
           pre-computes all output paths — callers may navigate them before generation
    2. generate_experiment_report(path, output_dir, include_html)
           copies figures, computes relative paths centrally, renders markdown,
           optionally renders HTML, writes provenance sidecar

Boundary:
    - Read-only with respect to experiment artefacts.
    - No recomputation, no rerunning, no data loading.
    - Figure paths are computed here and passed into renderers; renderers make
      no filesystem assumptions of their own.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.experiments.contracts import ARTEFACT_VERSION
from src.reporting.html import markdown_to_html
from src.reporting.markdown import render_report
from src.reporting.report_spec import ResearchReportSpec

_REPORT_VERSION = "1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExperimentArtefacts:
    """All content loaded from one experiment directory, plus pre-computed paths.

    Path fields (artefact_dir, markdown_path, html_path, figure_dir) are
    stable references computed at load time.  They describe intended destinations
    and may be read by future agents or automation layers before generation runs.
    """

    # Source
    artefact_dir: Path

    # Pre-computed output paths (directories/files will be created on write)
    markdown_path: Path
    html_path: Path
    figure_dir: Path

    # Loaded content
    metadata: dict[str, Any]
    metrics: dict[str, float]
    config: dict[str, Any] | None          # None when no config file exists
    source_figures: list[Path]             # PNGs discovered in artefact_dir/plots/
    ml_provenance: dict[str, Any] | None = None          # ml_provenance.json (v2 only)
    split_metrics: dict[str, Any] | None = None          # diagnostics/split_metrics.json
    ml_diagnostics: dict[str, Any] | None = None         # diagnostics/ml_diagnostics.json (v2 only)
    research_artefacts: dict[str, Any] | None = None     # research/*.json (data_summary, signal_transitions)
    backtest_diagnostics: dict[str, Any] | None = None   # diagnostics/backtest_diagnostics.json
    ml_model_diagnostics: dict[str, Any] | None = None   # diagnostics/ml_model_diagnostics.json (v2 only)
    wf_equity_curves: dict[str, Any] | None = None       # diagnostics/wf_equity_curves.json
    feature_summary: dict[str, Any] | None = None        # research/feature_summary.json (v2 only)
    feature_registry: dict[str, Any] | None = None       # research/feature_registry.json (v2 only)
    alignment_diagnostics: dict[str, Any] | None = None  # research/alignment_diagnostics.json (v2 only)
    feature_correlations: dict[str, Any] | None = None   # research/feature_correlations.json (v2 only)
    feature_families: dict[str, Any] | None = None       # research/feature_families.json (G2)
    plot_index: list[dict[str, Any]] | None = None       # plots/plot_index.json (semantic figure ordering)
    universe_coverage: dict[str, Any] | None = None      # diagnostics/universe_coverage.json (G1)
    allocation_diagnostics: dict[str, Any] | None = None # diagnostics/allocation_diagnostics.json (Phase 2)


@dataclass
class ReportPaths:
    """Filesystem paths to all generated report outputs."""

    markdown: Path
    html: Path | None                # None when include_html=False
    provenance: Path                 # JSON sidecar alongside markdown
    manifest: Path | None = None     # Frontend-facing manifest JSON (None if not written)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_experiment_artefacts(
    artefact_path: str | Path,
    output_dir: str | Path = Path("reports"),
) -> ExperimentArtefacts:
    """Load experiment artefacts from disk and pre-compute all output paths.

    Args:
        artefact_path: Path to the saved experiment directory (contains
            metadata.json, metrics.json, optionally plots/ and config files).
        output_dir: Base directory for report outputs.  Sub-directories
            markdown/, html/, and figures/ are used automatically.

    Returns:
        ExperimentArtefacts with loaded content and pre-computed paths.

    Raises:
        FileNotFoundError: If artefact_path does not exist, or if
            metadata.json or metrics.json are missing.
    """
    src = Path(artefact_path).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {src}")

    metadata = _load_json(src / "metadata.json")
    metrics = _load_json(src / "metrics.json")

    # Config: prefer normalized_config.json (D1), fall back to config.json (D0)
    config: dict[str, Any] | None = None
    for cfg_name in ("normalized_config.json", "config.json"):
        cfg_path = src / cfg_name
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                config = json.load(f)
            break

    # Figure discovery + semantic ordering via plot_index.json
    plots_dir = src / "plots"
    source_figures: list[Path] = (
        sorted(plots_dir.glob("*.png")) if plots_dir.is_dir() else []
    )
    plot_index: list[dict[str, Any]] | None = None
    pi_path = plots_dir / "plot_index.json"
    if pi_path.exists():
        with pi_path.open(encoding="utf-8") as f:
            plot_index = json.load(f)
        # Re-order source_figures to match semantic index order when index present
        if plot_index:
            index_names = [e.get("name", "") for e in plot_index]
            stem_to_path = {p.stem: p for p in source_figures}
            ordered = [stem_to_path[n] for n in index_names if n in stem_to_path]
            remaining = [p for p in source_figures if p.stem not in set(index_names)]
            source_figures = ordered + remaining

    # ML provenance sidecar (v2 experiments only)
    ml_provenance: dict[str, Any] | None = None
    ml_prov_path = src / "ml_provenance.json"
    if ml_prov_path.exists():
        with ml_prov_path.open(encoding="utf-8") as f:
            ml_provenance = json.load(f)

    # Persisted diagnostics (written by orchestrator when available)
    split_metrics: dict[str, Any] | None = None
    ml_diagnostics: dict[str, Any] | None = None
    backtest_diagnostics: dict[str, Any] | None = None
    ml_model_diagnostics: dict[str, Any] | None = None
    wf_equity_curves: dict[str, Any] | None = None
    feature_summary: dict[str, Any] | None = None
    feature_registry: dict[str, Any] | None = None
    alignment_diagnostics: dict[str, Any] | None = None
    feature_correlations: dict[str, Any] | None = None
    feature_families: dict[str, Any] | None = None
    diag_dir = src / "diagnostics"
    if diag_dir.is_dir():
        sm_path = diag_dir / "split_metrics.json"
        if sm_path.exists():
            with sm_path.open(encoding="utf-8") as f:
                split_metrics = json.load(f)
        ml_diag_path = diag_dir / "ml_diagnostics.json"
        if ml_diag_path.exists():
            with ml_diag_path.open(encoding="utf-8") as f:
                ml_diagnostics = json.load(f)
        bkd_path = diag_dir / "backtest_diagnostics.json"
        if bkd_path.exists():
            with bkd_path.open(encoding="utf-8") as f:
                backtest_diagnostics = json.load(f)
        ml_md_path = diag_dir / "ml_model_diagnostics.json"
        if ml_md_path.exists():
            with ml_md_path.open(encoding="utf-8") as f:
                ml_model_diagnostics = json.load(f)
        wf_ec_path = diag_dir / "wf_equity_curves.json"
        if wf_ec_path.exists():
            with wf_ec_path.open(encoding="utf-8") as f:
                wf_equity_curves = json.load(f)

    # Universe coverage artefact (G1 — written for all experiments)
    universe_coverage: dict[str, Any] | None = None
    uc_path = (src / "diagnostics" / "universe_coverage.json")
    if uc_path.exists():
        with uc_path.open(encoding="utf-8") as f:
            universe_coverage = json.load(f)

    # Allocation diagnostics (Phase 2 — panel experiments only)
    allocation_diagnostics: dict[str, Any] | None = None
    ad_path = src / "diagnostics" / "allocation_diagnostics.json"
    if ad_path.exists():
        with ad_path.open(encoding="utf-8") as f:
            _ad = json.load(f)
            if isinstance(_ad, dict) and _ad.get("available", False):
                allocation_diagnostics = _ad

    # Research artefacts (written by orchestrator — data_summary, signal_transitions)
    research_artefacts: dict[str, Any] | None = None
    research_dir = src / "research"
    if research_dir.is_dir():
        ra: dict[str, Any] = {}
        ds_path = research_dir / "data_summary.json"
        if ds_path.exists():
            with ds_path.open(encoding="utf-8") as f:
                ra["data_summary"] = json.load(f)
        st_path = research_dir / "signal_transitions.json"
        if st_path.exists():
            with st_path.open(encoding="utf-8") as f:
                ra["signal_transitions"] = json.load(f)
        if ra:
            research_artefacts = ra
        # Feature engineering artefacts (v2 only)
        fs_path = research_dir / "feature_summary.json"
        if fs_path.exists():
            with fs_path.open(encoding="utf-8") as f:
                feature_summary = json.load(f)
        fr_path = research_dir / "feature_registry.json"
        if fr_path.exists():
            with fr_path.open(encoding="utf-8") as f:
                feature_registry = json.load(f)
        al_path = research_dir / "alignment_diagnostics.json"
        if al_path.exists():
            with al_path.open(encoding="utf-8") as f:
                alignment_diagnostics = json.load(f)
        fc_path = research_dir / "feature_correlations.json"
        if fc_path.exists():
            with fc_path.open(encoding="utf-8") as f:
                feature_correlations = json.load(f)
        ff_path = research_dir / "feature_families.json"
        if ff_path.exists():
            with ff_path.open(encoding="utf-8") as f:
                feature_families = json.load(f)

    # Pre-compute output paths from experiment name
    exp_name = metadata.get("experiment_name", src.name)
    out = Path(output_dir).resolve()

    return ExperimentArtefacts(
        artefact_dir=src,
        markdown_path=out / "markdown" / f"{exp_name}.md",
        html_path=out / "html" / f"{exp_name}.html",
        figure_dir=out / "figures" / exp_name,
        metadata=metadata,
        metrics=metrics,
        config=config,
        source_figures=source_figures,
        ml_provenance=ml_provenance,
        split_metrics=split_metrics,
        ml_diagnostics=ml_diagnostics,
        research_artefacts=research_artefacts,
        backtest_diagnostics=backtest_diagnostics,
        ml_model_diagnostics=ml_model_diagnostics,
        wf_equity_curves=wf_equity_curves,
        feature_summary=feature_summary,
        feature_registry=feature_registry,
        alignment_diagnostics=alignment_diagnostics,
        feature_correlations=feature_correlations,
        feature_families=feature_families,
        plot_index=plot_index,
        universe_coverage=universe_coverage,
        allocation_diagnostics=allocation_diagnostics,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artefact not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _copy_figures(artefacts: ExperimentArtefacts) -> list[tuple[str, Path]]:
    """Copy source PNGs to figure_dir.

    Returns (display_name, destination_path) pairs for the copied files.
    Returns empty list if no source figures exist.
    """
    if not artefacts.source_figures:
        return []
    artefacts.figure_dir.mkdir(parents=True, exist_ok=True)
    result: list[tuple[str, Path]] = []
    for src_fig in artefacts.source_figures:
        dest = artefacts.figure_dir / src_fig.name
        shutil.copy2(src_fig, dest)
        display = src_fig.stem.replace("_", " ").title()
        result.append((display, dest))
    return result


def _relative_figure_paths(
    copied_figures: list[tuple[str, Path]],
    from_dir: Path,
) -> list[tuple[str, Path]]:
    """Compute relative paths from from_dir to each copied figure.

    Called separately for markdown and HTML renderers so each gets the
    correct relative path from its own output directory.
    """
    return [
        (name, Path(os.path.relpath(dest, from_dir)))
        for name, dest in copied_figures
    ]


def _resolve_spec_name(spec: ResearchReportSpec) -> str:
    """Return the canonical preset name for a spec, or 'custom' if unrecognised."""
    from src.reporting.report_spec import (
        AUDIT_REPORT,
        CANONICAL_SHOWCASE,
        COMPACT_REPORT,
        DIAGNOSTICS_REPORT,
        FULL_DEMO_REPORT,
        STANDARD_REPORT,
    )
    _NAMES: dict[ResearchReportSpec, str] = {
        STANDARD_REPORT: "STANDARD_REPORT",
        CANONICAL_SHOWCASE: "CANONICAL_SHOWCASE",
        DIAGNOSTICS_REPORT: "DIAGNOSTICS_REPORT",
        COMPACT_REPORT: "COMPACT_REPORT",
        AUDIT_REPORT: "AUDIT_REPORT",
        FULL_DEMO_REPORT: "FULL_DEMO_REPORT",
    }
    return _NAMES.get(spec, "custom")


def _build_figure_hierarchy(plot_index: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a figure hierarchy dict from plot_index entries.

    Groups figure names by their declared importance tier.  Frontends can use
    this to decide rendering priority without parsing the full plot_index.

    Returns:
        {"primary": [...names...], "secondary": [...names...]}
    """
    primary: list[str] = []
    secondary: list[str] = []
    for entry in plot_index:
        name = entry.get("name", "")
        importance = entry.get("importance", "secondary")
        if not name:
            continue
        if importance == "primary":
            primary.append(name)
        else:
            secondary.append(name)
    return {"primary": primary, "secondary": secondary}


def _derive_validation_verdict(split_metrics: dict[str, Any] | None) -> str:
    """Derive a one-word validation verdict from split_metrics summary.

    Returns one of: "pass", "marginal", "fail", "no_validation".
    """
    if not isinstance(split_metrics, dict):
        return "no_validation"
    summary = split_metrics.get("summary") or {}
    hit_rate = summary.get("hit_rate_positive_sharpe")
    mean_sharpe = summary.get("mean_sharpe")
    if hit_rate is None and mean_sharpe is None:
        return "no_validation"
    if (hit_rate is not None and hit_rate >= 0.6) and (mean_sharpe is not None and mean_sharpe > 0.3):
        return "pass"
    if (hit_rate is not None and hit_rate >= 0.4) or (mean_sharpe is not None and mean_sharpe > 0.0):
        return "marginal"
    return "fail"


def _write_report_manifest(
    artefacts: ExperimentArtefacts,
    paths: ReportPaths,
    generated_at: str,
    report_version: str,
    copied_figures: list[tuple[str, Path]],
    sections_rendered: list[str] | None = None,
    report_spec: ResearchReportSpec | None = None,
) -> Path:
    """Write a frontend-facing manifest JSON alongside the markdown report.

    All file paths in the manifest are relative to the manifest's own directory
    (reports/markdown/) for maximum portability.

    Canonical frontend primitive fields (always present, always deterministic):
        experiment_name, experiment_type, strategy_type, report_spec,
        generated_at, report_version, artefact_version, tags,
        files, figures, metrics_summary, sections_rendered,
        validation_verdict, plot_index.

    Legacy boolean fields (kept for backward compat; prefer sections_rendered):
        has_ml, has_validation, has_diagnostics, has_research_artefacts,
        has_ml_model_diagnostics, has_wf_equity_curves, has_rolling_sharpe,
        has_train_vs_test, has_ml_plots, has_feature_summary,
        has_feature_registry, has_feature_correlations,
        has_alignment_diagnostics, has_feature_engineering.

    Returns the path to the written manifest file.
    """
    exp_name = artefacts.metadata.get("experiment_name", artefacts.artefact_dir.name)
    manifest_path = paths.markdown.parent / f"{exp_name}_manifest.json"
    md_dir = paths.markdown.parent

    # Build relative file paths from manifest directory
    files: dict[str, str] = {
        "markdown": paths.markdown.name,
        "provenance": paths.provenance.name,
    }
    if paths.html is not None:
        files["html"] = os.path.relpath(paths.html, md_dir).replace("\\", "/")

    # Figure paths relative to manifest directory
    figure_list: list[str] = [
        os.path.relpath(dest, md_dir).replace("\\", "/")
        for _, dest in copied_figures
    ]

    # Tags from config if available
    tags: list[str] = []
    if isinstance(artefacts.config, dict):
        tags = artefacts.config.get("tags") or []

    # Metrics summary — key subset only
    metrics_summary: dict[str, float | None] = {}
    _SUMMARY_KEYS = ("sharpe_ratio", "annualized_return", "annualized_volatility",
                     "max_drawdown", "calmar_ratio", "hit_rate")
    if isinstance(artefacts.metrics, dict):
        for k in _SUMMARY_KEYS:
            v = artefacts.metrics.get(k)
            if v is not None:
                metrics_summary[k] = round(float(v), 6)

    # Experiment and strategy type classification
    experiment_type = "v2_ml" if isinstance(artefacts.ml_provenance, dict) else "v1_strategy"
    strategy_type: str | None = None
    if isinstance(artefacts.config, dict):
        strategy_type = (artefacts.config.get("strategy") or {}).get("type")

    manifest: dict[str, Any] = {
        "experiment_name": exp_name,
        "experiment_type": experiment_type,
        "strategy_type": strategy_type,
        "report_spec": _resolve_spec_name(report_spec) if report_spec is not None else "STANDARD_REPORT",
        "generated_at": generated_at,
        "report_version": report_version,
        "artefact_version": ARTEFACT_VERSION,
        "tags": tags,
        "files": files,
        "figures": figure_list,
        "metrics_summary": metrics_summary,
        "sections_rendered": sections_rendered or [],
        "validation_verdict": _derive_validation_verdict(artefacts.split_metrics),
        "plot_index": artefacts.plot_index or [],
        "figure_hierarchy": _build_figure_hierarchy(artefacts.plot_index or []),
        "has_ml": isinstance(artefacts.ml_provenance, dict),
        "has_validation": isinstance(artefacts.split_metrics, dict),
        "has_diagnostics": (
            isinstance(artefacts.split_metrics, dict)
            or isinstance(artefacts.ml_diagnostics, dict)
        ),
        "has_research_artefacts": isinstance(artefacts.research_artefacts, dict),
        "has_ml_model_diagnostics": isinstance(artefacts.ml_model_diagnostics, dict),
        "has_wf_equity_curves": isinstance(artefacts.wf_equity_curves, dict),
        "has_rolling_sharpe": any(
            n in ("rolling_sharpe", "Rolling Sharpe")
            for n, _ in copied_figures
        ),
        "has_train_vs_test": any(
            n in ("train_vs_test_sharpe", "Train Vs Test Sharpe")
            for n, _ in copied_figures
        ),
        "has_ml_plots": any(
            str(n).startswith(("ml_", "Ml "))
            for n, _ in copied_figures
        ),
        "has_feature_summary": isinstance(artefacts.feature_summary, dict),
        "has_feature_registry": isinstance(artefacts.feature_registry, dict),
        "has_feature_correlations": isinstance(artefacts.feature_correlations, dict),
        "has_alignment_diagnostics": isinstance(artefacts.alignment_diagnostics, dict),
        "has_feature_engineering": (
            isinstance(artefacts.feature_summary, dict)
            or isinstance(artefacts.feature_registry, dict)
        ),
        "has_allocation_diagnostics": isinstance(artefacts.allocation_diagnostics, dict),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def _write_provenance(
    artefacts: ExperimentArtefacts,
    provenance_path: Path,
    generated_at: str,
) -> None:
    """Write provenance JSON sidecar alongside the markdown output."""
    exp_name = artefacts.metadata.get("experiment_name", artefacts.artefact_dir.name)
    # config_hash is not currently persisted in metadata.json; record None honestly
    config_hash: str | None = artefacts.metadata.get("config_hash")

    provenance: dict[str, Any] = {
        "report_version": _REPORT_VERSION,
        "artefact_version": ARTEFACT_VERSION,
        "generated_at": generated_at,
        "source_experiment": exp_name,
        "config_hash": config_hash,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def generate_experiment_report(
    artefact_path: str | Path,
    output_dir: str | Path = Path("reports"),
    include_html: bool = True,
    report_spec: ResearchReportSpec | None = None,
) -> ReportPaths:
    """Generate markdown (and optionally HTML) reports from saved experiment artefacts.

    This function is strictly read-only with respect to experiment artefacts.
    It copies figures but never regenerates or recomputes them.

    All figure paths relative to each renderer's output directory are computed
    here and passed into the renderers — renderers make no filesystem assumptions.

    Args:
        artefact_path: Path to the saved experiment directory.
        output_dir: Root directory for generated reports.  Sub-directories
            markdown/, html/, and figures/ are created as needed.
        include_html: If True (default), also write an HTML version.

    Returns:
        ReportPaths with absolute paths to all generated files.

    Raises:
        FileNotFoundError: If the experiment directory or required artefacts
            are missing.
    """
    from src.reporting.report_spec import STANDARD_REPORT

    artefacts = load_experiment_artefacts(artefact_path, output_dir)
    generated_at = datetime.now(UTC).isoformat()

    # Resolve None → canonical default so spec identity is explicit everywhere
    effective_spec: ResearchReportSpec = report_spec if report_spec is not None else STANDARD_REPORT

    # Copy figures once; compute renderer-specific relative paths centrally
    copied_figures = _copy_figures(artefacts)
    md_figure_paths = _relative_figure_paths(
        copied_figures, artefacts.markdown_path.parent
    )

    # --- Markdown ---
    md_str = render_report(
        artefacts, md_figure_paths, generated_at, _REPORT_VERSION,
        report_spec=effective_spec,
    )
    artefacts.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    artefacts.markdown_path.write_text(md_str, encoding="utf-8")

    # Extract rendered section headings for manifest (## Heading lines)
    sections_rendered = [
        line[3:].strip()
        for line in md_str.splitlines()
        if line.startswith("## ")
    ]

    # --- Provenance sidecar (alongside markdown) ---
    exp_name = artefacts.metadata.get("experiment_name", artefacts.artefact_dir.name)
    provenance_path = artefacts.markdown_path.parent / f"{exp_name}_provenance.json"
    _write_provenance(artefacts, provenance_path, generated_at)

    # --- HTML (optional) ---
    html_path: Path | None = None
    if include_html:
        html_figure_paths = _relative_figure_paths(
            copied_figures, artefacts.html_path.parent
        )
        # render_report is pure; calling it again with HTML-relative paths is correct
        md_for_html = render_report(
            artefacts, html_figure_paths, generated_at, _REPORT_VERSION,
            report_spec=effective_spec,
        )
        title = f"Experiment Report: {exp_name}"
        html_str = markdown_to_html(md_for_html, title=title)
        artefacts.html_path.parent.mkdir(parents=True, exist_ok=True)
        artefacts.html_path.write_text(html_str, encoding="utf-8")
        html_path = artefacts.html_path

    # --- Frontend manifest ---
    report_paths_partial = ReportPaths(
        markdown=artefacts.markdown_path,
        html=html_path,
        provenance=provenance_path,
    )
    manifest_path = _write_report_manifest(
        artefacts, report_paths_partial, generated_at, _REPORT_VERSION,
        copied_figures, sections_rendered=sections_rendered,
        report_spec=effective_spec,
    )

    return ReportPaths(
        markdown=artefacts.markdown_path,
        html=html_path,
        provenance=provenance_path,
        manifest=manifest_path,
    )
