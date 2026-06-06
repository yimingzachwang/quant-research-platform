"""Build compact research-memory records from existing Zeto artefacts.

The indexer reads ONLY known Zeto artefact locations and produces compact
``MemoryRecord`` pointers.  It never:

  * executes experiments or calls any LLM;
  * inspects arbitrary paths, raw data, parquet, plots, or secrets;
  * mutates the source artefacts;
  * streams full artefact contents into a record (summaries are truncated).

Indexed sources (and the artefact_type assigned to each):
  results/experiments/*/metadata.json        -> experiment_metadata
  results/experiments/*/metrics.json         -> experiment_metrics
  results/llm_reviews/*/llm_review.json       -> llm_review
  results/llm_reviews/*/iteration_proposal.json -> iteration_proposal
  results/llm_reviews/*/draft_*.json          -> draft
  reports/markdown/*.md                        -> report
  results/research_sessions/*/session.json     -> session
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from src.orchestration.memory.memory_schema import (
    SUMMARY_MAX_CHARS,
    MemoryRecord,
    compute_memory_id,
)
from src.orchestration.utils.filesystem import (
    experiments_root,
    llm_reviews_root,
    reports_markdown_dir,
    sessions_root,
)
from src.orchestration.utils.serialization import load_json

logger = logging.getLogger(__name__)

# Small controlled keyword -> tag vocabulary.  Deterministic; no inference.
# Each entry maps a lowercase substring to a normalised research tag.
_TAG_VOCAB: tuple[tuple[str, str], ...] = (
    ("validation", "validation"),
    ("out-of-sample", "oos_consistency"),
    ("out of sample", "oos_consistency"),
    ("oos", "oos_consistency"),
    ("split", "splits"),
    ("drawdown", "drawdown"),
    ("sharpe", "sharpe"),
    ("regularis", "regularisation"),
    ("regulariz", "regularisation"),
    ("ridge", "regularisation"),
    ("feature", "features"),
    ("turnover", "turnover"),
    ("instab", "instability"),
    ("variance", "variance"),
    ("directional", "directional_accuracy"),
    ("coefficient", "coefficient_stability"),
    ("volatility", "volatility"),
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_memory_records(
    base: Path | str | None = None,
    llm_base: Path | str | None = None,
    reports_base: Path | str | None = None,
    sessions_base: Path | str | None = None,
) -> list[MemoryRecord]:
    """Scan the known artefact locations and return compact memory records.

    Deterministic and read-only: each source artefact maps to exactly one record
    with a stable id, so a re-index produces the same set with no duplicates.
    """
    records: list[MemoryRecord] = []
    records.extend(_index_experiments(base))
    records.extend(_index_llm_reviews(llm_base))
    records.extend(_index_reports(reports_base))
    records.extend(_index_sessions(sessions_base))
    return records


# ---------------------------------------------------------------------------
# Per-source indexers
# ---------------------------------------------------------------------------


def _index_experiments(base: Path | str | None) -> list[MemoryRecord]:
    root = experiments_root(base)
    if not root.exists():
        return []
    records: list[MemoryRecord] = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = exp_dir.name
        meta = load_json(exp_dir / "metadata.json")
        if isinstance(meta, dict):
            strategy = meta.get("strategy_name") or "(unknown strategy)"
            summary = f"Experiment '{name}' — strategy {strategy}."
            records.append(_make_record(
                artefact_type="experiment_metadata",
                path=exp_dir / "metadata.json",
                experiment_name=name,
                created_at=meta.get("created_at", ""),
                summary=summary,
                tag_text=f"{name} {strategy}",
            ))
        metrics = load_json(exp_dir / "metrics.json")
        if isinstance(metrics, dict):
            summary = _metrics_summary(name, metrics)
            records.append(_make_record(
                artefact_type="experiment_metrics",
                path=exp_dir / "metrics.json",
                experiment_name=name,
                summary=summary,
                tag_text=summary,
            ))
    return records


def _index_llm_reviews(llm_base: Path | str | None) -> list[MemoryRecord]:
    root = llm_reviews_root(llm_base)
    if not root.exists():
        return []
    records: list[MemoryRecord] = []
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = exp_dir.name

        review = load_json(exp_dir / "llm_review.json")
        if isinstance(review, dict):
            failure_modes = _failure_modes_from_flags(review.get("flags", []))
            summary = _review_summary(failure_modes)
            records.append(_make_record(
                artefact_type="llm_review",
                path=exp_dir / "llm_review.json",
                experiment_name=name,
                created_at=review.get("generated_at", ""),
                context_hash=review.get("context_hash", "") or "",
                summary=summary,
                failure_modes=failure_modes,
                tag_text=f"{summary} {' '.join(failure_modes)}",
            ))

        proposal = load_json(exp_dir / "iteration_proposal.json")
        if isinstance(proposal, dict):
            summary = _proposal_summary(proposal)
            records.append(_make_record(
                artefact_type="iteration_proposal",
                path=exp_dir / "iteration_proposal.json",
                experiment_name=name,
                created_at=proposal.get("generated_at", ""),
                context_hash=proposal.get("context_hash", "") or "",
                summary=summary,
                tag_text=summary,
            ))

        for draft_path in sorted(exp_dir.glob("draft_*.json")):
            draft = load_json(draft_path)
            if not isinstance(draft, dict):
                continue
            summary = _draft_summary(draft)
            records.append(_make_record(
                artefact_type="draft",
                path=draft_path,
                experiment_name=draft.get("base_experiment") or name,
                created_at=draft.get("generated_at", ""),
                context_hash=draft.get("draft_hash", "") or "",
                summary=summary,
                tag_text=summary,
            ))
    return records


def _index_reports(reports_base: Path | str | None) -> list[MemoryRecord]:
    root = reports_markdown_dir(reports_base)
    if not root.exists():
        return []
    records: list[MemoryRecord] = []
    for md_path in sorted(root.glob("*.md")):
        name = md_path.stem
        summary = _report_summary(md_path)
        records.append(_make_record(
            artefact_type="report",
            path=md_path,
            experiment_name=name,
            created_at=_mtime_iso(md_path),
            summary=summary,
            tag_text=f"{name} {summary}",
        ))
    return records


def _index_sessions(sessions_base: Path | str | None) -> list[MemoryRecord]:
    root = sessions_root(sessions_base)
    if not root.exists():
        return []
    records: list[MemoryRecord] = []
    for sess_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = load_json(sess_dir / "session.json")
        if not isinstance(data, dict):
            continue
        experiment_name = (
            data.get("active_experiment") or data.get("root_experiment") or ""
        )
        summary = _session_summary(data)
        records.append(_make_record(
            artefact_type="session",
            path=sess_dir / "session.json",
            experiment_name=experiment_name,
            created_at=data.get("updated_at", "") or data.get("created_at", ""),
            session_id=data.get("session_id"),
            summary=summary,
            tag_text=summary,
        ))
    return records


# ---------------------------------------------------------------------------
# Record + field helpers
# ---------------------------------------------------------------------------


def _make_record(
    artefact_type: str,
    path: Path,
    experiment_name: str,
    summary: str,
    tag_text: str = "",
    created_at: str = "",
    context_hash: str = "",
    session_id: str | None = None,
    failure_modes: list[str] | None = None,
) -> MemoryRecord:
    path_str = str(path)
    failure_modes = failure_modes or []
    tags = _derive_tags(tag_text or summary, failure_modes)
    return MemoryRecord(
        memory_id=compute_memory_id(artefact_type, path_str),
        experiment_name=experiment_name,
        artefact_type=artefact_type,
        path=path_str,
        short_summary=_truncate(summary),
        created_at=created_at or _mtime_iso(path),
        context_hash=context_hash,
        session_id=session_id,
        failure_modes=failure_modes,
        tags=tags,
    )


def _failure_modes_from_flags(flags: list) -> list[str]:
    """Extract bare failure-mode names from review flags.

    Flags look like ``"CRITICAL: poor_oos_consistency"``; we keep only the name
    (``poor_oos_consistency``), deduped and order-preserving.
    """
    names: list[str] = []
    for flag in flags or []:
        if not isinstance(flag, str):
            continue
        name = flag.split(":", 1)[-1].strip()
        if name and name not in names:
            names.append(name)
    return names


def _derive_tags(text: str, failure_modes: list[str]) -> list[str]:
    """Deterministic controlled-vocabulary tag extraction."""
    lowered = (text or "").lower()
    tags: list[str] = list(failure_modes)
    for keyword, tag in _TAG_VOCAB:
        if keyword in lowered and tag not in tags:
            tags.append(tag)
    return tags


def _metrics_summary(name: str, metrics: dict) -> str:
    def fmt(key: str) -> str:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return f"{val:.3f}"
        return "n/a"
    return (
        f"Metrics for '{name}': sharpe={fmt('sharpe_ratio')}, "
        f"max_drawdown={fmt('max_drawdown')}, "
        f"annualized_return={fmt('annualized_return')}."
    )


def _review_summary(failure_modes: list[str]) -> str:
    if failure_modes:
        return "LLM review flagged: " + ", ".join(failure_modes) + "."
    return "LLM review found no critical failure-mode flags."


def _proposal_summary(proposal: dict) -> str:
    focus = (proposal.get("research_focus") or "").strip()
    if focus:
        return f"Iteration proposal: {focus}"
    rationale = (proposal.get("rationale") or "").strip()
    if rationale:
        return f"Iteration proposal: {rationale}"
    return "Iteration proposal (no stated focus)."


def _draft_summary(draft: dict) -> str:
    proposed = draft.get("proposed_name") or "(unnamed)"
    changes = draft.get("changes") or []
    fields = []
    for ch in changes:
        if isinstance(ch, dict):
            section = ch.get("section", "")
            field = ch.get("field", "")
            fields.append(f"{section}.{field}".strip("."))
    diff = ", ".join(fields[:4]) or "no changes"
    return f"Config draft '{proposed}' changing {diff}."


def _report_summary(md_path: Path) -> str:
    """Compact one-line summary from a report's first meaningful line.

    Reads only the leading lines and never returns the full report body.
    """
    try:
        first = ""
        with md_path.open(encoding="utf-8") as fh:
            for _ in range(50):  # bounded read — never the whole report
                raw = fh.readline()
                if not raw:
                    break
                line = raw.strip().lstrip("#").strip()
                if line:
                    first = line
                    break
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read report head %s: %s", md_path, exc)
        first = ""
    title = first or md_path.stem
    return f"Report: {title}"


def _session_summary(data: dict) -> str:
    goal = (data.get("research_goal") or "").strip()
    status = data.get("status", "")
    n_events = len(data.get("events", []) or [])
    head = f"Research session ({status}, {n_events} event(s))"
    return f"{head}: {goal}" if goal else f"{head}."


def _mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _truncate(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
