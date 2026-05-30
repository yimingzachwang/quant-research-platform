"""Prompt template registry."""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent

EXPERIMENT_REVIEW = "experiment_review"
ITERATION_PROPOSAL = "iteration_proposal"
COMPARATIVE_REVIEW = "comparative_review"

_FILES = {
    EXPERIMENT_REVIEW: _TEMPLATE_DIR / "experiment_review.txt",
    ITERATION_PROPOSAL: _TEMPLATE_DIR / "iteration_proposal.txt",
    COMPARATIVE_REVIEW: _TEMPLATE_DIR / "comparative_review.txt",
}


def load_template(name: str) -> str:
    path = _FILES.get(name)
    if path is None:
        raise KeyError(f"No prompt template named {name!r}. Available: {list(_FILES)}")
    return path.read_text()
