"""Experiment discovery endpoints.

Wraps:
  list_all_experiments()
  rank_experiments_by_sharpe()
  get_experiment_summary()

Routes are ordered so the literal path /ranked is matched before the
parametric path /{name}/summary.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import _asdict
from src.orchestration.api import research_api as _api

router = APIRouter(prefix="/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# GET /api/experiments
# ---------------------------------------------------------------------------


@router.get("")
def list_experiments() -> dict:
    return {"experiments": _api.list_all_experiments()}


# ---------------------------------------------------------------------------
# GET /api/experiments/ranked  (must be declared before /{name}/summary)
# ---------------------------------------------------------------------------


@router.get("/ranked")
def get_ranked(by: str = Query(default="sharpe"), limit: int = Query(default=10)) -> dict:
    if by != "sharpe":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ranking field '{by}'. Only 'sharpe' is supported.",
        )
    summaries = _api.rank_experiments_by_sharpe()
    return {"experiments": [_asdict(s) for s in summaries[:limit]]}


# ---------------------------------------------------------------------------
# GET /api/experiments/{name}/summary
# ---------------------------------------------------------------------------


@router.get("/{name}/summary")
def get_summary(name: str) -> dict:
    summary = _api.get_experiment_summary(name)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{name}' not found",
        )
    return {"summary": _asdict(summary)}
