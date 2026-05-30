"""FastAPI application entry point.

Creates the app, registers all routers, and exposes a health endpoint.

Run locally with::

    uvicorn src.api.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routers import drafts, experiments, reviews, routing, sessions

app = FastAPI(
    title="Quant Research Platform API",
    description=(
        "Thin HTTP bridge over the research orchestration backend. "
        "All endpoints delegate directly to Research API functions. "
        "No business logic lives in this layer."
    ),
    version="5.0.0",
)

# ---------------------------------------------------------------------------
# Exception handler — never expose stack traces to clients
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["health"])
def health() -> dict:
    """Confirm the API process is alive."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(sessions.router, prefix="/api")
app.include_router(reviews.router, prefix="/api")
app.include_router(drafts.router, prefix="/api")
app.include_router(routing.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
