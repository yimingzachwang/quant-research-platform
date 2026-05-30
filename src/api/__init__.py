"""FastAPI bridge for the Quant Research Platform orchestration backend.

Thin HTTP transport layer over the existing Research API.  No business logic
lives here — every route is a direct wrapper over an orchestration function.

Usage::

    uvicorn src.api.app:app --reload
"""
