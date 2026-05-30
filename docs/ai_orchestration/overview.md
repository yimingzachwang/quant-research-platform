# AI Orchestration Layer — Overview

## Purpose

The `src/orchestration/` package provides a structured API between the quantitative research engine and LLM-assisted interpretation. The quantitative engine remains deterministic and authoritative; the LLM receives only curated, structured context.

## Architecture

```
User / LLM Frontend
        │
        ▼
src/orchestration/api/research_api.py    ← Single entry-point
        │
        ├─► registry/          ← Experiment discovery and listing
        ├─► retrieval/         ← Filesystem artefact loading
        ├─► context/           ← Deterministic semantic summarizers
        └─► llm/               ← Provider abstraction + review engine
                │
                ▼
        results/llm_reviews/<experiment>/
            llm_context.json   ← Structured context bundle
            llm_review.json    ← Persisted LLM output
```

## Key Invariants

- **LLM never calls the quant engine.** Only reads persisted artefacts from `results/experiments/`.
- **No computation in the orchestration layer.** Context is assembled from pre-computed artefacts.
- **Provider abstraction.** Set `provider="anthropic"` or `provider="openai"` and supply the appropriate API key.
- **Filesystem-first persistence.** All context and review outputs are written to `results/llm_reviews/`.
- **Graceful degradation.** Missing artefacts produce None; the API never raises for absent files.

## Quick Start

```python
from src.orchestration import run_llm_review

# Stub provider (no API key required — for testing)
review = run_llm_review("canonical_ml_multi_asset", provider="stub")
print(review.review_text)

# Anthropic Claude (requires ANTHROPIC_API_KEY)
review = run_llm_review("canonical_ml_multi_asset", provider="anthropic")

# OpenAI (requires OPENAI_API_KEY)
review = run_llm_review("canonical_ml_multi_asset", provider="openai")
```

## Module Map

| Module | Responsibility |
|--------|---------------|
| `api/research_api.py` | Top-level API — single import point for all orchestration |
| `api/schemas.py` | Typed dataclasses (ExperimentSummary, LLMContext, etc.) |
| `registry/experiment_registry.py` | List, find, and rank experiments from disk |
| `registry/artefact_registry.py` | Static catalog of known artefact types |
| `retrieval/artefact_retriever.py` | Load any artefact by key |
| `retrieval/diagnostics_retriever.py` | Load all diagnostics and research JSONs |
| `retrieval/plot_retriever.py` | Load plot index and metadata |
| `retrieval/manifest_retriever.py` | Load report manifests |
| `context/context_builder.py` | Assemble LLMContext from disk artefacts |
| `context/metric_summarizer.py` | Performance metric interpretation |
| `context/validation_summarizer.py` | Walk-forward validation summary |
| `context/ml_diagnostic_summarizer.py` | ML model diagnostic summary |
| `context/failure_mode_detector.py` | Rule-based failure mode detection |
| `llm/llm_interface.py` | Provider abstraction (Anthropic / OpenAI / Stub) |
| `llm/review_engine.py` | Render prompt, call LLM, parse and persist output |
| `llm/prompt_templates/` | Jinja2 prompt templates |
| `utils/filesystem.py` | Path resolution for all experiment artefacts |
| `utils/serialization.py` | Safe JSON/Parquet loading |

## Environment Variables

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `OPENAI_API_KEY` | OpenAI |

## Output Layout

```
results/llm_reviews/
    <experiment_name>/
        llm_context.json    ← Structured context (built from artefacts)
        llm_review.json     ← LLM review output (sections + flags + text)
```
