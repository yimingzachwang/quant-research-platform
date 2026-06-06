# AI Orchestration Architecture Audit

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Subject:** `src/orchestration/` — AI-assisted quantitative research layer

---

## 1. Purpose and Scope

The orchestration layer is a read-only, advisory AI subsystem layered on top of the quantitative research engine. Its sole function is to assist a human researcher in interpreting experiment results. It never executes experiments, never modifies data, and never touches the backtest or ML pipeline.

This document audits the implementation as shipped: what exists, how the pieces connect, and where the boundaries are enforced.

---

## 2. Module Inventory

| Subpackage | Files | Role |
|---|---|---|
| `api/` | `research_api.py`, `schemas.py`, `experiment_loader.py`, `comparison_api.py`, `artefact_api.py` | Single external entry point; all typed dataclasses |
| `context/` | `context_builder.py`, `failure_mode_detector.py`, `context_schema.py`, `metric_summarizer.py`, `ml_diagnostic_summarizer.py`, `validation_summarizer.py` | Artefact assembly and deterministic diagnostics |
| `llm/` | `llm_interface.py`, `review_engine.py`, `comparison_engine.py`, `iteration_engine.py`, `review_schema.py`, `prompt_templates/` | LLM calls, prompt rendering, output parsing |
| `config_generation/` | `draft_generator.py`, `draft_schema.py`, `draft_validator.py`, `yaml_renderer.py` | Config draft synthesis (advisory only) |
| `evolution/` | `evolution_builder.py` | Deterministic research lineage chain |
| `intents/` | `intent_parser.py`, `intent_schema.py`, `intent_examples.py` | Natural-language request classification |
| `router/` | `workflow_router.py`, `routing_schema.py` | Intent-to-API dispatch |
| `session/` | `session_manager.py`, `session_schema.py` | Optional research session tracking |
| `registry/` | `experiment_registry.py`, `artefact_registry.py` | Experiment enumeration and artefact lookup |
| `retrieval/` | `artefact_retriever.py`, `diagnostics_retriever.py`, `manifest_retriever.py`, `plot_retriever.py` | Typed artefact loading |
| `utils/` | `filesystem.py`, `serialization.py` | Path resolution and JSON I/O |

**Total subpackages:** 11. All imports flow inward toward `utils/`; no subpackage imports from a peer at the same level except through `api/`.

---

## 3. Dependency Architecture

```
External caller (API router, CLI, test)
    ↓
src/orchestration/api/research_api.py          ← SOLE public entry point
    ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestration internals                      │
│                                                                  │
│  context/context_builder  →  context/failure_mode_detector      │
│         ↓                                                        │
│  llm/review_engine        →  llm/llm_interface                  │
│  llm/comparison_engine    →  llm/prompt_templates/              │
│  llm/iteration_engine                                            │
│         ↓                                                        │
│  config_generation/draft_generator                               │
│         ↓                                                        │
│  evolution/evolution_builder                                     │
│  session/session_manager                                         │
│  intents/intent_parser    →  router/workflow_router             │
│  registry/ + retrieval/                                          │
│                                                                  │
│  utils/filesystem         (path resolution — no logic)          │
│  utils/serialization      (JSON I/O — no logic)                 │
└─────────────────────────────────────────────────────────────────┘
    ↓
results/experiments/<name>/    ← READ ONLY
results/llm_reviews/<name>/    ← WRITE (LLM outputs)
results/comparisons/           ← WRITE (comparative reviews)
results/evolution/             ← WRITE (evolution chains)
results/research_sessions/     ← WRITE (session logs)
```

**Architectural invariant:** `src/orchestration/` never imports from `src/strategies/`, `src/models/`, `src/backtest/`, `src/features/`, `src/data/`, or `src/validation/`. This is verified programmatically by `tests/integration/test_api_non_coupling.py`.

---

## 4. Entry Point: `research_api.py`

All external callers (FastAPI routers, CLI scripts, tests) invoke functions from `research_api.py` only. Internal subpackage functions are not part of the public API.

**Public functions:**

| Function | LLM call? | Persists? |
|---|---|---|
| `build_llm_context(name)` | No | Optional |
| `run_llm_review(name, provider, model)` | Yes | Yes |
| `run_llm_comparative_review(baseline, candidate, ...)` | Yes | Yes |
| `generate_iteration_proposal(name, provider, model)` | Yes | Yes |
| `register_experiment_lineage(name, parent, ...)` | No | Yes |
| `build_research_evolution_chain(root, ...)` | No | Optional |
| `generate_experiment_draft(name, provider, ...)` | Yes | Yes |
| `approve_experiment_draft(name, draft_id, ...)` | No | Yes |
| `render_draft_to_yaml(name, draft_id, ...)` | No | Yes |
| `list_all_experiments(base)` | No | No |
| `find_experiments(tag, strategy_pattern, ...)` | No | No |
| `rank_experiments_by_sharpe(base, descending)` | No | No |
| `retrieve_artefact(name, key, base)` | No | No |
| `create_research_session(...)` | No | Yes |
| `record_session_event(...)` | No | Yes |
| `update_session_status(...)` | No | Yes |
| `get_session_summary(session_id, ...)` | No | No |

---

## 5. Data Flow: Context Assembly

Context assembly is fully deterministic and runs before any LLM call:

```
results/experiments/<name>/
    metadata.json            → experiment_name, strategy_name, created_at, tags
    metrics.json             → sharpe, max_drawdown, annual_return
    diagnostics/
        ml_diagnostics.json  → IC, directional accuracy, coefficient stability
        split_metrics.json   → per-split OOS Sharpe
        backtest_diagnostics.json
        universe_coverage.json
    research/
        feature_summary.json
        alignment_diagnostics.json
    plots/
        plot_index.json      → PlotMetadata (name, group, importance, caption)

    → context_builder.build_context()
    → LLMContext (pure Python primitives only; no DataFrames, no numpy)
    → failure_mode_detector (deterministic rules — no LLM)
    → SHA256 context hash
```

**`LLMContext` fields** (all plain dicts/lists/strings/numbers):

- `experiment_name`, `strategy_name`, `created_at`, `tags`
- `performance`: `{sharpe, max_drawdown, annual_return, ...}`
- `validation`: `{mean_oos_sharpe, std_oos_sharpe, n_splits, ...}`
- `ml_diagnostics`: `{ic, directional_accuracy, coefficient_stability, feature_contributions}`
- `feature_summary`: `{families, dominant_family, mean_hhi, ...}`
- `universe_summary`: `{n_assets, asset_tickers, mean_coverage_pct}`
- `available_plots`: list of `PlotMetadata` (importance == "primary" only)
- `failure_modes`: list of `FailureMode` (deterministic)

---

## 6. LLM Interface

**File:** `src/orchestration/llm/llm_interface.py`

This is the only file that imports any vendor SDK. All other orchestration code goes through `call_llm()`.

**Providers:**

| Provider ID | SDK | Default model | API key env var |
|---|---|---|---|
| `"anthropic"` | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `"openai"` | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `"stub"` | (none) | `"stub"` | (none required) |

The stub provider is deterministic, returns a fixed placeholder string, and is used in all tests. No external API call is ever made in CI.

The OpenAI provider supports `base_url` override for local OpenAI-compatible endpoints (e.g., LM Studio).

---

## 7. Prompt Infrastructure

**File:** `src/orchestration/llm/prompt_templates/__init__.py`

Three Jinja2 templates registered by string constant:

| Constant | File | Used by |
|---|---|---|
| `EXPERIMENT_REVIEW` | `experiment_review.txt` | `review_engine.run_review()` |
| `ITERATION_PROPOSAL` | `iteration_proposal.txt` | `iteration_engine.run_iteration_proposal()` |
| `COMPARATIVE_REVIEW` | `comparative_review.txt` | `comparison_engine.run_comparative_review()` |

All templates are rendered with `jinja2.Environment(undefined=StrictUndefined)`. Any template variable that references a field not present in the render context raises `RuntimeError` at prompt construction time — before the LLM call is made.

After rendering, `_assert_no_unresolved_tokens()` performs a regex check for `{{`, `}}`, `{%`, `%}` sequences. If any survive rendering, `RuntimeError` is raised.

---

## 8. Output Artefact Tree

```
results/
├── llm_reviews/
│   └── <experiment_name>/
│       ├── llm_context.json         (optional persist)
│       ├── llm_review.json          (review output + provenance)
│       ├── iteration_proposal.json  (iteration output + provenance)
│       ├── iteration_proposal.md    (raw LLM markdown)
│       └── draft_<uuid>.json        (config draft)
├── comparisons/
│   └── <baseline>__vs__<candidate>/
│       ├── comparative_review.json
│       └── comparative_review.md
├── evolution/
│   └── <root_experiment>/
│       ├── evolution_chain.json
│       └── evolution_chain.md
└── research_sessions/
    └── <session_id>/
        └── session.json
```

All write paths use `utils/filesystem.py` path helpers — no hardcoded string paths in engine code.

---

## 9. Boundary Enforcement

| Boundary | Mechanism | Test |
|---|---|---|
| Orchestration never calls quant engine | Import-level: `src/orchestration/` has zero imports from quant engine subpackages | `test_api_non_coupling.py` |
| LLM is never called during failure mode detection | `failure_mode_detector.py` has no `llm_interface` import | Structural: no import chain |
| Broken context never reaches LLM | `StrictUndefined` + `_assert_no_unresolved_tokens` raise before `call_llm()` | `test_iteration.py:41`, `test_review.py` |
| Draft requires human approval | `ExperimentDraft.approved = False` default; `render_to_yaml` raises if not approved | `test_config_generation.py` |
| Lineage is human-triggered | `register_experiment_lineage()` is not called by any other API function | Source inspection |
| LLM output is advisory | No API function calls quant engine after receiving LLM response | Source inspection + coupling test |
