# AI Orchestration Research Cycle Audit

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Subject:** End-to-end research workflow through the AI orchestration layer

---

## 1. Overview

The orchestration layer supports two research workflows:

**Linear research cycle** (single experiment):
```
build_context → run_llm_review → generate_iteration_proposal
    → generate_experiment_draft → approve_experiment_draft → render_draft_to_yaml
```

**Comparative research cycle** (two experiments):
```
build_context (A) + build_context (B) → run_llm_comparative_review
    → [human reads tradeoffs] → register_experiment_lineage → build_research_evolution_chain
```

Both workflows are advisory only. The YAML output from `render_draft_to_yaml` is a config file for the researcher to review before manually running an experiment.

---

## 2. Step 1 — Context Assembly

**Function:** `research_api.build_llm_context(experiment_name, base, persist)`

**Source:** `context/context_builder.py`

The context builder reads from the experiment's artefact tree and assembles a typed `LLMContext`. No LLM is involved.

**Reads:**

| File | Fields extracted |
|---|---|
| `metadata.json` | `experiment_name`, `strategy_name`, `created_at`, `tags` |
| `metrics.json` | `sharpe`, `max_drawdown`, `annual_return`, `annual_volatility`, `calmar` |
| `diagnostics/ml_diagnostics.json` | `ic.mean_ic`, `ic.ic_tier`, `directional_accuracy.*`, `coefficient_stability.*` |
| `diagnostics/split_metrics.json` | `mean_oos_sharpe`, `std_oos_sharpe`, `n_splits`, `n_negative_sharpe_splits`, `worst_split_sharpe`, `consistency_tier` |
| `diagnostics/backtest_diagnostics.json` | `monthly_avg_turnover`, drawdown severity tier |
| `research/feature_summary.json` | `dominant_family`, `mean_hhi`, `n_family_transitions`, `most_volatile_feature`, `concentration_tier` |
| `research/alignment_diagnostics.json` | `alignment_loss_pct` |
| `diagnostics/universe_coverage.json` | `n_assets`, `asset_tickers`, `mean_coverage_pct` |
| `plots/plot_index.json` | `PlotMetadata` — name, group, importance, caption; filtered to `importance == "primary"` |

**Failure mode detection** runs immediately after context assembly. It is deterministic, rule-based, and has no LLM dependency. Results are embedded in `LLMContext.failure_modes`.

**`_prune_nulls()`** recursively removes `None`, empty dicts, and empty lists from the assembled payload before it is passed to any downstream component. Zero values and `False` are preserved.

**If `persist=True`**, the assembled `LLMContext` is written to `results/llm_reviews/<name>/llm_context.json`.

---

## 3. Step 2 — LLM Review

**Function:** `research_api.run_llm_review(experiment_name, provider, model, base, persist)`

**Source:** `llm/review_engine.py`

**Sequence:**

1. Call `build_context()` — assembles `LLMContext`
2. Compute SHA256 context hash (`_compute_context_hash`) — deterministic provenance ID
3. Load template `EXPERIMENT_REVIEW` from `prompt_templates/`
4. Render with `jinja2.Environment(undefined=StrictUndefined)` — raises `RuntimeError` on any missing variable
5. Assert no unresolved tokens survive rendering
6. Call `call_llm(prompt, provider, model, max_tokens=4096, temperature=0.2)`
7. Extract `###` sections from response into keyed dict
8. Build `LLMReviewOutput` with sections + provenance fields
9. If `persist=True`, write `llm_review.json` to `results/llm_reviews/<name>/`

**`LLMReviewOutput` sections:**

- `performance_interpretation`
- `signal_quality`
- `validation_assessment`
- `failure_mode_analysis`
- `feature_contribution_analysis`
- `recommendations`
- `flags` (list of failure mode name strings, derived from `LLMContext.failure_modes`)

**Provenance fields written to JSON:**

```json
{
  "experiment_name": "...",
  "generated_at": "2026-05-30T...",
  "context_hash": "<64-char SHA256>",
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "prompt_template": "experiment_review",
  "review_version": "1.0",
  "sections": { ... },
  "flags": [ ... ]
}
```

---

## 4. Step 3 — Iteration Proposal

**Function:** `research_api.generate_iteration_proposal(experiment_name, provider, model, base, persist)`

**Source:** `llm/iteration_engine.py`

The iteration engine mirrors the review engine in architecture. It uses the same context hash, the same rendering guard, and the same persistence philosophy.

**Sequence:**

1. Call `build_context()` — same assembly as review
2. Render template `ITERATION_PROPOSAL` with `StrictUndefined`
3. Call `call_llm(prompt, provider, model, max_tokens=2048, temperature=0.3)` (lower temperature than review for reproducibility)
4. Parse `###` sections from response into `IterationProposal`
5. If `persist=True`, write `iteration_proposal.json` + `iteration_proposal.md`

**`IterationProposal` fields:**

| Field | Type | Source |
|---|---|---|
| `research_focus` | str | `### Research Focus` section |
| `rationale` | str | `### Rationale` section |
| `supporting_evidence` | list[str] | `### Supporting Evidence` bullets |
| `suggested_experiments` | list[str] | `### Suggested Experiments` bullets |
| `instability_signals` | list[str] | `### Instability Signals` bullets |
| `validation_concerns` | list[str] | `### Validation Concerns` bullets |
| `feature_risks` | list[str] | `### Feature Risks` bullets |
| `confidence` | str | `### Confidence` section |
| `experiment_name` | str | From context |
| `context_hash` | str | SHA256 |
| `generated_at` | str | ISO timestamp |
| `provider`, `model`, `prompt_template` | str | Provenance |

---

## 5. Step 4 — Config Draft Synthesis

**Function:** `research_api.generate_experiment_draft(experiment_name, provider, model, proposal_hash, ...)`

**Source:** `config_generation/draft_generator.py`

This step converts an `IterationProposal` into a typed list of config parameter changes. The LLM receives a structured prompt with the base config and the proposal; it returns JSON only.

**Sequence:**

1. Load YAML config from `configs/experiments/<name>.yaml` (must be version 2)
2. Load `IterationProposal` from `results/llm_reviews/<name>/iteration_proposal.json`
3. If `proposal_hash` provided, verify hash matches loaded proposal
4. Build prompt: experiment name + config JSON summary + proposal fields
5. Call `call_llm(prompt, system=_LLM_SYSTEM, max_tokens=1024, temperature=0.1)` — lowest temperature for deterministic extraction
6. Parse JSON response: strip markdown fences, parse `{"proposed_name": ..., "changes": [...]}`
7. For each change, read `current_value` from base config (not from LLM)
8. Assemble `ExperimentDraft` with `approved=False`
9. Write `draft_<uuid>.json` to `results/llm_reviews/<name>/`

**LLM system prompt restricts changes to a fixed vocabulary:**

```
model                  → type, params.alpha, params.C, params.l1_ratio, params.max_iter
labels                 → type, params.horizon
signal                 → type, params.n, params.n_long, params.n_short, params.threshold
validation             → parameters.train_months, parameters.test_months, parameters.gap_days
execution              → transaction_cost_bps
portfolio_construction → weighting.scheme, weighting.prediction_normalization, weighting.temperature
features               → entries.add, entries.remove
```

Any LLM response outside this vocabulary is ignored.

**`current_value` is always read from the base config by Python code — never from the LLM.** The LLM only proposes `proposed_value` and `rationale`.

---

## 6. Step 5 — Human Approval Gate

**Function:** `research_api.approve_experiment_draft(experiment_name, draft_id, ...)`

**Source:** `api/research_api.py` (delegates to `config_generation/`)

Sets `ExperimentDraft.approved = True` and `approved_at = <ISO timestamp>`. Persists the updated draft JSON. No LLM involvement.

This is an explicit, human-triggered action. The draft file is not modified automatically by any other function.

---

## 6. Step 6 — YAML Rendering

**Function:** `research_api.render_draft_to_yaml(experiment_name, draft_id, output_path, ...)`

**Source:** `config_generation/yaml_renderer.py`

Renders an approved `ExperimentDraft` to a YAML config file.

**Guards:**

- Raises `ValueError` if `draft.approved` is False
- Runs `validate_draft()` before applying changes — validates that all change paths are in the permitted vocabulary and that proposed value types are correct
- Loads base config from `configs/experiments/<name>.yaml`
- Applies changes via `apply_changes()` (deep copy, never mutates base)
- Writes YAML to `output_path` (or `configs/experiments/<proposed_name>.yaml`)

**The researcher then manually runs the experiment from this config file.** No automation bridges YAML generation and experiment execution.

---

## 7. Comparative Research Cycle

**Function:** `research_api.run_llm_comparative_review(baseline, candidate, provider, model, ...)`

**Source:** `llm/comparison_engine.py`

**Sequence:**

1. Build `LLMContext` for both experiments (two independent `build_context` calls)
2. Build pre-computed delta payload (`_build_comparative_payload`) — all numeric deltas computed in Python before LLM call:
   - Performance deltas: Sharpe, drawdown, return
   - Validation deltas: mean OOS Sharpe, std OOS Sharpe, hit rate, n_negative_splits
   - ML signal deltas: mean IC, IC tier, DA tier, n_sign_reversal_features
   - Feature deltas: dominant family, mean HHI, n_family_transitions, most_volatile_feature
   - Universe deltas: n_assets, tickers, coverage
   - Failure mode comparison: baseline-only, candidate-only, shared
3. Compute comparison hash: SHA256 of both contexts concatenated
4. Render `COMPARATIVE_REVIEW` template with both contexts + pre-computed deltas
5. Call LLM; parse `###` sections into `ComparativeReview`
6. Persist to `results/comparisons/<baseline>__vs__<candidate>/`

**The LLM is explicitly instructed to use pre-computed deltas directly — not to re-derive or contradict them.**

---

## 8. Evolution Chain (Deterministic — No LLM)

**Function:** `research_api.build_research_evolution_chain(root_experiment, base, persist)`

**Source:** `evolution/evolution_builder.py`

This function never calls the LLM. It builds a `ResearchEvolutionChain` from lineage records and diagnostic artefacts.

**Sequence:**

1. Scan all experiments for `lineage.json` records
2. Resolve the ordered chain starting from `root_experiment` by following parent links (up to 50 steps)
3. For each step, derive an `EvolutionStep`:
   - If a `comparative_review.json` exists for the `(prev, curr)` pair: extract from it
   - Otherwise: compute deltas directly from two `LLMContext` summaries
4. Generate a deterministic `evolution_summary` string from the list of steps
5. Return `ResearchEvolutionChain` with optional persist to `results/evolution/`

**Lineage registration** (`register_experiment_lineage`) is human-triggered. It is never called automatically by any other orchestration function.

---

## 9. Optional Session Layer

**Functions:** `create_research_session`, `record_session_event`, `update_session_status`, `get_session_summary`

**Source:** `session/session_manager.py`

Sessions are optional and non-intrusive. No core research API function requires a session. Sessions are a logging layer that tracks which functions were called, in what order, for which experiments.

**Session event types:**

- `REVIEW_GENERATED`, `ITERATION_PROPOSAL_GENERATED`, `DRAFT_GENERATED`, `DRAFT_APPROVED`, `YAML_RENDERED`, `EXPERIMENT_LINKED`, `COMPARISON_GENERATED`

`get_session_summary()` is a pure in-memory projection over the event log — no disk I/O, no LLM calls.

---

## 10. Intent Parser and Router

**Functions:** `intents/intent_parser.parse()` → `router/workflow_router.route()`

These components are optional convenience wrappers. The router dispatches typed `Intent` objects to `research_api` functions. External callers can bypass the router entirely and call `research_api` functions directly.

**Intent parsing is two-stage:**

1. Rule-based keyword regex — deterministic, zero latency, no API key required
2. LLM fallback — only when rule-based classification fails AND caller provides a live provider (default is `"stub"`, which skips the fallback entirely)

**Nine intent types:**

`ReviewExperimentIntent`, `CompareExperimentsIntent`, `GenerateIterationIntent`, `BuildEvolutionChainIntent`, `ListExperimentsIntent`, `RankExperimentsIntent`, `RetrieveArtefactIntent`, `BuildContextIntent`, `GenerateDraftIntent`, `UnrecognisedIntent`
