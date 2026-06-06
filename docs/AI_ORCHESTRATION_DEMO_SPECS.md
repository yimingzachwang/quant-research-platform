# AI Orchestration Demo Specifications

**Repository:** `quant-research-platform`
**Date:** 2026-05-30
**Scope:** Canonical demo video and walkthrough specifications

---

## Format

**Length:** 90–150 seconds  
**Format:** Screen recording with narration or captions. No live editing — pre-recorded.  
**Resolution:** 1920×1080 minimum.  
**Audio:** Optional. Captions required if narrated.  
**Terminal font size:** ≥14pt for readability at compressed video dimensions.

All scenes must reflect actual implemented behaviour. No mockups, no placeholder data, no simulated output.

---

## Scene 1 — Natural-Language Research Request

**Duration:** 10–15s

**What to show:**

A researcher enters one of the following requests into a terminal, API client, or the `POST /api/route` endpoint:

> `Review the canonical ML showcase, identify the main robustness concern, and propose the next controlled experiment.`

or:

> `Compare the canonical ML showcase against the multi-asset experiment and suggest the next controlled iteration.`

Show the raw text entering the system. Show the `POST /api/route` call or equivalent if using the HTTP layer.

**Must show:**
- The natural-language input
- That it is being submitted to a parsing layer, not directly to an LLM

**Caption:**

> A natural-language research request enters the orchestration layer.

**Forbidden:** Showing the request going directly to an LLM. Showing an LLM chatbot interface.

---

## Scene 2 — Intent Parsing and Routing

**Duration:** 8–12s

**What to show:**

The intent parser output. Show the typed intent dataclass:

```json
{
  "intent_type": "ReviewExperimentIntent",
  "experiment_name": "canonical_ml_showcase",
  "provider": "anthropic"
}
```

or:

```json
{
  "intent_type": "CompareExperimentsIntent",
  "baseline": "canonical_ml_showcase",
  "candidate": "canonical_ml_multi_asset"
}
```

If showing the workflow router, show which API function it dispatches to (`run_llm_review`, `run_llm_comparative_review`, etc.).

**Caption:**

> Natural language is converted into a typed research intent before any analysis is performed.

**Must show:**
- The concrete intent type name
- That rule-based parsing (not the LLM) produced this output for standard inputs

---

## Scene 3 — Iteration Proposal

**Duration:** 15–20s

**What to show:**

The `IterationProposal` output from `generate_iteration_proposal()`. Show the structured JSON or the rendered Markdown. Highlight these fields:

```
research_focus:       [1–2 sentence hypothesis grounded in named diagnostics]
rationale:            [2–3 sentences citing specific metric values]
supporting_evidence:  [bulleted list — each names a metric/feature/failure mode]
suggested_experiments:[bulleted list — testable research directions]
validation_concerns:  [named OOS validation weaknesses]
feature_risks:        [named feature-level risks]
confidence:           medium — [one sentence justification]
```

**Caption:**

> The iteration proposal is advisory. It cites named diagnostics and proposes testable directions — it does not prescribe parameter values or execution orders.

**Must show:**
- At least one named metric (e.g., `std_oos_sharpe`, `mean_hhi`)
- At least one named failure mode (e.g., `catastrophic_split`, `weak_ic`)
- The `confidence` field

**Forbidden:** Showing a proposal that contains generic advice ("add stop-losses", "diversify the universe") without diagnostic grounding.

---

## Scene 4 — Config Draft Generation

**Duration:** 12–18s

**What to show:**

The `ExperimentDraft` output from `generate_experiment_draft()`. Show the structured change list:

```json
{
  "proposed_name": "canonical_ml_showcase_v2",
  "approved": false,
  "changes": [
    {
      "section": "model",
      "field": "params.alpha",
      "current_value": 1.0,
      "proposed_value": 0.5,
      "rationale": "Reduce regularization strength to investigate..."
    },
    {
      "section": "features",
      "field": "entries.remove",
      "current_value": null,
      "proposed_value": "breakout_63d",
      "rationale": "Remove most_volatile_feature to test split-to-split stability..."
    }
  ]
}
```

Highlight:
- `"approved": false` — the draft starts unapproved
- `"current_value"` — always read from the base config by Python, not from the LLM
- The bounded vocabulary: only permitted change paths appear (`model.params.alpha`, `features.entries.remove`, etc.)

**Caption:**

> The LLM proposes constrained config changes. Python reads current values from the base config and validates only permitted change paths.

---

## Scene 5 — Human Approval Gate

**Duration:** 8–12s

**What to show:**

The approval step. Show one of:

**Via HTTP:**
```
POST /api/sessions/{session_id}/draft/approve
Body: {"experiment_name": "canonical_ml_showcase", "draft_id": "<uuid>"}
Response: {"draft": {"approved": true, "approved_at": "2026-05-30T..."}, ...}
```

**Via Python:**
```python
approved = approve_experiment_draft(draft)
assert approved.approved is True
```

Then show what happens if you try to render without approval:

```python
render_draft_to_yaml(unapproved_draft)
# ValueError: Draft must be approved before rendering to YAML.
```

**Caption:**

> The researcher must approve the draft before a YAML config can be rendered. This gate cannot be bypassed.

---

## Scene 6 — YAML Rendering and the Execution Boundary

**Duration:** 15–20s

**What to show (two parts):**

**Part A — YAML output:**

Show the rendered YAML file being written to `configs/experiments/canonical_ml_showcase_v2.yaml`. Show the provenance header:

```yaml
# Generated by Quant Research Platform — Config Synthesis
# Base experiment:        canonical_ml_showcase
# Draft hash:             <12-char hash>
# Source proposal hash:   <64-char SHA256>
#
# Review and run with:
#   python scripts/run_from_config.py configs/experiments/canonical_ml_showcase_v2.yaml
```

**Part B — The boundary (critical):**

After the YAML appears, show a visible separator on screen:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       Human-Controlled Execution Boundary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Then show the CLI command the researcher runs manually:

```bash
python scripts/run_from_config.py \
  configs/experiments/canonical_ml_showcase_v2.yaml \
  --report --preset canonical
```

**Caption:**

> Execution is intentionally human-controlled. The orchestration layer renders the config; the researcher reviews and runs it.

**This scene is mandatory. Do not cut it. The boundary must be visible.**

---

## Scene 7 — Research Engine Run

**Duration:** 8–12s

**What to show:**

Terminal output from the experiment run. Show progress indicators — data loading, feature computation, ML training, walk-forward splits, report generation. The output should feel like a real computational process, not an API call.

Show that this produces output on disk:

```
results/experiments/canonical_ml_showcase_v2/
  metadata.json
  metrics.json
  diagnostics/
  research/
  plots/
```

**Caption:**

> The quant engine runs independently. It is the source of truth.

---

## Scene 8 — Research Artefacts

**Duration:** 8–10s

**What to show:**

The artefact tree. Either a terminal `find` output or a file explorer view of:

```
results/experiments/canonical_ml_showcase_v2/
├── metadata.json
├── metrics.json
├── diagnostics/
│   ├── ml_diagnostics.json
│   ├── split_metrics.json
│   ├── backtest_diagnostics.json
│   └── universe_coverage.json
├── research/
│   ├── feature_summary.json
│   └── alignment_diagnostics.json
└── plots/
    ├── equity_and_drawdown.png
    ├── rolling_sharpe.png
    └── plot_index.json
```

**Caption:**

> Reproducible artefacts with full provenance. All subsequent analysis reads from these files — nothing is recomputed.

---

## Scene 9 — Deterministic Context Assembly

**Duration:** 10–14s

**What to show:**

The `build_llm_context()` call and its output. Show the assembled `LLMContext` JSON (partially):

```json
{
  "experiment_name": "canonical_ml_showcase",
  "strategy_name": "MLStrategy(Ridge)",
  "performance": {
    "sharpe": 1.47,
    "max_drawdown": -0.283,
    "annual_return": 0.114
  },
  "validation": {
    "mean_oos_sharpe": 0.89,
    "std_oos_sharpe": 0.63,
    "n_splits": 7,
    "worst_split_sharpe": -0.41
  },
  "failure_modes": [ ... ]
}
```

Highlight that this assembly involves no LLM call. It reads from the artefact files and applies `_prune_nulls()` to produce a clean, structured payload.

**Caption:**

> Structured research context is assembled deterministically from persisted artefacts before any LLM review.

---

## Scene 10 — Failure Mode Detection

**Duration:** 8–10s

**What to show:**

The detected failure modes. Show the list from the `LLMContext`:

```json
"failure_modes": [
  {
    "name": "high_split_sharpe_variance",
    "severity": "warning",
    "description": "std_oos_sharpe 0.63 — high dispersion across walk-forward splits.",
    "evidence": "std_oos_sharpe = 0.63"
  },
  {
    "name": "catastrophic_split",
    "severity": "warning",
    "description": "Worst split Sharpe -0.41 — one split produced significantly negative OOS returns.",
    "evidence": "worst_split_sharpe = -0.41"
  }
]
```

**Caption:**

> Failure modes are detected by deterministic rules before any LLM is involved. The LLM receives labelled evidence, not raw numbers.

**Must show:** That these are rule-based outputs, not LLM-generated labels.

---

## Scene 11 — LLM Review

**Duration:** 18–25s

**What to show:**

The `LLMReviewOutput` structured sections. Scroll through or highlight:

- `performance_interpretation`: Risk-adjusted analysis citing `sharpe`, `max_drawdown`, `calmar`
- `signal_quality`: IC tier, directional accuracy, coefficient stability
- `validation_assessment`: OOS analysis citing `std_oos_sharpe`, `worst_split_sharpe`, n_negative_splits — must address all required metrics
- `failure_mode_analysis`: Named modes with root cause and mitigation
- `feature_contribution_analysis`: Dominant family, HHI, `n_family_transitions`, `most_volatile_feature`
- `recommendations`: 3–5 items, each citing a specific diagnostic value or failure mode name

Optionally show the Jinja2 template rendering call or the provenance header of the persisted JSON (`context_hash`, `provider`, `model`, `generated_at`).

**Caption:**

> The LLM interprets structured evidence from the experiment artefacts. It does not invent metrics, recompute results, or produce generic advice.

---

## Scene 12 — Comparative Review

**Duration:** 12–18s

**What to show:**

The `ComparativeReview` between `canonical_ml_showcase` and `canonical_ml_multi_asset`. Show:

- The pre-computed delta block (before LLM call):

```json
"metric_comparison": {
  "sharpe": {"baseline": 1.47, "candidate": 1.63, "delta": 0.16},
  "max_drawdown": {"baseline": -0.283, "candidate": -0.341, "delta": -0.058}
},
"failure_mode_comparison": {
  "baseline_only": ["catastrophic_split"],
  "candidate_only": ["high_alignment_loss"],
  "shared": ["high_split_sharpe_variance"]
}
```

- The LLM's `key_tradeoffs` bullets (named, evidence-grounded tradeoffs)
- The `validation_changes` bullets (named metric changes)

**Caption:**

> Numeric deltas are computed before the LLM call. The LLM explains tradeoffs; the numbers are determined by Python.

---

## Scene 13 — Research Evolution Chain

**Duration:** 8–12s

**What to show:**

The `ResearchEvolutionChain` output. Show the chain:

```
canonical_ml_showcase → canonical_ml_multi_asset
```

Show an `EvolutionStep`:

```json
{
  "experiment_name": "canonical_ml_multi_asset",
  "key_improvements": ["Resolved failure mode: catastrophic_split"],
  "new_risks": ["New failure mode: high_alignment_loss"],
  "validation_changes": ["mean_oos_sharpe improved: 0.89 → 1.12 (Δ+0.230)"],
  "research_direction": "Expand from single-asset to 15-ETF universe..."
}
```

**Caption:**

> The platform records how research ideas evolve across experiments. Lineage is human-registered; the chain is deterministically assembled from artefacts.

**Must show:** That lineage registration is a human step, not automated.

---

## Scene 14 — Closing Frame

**Duration:** 10–15s

**What to show:**

The full research loop diagram on screen, text or graphic:

```
Research Request
        ↓
Intent Parsing
        ↓
Proposal → Draft → Human Approval → YAML
        ↓
[Human-Controlled Execution Boundary]
        ↓
Research Engine → Artefacts
        ↓
Context Assembly → Failure Modes → LLM Review
        ↓
Comparative Review → Iteration Proposal → Evolution Chain
        ↻ (loop)
```

**No voiceover needed.** Let the diagram hold.

Optional text overlay:

> Intent-governed research orchestration. Evidence-grounded. Human-controlled. Reproducible.

---

## Acceptance Criteria

A demo is acceptable only when all of the following are true:

| Criterion | Required |
|---|---|
| Shows both planning flow and post-experiment review flow | Yes |
| Human-Controlled Execution Boundary is visible on screen | Yes |
| No implication of autonomous experiment execution | Yes |
| At least one LLM review or proposal output is shown | Yes |
| At least one deterministic diagnostic stage is shown | Yes |
| Persisted artefacts are shown | Yes |
| Context hash or provenance appears at least once | Yes |
| Approval gate is shown | Yes |
| Understandable to a senior quant engineer audience | Yes |
| Consistent with website lifecycle diagram | Yes |
| All data shown is from actual implemented system | Yes |
