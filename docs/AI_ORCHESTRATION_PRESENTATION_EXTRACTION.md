# AI Orchestration Presentation and Extraction Audit

**Repository:** `quant-research-platform`
**Audit date:** 2026-05-30
**Subject:** Prompt templates, section structure, and structured output extraction

---

## 1. Template Infrastructure

**File:** `src/orchestration/llm/prompt_templates/__init__.py`

Three templates are registered by string constant. All templates are Jinja2 with `StrictUndefined` enforcement.

| Constant | File | Engine | Default tokens | Default temperature |
|---|---|---|---|---|
| `EXPERIMENT_REVIEW` | `experiment_review.txt` | `review_engine` | 4096 | 0.2 |
| `ITERATION_PROPOSAL` | `iteration_proposal.txt` | `iteration_engine` | 2048 | 0.3 |
| `COMPARATIVE_REVIEW` | `comparative_review.txt` | `comparison_engine` | 4096 | 0.2 |

Templates are plain `.txt` files on disk — not embedded strings. Loading is via `path.read_text()`.

---

## 2. Experiment Review Template

**File:** `src/orchestration/llm/prompt_templates/experiment_review.txt`

### Context blocks injected

| Block | Jinja2 variable | Format |
|---|---|---|
| Experiment header | `experiment_name`, `strategy_name`, `tags`, `created_at` | Plain text |
| Performance metrics | `performance` | JSON (via `tojson(indent=2)` filter) |
| Walk-forward validation | `validation` | JSON |
| ML model diagnostics | `ml_diagnostics` | JSON |
| Feature context | `feature_summary` | JSON |
| Universe | `universe_summary` | JSON |
| Failure modes | `failure_modes` | Jinja2 loop: `[SEVERITY] name: description\nEvidence: evidence` |
| Available plots | `available_plots` | Jinja2 loop: `- name (group, importance): caption` |

All JSON blocks use a custom `tojson` filter (`json.dumps(..., default=str)`) to handle any non-serializable types gracefully.

### Failure mode rendering (template loop)

```jinja2
{% if failure_modes %}
{% for fm in failure_modes %}
[{{ fm.severity | upper }}] {{ fm.name }}: {{ fm.description }}
Evidence: {{ fm.evidence }}
{% endfor %}
{% else %}
No failure modes detected.
{% endif %}
```

Failure modes are rendered as labeled diagnostic facts before the instruction block, giving the LLM labeled evidence rather than raw numbers.

### Instruction section structure

The template requires exactly these `###` sections in the output:

| Section heading | `{{ SECTION_* }}` variable | Expected content |
|---|---|---|
| `### Performance Interpretation` | `SECTION_PERFORMANCE` | Risk-adjusted performance analysis |
| `### Signal Quality` | `SECTION_SIGNAL_QUALITY` | IC tier, directional accuracy, coefficient stability |
| `### Validation Assessment` | `SECTION_VALIDATION` | OOS consistency depth analysis |
| `### Failure Mode Analysis` | `SECTION_FAILURE_MODES` | Per-mode root cause + mitigation |
| `### Feature Contribution Analysis` | `SECTION_FEATURE_CONTRIBUTION` | Dominant family, HHI, n_family_transitions, most_volatile_feature |
| `### Recommendations` | `SECTION_RECOMMENDATIONS` | 3–5 actionable, diagnostic-grounded next steps |

**Section constants** are interpolated from the rendering context — they are not hardcoded strings in the template, which means the section heading names are controlled by the Python code, not the template file.

### Validation assessment requirements (verbatim from template)

The template explicitly mandates that the Validation section address all of the following if data is present:
- `std_oos_sharpe` and coefficient of variation
- Worst split Sharpe and worst split drawdown
- Number of negative-Sharpe splits
- Whether dispersion risk undermines or corroborates the in-sample Sharpe narrative

---

## 3. Iteration Proposal Template

**File:** `src/orchestration/llm/prompt_templates/iteration_proposal.txt`

### Context blocks injected

| Block | Jinja2 variable | Format |
|---|---|---|
| Experiment header | `experiment_name`, `strategy_name`, `created_at` | Plain text |
| Performance summary | `performance` | JSON |
| Walk-forward validation | `validation` | JSON |
| ML model diagnostics | `ml_diagnostics` | JSON |
| Feature context | `feature_summary` | JSON |
| Failure modes | `failure_modes` | Same loop as review template |
| Universe | `universe_summary` | JSON |
| Primary diagnostic plots | `available_plots` | Loop: `- name (group): caption` |

The template label for plots is "PRIMARY DIAGNOSTIC PLOTS AVAILABLE" (vs "AVAILABLE VISUALISATIONS" in the review template). In both cases, the LLM is told it cannot view the plots — it can only reference them by name.

### Instruction section structure

The template requires exactly these `###` sections:

| Section heading | Type | Expected content |
|---|---|---|
| `### Research Focus` | prose | 1–2 sentence hypothesis, specific + testable |
| `### Rationale` | prose | 2–3 sentences citing metric values and failure mode names |
| `### Supporting Evidence` | bullets | 3–5 items, each naming a metric/feature/failure mode/split |
| `### Suggested Experiments` | bullets | 2–4 testable research directions (no parameter prescriptions) |
| `### Instability Signals` | bullets | Named signals of regime/feature/model instability |
| `### Validation Concerns` | bullets | Named validation weaknesses with metric references |
| `### Feature Risks` | bullets | Named feature-level risks with supporting evidence |
| `### Confidence` | prose | `low | medium | high` + one sentence justification |

---

## 4. Comparative Review Template

**File:** `src/orchestration/llm/prompt_templates/comparative_review.txt`

### Context blocks injected

The template receives both experiment contexts plus a pre-computed delta block:

| Block | Jinja2 variable | Format |
|---|---|---|
| Baseline header | `baseline_experiment`, `generated_at` | Plain text |
| Baseline performance | `baseline.performance` | JSON |
| Baseline validation | `baseline.validation` | JSON |
| Baseline ML diagnostics | `baseline.ml_diagnostics` | JSON |
| Baseline failure modes | `baseline.failure_modes` | Loop |
| Baseline universe | `baseline.universe_summary` | JSON |
| Candidate header | `candidate_experiment` | Plain text |
| Candidate blocks | `candidate.*` | Same structure as baseline |
| Pre-computed deltas | `metric_comparison` | JSON |
| Failure mode evolution | `failure_mode_comparison` | JSON |
| ML signal delta | `ml_comparison` | JSON |
| Feature behaviour delta | `feature_comparison` | JSON |

### Instruction section structure

| Section heading | Type | Expected content |
|---|---|---|
| `### Overall Assessment` | prose | 2–3 sentences on research evolution, explicit tradeoffs |
| `### Validation Changes` | bullets | Named metric changes with direction and values |
| `### Instability Changes` | bullets | Named regime/split variance signals |
| `### Feature Behavior Changes` | bullets | Named feature/family before-after values |
| `### Robustness Changes` | bullets | Drawdown and catastrophic period changes |
| `### Failure Mode Changes` | bullets | `gained/lost/persistent: name — implication` format |
| `### Key Tradeoffs` | bullets | `tradeoff: benefit at cost of cost` format |
| `### Research Progression Summary` | prose | 1–2 sentences on research direction |
| `### Confidence` | prose | `low | medium | high` + one sentence |

---

## 5. Section Parsing: `_split_sections()` and `_normalise_heading()`

All three engines use the same parsing strategy.

**`_split_sections(text: str) → dict[str, str]`**

```python
for line in text.split("\n"):
    if line.startswith("###"):
        heading = line.lstrip("#").strip()
        current_key = _normalise_heading(heading)
    else:
        buffer.append(line)
```

Splits on `###` level headings only. Lower-level headings (`####`, `#####`) are treated as body text.

**`_normalise_heading(heading: str) → str`**

```python
re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
```

"Research Focus" → `research_focus`
"Suggested Experiments" → `suggested_experiments`
"Feature Behavior Changes" → `feature_behavior_changes`

Section keys are snake_case — consistent with the dataclass field names they populate.

---

## 6. Bullet Extraction

For list-valued fields, all engines use the same bullet extraction logic:

```python
def _bullets(key: str) -> list[str]:
    raw = sections.get(key, "")
    return [
        line.lstrip("-").strip()
        for line in raw.splitlines()
        if line.strip().startswith("-") and line.lstrip("-").strip()
    ]
```

Only lines starting with `-` are captured. The leading dash is stripped. Empty items after stripping are discarded. This is tolerant of both `- item` and `  - item` indentation styles.

---

## 7. Output Schema: `LLMReviewOutput`

```python
@dataclass
class LLMReviewOutput:
    experiment_name: str
    generated_at: str
    context_hash: str
    provider: str
    model: str
    prompt_template: str
    sections: dict[str, str]   # all ### sections as raw text
    flags: list[str]           # failure mode names from LLMContext.failure_modes
```

`flags` is derived deterministically from `LLMContext.failure_modes` — it is not extracted from the LLM response. It provides a pre-computed summary of which failure modes were active when the review was generated.

---

## 8. Output Schema: `IterationProposal`

```python
@dataclass
class IterationProposal:
    experiment_name: str
    generated_at: str
    context_hash: str
    research_focus: str           # prose
    rationale: str                # prose
    supporting_evidence: list[str]
    suggested_experiments: list[str]
    instability_signals: list[str]
    validation_concerns: list[str]
    feature_risks: list[str]
    confidence: str               # "low | medium | high — justification"
    provider: str
    model: str
    prompt_template: str
```

---

## 9. Output Schema: `ComparativeReview`

```python
@dataclass
class ComparativeReview:
    baseline_experiment: str
    candidate_experiment: str
    generated_at: str
    context_hash: str
    overall_assessment: str       # prose
    validation_changes: list[str]
    instability_changes: list[str]
    feature_behavior_changes: list[str]
    robustness_changes: list[str]
    failure_mode_changes: list[str]
    key_tradeoffs: list[str]
    research_progression_summary: str  # prose
    confidence: str
    provider: str
    model: str
    prompt_template: str
```

---

## 10. Persisted JSON Provenance Fields

Every persisted JSON artefact includes these provenance fields in addition to content:

| Field | Description |
|---|---|
| `experiment_name` | Source experiment |
| `generated_at` | ISO 8601 UTC timestamp |
| `context_hash` | 64-char SHA256 of the LLMContext that produced this output |
| `provider` | LLM provider string (`"anthropic"`, `"openai"`, `"stub"`) |
| `model` | Provider model ID |
| `prompt_template` | Template name constant used |
| `review_version` / `iteration_version` / `comparison_version` | Schema version string (currently `"1.0"`) |

The version field allows schema evolution to be tracked without breaking existing artefacts.

---

## 11. Plot Metadata Presentation

**Source:** `plots/plot_index.json` → `PlotMetadata` in `LLMContext.available_plots`

Only plots with `importance == "primary"` are included in the LLM context. Secondary plots are excluded to reduce token count and focus the review on key diagnostics.

Each `PlotMetadata` in the prompt includes:
- `name`: plot stem (e.g., `equity_and_drawdown`)
- `group`: logical group (e.g., `performance`, `ml_diagnostics`, `allocation`)
- `importance`: always `"primary"` in the LLM context
- `caption`: pre-written description of what the plot shows

The LLM is explicitly told it cannot view the plots and should reference them by name when relevant. The caption provides enough semantic content for the LLM to incorporate plot references in its analysis without actually seeing the image.

---

## 12. Config Draft Extraction

**File:** `src/orchestration/config_generation/draft_generator.py`

The draft generator uses a different prompt pattern from the review/iteration engines:

- Uses a `system` parameter (for Anthropic: system turn; for OpenAI: system message)
- Expects pure JSON output only — no prose, no markdown fences
- Temperature is 0.1 (lowest of all three engines)

The system prompt (`_LLM_SYSTEM`) defines a strict JSON schema:

```json
{
  "proposed_name": "<base_name>_v2",
  "changes": [
    {
      "section": "<section>",
      "field": "<field>",
      "proposed_value": <typed_value>,
      "rationale": "<concise rationale>"
    }
  ]
}
```

`_parse_llm_response()` strips markdown code fences (```` ```json ```) before JSON parsing, making the extraction tolerant of model-specific formatting habits.

The `current_value` field is explicitly not in the LLM's output schema — it is always read from the base config by Python.
