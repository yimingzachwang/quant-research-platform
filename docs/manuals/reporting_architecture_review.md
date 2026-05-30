# Reporting Architecture Review
## Quant ETF Research Platform — Phase D Evolution Analysis

**Date:** 2026-05-23  
**Scope:** `src/reporting/`, `src/visualization/`, `src/experiments/orchestrator.py`, artefact contracts, export/gallery structure  
**Purpose:** Determine how the current reporting system can evolve into a configurable, section-based, narrative-oriented, institutional-style research reporting layer — without unnecessary complexity.

---

## Part 1 — Existing Reporting System: What Is Actually There

### 1.1 Implemented Abstractions

**`src/reporting/report_builder.py`** — The load/generate entrypoint.

- `ExperimentArtefacts` — unified data carrier: source directory, pre-computed output paths, loaded JSON content (`metadata`, `metrics`, `config`, `ml_provenance`). Clean. Single responsibility.
- `ReportPaths` — output path bag: markdown, html (optional), provenance sidecar. Clean.
- `load_experiment_artefacts(path, output_dir)` — reads from disk, returns `ExperimentArtefacts`. Pre-computes all output paths before any generation runs.
- `generate_experiment_report(path, output_dir, include_html)` — orchestrates copy → render → write. Pure with respect to experiment artefacts.
- Figure path computation is centralized here and passed into renderers. This is architecturally correct.

**`src/reporting/markdown.py`** — The section assembly engine.

- Ten private section renderers: `_title`, `_summary`, `_metadata`, `_configuration`, `_ml_section`, `_metrics`, `_walk_forward`, `_figures`, `_provenance_section`, `_footer`.
- `render_report(artefacts, figure_paths, generated_at, report_version) → str` — pure function. No I/O. No filesystem assumptions. Assembles sections with conditional inclusion.
- Current section order is hardcoded in `render_report()`. No configuration possible.

**`src/reporting/html.py`** — Intentionally fixed-scope line-by-line state machine.

- ~240 lines of stdlib-only code. Handles exactly the constructs that `markdown.py` produces.
- No external dependencies. Produces standalone self-contained HTML.
- Minimal CSS embedded inline: system font, 900px max-width, clean table style, white background. Functional but not publication-grade.
- Architectural note: this is permanently fixed-scope by design. It is NOT the right place to invest for styling improvements.

**`src/reporting/interfaces.py` + `placeholders.py`** — Legacy scaffold (dead code in current pipeline).

- `Report` dataclass and `ReportGenerator` Protocol exist but are not called by any real pipeline code.
- `MarkdownReportGenerator` is a non-functional placeholder from the initial scaffold phase.
- These are forward-compatible stubs. The Protocol is actually a reasonable foundation for a section interface if extended.

**`src/experiments/orchestrator.py`** — Composite `run_and_report()` helper.

- Two-line delegation: `run_experiment_from_config()` then `generate_experiment_report()`.
- No business logic. Correct scope.
- `format_run_summary()` produces a human-readable CLI summary from `ExperimentRun` fields.

### 1.2 Orchestration Already in Place

The end-to-end pipeline exists and works:

```
Config YAML
  → run_experiment_from_config()  [orchestrator.py]
      → validate + normalize
      → factory (pure)
      → load_universe + align_prices
      → run_strategy / strategy.fit
      → run_walk_forward_validation
      → save_run (artefacts)
      → _write_ml_provenance
      → registry.register
  → generate_experiment_report()  [report_builder.py]
      → load_experiment_artefacts
      → _copy_figures
      → render_report (markdown.py)
      → markdown_to_html (html.py)
      → _write_provenance sidecar
```

`run_and_report()` wires these two stages together in a single call.

### 1.3 What Is Already Reusable

| Component | Status | Notes |
|---|---|---|
| `ExperimentArtefacts` | Reuse as-is | Clean data carrier. Extend fields, don't replace. |
| `load_experiment_artefacts()` | Reuse as-is | Already loads ml_provenance conditionally |
| `render_report()` | Extend | Add `spec` parameter — no structural change needed |
| Private section renderers | Reuse / extend | Already section-shaped functions |
| `_pipe_table()` | Reuse | Works for any-width tables |
| `_render_section()` | Reuse | Heading + body combinator |
| `markdown_to_html()` | Reuse as-is | Keep fixed-scope |
| `ReportGenerator` Protocol | Reuse | Good foundation for section interface |
| `styles.py` (COLORS, make_figure, etc.) | Reuse | Already institutional-quality |
| `check_artefact_dir()` / `check_ml_artefacts()` | Reuse | Advisory checker pattern is right |
| `run_and_report()` | Reuse | Correct scope, zero logic |

### 1.4 What Is Currently Fragmented

**ML diagnostics are computed but never surfaced in reports.**  
`src/ml/diagnostics/` contains `prediction_correlation`, `information_coefficient`, `rolling_directional_accuracy`, `coefficient_stability`, `prediction_drift`, `signal_turnover`, `average_turnover`, `turnover_by_split` — none of these are called by the orchestrator or included in any report. They exist as standalone functions with no bridge to the reporting layer.

**ML diagnostic plots exist but are not generated during experiment runs.**  
`src/visualization/ml_plots.py` has six diagnostic plot functions (`plot_prediction_vs_actual`, `plot_information_coefficient`, `plot_coefficient_stability`, etc.). The orchestrator's `_build_plots()` only generates `equity_and_drawdown`, `walk_forward_stitched`, `split_sharpes`. None of the ML-specific diagnostic plots are currently saved to artefacts.

**The `diagnostics/` subdirectory is always empty.**  
`save_run()` creates it but nothing ever writes to it. This is the natural home for serialized diagnostic results (IC series, coefficient tables, prediction DataFrames) that should feed both the report and future frontend consumption.

**No narrative text exists anywhere in the pipeline.**  
Every table, figure, and section is purely structural. There is no interpretation layer — no statement about what a Sharpe of 0.8 means in context, no validation commentary, no risk annotation.

**The HTML style is functional but not showcase-grade.**  
The inline CSS in `html.py` produces correct output but is minimal. There is no way to apply a different stylesheet, use a template, or produce a publication-quality layout without modifying `html.py` directly — which violates its fixed-scope constraint.

**No report mode or configuration concept exists.**  
`render_report()` has no way to suppress or include sections. Every call produces the same structure. There is no "compact", "full", "publication" or "audit" mode.

**Report spec (`ResearchReportSpec`) does not exist yet.**  
The concept is fully supportable by the current architecture but has not been implemented.

### 1.5 What Is Tightly Coupled

- **Section ordering is hardcoded** in `render_report()` as a Python list literal. Changing order requires editing the function body.
- **Section inclusion logic is scattered** across individual section functions (some return `""` if conditions aren't met; some are conditionally appended in `render_report()`). There is no single governing structure.
- **HTML styling is inline** in `html.py`. Changing the look requires touching the fixed-scope converter.
- **`ExperimentArtefacts` is the only context object**. Any new section that needs additional data (e.g., per-split metrics, prediction series) must either add a field to `ExperimentArtefacts` or load from disk separately. This is manageable today but needs a strategy as the report grows.

### 1.6 What Is Well-Designed

- **Path computation centralized.** All figure paths are computed in `report_builder.py` and passed into renderers. Renderers make no filesystem assumptions. This is architecturally correct and should not change.
- **`render_report()` is pure.** No I/O, no side effects. The same inputs always produce the same output. This is essential for determinism.
- **`ExperimentArtefacts` as the single data carrier.** One object contains everything a report needs. New fields can be added as optional with `= None` defaults, maintaining backward compatibility.
- **Section functions are independent.** Each section function receives `artefacts` and has no dependency on other sections. They can be reordered, removed, or replaced without side effects.
- **Version-conditional rendering already works.** `_ml_section()` uses `isinstance(artefacts.ml_provenance, dict)` as a safe runtime guard. This pattern is correct and should be used for any new version-conditional content.
- **`styles.py` is genuinely institutional-quality.** The color palette, RC parameters, DPI settings, and helper functions produce research-grade matplotlib output. This does not need to change.
- **Advisory contract checkers** (`check_artefact_dir`, `check_ml_artefacts`) return violation lists and never raise. This is the right pattern.

---

## Part 2 — ResearchReportSpec: Architectural Fit

### 2.1 Assessment

A `ResearchReportSpec` fits naturally into the current architecture. The existing section functions are already the implementation — they just need a governing spec to determine which run.

The addition is minimal: one dataclass and one additional parameter to `render_report()`. No structural changes to any existing component.

### 2.2 Proposed Design

```python
# src/reporting/spec.py  (new, ~30 lines)

from dataclasses import dataclass, field

@dataclass(frozen=True)
class ReportMode:
    COMPACT     = "compact"
    RESEARCH    = "research"
    DIAGNOSTICS = "diagnostics"
    PUBLICATION = "publication"
    AUDIT       = "audit"

@dataclass(frozen=True)
class ResearchReportSpec:
    # Core sections (always present for any useful report)
    include_summary:      bool = True
    include_metadata:     bool = True
    include_configuration: bool = True
    include_metrics:      bool = True

    # Conditional sections
    include_ml_analysis:  bool = True   # Model & Features (v2 only — auto-guarded)
    include_validation:   bool = True   # Walk-Forward section
    include_diagnostics:  bool = False  # ML diagnostics (IC, stability, turnover)
    include_figures:      bool = True
    include_provenance:   bool = True
    include_narrative:    bool = False  # Threshold-based commentary (future)
    include_appendix:     bool = False  # Raw config dump, feature list (future)

    @classmethod
    def compact(cls) -> "ResearchReportSpec":
        return cls(include_validation=False, include_figures=False, include_provenance=False)

    @classmethod
    def research(cls) -> "ResearchReportSpec":
        return cls()  # all defaults

    @classmethod
    def diagnostics(cls) -> "ResearchReportSpec":
        return cls(include_diagnostics=True)

    @classmethod
    def publication(cls) -> "ResearchReportSpec":
        return cls(include_narrative=True, include_diagnostics=True)

    @classmethod
    def audit(cls) -> "ResearchReportSpec":
        return cls(include_narrative=True, include_diagnostics=True, include_appendix=True)
```

### 2.3 Integration Point

`render_report()` gains one optional parameter:

```python
def render_report(
    artefacts: ExperimentArtefacts,
    figure_paths: list[tuple[str, Path]],
    generated_at: str,
    report_version: str,
    spec: ResearchReportSpec | None = None,  # NEW — None = use defaults
) -> str:
```

When `spec` is `None`, a `ResearchReportSpec()` instance (all defaults) is used. This preserves 100% backward compatibility. All existing tests pass unchanged.

### 2.4 What Must Not Change

- `render_report()` remains pure.
- `ExperimentArtefacts` remains the single data carrier.
- Figure paths remain computed in `report_builder.py`, not in renderers.
- `markdown_to_html()` remains fixed-scope.

---

## Part 3 — Section-Based Reporting Architecture

### 3.1 Assessment

The current codebase already has a section-based structure inside `markdown.py` — it is just not formally abstracted. Each private function is a section. The question is whether to extract them.

**Verdict: partial extraction is appropriate, but a full `sections/` package is premature.**

A `ReportSection` abstraction adds value because it provides a single named type that:
1. `render_report()` can iterate over (replacing the hardcoded list of function calls)
2. External code can import and test individually
3. Future diagnostic sections can be added without modifying `render_report()`'s body

But the extraction should be internal to `src/reporting/` — not a `sections/` sub-package. A sub-package at this stage adds filesystem fragmentation without architectural benefit.

### 3.2 Minimal Section Abstraction

```python
# src/reporting/sections.py  (new, ~20 lines + section implementations moved here)

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.reporting.report_builder import ExperimentArtefacts
    from src.reporting.spec import ResearchReportSpec

@dataclass(frozen=True)
class ReportSection:
    name: str
    enabled_by: Callable[[ResearchReportSpec], bool]
    render: Callable[["ExperimentArtefacts"], str]
```

`render_report()` becomes:

```python
def render_report(artefacts, figure_paths, generated_at, report_version, spec=None):
    spec = spec or ResearchReportSpec()
    sections_out = []
    for section in _SECTION_REGISTRY:
        if section.enabled_by(spec):
            content = section.render(artefacts)
            if content:
                sections_out.append(content)
    # figures and footer handled separately (figure_paths argument dependency)
    ...
```

The `_SECTION_REGISTRY` is a module-level ordered list of `ReportSection` instances. Section order is explicit and inspectable. Adding a new section is registering one entry.

### 3.3 What Should NOT Be a Section

- **Footer** — always rendered, carries generated_at/version metadata not in artefacts. Keep as a direct call.
- **Figures** — depends on `figure_paths` argument, not just artefacts. Keep as a direct call with the paths argument.
- These two are caller-provided context, not artefact-derived content.

### 3.4 Recommended Sub-Package Boundary (Future)

If the section count grows beyond ~10, or if sections begin requiring their own test files, introduce a `sections/` sub-package at that point:

```
src/reporting/
  sections/
    __init__.py
    core.py          ← title, summary, metadata, configuration, metrics, footer
    ml.py            ← ml_analysis (Model & Features)
    validation.py    ← walk_forward
    diagnostics.py   ← ML diagnostics (IC, stability, turnover)
    narrative.py     ← threshold-based commentary
    appendix.py      ← raw config dump
```

This split is not warranted today with 10 sections. Trigger: when any single section file exceeds ~150 lines or requires isolated test coverage.

---

## Part 4 — Research Narrative Layer

### 4.1 What Raw Material Already Exists

The platform has substantial structured evidence available for interpretation:

| Evidence | Source | Available |
|---|---|---|
| Sharpe ratio | `metrics.json` | Yes |
| Calmar ratio | `metrics.json` | Yes |
| Max drawdown | `metrics.json` | Yes |
| Hit rate | `metrics.json` | Yes |
| Annualized return / volatility | `metrics.json` | Yes |
| Per-split Sharpe series | `WalkForwardResult.splits` | Yes (in-memory during run) |
| Split Sharpe std (stability) | `stability.py:summarize_stability` | Yes (callable) |
| Directional accuracy | `ml/diagnostics/prediction.py` | Yes |
| Information coefficient (IC) | `ml/diagnostics/prediction.py` | Yes |
| Coefficient stability | `ml/diagnostics/stability.py` | Yes |
| Signal turnover | `ml/diagnostics/turnover.py` | Yes |
| Model type / parameters | `ml_provenance.json` | Yes |
| Validation config | `normalized_config.json` | Yes |

The gap is that per-split metrics and ML diagnostics are currently computed transiently during the run and discarded — they are not persisted as artefacts. The `diagnostics/` subdirectory exists but is never written to.

### 4.2 What "Narrative" Should Mean Here

The goal is NOT AI-generated prose. It is threshold-based, rules-driven interpretation that transforms numbers into institutional-grade context:

```
Sharpe 0.82, Calmar 0.41 → "Moderate risk-adjusted return; drawdown recovery adequacy is marginal."
IC mean 0.063, IC std 0.21 → "Weak-to-moderate signal quality; high IC volatility suggests instability."
Split Sharpe std / mean = 0.8 → "Low cross-period consistency; performance varies substantially across validation windows."
Directional accuracy 53.1% → "Marginally above chance; insufficient for high-conviction signal."
Average turnover 0.18 / day → "High turnover; transaction costs will materially erode live performance."
```

These are deterministic, reproducible interpretations. The thresholds are institutional conventions (Sharpe >1.0 competitive, IC >0.05 meaningful, turnover <0.05/day low-cost-viable). They belong in a `_NARRATIVE_THRESHOLDS` dict, not in a model.

### 4.3 Minimal Narrative Layer Design

```python
# src/reporting/narrative.py  (new, ~80 lines)

from dataclasses import dataclass

@dataclass(frozen=True)
class NarrativeComment:
    label: str      # "Signal Quality", "Drawdown Profile", etc.
    finding: str    # "Sharpe 0.82 — moderate risk-adjusted return"
    level: str      # "positive" | "neutral" | "caution" | "warning"
```

```python
def generate_performance_commentary(metrics: dict) -> list[NarrativeComment]: ...
def generate_validation_commentary(split_sharpes: list[float]) -> list[NarrativeComment]: ...
def generate_ml_commentary(ic_mean: float, directional_accuracy: float, avg_turnover: float) -> list[NarrativeComment]: ...
```

Each function is pure: metrics in → comments out. No I/O. Fully testable. The rendering side converts `NarrativeComment` objects into bullet points or a formatted callout block in markdown.

### 4.4 Pre-condition: Persist Diagnostics First

Narrative generation requires per-split Sharpe and ML diagnostic results at report-generation time, not just at run time. The prerequisite step is writing them to the `diagnostics/` subdirectory during the experiment run:

- `diagnostics/split_metrics.json` — per-split Sharpe, return, max DD
- `diagnostics/ml_diagnostics.json` — IC mean/std, directional accuracy, average turnover (v2 only)

Until these are persisted, narrative generation at report time requires re-computation, which violates the read-only contract. This is the single most important prerequisite for the full narrative layer.

---

## Part 5 — Output Modes

### 5.1 Assessment

`ResearchReportSpec` with class-method factories (`compact()`, `research()`, `diagnostics()`, `publication()`, `audit()`) is sufficient. No separate "mode" enum is needed — the spec is the mode.

The five natural modes map directly to `ResearchReportSpec` flag combinations:

| Mode | include_summary | include_validation | include_diagnostics | include_narrative | include_figures |
|---|---|---|---|---|---|
| `compact` | ✓ | — | — | — | — |
| `research` | ✓ | ✓ | — | — | ✓ |
| `diagnostics` | ✓ | ✓ | ✓ | — | ✓ |
| `publication` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `audit` | ✓ | ✓ | ✓ | ✓ | ✓ + appendix |

### 5.2 CLI Integration

`generate_report.py` gains a `--mode` flag:

```
python scripts/generate_report.py results/experiments/my_exp --mode publication
python scripts/generate_report.py results/experiments/my_exp --mode compact
```

`run_from_config.py` gains `--report-mode`:

```
python scripts/run_from_config.py config.yaml --report --report-mode diagnostics
```

### 5.3 HTML Template Mode (Future, Not Now)

The current HTML output is appropriate for `compact` and `research` modes. For `publication` quality, a separate HTML template (an HTML string with `{body}` / `{title}` placeholders) would allow richer styling without modifying `html.py`. This is a one-function addition to `report_builder.py` — not a template engine.

---

## Part 6 — Static Website Readiness

### 6.1 What Is Already Website-Ready

| Artefact | Status |
|---|---|
| `metadata.json` | Clean JSON — directly parseable by frontend |
| `metrics.json` | Clean JSON — directly parseable |
| `ml_provenance.json` | Clean JSON — directly parseable |
| `normalized_config.json` | Clean JSON — directly parseable |
| Plot PNGs in `plots/` | Static files — directly servable |
| `*.html` reports | Standalone — self-contained, no external dependencies |
| `*.md` reports | Markdown — directly renderable |
| `*_provenance.json` | Clean JSON — report metadata |

The core output is fundamentally static-website-compatible. Every file is a standalone, self-contained artefact with no runtime dependencies.

### 6.2 What Is Missing for Clean Frontend Consumption

**No experiment-level manifest file.** A static website that wants to render an experiment card needs to discover all experiments. Currently it would have to read `registry.json` (which is run-output-relative) or enumerate directories. A per-experiment `report_manifest.json` written alongside the report would fix this:

```json
{
  "experiment_name": "example_ridge_forecast",
  "generated_at": "2026-05-23T14:22:00Z",
  "report_version": "1",
  "artefact_version": "1",
  "files": {
    "markdown": "markdown/example_ridge_forecast.md",
    "html": "html/example_ridge_forecast.html",
    "provenance": "markdown/example_ridge_forecast_provenance.json"
  },
  "figures": ["figures/example_ridge_forecast/equity_and_drawdown.png"],
  "tags": ["ml", "regression", "single_asset"],
  "metrics_summary": {
    "sharpe_ratio": 0.82,
    "annualized_return": 0.094,
    "max_drawdown": -0.183
  }
}
```

This is a 20-line addition to `_write_provenance()` in `report_builder.py`.

**No gallery index.** The `exports/experiment_gallery/` directory has no machine-readable index. A `gallery_index.json` generated by `export_gallery.py` would allow a static frontend to discover and filter experiments without traversing directories.

**`registry.json` is not path-stable.** It contains absolute filesystem paths (`"path": "/absolute/path/..."`) which break when the output is deployed to a web server or shared with collaborators. Report paths in any web-facing manifest should be relative.

**The HTML stylesheet is minimal.** For showcase quality, the HTML output needs typography improvements, print-friendly layout, and better figure handling. This is a CSS-only change to the template in `html.py` — no structural changes required.

### 6.3 Minimal Additions for Frontend Compatibility

Priority order:

1. **`report_manifest.json`** per generated report — enables card-based gallery UIs
2. **`gallery_index.json`** generated by `export_gallery.py` — enables search/filter
3. **Relative paths** in all manifests — enables portability
4. **Improved HTML CSS** — enables publication-quality visual output

None of these require architectural changes. They are all additions to existing write paths.

---

## Part 7 — Minimal Implementation Plan

### A. Minimal Architectural Additions Required

Ranked by dependency order (each item enables the next):

**1. Persist diagnostics during experiment runs** *(prerequisite for narrative)*

Write to `diagnostics/split_metrics.json` and `diagnostics/ml_diagnostics.json` in the orchestrator. Exactly two new JSON writes in `_run_ml_experiment()` and the shared run path. No new modules needed — calls to existing `stability.py` and `ml/diagnostics/` functions.

**2. `src/reporting/spec.py` — `ResearchReportSpec` dataclass** *(~40 lines)*

Frozen dataclass. Boolean flags with defaults. Five class-method factories. No imports except stdlib dataclasses. No dependencies on any other reporting module.

**3. `render_report()` + `spec` parameter** *(~10-line change)*

Add `spec: ResearchReportSpec | None = None` to `render_report()`. Sections already exist — they just need to be guarded by `if spec.include_X`. Full backward compatibility: `spec=None` uses defaults, all existing tests unchanged.

**4. `src/reporting/narrative.py` — threshold-based commentary** *(~80 lines)*

Pure functions: `dict[str, float] → list[NarrativeComment]`. One function per domain (performance, validation, ML signal). No external dependencies. Testable in isolation. Integrated into `render_report()` only when `spec.include_narrative=True`.

**5. `report_manifest.json` per report** *(~20-line addition to `report_builder.py`)*

Written by `_write_provenance()` alongside the existing provenance sidecar. Contains relative file paths, tags, and a metrics summary object. Enables frontend consumption.

**6. `gallery_index.json` in `export_gallery.py`** *(~20-line addition)*

Written after all experiments are exported. Array of manifest objects. Enables a static site's gallery page to render without directory enumeration.

**7. HTML CSS upgrade** *(self-contained change to `html.py` `_CSS` constant)*

Better typography, inter-section spacing, figure captions, a narrower max-width for print. ~40 lines of CSS replacement. Zero logic change.

### B. Existing Components to Reuse Without Modification

- `ExperimentArtefacts` — extend with new optional fields only
- `load_experiment_artefacts()` — extend by loading `diagnostics/*.json` if present
- All ten private section renderer functions — unchanged
- `_pipe_table()`, `_render_section()` — unchanged
- `markdown_to_html()` — unchanged (CSS upgrade is in `_CSS`, not the converter logic)
- `run_and_report()` — extend signature with `spec` parameter only
- `check_artefact_dir()`, `check_ml_artefacts()` — unchanged
- All visualization functions — unchanged; more are called from orchestrator
- All ML diagnostics functions — unchanged; called from orchestrator at run time

### C. Existing Components That Must NOT Be Modified

- Core `render_report()` purity contract — must remain side-effect-free
- `markdown_to_html()` scope — must remain a fixed-scope converter, not a general engine
- Figure path computation in `report_builder.py` — renderers must never compute paths
- `save_run()` / `load_run()` filesystem layout — artefact structure must remain stable
- Version routing in `orchestrator.py` — single `if version == "2"` check only
- Any factory layer function — must remain pure (no I/O, no sklearn imports)
- `interfaces.py` + `placeholders.py` — legacy scaffold; leave untouched unless formally replacing

### D. Recommended Implementation Order

```
Step 1 ── ResearchReportSpec (spec.py)
          Add spec parameter to render_report() + generate_experiment_report()
          Update generate_report.py CLI with --mode flag
          ↓ (all existing tests still pass; new mode tests added)

Step 2 ── Persist diagnostics in orchestrator
          Write diagnostics/split_metrics.json + diagnostics/ml_diagnostics.json
          Update load_experiment_artefacts() to load them
          Add diagnostics fields to ExperimentArtefacts
          ↓ (enables narrative and diagnostics sections)

Step 3 ── Diagnostics section in report (spec.include_diagnostics)
          _diagnostics_section() renderer in markdown.py
          Registered in _SECTION_REGISTRY
          ↓ (surfacing ML IC, turnover, coefficient stability in report)

Step 4 ── Narrative layer (narrative.py + spec.include_narrative)
          Pure threshold-based commentary functions
          _narrative_section() renderer
          ↓ (publication-quality interpretation layer)

Step 5 ── report_manifest.json + gallery_index.json
          Relative-path manifests for frontend consumption
          gallery_index.json from export_gallery.py
          ↓ (static website ready)

Step 6 ── HTML CSS upgrade
          Replace _CSS constant in html.py
          ↓ (publication visual quality)
```

Each step is independently deployable. The test suite validates each step before the next begins.

### E. What to Intentionally Avoid

- **`ReportSection` as a plugin system.** Register functions, not classes. No `__init_subclass__`, no dynamic discovery, no hook mechanisms.
- **Section sub-package before >10 sections.** Premature extraction adds indirection without benefit.
- **Jinja2 or any template engine.** The current string concatenation approach is appropriate for the output complexity. A template engine introduces a parse/render dependency chain with no upside.
- **Narrative AI generation.** Threshold-based rules are deterministic, auditable, and reproducible. LLM-based generation is neither.
- **Any modification to the core research engine** (backtesting, features, ML models, validation) to accommodate reporting needs.
- **Absolute paths in any web-facing output.** All manifests must use paths relative to the output directory.
- **`diagnostics/` sub-package extraction** until the narrative and diagnostics sections are validated and the section count warrants it.

---

## Part 8 — Final Assessment

### 8.1 Maturity Ratings

| Dimension | Rating | Notes |
|---|---|---|
| **Reporting infrastructure** | Solid (7/10) | Clean pipeline, pure rendering, deterministic output. Gap: no spec/mode, no diagnostics surfacing. |
| **Narrative / commentary** | Not started (1/10) | All raw material exists; zero integration to report. Gap: diagnostics not persisted, no interpretation layer. |
| **Showcase readiness** | Functional (5/10) | HTML output works but is not publication-quality. No manifest/index for frontend discovery. |
| **Frontend readiness** | Partial (4/10) | Artefacts are static-friendly; missing relative-path manifests, gallery index, machine-readable structure. |
| **Architectural cleanliness** | Strong (8/10) | Pure rendering, central path computation, clean data carrier, advisory contracts. Dead code in legacy scaffold. |
| **Separation of concerns** | Good (7/10) | Load / render / write are cleanly separated. Gap: no spec governs section inclusion. Diagnostics gap between ml/ and reporting/. |

### 8.2 Final Answers

**1. Is the platform ready for a true institutional-style research reporting layer?**

Yes, conditionally. The architecture is sound and the building blocks are in place. The gap is not structural — it is surface-level: sections exist but aren't governed by a spec, diagnostics are computed but not persisted, narrative is absent. These are additive concerns, not redesign concerns.

**2. Is the current architecture already strong enough to support it cleanly?**

Yes. The key invariants that enable clean evolution are already in place: `render_report()` is pure, `ExperimentArtefacts` is a stable data carrier, figure paths are centralized, section functions are independent. The legacy scaffold (`interfaces.py`, `placeholders.py`) is dead weight but harmless. The path forward requires extending, not refactoring.

**3. What is the highest ROI next implementation step?**

**`ResearchReportSpec` + `spec` parameter to `render_report()`**, paired with **persisting diagnostics to `diagnostics/`** during the experiment run.

These two steps together unlock the full roadmap: the spec enables mode-based output immediately, and persisting diagnostics makes the narrative layer and diagnostics section implementable without re-computation. Both steps are small (one new file + two small edits to existing files), fully backward-compatible, and independently testable.

The `report_manifest.json` step immediately follows and costs ~20 lines — but its payoff (frontend-ready artefacts) is high relative to the implementation cost.

---

## Appendix: Current Artefact Directory Layout

```
results/experiments/<experiment_name>/
  metadata.json               ← run provenance (experiment_name, strategy, created_at)
  metrics.json                ← scalar performance metrics
  config.json                 ← legacy ExperimentSpec (if present)
  normalized_config.json      ← full normalized config (D1+)
  raw_config.yaml             ← verbatim source config file
  ml_provenance.json          ← ML spec hash, features, labels, model, signal (v2 only)
  equity_curve.parquet        ← daily equity curve
  returns.parquet             ← daily net returns
  weights.parquet             ← daily position weights
  plots/
    equity_and_drawdown.png
    walk_forward_stitched.png   (if walk-forward ran)
    split_sharpes.png           (if walk-forward ran)
  diagnostics/                ← currently empty; reserved

reports/
  markdown/<experiment_name>.md
  markdown/<experiment_name>_provenance.json
  html/<experiment_name>.html
  figures/<experiment_name>/*.png

exports/experiment_gallery/
  examples/
    <experiment_name>/        ← copied report outputs
  README.md
```

**Recommended additions (marked **):**

```
results/experiments/<experiment_name>/
  diagnostics/
    split_metrics.json        ← *per-split Sharpe, return, max DD
    ml_diagnostics.json       ← *IC mean/std, directional accuracy, turnover (v2 only)

reports/
  markdown/<experiment_name>_manifest.json   ← *relative paths, tags, metrics summary
  
exports/experiment_gallery/
  gallery_index.json          ← *array of experiment summaries for frontend
```
