# Research Report Framework Proposal

Status: design proposal with implemented foundations  
Last updated: 2026-05-23

This document proposes a configurable research reporting framework for the
quant research platform. It is a design proposal, not a request to add a new
reporting engine. The proposal extends existing implemented pieces:

- `ExperimentArtefacts`
- `ReportPaths`
- `ResearchReportSpec`
- `render_report`
- `generate_experiment_report`
- report provenance sidecars
- report manifests

## 1. Current Implemented Reporting Foundation

The current reporting system is artefact-driven and read-only. It loads saved
experiment artefacts from disk, discovers figures, renders markdown, optionally
renders HTML, writes provenance, and writes a frontend-facing manifest.

Implemented report section governance is provided by `ResearchReportSpec`, a
frozen dataclass with boolean section flags:

- summary
- metadata
- configuration
- metrics
- ML analysis
- validation
- diagnostics
- figures
- provenance

Implemented presets:

- `FULL_DEMO_REPORT`
- `COMPACT_REPORT`
- `DIAGNOSTICS_REPORT`
- `AUDIT_REPORT`

Current renderer behaviour is fixed-section and deterministic. This is a
strength: reports are predictable, testable, and do not rely on plugins or
template engines.

## 2. Proposed Reporting Modes

The platform should formalise three user-facing report modes on top of the
existing `ResearchReportSpec`.

### Compact Mode

Purpose:

- quick review
- email or chat summary
- rapid triage of many experiments

Sections:

- summary
- metadata
- compact config
- key metrics
- no figures by default
- no diagnostics by default

Current implementation mapping:

- close to `COMPACT_REPORT`

### Research Mode

Purpose:

- day-to-day research review
- strategy debugging
- ML experiment inspection
- walk-forward analysis

Sections:

- summary
- metadata
- configuration
- metrics
- ML analysis when available
- validation
- diagnostics
- selected figures
- provenance

Current implementation mapping:

- close to `DIAGNOSTICS_REPORT`

### Showcase Mode

Purpose:

- portfolio presentation
- interview artefacts
- polished technical demonstrations

Sections:

- concise summary
- visual strategy narrative
- key metrics
- selected figures
- methodology note
- limitations
- provenance

Current implementation mapping:

- partially supported by `FULL_DEMO_REPORT`
- a future preset could tune section order and figure selection for
  presentation quality

## 3. Proposed `ResearchReportSpec` Evolution

The implemented `ResearchReportSpec` is intentionally minimal. Future
evolution should preserve the same dataclass-centred design.

Potential additions:

```python
@dataclass(frozen=True)
class ResearchReportSpec:
    mode: Literal["compact", "research", "showcase", "audit"]
    include_summary: bool
    include_metadata: bool
    include_configuration: bool
    include_metrics: bool
    include_ml_analysis: bool
    include_validation: bool
    include_diagnostics: bool
    include_figures: bool
    include_provenance: bool
    figure_whitelist: tuple[str, ...] = ()
    metric_whitelist: tuple[str, ...] = ()
```

This remains simple and inspectable. It avoids a plugin system and avoids
turning reporting into a second orchestration framework.

## 4. Configurable Sections

Future section controls should remain explicit:

- performance summary
- data/universe summary
- strategy or ML configuration
- metrics table
- walk-forward validation table
- ML diagnostics
- turnover diagnostics
- figures
- provenance
- limitations

Sections should render only from existing artefacts. If an artefact is absent,
the section should either be omitted or render a clear note. It should not
trigger recomputation.

## 5. Visualisation Integration

Visualisation integration should remain path-based and read-only.

Current behaviour:

- experiments save figures
- report builder copies figures
- renderers receive precomputed relative paths

Proposed future behaviour:

- figure groups can be selected by report mode
- showcase mode can prioritise equity, drawdown, validation, and portfolio
  diagnostics
- research mode can include broader diagnostics
- audit mode can omit figures and focus on tables/provenance

The visualisation layer should continue returning matplotlib figures from
computed data. Reporting should continue consuming saved images.

## 6. Output Formats

Implemented:

- markdown
- HTML through a minimal fixed-scope converter
- provenance JSON
- report manifest JSON

Proposed future:

- PDF export as an optional external conversion step
- DOCX export as a standalone conversion step if needed
- static site/gallery output driven by report manifests

PDF/DOCX generation should not become core research logic. It should consume
rendered markdown/HTML or saved report artefacts.

## 7. Architecture Integration

The reporting framework should integrate with existing architecture as follows:

```text
ExperimentRun
-> saved artefacts
-> load_experiment_artefacts()
-> ResearchReportSpec
-> render_report()
-> markdown/html/provenance/manifest
```

No report mode should load market data, fit models, run backtests, generate
validation splits, or recompute metrics.

The report manifest should serve future frontend consumers. A dashboard can
read report metadata, files, figures, tags, and metric summaries without
understanding the full Python research stack.

## 8. Automated Report Generation

Implemented:

- `run_and_report(config_path, report_output_dir, include_html)` as a thin
  wrapper around experiment execution and report generation
- `scripts/run_from_config.py` supports report generation flags

Proposed:

- allow a report spec/mode to be selected from config
- persist selected mode in report provenance
- generate compact and research reports from the same experiment artefact
  directory without rerunning the experiment

This should be added by extending existing function parameters and config
normalisation, not by creating a new reporting orchestrator.

## 9. Future Frontend Integration

The report manifest creates a natural boundary for frontend integration.

Potential frontend flow:

```text
reports/markdown/<experiment>_manifest.json
-> list available reports
-> display metric summary
-> link markdown/html
-> show copied figures
```

This should remain a consumer of generated artefacts. The frontend should not
directly run experiments or mutate registries.

## 10. Non-Goals

The reporting framework should not introduce:

- a plugin system
- a general markdown engine
- a template marketplace
- database-backed report storage
- live dashboard computation
- hidden experiment reruns
- report-time metric recomputation

The existing fixed-section renderer is an appropriate foundation for the
current platform. Future work should preserve its determinism.

## 11. Recommended Next Steps

1. Expose report mode selection through existing CLI/config paths.
2. Add a `SHOWCASE_REPORT` preset if presentation workflows need it.
3. Add optional figure selection controls.
4. Persist selected report spec/mode in provenance.
5. Add tests proving reports generated with different specs do not recompute
   experiment artefacts.
6. Keep PDF/DOCX export as an external conversion layer unless repeated use
   justifies package integration.

