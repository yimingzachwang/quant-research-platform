# Extension Patterns

This file describes extension paths that fit the current architecture. It is
not a claim that the extensions are already implemented.

## Add A New Dataset Source

1. Extend `DataSource` and supported config only if needed.
2. Implement a downloader matching the `DataDownloader` protocol.
3. Add or reuse a standardizer that outputs canonical columns.
4. Validate through `DatasetValidator`.
5. Persist through `DataStorage`.
6. Register `DatasetManifest`.
7. Add offline tests with fake downloader data.

Avoid: parallel registries, hardcoded loader paths, strategy logic in data code.

## Add A New Feature

1. Implement a pure function under `src/features`.
2. Preserve input index and deterministic output.
3. Use trailing windows unless explicitly creating a label.
4. Add tests for NaN warm-up, naming, and shape.
5. Compose into ML matrices with `build_feature_matrix` when needed.

Avoid: data loading or persistence inside feature functions.

## Add A New Strategy

1. Subclass `Strategy`.
2. Implement `generate_weights(prices)`.
3. Do not lag weights internally.
4. Add tests for shape, warm-up behavior, and row sums/exposure.
5. Run via `run_strategy` or D1 config factory if registered.

Avoid: plotting, file I/O, and hidden data fetches.

## Add A New ML Model

1. Satisfy `BaseMLModel`.
2. Fit only on `SupervisedDataset`.
3. Predict on `X` and return `PredictionSeries`.
4. Preserve prediction index exactly.
5. Add tests for fit-before-predict, dtype, index alignment, and edge cases.

Avoid: generating splits inside the model.

## Add A New Report

1. Consume saved experiment artifacts.
2. Do not rerun experiments or recompute metrics.
3. Keep path computation centralized.
4. Preserve provenance sidecars.

