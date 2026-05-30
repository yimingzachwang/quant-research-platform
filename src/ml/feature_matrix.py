"""Feature matrix composition for supervised ML research.

Composes existing src/features/ outputs into aligned, ML-ready matrices.
Does NOT re-implement feature computation — all logic lives in src/features/.

align_features_and_labels() is the canonical leakage-safe alignment step and
is the only function that drops NaN rows.  build_feature_matrix() is purely
additive; it never drops rows.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def build_feature_matrix(
    prices: pd.DataFrame,
    feature_fns: dict[str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]],
) -> pd.DataFrame:
    """Compose feature callables into a wide feature matrix.

    Each callable receives the full price DataFrame and returns either:

    - pd.Series: renamed to the dict key and added as one column.
    - pd.DataFrame with a single column: renamed to the dict key.
    - pd.DataFrame with multiple columns: each column is prefixed with
      ``{key}_`` to avoid name collisions and make provenance clear.

    All results are aligned on the DatetimeIndex and concatenated
    horizontally.  NaN rows are NOT dropped here — call
    align_features_and_labels() downstream to remove them before dataset
    construction.

    Args:
        prices: Price DataFrame (DatetimeIndex × assets).  Passed verbatim
            to each callable.
        feature_fns: Ordered mapping of feature_name → callable.  Evaluated
            in insertion order; column names are derived from the keys.

    Returns:
        Wide feature DataFrame with DatetimeIndex and one or more columns.
        Returns an empty DataFrame (with prices.index) if feature_fns is empty.
    """
    frames: list[pd.DataFrame] = []
    for name, fn in feature_fns.items():
        result = fn(prices)
        if isinstance(result, pd.Series):
            frames.append(result.rename(name).to_frame())
        elif isinstance(result, pd.DataFrame):
            if len(result.columns) == 1:
                frames.append(result.rename(columns={result.columns[0]: name}))
            else:
                frames.append(result.add_prefix(f"{name}_"))
    if not frames:
        return pd.DataFrame(index=prices.index)
    return pd.concat(frames, axis=1)


def align_features_and_labels(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    """Inner-join X and y on index; drop all rows with any NaN.

    This is the canonical leakage-safe alignment step.  The returned pair
    has an identical DatetimeIndex containing only rows where every feature
    and every label value is present.

    NaN rows are dropped, not imputed.  Callers who need imputation should
    apply it before calling this function.

    Why this is leakage-safe:
    - Labels generated via shift(-horizon) already have NaN on the last
      horizon rows.  Those rows are removed here before any model sees them.
    - Features generated via rolling windows have NaN on the first window-1
      rows.  Those rows are also removed here.
    - The resulting index only contains dates where both features and labels
      are fully observed.

    Args:
        X: Feature matrix (DatetimeIndex × features).
        y: Label series or DataFrame (DatetimeIndex × targets).

    Returns:
        (X_aligned, y_aligned) with identical index and no NaN values.
        y_aligned preserves the original type (Series or DataFrame).
    """
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]

    x_valid = X_aligned.notna().all(axis=1)
    if isinstance(y_aligned, pd.Series):
        y_valid = y_aligned.notna()
    else:
        y_valid = y_aligned.notna().all(axis=1)

    clean = x_valid & y_valid
    return X_aligned.loc[clean], y_aligned.loc[clean]
