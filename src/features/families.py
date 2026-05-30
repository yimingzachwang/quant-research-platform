"""Feature family registry — institutional semantic grouping.

Defines the five canonical feature families used in this research platform.
Family membership drives report narrative, IC grouping, and diagnostic labels.

Design:
  - Family membership is defined by feature *type* (config-level) and by
    naming convention (feature *name* prefix).  Both lookups are provided.
  - Pure data — no computation, no imports from other src modules.
  - The five families mirror the institutional hypothesis taxonomy:
      Trend         — directional momentum and price continuation
      Volatility    — risk-regime indicators and vol structure
      Mean-Reversion — price normalisation and reversion pressure
      Market Structure — distributional and autocorrelation properties
      Relative Strength — cross-sectional ranking (multi-asset)
  - generate_feature_label() is the single canonical label authority for all
    feature display names — both visualisations and report sections call it.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Family definitions
# ---------------------------------------------------------------------------

FEATURE_FAMILIES: dict[str, list[str]] = {
    "Trend": [
        "momentum",
        "trend_strength",
        "trend_persistence",
        "breakout_strength",
        "risk_adjusted_momentum",
        "sma",
        "ema",
        "sma_crossover",
    ],
    "Volatility": [
        "rolling_volatility",
        "ewm_volatility",
        "downside_volatility",
        "vol_of_vol",
        "vol_percentile",
        "vol_expansion",
        "vol_compression",
    ],
    "Mean-Reversion": [
        "rolling_zscore",
        "bollinger_distance",
        "rolling_rank",
        "rolling_minmax",
        "drawdown_distance",
    ],
    "Market Structure": [
        "rolling_skewness",
        "rolling_kurtosis",
        "rolling_autocorrelation",
        "rolling_beta",
    ],
    "Relative Strength": [
        "cross_sectional_rank",
        "relative_momentum",
    ],
}

FEATURE_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "Trend": (
        "Directional momentum and price-continuation signals. "
        "Positive IC in trending regimes; may invert during mean-reverting markets."
    ),
    "Volatility": (
        "Risk-regime indicators capturing realised and implied volatility structure. "
        "Tend to be negatively correlated with Trend signals in high-vol regimes."
    ),
    "Mean-Reversion": (
        "Price normalisation and short-horizon reversion pressure. "
        "Effective when markets are range-bound or temporarily dislocated."
    ),
    "Market Structure": (
        "Systematic risk exposure and cross-asset sensitivity dynamics. "
        "Rolling beta captures time-varying market linkage; relevant to regime positioning and defensive vs. high-beta rotation."
    ),
    "Relative Strength": (
        "Cross-sectional ranking and asset-vs-universe momentum. "
        "Require multi-asset universe; not applicable for single-asset experiments."
    ),
}

FEATURE_FAMILY_COLORS: dict[str, str] = {
    "Trend": "#2166AC",
    "Volatility": "#D73027",
    "Mean-Reversion": "#1A9850",
    "Market Structure": "#984EA3",
    "Relative Strength": "#FF7F00",
}

# ---------------------------------------------------------------------------
# Type → family mapping (inverted from FEATURE_FAMILIES)
# ---------------------------------------------------------------------------

_TYPE_TO_FAMILY: dict[str, str] = {
    feat_type: family
    for family, types in FEATURE_FAMILIES.items()
    for feat_type in types
}

# ---------------------------------------------------------------------------
# Name prefix → family mapping (for runtime column names)
# ---------------------------------------------------------------------------

_NAME_PREFIX_FAMILIES: list[tuple[str, str]] = [
    ("mom_", "Trend"),
    ("momentum_", "Trend"),
    ("trend_persist_", "Trend"),
    ("trend_", "Trend"),
    ("breakout_", "Trend"),
    ("sharpe_mom_", "Trend"),
    ("sma_", "Trend"),
    ("ema_", "Trend"),
    ("vol_compress_", "Volatility"),
    ("vol_", "Volatility"),
    ("ewm_vol_", "Volatility"),
    ("downside_vol_", "Volatility"),
    ("vol_of_vol_", "Volatility"),
    ("vol_pct_", "Volatility"),
    ("vol_exp_", "Volatility"),
    ("drawdown_dist_", "Mean-Reversion"),
    ("zscore_", "Mean-Reversion"),
    ("bollinger_", "Mean-Reversion"),
    ("rank_", "Mean-Reversion"),
    ("minmax_", "Mean-Reversion"),
    ("beta_", "Market Structure"),
    ("skew_", "Market Structure"),
    ("kurt_", "Market Structure"),
    ("autocorr_", "Market Structure"),
    ("cross_sec_", "Relative Strength"),
    ("rel_mom_", "Relative Strength"),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

UNKNOWN_FAMILY = "Other"


def get_family_for_type(feature_type: str) -> str:
    """Return the canonical family name for a config-level feature type.

    Returns UNKNOWN_FAMILY if the type is not registered.
    """
    return _TYPE_TO_FAMILY.get(feature_type, UNKNOWN_FAMILY)


def get_family_for_name(feature_name: str) -> str:
    """Infer the canonical family name from a runtime feature column name.

    Uses prefix matching against _NAME_PREFIX_FAMILIES.  Returns UNKNOWN_FAMILY
    if no prefix matches.
    """
    lower = feature_name.lower()
    for prefix, family in _NAME_PREFIX_FAMILIES:
        if lower.startswith(prefix):
            return family
    return UNKNOWN_FAMILY


def group_by_family(
    feature_names: list[str],
    *,
    fallback_to_name: bool = True,
    feature_types: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Group a list of feature names by their canonical family.

    Args:
        feature_names: Runtime column names (e.g. ["mom_5", "vol_21"]).
        fallback_to_name: If True, use name-prefix lookup when the type lookup
            fails.  If False, unregistered features go to UNKNOWN_FAMILY.
        feature_types: Optional mapping of feature_name → feature_type for
            type-based lookup (e.g. from the feature registry).

    Returns:
        Dict mapping family_name → [feature_name, ...], only including
        families that have at least one member.  Order mirrors FEATURE_FAMILIES.
    """
    result: dict[str, list[str]] = {}
    type_map = feature_types or {}

    for name in feature_names:
        ftype = type_map.get(name)
        if ftype:
            family = get_family_for_type(ftype)
        else:
            family = UNKNOWN_FAMILY

        if family == UNKNOWN_FAMILY and fallback_to_name:
            family = get_family_for_name(name)

        result.setdefault(family, []).append(name)

    # Reorder according to canonical family order
    canonical_order = list(FEATURE_FAMILIES.keys()) + [UNKNOWN_FAMILY]
    return {
        fam: result[fam]
        for fam in canonical_order
        if fam in result
    }


# ---------------------------------------------------------------------------
# Canonical label authority (G-SYNC-1)
# ---------------------------------------------------------------------------

# Hardcoded overrides take strict precedence over pattern-derived labels.
_LABEL_OVERRIDES: dict[str, str] = {
    "mom_5": "5D Momentum",
    "mom_20": "20D Momentum",
    "mom_60": "60D Momentum",
    "mom_252": "252D Momentum",
    "vol_21": "21D Realized Volatility",
    "zscore_20": "20D Z-Score",
    "trend_20": "20D Trend Strength",
    # Phase H-1 additions
    "trend_persist_20d": "20D Trend Persistence",
    "breakout_63d": "63D Breakout Strength",
    "drawdown_dist_252d": "252D Drawdown Distance",
    "vol_compress_21_63": "Vol Compression (21/63D)",
    "beta_60d": "60D Market Beta",
    "sharpe_mom_252d": "252D Risk-Adj Momentum",
}

# Ordered patterns: (compiled_regex, format_string_with_positional_groups)
_LABEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Momentum
    (re.compile(r"^mom_(\d+)$"), "{0}D Momentum"),
    (re.compile(r"^momentum_(\d+)d?$"), "{0}D Momentum"),
    # Realized volatility
    (re.compile(r"^vol_(\d+)d?(?:_ann)?$"), "{0}D Realized Volatility"),
    (re.compile(r"^ewm_vol_(\d+)span(?:_ann)?$"), "EWM Volatility (span={0})"),
    (re.compile(r"^downside_vol_(\d+)d?$"), "{0}D Downside Volatility"),
    (re.compile(r"^vol_of_vol_(\d+)_(\d+)$"), "Vol-of-Vol ({0}D/{1}D)"),
    (re.compile(r"^vol_pct_(\d+)_(\d+)$"), "Vol Pct ({0}D/{1}D)"),
    # Z-score / rolling normalisation
    (re.compile(r"^zscore_(\d+)d?$"), "{0}D Z-Score"),
    (re.compile(r"^.*_zscore_(\d+)d?$"), "{0}D Z-Score"),
    # Mean-reversion
    (re.compile(r"^bollinger_(\d+)d?$"), "{0}D Bollinger Dist"),
    # Market structure
    (re.compile(r"^skew_(\d+)d?$"), "{0}D Skewness"),
    (re.compile(r"^autocorr_(\d+)_(\d+)d?$"), "Lag-{0} Autocorr ({1}D)"),
    # Trend
    (re.compile(r"^trend_(?:strength_)?(\d+)d?$"), "{0}D Trend Strength"),
    (re.compile(r"^sma_(\d+)$"), "{0}D SMA"),
    (re.compile(r"^ema_(\d+)$"), "{0}D EMA"),
    (re.compile(r"^sma_cross_(\d+)_(\d+)$"), "SMA Cross ({0}/{1})"),
    # Rolling utilities
    (re.compile(r"^.*_rank_(\d+)d?$"), "{0}D Rolling Rank"),
    (re.compile(r"^rank_(\d+)d?$"), "{0}D Rolling Rank"),
    (re.compile(r"^.*_minmax_(\d+)d?$"), "{0}D Min-Max"),
    (re.compile(r"^minmax_(\d+)d?$"), "{0}D Min-Max"),
    # Phase H-1 additions
    (re.compile(r"^trend_persist_(\d+)d?$"), "{0}D Trend Persistence"),
    (re.compile(r"^breakout_(\d+)d?$"), "{0}D Breakout Strength"),
    (re.compile(r"^drawdown_dist_(\d+)d?$"), "{0}D Drawdown Distance"),
    (re.compile(r"^vol_compress_(\d+)_(\d+)$"), "Vol Compression ({0}/{1}D)"),
    (re.compile(r"^beta_(\d+)d?$"), "{0}D Market Beta"),
    (re.compile(r"^sharpe_mom_(\d+)d?$"), "{0}D Risk-Adj Momentum"),
]


def generate_feature_label(name: str) -> str:
    """Return a publication-ready display label for any feature column name.

    This is the single canonical label authority for all feature display
    names across plots, tables, and report prose.  Priority order:

    1. Hardcoded overrides in _LABEL_OVERRIDES (for established canonical names).
    2. Regex pattern matching on the naming convention.
    3. Title-case formatting of the raw name as a last-resort fallback.

    Args:
        name: Raw feature column name (e.g. "mom_20", "bollinger_20d").

    Returns:
        Publication-ready label string.
    """
    if name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[name]

    for pattern, template in _LABEL_PATTERNS:
        m = pattern.match(name)
        if m:
            return template.format(*m.groups())

    # Fallback: replace underscores and title-case
    return name.replace("_", " ").title()
