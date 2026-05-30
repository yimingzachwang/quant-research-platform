"""Tests for src.features.families — feature family registry and lookups."""

from __future__ import annotations

from src.features.families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_COLORS,
    FEATURE_FAMILY_DESCRIPTIONS,
    UNKNOWN_FAMILY,
    get_family_for_name,
    get_family_for_type,
    group_by_family,
)

# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

def test_feature_families_has_five_canonical_families():
    families = set(FEATURE_FAMILIES.keys())
    expected = {"Trend", "Volatility", "Mean-Reversion", "Market Structure", "Relative Strength"}
    assert families == expected


def test_feature_families_all_non_empty():
    for family, members in FEATURE_FAMILIES.items():
        assert len(members) > 0, f"{family} has no members"


def test_feature_family_colors_covers_all_families():
    for family in FEATURE_FAMILIES:
        assert family in FEATURE_FAMILY_COLORS, f"Missing color for {family}"


def test_feature_family_descriptions_covers_all_families():
    for family in FEATURE_FAMILIES:
        assert family in FEATURE_FAMILY_DESCRIPTIONS, f"Missing description for {family}"


# ---------------------------------------------------------------------------
# get_family_for_type
# ---------------------------------------------------------------------------

def test_get_family_for_type_momentum():
    assert get_family_for_type("momentum") == "Trend"


def test_get_family_for_type_rolling_volatility():
    assert get_family_for_type("rolling_volatility") == "Volatility"


def test_get_family_for_type_rolling_zscore():
    assert get_family_for_type("rolling_zscore") == "Mean-Reversion"


def test_get_family_for_type_trend_strength():
    assert get_family_for_type("trend_strength") == "Trend"


def test_get_family_for_type_downside_volatility():
    assert get_family_for_type("downside_volatility") == "Volatility"


def test_get_family_for_type_vol_of_vol():
    assert get_family_for_type("vol_of_vol") == "Volatility"


def test_get_family_for_type_bollinger_distance():
    assert get_family_for_type("bollinger_distance") == "Mean-Reversion"


def test_get_family_for_type_rolling_skewness():
    assert get_family_for_type("rolling_skewness") == "Market Structure"


def test_get_family_for_type_rolling_autocorrelation():
    assert get_family_for_type("rolling_autocorrelation") == "Market Structure"


def test_get_family_for_type_unknown_returns_unknown():
    assert get_family_for_type("some_exotic_feature") == UNKNOWN_FAMILY


# ---------------------------------------------------------------------------
# get_family_for_name
# ---------------------------------------------------------------------------

def test_get_family_for_name_mom_prefix():
    assert get_family_for_name("mom_5") == "Trend"
    assert get_family_for_name("mom_20") == "Trend"
    assert get_family_for_name("mom_252") == "Trend"


def test_get_family_for_name_vol_prefix():
    assert get_family_for_name("vol_21") == "Volatility"
    assert get_family_for_name("vol_pct_21_252") == "Volatility"


def test_get_family_for_name_zscore_prefix():
    assert get_family_for_name("zscore_20") == "Mean-Reversion"


def test_get_family_for_name_trend_prefix():
    assert get_family_for_name("trend_20") == "Trend"


def test_get_family_for_name_bollinger_prefix():
    assert get_family_for_name("bollinger_20d") == "Mean-Reversion"


def test_get_family_for_name_skew_prefix():
    assert get_family_for_name("skew_20d") == "Market Structure"


def test_get_family_for_name_autocorr_prefix():
    assert get_family_for_name("autocorr_1_60d") == "Market Structure"


def test_get_family_for_name_unknown_returns_unknown():
    assert get_family_for_name("mystery_feature_xyz") == UNKNOWN_FAMILY


# ---------------------------------------------------------------------------
# group_by_family
# ---------------------------------------------------------------------------

def test_group_by_family_standard_features():
    names = ["mom_5", "mom_20", "vol_21", "zscore_20", "trend_20"]
    groups = group_by_family(names)

    assert "Trend" in groups
    assert "mom_5" in groups["Trend"]
    assert "mom_20" in groups["Trend"]
    assert "trend_20" in groups["Trend"]

    assert "Volatility" in groups
    assert "vol_21" in groups["Volatility"]

    assert "Mean-Reversion" in groups
    assert "zscore_20" in groups["Mean-Reversion"]


def test_group_by_family_with_type_map():
    names = ["mom_5", "vol_21"]
    type_map = {"mom_5": "momentum", "vol_21": "rolling_volatility"}
    groups = group_by_family(names, feature_types=type_map)

    assert groups["Trend"] == ["mom_5"]
    assert groups["Volatility"] == ["vol_21"]


def test_group_by_family_canonical_order():
    names = ["zscore_20", "mom_5", "vol_21", "trend_20"]
    groups = group_by_family(names)
    keys = list(groups.keys())
    # Trend should appear before Volatility before Mean-Reversion
    assert keys.index("Trend") < keys.index("Volatility")
    assert keys.index("Volatility") < keys.index("Mean-Reversion")


def test_group_by_family_empty_list():
    groups = group_by_family([])
    assert groups == {}


def test_group_by_family_unknown_family_included():
    names = ["mystery_feature"]
    groups = group_by_family(names, fallback_to_name=False)
    assert UNKNOWN_FAMILY in groups


def test_group_by_family_no_duplicates():
    names = ["mom_5", "mom_20", "vol_21"]
    groups = group_by_family(names)
    all_assigned = [f for fam_list in groups.values() for f in fam_list]
    assert len(all_assigned) == len(names)
    assert set(all_assigned) == set(names)
