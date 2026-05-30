"""Tests for src.features.families.generate_feature_label — canonical label authority."""

from __future__ import annotations

from src.features.families import generate_feature_label

# ---------------------------------------------------------------------------
# Hardcoded overrides
# ---------------------------------------------------------------------------

def test_mom_5_override():
    assert generate_feature_label("mom_5") == "5D Momentum"

def test_mom_20_override():
    assert generate_feature_label("mom_20") == "20D Momentum"

def test_vol_21_override():
    assert generate_feature_label("vol_21") == "21D Realized Volatility"

def test_zscore_20_override():
    assert generate_feature_label("zscore_20") == "20D Z-Score"

def test_trend_20_override():
    assert generate_feature_label("trend_20") == "20D Trend Strength"


# ---------------------------------------------------------------------------
# Pattern-derived labels
# ---------------------------------------------------------------------------

def test_momentum_pattern_60():
    label = generate_feature_label("mom_60")
    assert "60" in label and "Momentum" in label

def test_momentum_pattern_252():
    label = generate_feature_label("mom_252")
    assert "252" in label and "Momentum" in label

def test_vol_pattern_63():
    label = generate_feature_label("vol_63")
    assert "63" in label and "Volatility" in label

def test_downside_vol_pattern():
    label = generate_feature_label("downside_vol_21d")
    assert "21" in label and ("Downside" in label or "Volatility" in label)

def test_vol_of_vol_pattern():
    label = generate_feature_label("vol_of_vol_21_63")
    assert "21" in label and "63" in label

def test_vol_percentile_pattern():
    label = generate_feature_label("vol_pct_21_252")
    assert "21" in label and "252" in label

def test_zscore_pattern_60():
    label = generate_feature_label("zscore_60")
    assert "60" in label and ("Z-Score" in label or "Zscore" in label or "Z Score" in label)

def test_bollinger_pattern():
    label = generate_feature_label("bollinger_20d")
    assert "20" in label and "Bollinger" in label

def test_skewness_pattern():
    label = generate_feature_label("skew_60d")
    assert "60" in label and "Skew" in label

def test_autocorr_pattern():
    label = generate_feature_label("autocorr_1_60d")
    assert "1" in label and "60" in label and "Autocorr" in label

def test_trend_pattern_different_window():
    label = generate_feature_label("trend_60")
    assert "60" in label and "Trend" in label

def test_sma_pattern():
    label = generate_feature_label("sma_50")
    assert "50" in label and "SMA" in label

def test_ema_pattern():
    label = generate_feature_label("ema_12")
    assert "12" in label and "EMA" in label


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

def test_unknown_name_falls_back_to_title_case():
    label = generate_feature_label("mystery_signal_x")
    assert isinstance(label, str)
    assert len(label) > 0

def test_empty_string_returns_string():
    label = generate_feature_label("")
    assert isinstance(label, str)

def test_return_type_always_str():
    for name in ["mom_5", "vol_21", "bollinger_20d", "skew_60d", "completely_unknown_xyz"]:
        result = generate_feature_label(name)
        assert isinstance(result, str), f"Expected str for {name!r}"


# ---------------------------------------------------------------------------
# Consistency: labels are non-empty and don't contain raw underscores for known features
# ---------------------------------------------------------------------------

def test_known_features_no_trailing_underscores():
    known = ["mom_5", "mom_20", "vol_21", "zscore_20", "trend_20",
             "bollinger_20d", "skew_60d", "autocorr_1_60d"]
    for name in known:
        label = generate_feature_label(name)
        assert not label.startswith("_")
        assert not label.endswith("_")
