# Phase 2.5 — Controlled Allocation Research Synthesis

*Generated: 2026-05-28*
*Experiments: alloc_study_ew, alloc_study_sm05, alloc_study_sm10, alloc_study_sm20*

---

## Research Design

**Objective:** Determine whether confidence-weighted (softmax) allocation produces economically meaningful improvement over equal-weight for a cross-sectional Ridge regression model on a 9-asset ETF universe (2013-2024).

**Experimental control:** Identical universe (SPY/QQQ/IWM/EEM/TLT/GLD/XLF/XLK/XLE), features (13-feature set: momentum, volatility, trend, breakout, drawdown, beta, risk-adjusted momentum), labels (21-day cross-sectional return rank), model (Ridge α=0.5), signal (top-3 by predicted score), and validation (rolling 48m train / 12m test). **Only the `portfolio_construction.weighting` block differs across experiments.**

| Arm | Scheme | Temperature | Design hypothesis |
| --- | --- | --- | --- |
| alloc_study_ew | equal_weight | — | Null: 1/k per selected asset |
| alloc_study_sm05 | zscore_softmax | τ = 0.5 | High concentration; amplifies score differences |
| alloc_study_sm10 | zscore_softmax | τ = 1.0 | Moderate concentration; standard softmax |
| alloc_study_sm20 | zscore_softmax | τ = 2.0 | Near-EW; minimal score amplification |

**Excluded by design:** τ > 2, leverage, volatility targeting, dynamic breadth, regime-conditioned scaling.

---

## A. Experiment Summary

All 4 experiments completed successfully over the 2013-2024 period with 2,767 active allocation periods (daily rebalance, top-3 selection). Walk-forward validation: 7 splits (48m train / 12m test), OOS coverage 2017-2024.

---

## B. Comparative Results

### B1. Risk-Adjusted Performance

| Scheme | Sharpe | Ann. Return | Volatility | Max DD | Calmar | Hit Rate |
| --- | --- | --- | --- | --- | --- | --- |
| Equal Weight | **0.5481** | 9.76% | 21.06% | -45.00% | 0.217 | 49.7% |
| Softmax τ=0.5 | 0.5484 | 9.78% | 21.12% | -45.25% | 0.216 | 49.7% |
| Softmax τ=1.0 | 0.5482 | 9.77% | 21.09% | -45.12% | 0.216 | 49.7% |
| Softmax τ=2.0 | 0.5482 | 9.76% | 21.07% | -45.06% | 0.217 | 49.7% |

**Walk-forward out-of-sample summary:**

| Scheme | OOS mean Sharpe | OOS hit rate (positive splits) |
| --- | --- | --- |
| Equal Weight | **0.645** | 71.4% (5 of 7 splits) |
| Softmax τ=0.5 | 0.644 | 71.4% |
| Softmax τ=1.0 | 0.645 | 71.4% |
| Softmax τ=2.0 | 0.645 | 71.4% |

**Interpretation:** Performance differences across all four schemes are negligible. The maximum Sharpe difference is Δ=0.0003 (0.5481 vs 0.5484), which is well within measurement noise for an 11-year backtest. In-sample and OOS conclusions are identical: the allocation policy is not the determinant of this strategy's returns. The signal quality — which assets the model selects — drives cross-sectional alpha entirely.

![Equity Comparison](figures/equity_comparison.png)

*All four equity curves are visually indistinguishable. Any divergence between schemes is attributable solely to portfolio construction; the convergent trajectories confirm that portfolio construction policy does not add or subtract material alpha in this setting.*

![Metrics Comparison](figures/metrics_bar_comparison.png)

---

## C. Concentration & Dispersion Findings

### C1. Concentration Diagnostics

| Scheme | Mean HHI | Eff. Breadth | Entropy Eff.-N | Mean max weight | HHI vs EW |
| --- | --- | --- | --- | --- | --- |
| Equal Weight | 0.33333 | 3.000 | 3.000 | 0.3333 | baseline |
| Softmax τ=0.5 | 0.33342 | 2.999 | 2.9996 | 0.3380 | +0.0001 |
| Softmax τ=1.0 | 0.33335 | 2.9998 | 2.9999 | 0.3357 | +0.0000 |
| Softmax τ=2.0 | 0.33334 | 2.9999 | 3.0000 | 0.3345 | +0.0000 |

**The softmax schemes show negligible concentration increase relative to equal-weight.** Equal-weight top-3 yields a theoretical HHI of 1/3 = 0.3333. The most aggressive softmax (τ=0.5) achieves HHI = 0.3334 — an increase of 0.0001, or 0.03%. Mean max weight rises from 0.333 to 0.338, a difference of 0.5%.

![HHI Comparison](figures/hhi_comparison.png)

*Rolling 63-day HHI time-series. All four curves are visually indistinguishable. The expected concentration gradient (τ=0.5 > τ=1.0 > τ=2.0 > EW) is mechanically correct but the magnitude is too small to see at this scale.*

![Breadth and Entropy Comparison](figures/breadth_entropy_comparison.png)

### C2. Root Cause: Prediction Dispersion Compression

The concentration data demands explanation. Why does τ=0.5 — which should concentrate weight aggressively — produce HHI nearly identical to equal-weight?

**The answer lies in the prediction dispersion diagnostics:**

| Dispersion metric | Value | Interpretation |
| --- | --- | --- |
| Mean cross-sectional prediction σ | **0.0362** | Very small — scores are tightly clustered |
| Mean top-minus-bottom spread (all 9 assets) | **0.102** | Total range across the universe is only ~10bp |
| Min CS std (lowest-dispersion period) | 0.0065 | Severe compression in low-signal regimes |
| Max CS std (highest-dispersion period) | 0.0799 | Maximum spread barely exceeds 8bp |

**The Ridge regression model (α=0.5) imposes strong L2 regularisation, which systematically shrinks coefficient magnitudes toward zero.** For a 21-day cross-sectional ranking label across 9 assets, the regularised predictions are highly compressed — the model assigns similar scores across all assets, with differences on the order of 3-10bp.

For softmax allocation within the **selected top-3 basket**, the relevant dispersion is even smaller than the full 9-asset spread. If the universe-wide score range is ~10bp, the intra-basket range (scores for the 1st vs 3rd ranked asset) might be 2-4bp. With τ=0.5:

```
stable_scores = (scores - max_score) / 0.5
If (s2 - s1) ≈ -0.02 and (s3 - s1) ≈ -0.04:
  weights ≈ softmax([0, -0.04, -0.08]) ≈ [0.353, 0.340, 0.327]
  HHI ≈ 0.353² + 0.340² + 0.327² ≈ 0.124 + 0.116 + 0.107 ≈ 0.347
```

But when intra-basket differences are ~1-2bp:
```
stable_scores = (scores - max_score) / 0.5
If (s2 - s1) ≈ -0.003 and (s3 - s1) ≈ -0.006:
  weights ≈ softmax([0, -0.006, -0.012]) ≈ [0.336, 0.334, 0.331]
  HHI ≈ 0.336² + 0.334² + 0.331² ≈ 0.113 + 0.112 + 0.110 ≈ 0.334
```

**This is the precise diagnostic finding:** the Ridge model's L2 regularisation compresses intra-basket score differences to a scale where softmax, regardless of temperature, produces weights indistinguishable from uniform.

![Concentration vs Temperature](figures/concentration_vs_temperature.png)

*HHI (left) and entropy eff.-N (right) vs τ. The monotonic gradient is mechanically correct — the implementation is working — but the magnitude is too small to create economically distinct concentration profiles. The dashed line shows the equal-weight reference.*

### C3. Turnover Analysis

| Scheme | Mean daily turnover | vs EW |
| --- | --- | --- |
| Equal Weight | 0.12660 | baseline |
| Softmax τ=0.5 | 0.12757 | +0.77% |
| Softmax τ=1.0 | 0.12708 | +0.38% |
| Softmax τ=2.0 | 0.12684 | +0.19% |

Turnover differences are minimal (< 1%). This confirms that the allocation scheme changes are too small to materially affect rebalancing frequency.

![Turnover Comparison](figures/turnover_comparison.png)

![Sharpe vs Concentration](figures/sharpe_vs_concentration.png)

*Sharpe vs HHI scatter. All four points cluster at HHI ≈ 0.333, confirming that meaningful concentration differentiation was not achieved. The near-flat relationship across this cluster is consistent with the null hypothesis that allocation policy is irrelevant when prediction dispersion is below the softmax activation threshold.*

---

## D. Confidence Legitimacy Findings

### D1. Calibration Results

**Quintile calibration is monotonically increasing across all schemes:**

| Quintile | Mean forward return (21d) |
| --- | --- |
| Q1 (lowest scores) | -0.33% |
| Q2 | +0.51% |
| Q3 | +1.10% |
| Q4 | +1.16% |
| Q5 (highest scores) | +1.27% |

**Top-minus-bottom spread: +1.59% per 21-day period** (Q5 vs Q1)

All four schemes share identical calibration because they use the same underlying Ridge model; calibration is a property of the predictions, not the allocation.

![Calibration Comparison](figures/calibration_comparison.png)

*Quintile bar chart: the monotonic pattern (Q1 negative → Q5 positive) confirms that prediction **sign** carries economic information. The model correctly identifies which assets will outperform vs underperform in the cross-section. However, monotonic quintile calibration alone does not justify confidence weighting — that requires the additional condition that within-basket score magnitudes are also predictive of relative return magnitudes.*

### D2. The Critical Distinction

The calibration finding must be interpreted carefully:

1. **What monotonic calibration confirms:** The model ranks assets correctly. Higher-scored assets outperform lower-scored assets on average. The *selection* step (choosing the top 3) is economically meaningful. This validates the strategy's core alpha mechanism.

2. **What monotonic calibration does NOT confirm:** That within the selected top-3 basket, score magnitudes predict relative return magnitudes. A model can have perfect quintile calibration (Q1 < Q2 < Q3 < Q4 < Q5) while the within-Q5 score differences are economically arbitrary.

3. **The evidence for the second condition is weak:** The compressed prediction range (CS std = 0.036) suggests the model has difficulty discriminating between the 1st, 2nd, and 3rd ranked assets. The Ridge regularisation is appropriate for ranking but destroys the amplitude information needed for confidence weighting.

**Conclusion on confidence legitimacy:** The model's confidence ordering is economically valid (the ranking is predictive). The model's confidence magnitude within the top basket is not distinguishable from noise. Equal-weight is the institutionally appropriate policy for this signal.

---

## E. Validation Safety Confirmation

**Chronology integrity:** All four experiments use rolling walk-forward validation (48m train / 12m test, 0-day gap). Each allocation decision on date *t* is made using predictions from a model trained only on data up to *t−1* (enforced by `shift(1)` in the backtest engine).

**Row-wise allocation:** All softmax operations are timestamp-local. For each date *t*, the softmax is computed using only the prediction scores at *t*, with no reference to other dates. No cross-date normalization is applied.

**No cross-experiment contamination:** Each experiment is an independent run with its own fitted model, independent walk-forward splits, and independent provenance hash. The same model object is not shared across experiments.

**No post-hoc optimization:** The experiment matrix was specified before running. No configuration adjustments were made after observing results.

**Provenance:**
- alloc_study_ew: spec_version="2", weighting scheme="equal_weight"
- alloc_study_sm05: spec_version="2", weighting scheme="zscore_softmax", temperature=0.5
- alloc_study_sm10: spec_version="2", weighting scheme="zscore_softmax", temperature=1.0
- alloc_study_sm20: spec_version="2", weighting scheme="zscore_softmax", temperature=2.0

---

## F. Research Assessment

### F1. Does confidence weighting improve risk-adjusted returns?

**No — not for this model configuration.** The Sharpe difference between equal-weight (0.5481) and the best softmax variant (0.5484, τ=0.5) is Δ=0.0003, within measurement noise for this backtest period. The OOS walk-forward results confirm the same conclusion: identical hit rates and negligible mean OOS Sharpe differences.

### F2. Does concentration amplify instability?

**Not meaningfully** — because meaningful concentration was not achieved. The maximum HHI is 0.3334 vs 0.3333 for equal-weight. There is no concentration-induced drawdown amplification because no concentration occurred.

### F3. Does model confidence contain economic information?

**Partially:** The model's ranking order is economically valid (monotonic quintile calibration, +1.59% Q5-Q1 spread). The model's score magnitudes within the selected basket are not economically distinguishable from noise at this regularisation level.

### F4. Does prediction dispersion behave regime-sensitively?

**Yes:** The cross-sectional prediction std ranges from 0.0065 (severe compression) to 0.080 (maximum signal). This 12× variation suggests there are distinct regimes where the model produces more or less discriminating scores. However, even at maximum dispersion, intra-basket score differences remain small relative to what softmax needs to create meaningful concentration.

### F5. Is this platform's allocation behaviour institutionally plausible?

**Yes.** The research conclusion — that equal-weight is appropriate for a heavily regularised Ridge model — is institutionally defensible and research-honest. Softmax allocation requires a model that produces meaningfully dispersed confidence scores within the selected basket. This Ridge configuration does not; identifying this limitation through controlled experimentation is the correct outcome.

---

## G. Directions for Future Research

Based on this study, confidence-weighted allocation could be revisited under:

1. **Reduced regularisation:** Ridge α ∈ {0.01, 0.1} would produce less compressed predictions, potentially creating intra-basket score differences large enough for softmax to exploit. Risk: higher overfitting probability.

2. **Prediction normalization:** Using `prediction_normalization: zscore` would re-scale cross-sectional predictions to mean=0, std=1 before passing to softmax. This would mechanically amplify intra-basket differences regardless of raw score magnitude. Risk: the z-scores may inflate noise rather than signal.

3. **Alternative model types:** Gradient boosting or random forests typically produce more dispersed predictions than regularised linear models, with score distributions that may better support confidence weighting.

4. **Larger universes:** With 9 assets and top-3 selection, the within-basket rank compression is acute. Larger universes (30+ assets, top-5 selection) would produce greater spread between basket members.

5. **Longer prediction horizons:** Shorter horizons (e.g., 5-day) introduce more noise; 21-day is appropriate. Longer horizons (63-day quarterly) might produce more stable, dispersed scores.

---

## H. Synthesis Conclusion

> **The Phase 2.5 controlled allocation research experiment demonstrates that the platform's diagnostic infrastructure correctly identifies when confidence-weighted allocation is not institutionally justified. The Ridge model with α=0.5 produces prediction scores with insufficient intra-basket dispersion (CS std = 0.036) for softmax to differentiate meaningfully from equal-weight. The model does exhibit economically valid ranking (monotonic quintile calibration, 1.59% Q5-Q1 spread), validating the selection step. The allocation step — how to distribute weight within the selected basket — is irrelevant at this level of prediction compression. Equal-weight is the correct institutional choice for this signal configuration.**

This is a research success, not a research failure. Identifying that a signal generates alpha through selection rather than through confidence-proportional weighting is a precise, actionable research finding. The platform now possesses the diagnostic infrastructure — prediction dispersion monitoring, calibration analysis, concentration evolution, and controlled comparative experimentation — to evaluate this question rigorously for any future model configuration.

---

*Phase 2.5 Allocation Research Synthesis*
*Allocation Study: alloc_study_ew, alloc_study_sm05, alloc_study_sm10, alloc_study_sm20*
*Universe: SPY, QQQ, IWM, EEM, TLT, GLD, XLF, XLK, XLE (2013–2024)*
*Model: Ridge α=0.5, 13 features, ranking_target label, top-3 signal, 48m/12m WFV*
