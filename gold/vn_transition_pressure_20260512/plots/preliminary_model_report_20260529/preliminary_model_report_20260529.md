# Preliminary VN Model Report — Validation Freeze Draft

Date: 2026-05-29  
Scope: VN validation only. Holdout/test is not used.  
Current objective: freeze the validation methodology before any final holdout readout.

## 1. Executive Summary

The current best validation pipeline is no longer a single LSTM, but a calibrated ensemble stack:

1. Frozen 5-seed `hetero_combined_full5` prediction anchor.
2. Train-selected 2D market-regime calibration using lagged volatility and lagged cross-sectional tail proxy.
3. Lightweight meta-ensemble calibration over frozen seed predictions and regime features.
4. Optional selective/uncertainty layer for low-risk coverage and conformal interval reporting.

The strongest full-coverage validation result is `hgb_abs_blend` from meta-ensemble calibration:

| Model | rel_score | Q90(\|E\|) | Q95(\|E\|) | Share \|E\|>5% | Violation days Q90>3.5% | DA |
|---|---:|---:|---:|---:|---:|---:|
| Anchor baseline | 0.04472 | 4.695% | 6.218% | 8.856% | 364 | 51.83% |
| 2D regime calibration | 0.04855 | 4.665% | 6.212% | 8.753% | 361 | 51.83% |
| Meta best rel_score: `hgb_abs_blend` | **0.04881** | 4.672% | 6.244% | 8.796% | **360** | 51.83% |
| Meta best tail: `et_tail_blend` | 0.04788 | **4.659%** | 6.222% | **8.682%** | **360** | 51.75% |

Recommendation for paper/reporting:

- Use `hgb_abs_blend` as the best full-coverage rel_score candidate.
- Use `et_tail_blend` as a tail-risk robustness candidate.
- Keep `2d_regime` as the interpretable model because it is simpler and almost as strong.
- Treat static-error riskaux retraining as ablation, not as the main model.

## 2. What Changed From Earlier Attempts

Early validation attempts improved slowly because the full-coverage point-forecast task appears close to a validation ceiling under current features. Retraining LSTM variants and static-error risk heads improved single models only marginally and did not beat the calibrated 5-seed ensemble.

The useful improvements came from post-model calibration over frozen predictions:

- 2D regime calibration improved rel_score from `0.04472` to `0.04855`.
- Meta-ensemble calibration improved best rel_score further to `0.04881`.
- Tail-focused meta calibration lowered Q90 and >5% error share more than the rel_score-best candidate.

## 3. Full-Coverage Candidate Comparison

Source: `gold/vn_transition_pressure_20260512/plots/meta_ensemble_calibration_20260528/comparison.csv`.

| Variant | rel_score | Q90(\|E\|) | Q95(\|E\|) | Share \|E\|>3.5% | Share \|E\|>5% | Daily violations | DA |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hgb_abs_blend` | **0.04881** | 4.672% | 6.244% | 16.369% | 8.796% | **360** | 51.83% |
| `2d_regime` | 0.04855 | 4.665% | 6.212% | **16.342%** | 8.753% | 361 | 51.83% |
| `et_tail_blend` | 0.04788 | **4.659%** | 6.222% | 16.415% | **8.682%** | **360** | 51.75% |
| `anchor` | 0.04472 | 4.695% | 6.218% | 16.372% | 8.856% | 364 | 51.83% |

Interpretation:

- The improvement is real but numerically modest.
- `hgb_abs_blend` is best for the paper's primary metric (`rel_score`).
- `et_tail_blend` is better when the objective is reducing error tails.
- `2d_regime` is the cleanest interpretable method and should remain central in the narrative.

## 4. Tail-Risk And Selective Forecasting

Selective forecasting gives a visibly stronger tail reduction when the system is allowed to abstain from high-risk predictions.

Source: `gold/vn_transition_pressure_20260512/plots/selective_2d_regime_20260527/coverage_metrics.csv`.

| Coverage kept | Q90(\|E\|) | Q95(\|E\|) | Share \|E\|>5% | DA |
|---:|---:|---:|---:|---:|
| 50% | 3.496% | 4.874% | 4.735% | 48.60% |
| 60% | 3.650% | 5.077% | 5.192% | 49.20% |
| 70% | 3.856% | 5.392% | 5.899% | 49.67% |
| 90% | 4.289% | 5.969% | 7.559% | 50.56% |
| 100% | 4.665% | 6.212% | 8.753% | 51.83% |

Interpretation:

- Selective mode is not a replacement for full-coverage forecasting.
- It is useful as a risk-aware decision layer.
- It supports a paper claim around selective prediction / uncertainty-aware forecasting.

## 5. Static-Error RiskAux Ablation

The two-stage static-error risk auxiliary model was tested on the stronger hetero anchor path with 3 seeds.

| Variant | Mean rel_score | Mean Q90(\|E\|) | Mean DA | Seeds |
|---|---:|---:|---:|---:|
| Stage-1 hetero | 0.02535 | 4.840% | 50.28% | 3 |
| Stage-2 static-error riskaux | 0.02548 | 4.812% | 50.44% | 3 |

Interpretation:

- The risk head improves the single model slightly.
- It does not beat the frozen 5-seed ensemble stack.
- This is useful as an ablation result: auxiliary tail-risk supervision helps, but calibrated ensembles remain stronger.

## 6. Current Paper Framing

Recommended framing:

> A regime-aware calibrated ensemble for Vietnamese equity return forecasting, with selective risk-aware prediction and conformal uncertainty diagnostics.

Core contributions:

1. A heteroscedastic multi-seed LSTM ensemble for next-return prediction.
2. Train-selected market-regime calibration using lagged volatility/tail proxies.
3. Meta-ensemble calibration using seed disagreement and regime context.
4. Selective forecasting analysis for tail-risk control.
5. Ablation showing static-error risk auxiliary training improves single models but does not dominate the calibrated ensemble.

Claims to avoid:

- Do not claim tail errors are solved.
- Do not claim production readiness.
- Do not claim holdout performance before opening holdout.

## 7. Validation Freeze Recommendation

Freeze validation methodology as follows:

- Main full-coverage model: `hgb_abs_blend`.
- Interpretable full-coverage model: `2d_regime`.
- Tail-focused model: `et_tail_blend`.
- Risk layer: selective 2D-regime coverage curve.
- Uncertainty layer: heteroscedastic conformal interval readout.
- Ablation: static-error riskaux 3-seed result.

Do not continue broad validation search unless adding genuinely new external data or changing the research question.

## 8. VN30 Simulation Next

VN30 simulation should be treated as a separate validation-style experiment before holdout. Suggested plan:

1. Build or identify a VN30 constituent universe for the validation period.
2. Recompute the same prediction/candidate outputs on VN30 subset only, without retraining if possible.
3. Compare:
   - full VN universe vs VN30 subset;
   - anchor vs 2D regime vs hgb/meta vs et-tail;
   - top-k portfolio simulation under VN30 universe.
4. Generate advisor-style plots:
   - error histogram;
   - Q90(\|E\|) time series;
   - selective coverage curve;
   - equity/backtest overlay if portfolio rules are used.
5. Keep holdout/test closed until VN30 methodology is frozen.

## 9. Key Artifacts

- Meta-ensemble summary: `gold/vn_transition_pressure_20260512/plots/meta_ensemble_calibration_20260528/summary.md`
- Meta histogram: `gold/vn_transition_pressure_20260512/plots/meta_ensemble_calibration_20260528/meta_ensemble_histogram.png`
- Regime summary: `gold/vn_transition_pressure_20260512/plots/regime_calibration_20260527/summary.md`
- Selective coverage: `gold/vn_transition_pressure_20260512/plots/selective_2d_regime_20260527/selective_coverage.png`
- Static-error ablation: `gold/vn_transition_pressure_20260512/plots/static_error_hetero_anchor_3seed_20260528/summary.md`
