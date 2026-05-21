# Tail-Aware LSTM Multiseed Readout

Scope: VN train/validation only. Holdout/test is not used.

Setup: LSTM `[64, 32]`, window `15`, 29 portable features, epochs `18`, patience `5`, seeds `43, 52, 71`.

## By-Seed Validation

| seed | variant | rel_score | pooled q90 error | daily q90 p90 | daily q90 max | days >= 7% | days >= 8% | pred/actual abs q90 |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 43 | `plain_global_instance_rel` | 0.00448 | 5.010% | 6.571% | 8.183% | 36 | 2 | 0.116 |
| 43 | `plain_global_rel` | 0.01719 | 4.918% | 6.497% | 9.361% | 39 | 14 | 0.142 |
| 43 | `plain_global_weighted_mild` | 0.02968 | 4.831% | 6.460% | 10.508% | 40 | 10 | 0.150 |
| 52 | `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |
| 52 | `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| 52 | `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 |
| 71 | `plain_global_instance_rel` | 0.00889 | 4.990% | 6.554% | 8.615% | 35 | 3 | 0.133 |
| 71 | `plain_global_rel` | 0.01744 | 4.927% | 6.606% | 9.534% | 44 | 16 | 0.123 |
| 71 | `plain_global_weighted_mild` | 0.00391 | 4.974% | 6.523% | 7.992% | 45 | 0 | 0.091 |

## Aggregate Validation

| variant | rel_score mean | rel_score std | pooled q90 mean | daily q90 p90 mean | daily q90 max mean | days >= 7% mean | days >= 8% mean | pred/actual abs q90 mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild` | 0.02021 | 0.01419 | 4.880% | 6.458% | 10.229% | 42.0 | 9.0 | 0.130 |
| `plain_global_rel` | 0.01667 | 0.00112 | 4.926% | 6.563% | 9.506% | 40.3 | 13.0 | 0.134 |
| `plain_global_instance_rel` | 0.00650 | 0.00223 | 4.997% | 6.598% | 8.447% | 36.0 | 2.0 | 0.115 |

## Target Status

- `plain_global_weighted_mild` improves mean `rel_score` versus baseline and keeps pooled q90 error below `5%` on all three seeds.
- `plain_global_weighted_mild` does not reliably control tail spikes: it worsens max daily q90 error on seeds 43 and 52, but improves seed 71.
- `plain_global_instance_rel` consistently reduces `>=8%` spike days, but its `rel_score` is much lower than baseline.
- No tested all-days LSTM variant reaches daily q90 p90 below `5%` or daily q90 max below `7%` across seeds.

## Decision

Do not freeze a new gold model from this run.

The useful research result is now clear: mild imbalance-aware training improves the central forecasting objective, while instance normalization improves spike stability. The next model should combine these as a calibrated hybrid instead of choosing one directly.

Recommended next experiment:

1. Train `plain_global_weighted_mild` as the main return forecaster.
2. Add a validation-fitted calibration/shrinkage layer for high predicted amplitude or high estimated tail stress.
3. Compare against `plain_global_rel` with the same 3 seeds before promoting anything to `gold`.