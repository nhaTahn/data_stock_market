# StressAux LSTM Multiseed Readout

Scope: VN train/validation only. Holdout/test is not used.

StressAux keeps LSTM as the return forecaster and adds an auxiliary market-stress classification head.

## Validation By Seed

| seed | variant | rel_score | q90 error | daily q90 p90 | daily max | days >=7% | days >=8% | pred/actual q90 |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 43 | `plain_global_weighted_mild` | 0.02968 | 4.831% | 6.460% | 10.508% | 40 | 10 | 0.150 |
| 43 | `plain_global_weighted_mild_tail35_p05_tailstress` | 0.02652 | 4.830% | 6.265% | 10.549% | 34 | 14 | 0.179 |
| 43 | `plain_global_weighted_mild_tail35_p05` | 0.02580 | 4.836% | 6.405% | 10.007% | 37 | 11 | 0.168 |
| 43 | `plain_global_weighted_mild_tail35_stressaux_w20` | 0.02168 | 4.878% | 6.434% | 9.423% | 30 | 8 | 0.164 |
| 43 | `plain_global_rel` | 0.01719 | 4.918% | 6.497% | 9.361% | 39 | 14 | 0.142 |
| 43 | `plain_global_instance_rel` | 0.00448 | 5.010% | 6.571% | 8.183% | 36 | 2 | 0.116 |
| 52 | `plain_global_weighted_mild_tail35_p05_tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 |
| 52 | `plain_global_weighted_mild_tail35_stressaux_w20` | 0.03300 | 4.796% | 6.372% | 9.624% | 29 | 12 | 0.171 |
| 52 | `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 |
| 52 | `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 |
| 52 | `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| 52 | `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |
| 71 | `plain_global_weighted_mild_tail35_stressaux_w20` | 0.01963 | 4.898% | 6.375% | 9.279% | 28 | 3 | 0.163 |
| 71 | `plain_global_weighted_mild_tail35_p05_tailstress` | 0.01896 | 4.885% | 6.320% | 8.466% | 19 | 3 | 0.134 |
| 71 | `plain_global_rel` | 0.01744 | 4.927% | 6.606% | 9.534% | 44 | 16 | 0.123 |
| 71 | `plain_global_weighted_mild_tail35_p05` | 0.01482 | 4.912% | 6.477% | 9.084% | 32 | 4 | 0.150 |
| 71 | `plain_global_instance_rel` | 0.00889 | 4.990% | 6.554% | 8.615% | 35 | 3 | 0.133 |
| 71 | `plain_global_weighted_mild` | 0.00391 | 4.974% | 6.523% | 7.992% | 45 | 0 | 0.091 |

## Aggregate Validation

| variant | rel_score mean | rel_score std | q90 error mean | daily q90 p90 mean | daily max mean | days >=7% mean | days >=8% mean | pred/actual q90 mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05_tailstress` | 0.02670 | 0.00784 | 4.833% | 6.286% | 10.556% | 28.7 | 11.0 | 0.170 |
| `plain_global_weighted_mild_tail35_stressaux_w20` | 0.02477 | 0.00720 | 4.857% | 6.393% | 9.442% | 29.0 | 7.7 | 0.166 |
| `plain_global_weighted_mild_tail35_p05` | 0.02271 | 0.00689 | 4.846% | 6.399% | 9.869% | 31.3 | 8.0 | 0.170 |
| `plain_global_weighted_mild` | 0.02021 | 0.01419 | 4.880% | 6.458% | 10.229% | 42.0 | 9.0 | 0.130 |
| `plain_global_rel` | 0.01667 | 0.00112 | 4.926% | 6.563% | 9.506% | 40.3 | 13.0 | 0.134 |
| `plain_global_instance_rel` | 0.00650 | 0.00223 | 4.997% | 6.598% | 8.447% | 36.0 | 2.0 | 0.115 |

## Read

- StressAux improves mean `rel_score` versus `tail_loss`, but the gain is not as strong as direct tailstress.
- StressAux reduces mean daily max versus `tail_loss` and direct tailstress.
- StressAux does not beat `tail_loss` on `>=8%` spike days, so it is a stability/regularization candidate, not a final gold model.

## Decision

Keep `plain_global_weighted_mild_tail35_p05` as the current best frozen-candidate baseline. Keep `stressaux_w20` as the next research candidate if the priority is reducing daily max while preserving better rel_score than base.