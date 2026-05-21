# Tail-Loss LSTM Multiseed Readout

Scope: VN train/validation only. Holdout/test is not used.

Setup: LSTM `[64, 32]`, window `15`, 29 portable features, epochs `18`, patience `5`, seeds `43, 52, 71`.

## Validation By Seed

| seed | variant | rel_score | q90 error | daily q90 p90 | daily q90 max | days >=7% | days >=8% | pred/actual abs q90 |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 43 | `plain_global_weighted_mild` | 0.02968 | 4.831% | 6.460% | 10.508% | 40 | 10 | 0.150 |
| 43 | `plain_global_weighted_mild_tail35_p05` | 0.02580 | 4.836% | 6.405% | 10.007% | 37 | 11 | 0.168 |
| 43 | `plain_global_rel` | 0.01719 | 4.918% | 6.497% | 9.361% | 39 | 14 | 0.142 |
| 43 | `plain_global_instance_rel` | 0.00448 | 5.010% | 6.571% | 8.183% | 36 | 2 | 0.116 |
| 52 | `plain_global_weighted_mild_tail35_p05` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 |
| 52 | `plain_global_weighted_mild` | 0.02705 | 4.835% | 6.390% | 12.188% | 41 | 17 | 0.149 |
| 52 | `plain_global_rel` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| 52 | `plain_global_instance_rel` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |
| 71 | `plain_global_rel` | 0.01744 | 4.927% | 6.606% | 9.534% | 44 | 16 | 0.123 |
| 71 | `plain_global_weighted_mild_tail35_p05` | 0.01482 | 4.912% | 6.477% | 9.084% | 32 | 4 | 0.150 |
| 71 | `plain_global_instance_rel` | 0.00889 | 4.990% | 6.554% | 8.615% | 35 | 3 | 0.133 |
| 71 | `plain_global_weighted_mild` | 0.00391 | 4.974% | 6.523% | 7.992% | 45 | 0 | 0.091 |

## Aggregate Validation

| variant | rel_score mean | rel_score std | q90 error mean | daily q90 p90 mean | daily q90 max mean | days >=7% mean | days >=8% mean | pred/actual abs q90 mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_p05` | 0.02271 | 0.00689 | 4.846% | 6.399% | 9.869% | 31.3 | 8.0 | 0.170 |
| `plain_global_weighted_mild` | 0.02021 | 0.01419 | 4.880% | 6.458% | 10.229% | 42.0 | 9.0 | 0.130 |
| `plain_global_rel` | 0.01667 | 0.00112 | 4.926% | 6.563% | 9.506% | 40.3 | 13.0 | 0.134 |
| `plain_global_instance_rel` | 0.00650 | 0.00223 | 4.997% | 6.598% | 8.447% | 36.0 | 2.0 | 0.115 |

## Read

- `plain_global_weighted_mild_tail35_p05` is the best current forecasting candidate by mean `rel_score`.
- It improves pooled q90 error and reduces average `>=8%` spike days versus both baseline and weighted mild.
- It does not solve the strict daily max target; mean max daily q90 error remains above `10%`.
- This clears the next hypothesis partially: tail-aware loss helps, but daily max spikes likely require either better tail/regime features or a constrained confidence layer.

## Decision

Do not freeze yet. Promote `plain_global_weighted_mild_tail35_p05` to the next candidate for longer validation/walk-forward, while keeping `plain_global_rel` as baseline.