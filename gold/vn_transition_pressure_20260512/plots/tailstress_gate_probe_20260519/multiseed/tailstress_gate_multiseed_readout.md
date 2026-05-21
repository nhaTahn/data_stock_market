# Tailstress Gate Multiseed Readout

Scope: VN train/validation only. Rules fitted on train and evaluated on validation. Holdout/test is not used.

## Validation By Seed

| seed | candidate | rel_score | q90 error | daily q90 p90 | daily max | days >=7% | days >=8% | pred/actual q90 |
|--:|:--|--:|--:|--:|--:|--:|--:|--:|
| 43 | `simplex_rel` | 0.02818 | 4.826% | 6.378% | 10.131% | 35 | 10 | 0.158 |
| 43 | `tailstress` | 0.02652 | 4.830% | 6.265% | 10.549% | 34 | 14 | 0.179 |
| 43 | `tail_loss` | 0.02580 | 4.836% | 6.405% | 10.007% | 37 | 11 | 0.168 |
| 43 | `conservative_mean_positive_gap` | 0.02339 | 4.860% | 6.378% | 10.007% | 32 | 10 | 0.163 |
| 43 | `base` | 0.01719 | 4.918% | 6.497% | 9.361% | 39 | 14 | 0.142 |
| 43 | `tailstress_past` | 0.00692 | 4.950% | 6.480% | 8.361% | 35 | 6 | 0.105 |
| 43 | `instance` | 0.00448 | 5.010% | 6.571% | 8.183% | 36 | 2 | 0.116 |
| 52 | `conservative_mean_positive_gap` | 0.03590 | 4.767% | 6.259% | 9.957% | 21 | 11 | 0.185 |
| 52 | `tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 33 | 16 | 0.196 |
| 52 | `simplex_rel` | 0.03198 | 4.807% | 6.289% | 11.152% | 33 | 10 | 0.158 |
| 52 | `tail_loss` | 0.02751 | 4.789% | 6.314% | 10.516% | 25 | 9 | 0.192 |
| 52 | `tailstress_past` | 0.02271 | 4.846% | 6.476% | 11.469% | 24 | 8 | 0.141 |
| 52 | `base` | 0.01539 | 4.934% | 6.585% | 9.624% | 38 | 9 | 0.136 |
| 52 | `instance` | 0.00613 | 4.989% | 6.670% | 8.543% | 37 | 1 | 0.095 |
| 71 | `conservative_mean_positive_gap` | 0.01909 | 4.883% | 6.320% | 8.466% | 19 | 3 | 0.134 |
| 71 | `tailstress` | 0.01896 | 4.885% | 6.320% | 8.466% | 19 | 3 | 0.134 |
| 71 | `base` | 0.01744 | 4.927% | 6.606% | 9.534% | 44 | 16 | 0.123 |
| 71 | `tail_loss` | 0.01482 | 4.912% | 6.477% | 9.084% | 32 | 4 | 0.150 |
| 71 | `simplex_rel` | 0.01219 | 4.967% | 6.427% | 8.397% | 25 | 1 | 0.109 |
| 71 | `instance` | 0.00889 | 4.990% | 6.554% | 8.615% | 35 | 3 | 0.133 |
| 71 | `tailstress_past` | 0.00748 | 4.977% | 6.646% | 7.953% | 37 | 0 | 0.085 |

## Aggregate Validation

| candidate | rel_score mean | rel_score std | q90 error mean | daily q90 p90 mean | daily max mean | days >=7% mean | days >=8% mean | pred/actual q90 mean |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `tailstress` | 0.02670 | 0.00784 | 4.833% | 6.286% | 10.556% | 28.7 | 11.0 | 0.170 |
| `conservative_mean_positive_gap` | 0.02612 | 0.00873 | 4.837% | 6.319% | 9.477% | 24.0 | 8.0 | 0.161 |
| `simplex_rel` | 0.02411 | 0.01050 | 4.867% | 6.365% | 9.893% | 31.0 | 7.0 | 0.141 |
| `tail_loss` | 0.02271 | 0.00689 | 4.846% | 6.399% | 9.869% | 31.3 | 8.0 | 0.170 |
| `base` | 0.01667 | 0.00112 | 4.926% | 6.563% | 9.506% | 40.3 | 13.0 | 0.134 |
| `tailstress_past` | 0.01237 | 0.00896 | 4.924% | 6.534% | 9.261% | 32.0 | 4.7 | 0.110 |
| `instance` | 0.00650 | 0.00223 | 4.997% | 6.598% | 8.447% | 36.0 | 2.0 | 0.115 |

## Read

- Tailstress gate improves the best single-seed result, but it is not stable across seeds.
- `conservative_mean_positive_gap` wins seed 52 and 71, but loses to `tail_loss`/`weighted` on seed 43.
- Mean `rel_score` improves versus `tail_loss`, but mean `>=8%` spike days and max daily q90 error are worse than `tail_loss`.
- Therefore gate should remain an experimental confidence layer, not the next frozen model.

## Decision

Keep `plain_global_weighted_mild_tail35_p05` as the current best LSTM forecasting candidate. Tailstress is confirmed as a useful signal, but should be constrained more before becoming part of the model.