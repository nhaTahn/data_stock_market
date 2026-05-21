# FutureStressAux Readout

Scope: VN train/validation only. Holdout/test is not used.

Hypothesis: keep LSTM as the return forecaster, but add an auxiliary head that predicts next-day market tail stress from the same input window.

## Comparable Seed 52 Validation

| candidate | rel_score | q90 error | daily q90 p90 | daily max | days >=5% | days >=7% | days >=8% | pred/actual q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `tailstress` | 0.03463 | 4.783% | 6.274% | 12.653% | 162 | 33 | 16 | 0.196 | 0.505 |
| `stressaux_w20` | 0.03300 | 4.796% | 6.372% | 9.624% | 166 | 29 | 12 | 0.171 | 0.510 |
| `futurestress_w10` | 0.03281 | 4.766% | 6.352% | 11.812% | 159 | 29 | 9 | 0.167 | 0.504 |
| `tail_loss` | 0.02751 | 4.789% | 6.314% | 10.516% | 161 | 25 | 9 | 0.192 | 0.504 |
| `futurestress_w20` | 0.01299 | 4.888% | 6.460% | 9.675% | 163 | 31 | 5 | 0.137 | 0.495 |

## Quick Stability Screens

These use fewer epochs and larger batch size, so they are not comparable with the full runs. They are only used to decide whether a full multiseed run is worth the cost.

| seed | candidate | rel_score | q90 error | daily q90 p90 | daily max | days >=7% | days >=8% | pred/actual q90 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| `43_screen` | `futurestress_w20_screen` | 0.01173 | 4.927% | 6.520% | 9.229% | 43 | 11 | 0.125 |
| `43_screen` | `futurestress_w10_screen` | -0.00434 | 4.981% | 6.592% | 8.152% | 30 | 1 | 0.126 |
| `71_screen` | `futurestress_w10_screen` | 0.00224 | 4.990% | 6.534% | 8.750% | 43 | 9 | 0.114 |
| `71_screen` | `futurestress_w20_screen` | -0.00767 | 5.059% | 6.570% | 7.773% | 39 | 0 | 0.122 |

## Decision

- `futurestress_w10` is promising on seed 52 rel_score, but it does not reduce daily max and fails the quick stability screens.
- `futurestress_w20` can reduce extreme >=8% days on seed 52, but it lowers rel_score sharply and behaves like an over-conservative filter.
- Do not promote FutureStressAux to gold yet. Keep current best candidate as `plain_global_weighted_mild_tail35_p05`; keep `stressaux_w20` as the best stability-side research challenger.

## Next Hypothesis

The remaining spikes are not solved by output calibration or a simple auxiliary stress label. The next model-side work should target input processing: point-in-time cross-sectional normalization, market-relative residual target, and leader/market pressure features rather than stronger auxiliary penalties.