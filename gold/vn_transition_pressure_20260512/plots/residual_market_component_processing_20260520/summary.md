# Residual Market Component Processing

Scope: reuse residual-target LSTM predictions, tune market reconstruction component on late-train, evaluate on validation.

| policy | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `raw_baseline` | 3 | 0.02837 | 4.82% | 3.72% | 6.32% | 11.12% | 159.0 | 36.0 | 11.7 |
| `selected:ewm5_scale_0.25` | 1 | 0.02092 | 4.86% | 3.71% | 6.37% | 10.48% | 169.0 | 29.0 | 8.0 |
| `existing_residual_lagged_ar1` | 3 | 0.02436 | 4.85% | 3.70% | 6.39% | 10.68% | 160.3 | 33.3 | 10.0 |
| `selected:ewm3_scale_0.1` | 1 | 0.02005 | 4.90% | 3.74% | 6.42% | 9.37% | 160.0 | 35.0 | 10.0 |
| `selected:ewm3_scale_0.25` | 1 | 0.02428 | 4.81% | 3.70% | 6.45% | 11.05% | 157.0 | 30.0 | 12.0 |

## Read

- If selected lagged-market reconstruction does not beat `raw_baseline`, residual target is only useful with an oracle or a much stronger market nowcast.
- If `scale_0` wins, the best practical use is alpha/residual-only target, not adding noisy market prediction back.
- This is target processing, not a new architecture: it tests whether the target decomposition is operationally useful without future market information.