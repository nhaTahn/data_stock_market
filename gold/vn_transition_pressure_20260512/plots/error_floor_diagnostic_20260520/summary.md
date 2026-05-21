# Error Floor Diagnostic

Metric: daily `q90(|actual - prediction|)` on validation. Threshold target: 3.5%.

## Aggregate

| candidate | rel_score | daily median | daily p90 | daily max | days >=3.5% | days >=5% | days >=7% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `residual_oracle` | 0.21578 | 3.24% | 4.67% | 7.91% | 266.0 | 44.0 | 1.0 | 0.3 |
| `stressaux_w20` | 0.02477 | 3.69% | 6.39% | 9.44% | 363.3 | 166.7 | 29.0 | 7.7 |
| `residual_lagged_ar1` | 0.02436 | 3.70% | 6.39% | 10.68% | 365.3 | 160.3 | 33.3 | 10.0 |
| `tail_loss` | 0.02271 | 3.71% | 6.40% | 9.87% | 363.3 | 164.0 | 31.3 | 8.0 |
| `zero_prediction_floor` | 0.00000 | 3.73% | 6.84% | 6.99% | 359.0 | 171.0 | 0.0 | 0.0 |

## Interpretation

- If `zero_prediction_floor` already exceeds 3.5% on many days, full-coverage point prediction cannot guarantee daily q90 error below 3.5% without predicting the market/tail component.
- `residual_oracle` uses future market information and is an upper-bound, not a deployable model.
- A deployable model should be judged by both rel_score and spike days; reducing spikes by shrinking to zero is not enough if rel_score collapses.