# Current Best Error-Control Visuals

Policy: `risk_hgb/coverage_q40`.
Seeds pooled for visualization: `43, 52, 71`.

## Histogram Summary

| sample | rows | rel_score | base_score | abs_score | q90(|E|) | dir_acc | q20(E) | q80(E) | mean(E) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `full coverage` | 181404 | +0.02463 | 0.03769 | 0.03676 | 4.86% | 50.70% | -0.01830 | +0.01510 | -0.00095 |
| `accepted` | 30921 | +0.00339 | 0.01914 | 0.01907 | 2.47% | 47.28% | -0.01022 | +0.00800 | -0.00163 |

## Yearly q90(|E|)

| label | year | median | p90 | max | days >=3.5% | days >=8% |
|:--|--:|--:|--:|--:|--:|--:|
| `accepted` | 2020 | 2.04% | 3.30% | 5.90% | 20 | 0 |
| `accepted` | 2021 | 1.94% | 3.40% | 6.27% | 18 | 0 |
| `accepted` | 2022 | 2.22% | 3.50% | 5.40% | 13 | 0 |
| `full coverage` | 2020 | 3.18% | 5.35% | 7.37% | 228 | 0 |
| `full coverage` | 2021 | 3.67% | 6.00% | 8.87% | 422 | 3 |
| `full coverage` | 2022 | 4.30% | 7.08% | 9.62% | 440 | 20 |

## Plot Files

- `error_hist_full_coverage_seed_pooled.png`
- `error_hist_accepted_seed_pooled.png`
- `rel_score_proxy_hist_accepted_seed_pooled.png`
- `yearly_q90_abs_error_best_policy.png`