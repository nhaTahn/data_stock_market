# rel_score & Error Distribution — `plain_global_weighted_mild_tail35_stressaux_w20`

Scope: VN train + validation pooled over seeds 43, 52, 71. Holdout/test not used.

## Per-Seed Metrics

| split   |   seed |   n_obs |   rel_score | median_abs_error   | q90_abs_error   | daily_q90_p90   | daily_q90_max   |   directional_accuracy |   pred_actual_q90_ratio |   spike_days_ge_5pct |   spike_days_ge_7pct |   spike_days_ge_8pct |
|:--------|-------:|--------:|------------:|:-------------------|:----------------|:----------------|:----------------|-----------------------:|------------------------:|---------------------:|---------------------:|---------------------:|
| train   |     43 |  129720 |     0.02343 | 0.98%              | 3.60%           | 4.81%           | 8.84%           |                 0.476  |                  0.171  |                  163 |                    9 |                    1 |
| train   |     52 |  129720 |     0.02565 | 0.98%              | 3.59%           | 4.79%           | 8.93%           |                 0.477  |                  0.1789 |                  158 |                   11 |                    3 |
| train   |     71 |  129720 |     0.02209 | 0.98%              | 3.61%           | 4.88%           | 8.89%           |                 0.4744 |                  0.1644 |                  170 |                   12 |                    2 |
| val     |     43 |   60468 |     0.02168 | 1.25%              | 4.88%           | 6.43%           | 9.42%           |                 0.5067 |                  0.1642 |                  167 |                   30 |                    8 |
| val     |     52 |   60468 |     0.033   | 1.25%              | 4.80%           | 6.37%           | 9.62%           |                 0.5105 |                  0.171  |                  166 |                   29 |                   12 |
| val     |     71 |   60468 |     0.01963 | 1.25%              | 4.90%           | 6.37%           | 9.28%           |                 0.5037 |                  0.1627 |                  167 |                   28 |                    3 |

## Aggregate (mean ± std across seeds)

- **Train rel_score**: +0.0237 ± 0.0018
- **Validation rel_score**: +0.0248 ± 0.0072
- **Validation q90(|E|)**: 4.86% ± 0.05%
- **Validation daily_q90 max**: 9.44% ± 0.17%
- **Validation directional accuracy**: 50.70% ± 0.34%
- **Validation spike days ≥8%**: 7.7 ± 4.5
- **Validation pred/actual q90 ratio**: 0.166 ± 0.004

## Plots

- `error_histogram_pooled.png` — absolute error distribution, train vs val, pooled across seeds.
- `error_histogram_per_seed.png` — per-seed validation absolute-error histograms.
- `signed_error_histogram.png` — signed error showing direction bias and skew.
- `relscore_per_seed.png` — bar chart of rel_score per seed (train vs val).