# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `71`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                    | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:---------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_instance_rel  | val     |   60468 |      659 |     0.00889 |            0.0124  |         0.0499  |                      0.03743 |                   0.06554 |                   0.08615 |        0.0502214 |           0.00669108 |                           0.13323 |                0.49163 |                  176 |              0.26707 |                   35 |              0.05311 |                    3 |              0.00455 |
| plain_global_rel           | val     |   60468 |      659 |     0.01744 |            0.0124  |         0.04927 |                      0.03661 |                   0.06606 |                   0.09534 |        0.0502214 |           0.00618954 |                           0.12324 |                0.49357 |                  166 |              0.2519  |                   44 |              0.06677 |                   16 |              0.02428 |
| plain_global_weighted_mild | val     |   60468 |      659 |     0.00391 |            0.01267 |         0.04974 |                      0.03681 |                   0.06523 |                   0.07992 |        0.0502214 |           0.00456073 |                           0.09081 |                0.48419 |                  158 |              0.23976 |                   45 |              0.06829 |                    0 |              0       |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.