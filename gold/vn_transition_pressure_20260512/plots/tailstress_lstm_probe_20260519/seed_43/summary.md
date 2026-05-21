# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `43`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                               | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_p05_tailstress      | val     |   60468 |      659 |     0.02652 |            0.01254 |          0.0483 |                      0.03707 |                   0.06265 |                   0.10549 |        0.0502214 |           0.00898801 |                           0.17897 |                0.5104  |                  161 |              0.24431 |                   34 |              0.05159 |                   14 |              0.02124 |
| plain_global_weighted_mild_tail35_p05_tailstress_past | val     |   60468 |      659 |     0.00692 |            0.01268 |          0.0495 |                      0.03692 |                   0.0648  |                   0.08361 |        0.0502214 |           0.00526943 |                           0.10492 |                0.49117 |                  160 |              0.24279 |                   35 |              0.05311 |                    6 |              0.0091  |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.