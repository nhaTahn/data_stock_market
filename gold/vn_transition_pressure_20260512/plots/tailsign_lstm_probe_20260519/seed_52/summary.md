# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                    | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:-------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_sign_p05 | val     |   60468 |      659 |     0.02695 |            0.01263 |         0.04808 |                      0.03747 |                   0.06469 |                   0.12353 |        0.0502214 |           0.00966782 |                            0.1925 |                0.50463 |                  153 |              0.23217 |                   40 |              0.0607  |                   16 |              0.02428 |
| plain_global_weighted_mild_tail35_sign_p10 | val     |   60468 |      659 |     0.02827 |            0.0125  |         0.04825 |                      0.03699 |                   0.06342 |                   0.10793 |        0.0502214 |           0.00848237 |                            0.1689 |                0.50362 |                  159 |              0.24127 |                   33 |              0.05008 |                   12 |              0.01821 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.