# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                               | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:--------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_p05 | val     |   60468 |      659 |     0.02751 |            0.01271 |         0.04789 |                      0.03707 |                   0.06314 |                   0.10516 |        0.0502214 |           0.00963798 |                           0.19191 |                0.5044  |                  161 |              0.24431 |                   25 |              0.03794 |                    9 |              0.01366 |
| plain_global_weighted_mild_tail50_p10 | val     |   60468 |      659 |     0.02628 |            0.01257 |         0.04826 |                      0.03709 |                   0.06386 |                   0.10814 |        0.0502214 |           0.00936009 |                           0.18638 |                0.50744 |                  153 |              0.23217 |                   35 |              0.05311 |                   14 |              0.02124 |
| plain_global_weighted_mild_tail50_p20 | val     |   60468 |      659 |     0.02195 |            0.01256 |         0.04861 |                      0.03708 |                   0.06381 |                   0.10635 |        0.0502214 |           0.00720139 |                           0.14339 |                0.4995  |                  158 |              0.23976 |                   37 |              0.05615 |                    9 |              0.01366 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.