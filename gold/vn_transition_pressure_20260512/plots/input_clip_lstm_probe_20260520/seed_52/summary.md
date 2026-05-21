# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                               | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_stressaux_w20       | val     |   60468 |      659 |    -0.00238 |            0.0126  |         0.05036 |                      0.03747 |                   0.06552 |                   0.07549 |        0.0502214 |           0.00510741 |                           0.1017  |                0.48116 |                  176 |              0.26707 |                   26 |              0.03945 |                    0 |              0       |
| plain_global_weighted_mild_tail35_stressaux_w20_clip3 | val     |   60468 |      659 |    -0.00155 |            0.01291 |         0.04968 |                      0.03641 |                   0.06759 |                   0.08463 |        0.0502214 |           0.00681742 |                           0.13575 |                0.48543 |                  168 |              0.25493 |                   49 |              0.07436 |                    6 |              0.0091  |
| plain_global_weighted_mild_tail35_stressaux_w20_clip5 | val     |   60468 |      659 |     0.00889 |            0.01263 |         0.04946 |                      0.03663 |                   0.06531 |                   0.08403 |        0.0502214 |           0.00590054 |                           0.11749 |                0.49593 |                  160 |              0.24279 |                   46 |              0.0698  |                    7 |              0.01062 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.