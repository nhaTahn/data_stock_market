# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `71`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                               | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_p05_tailstress      | val     |   60468 |      659 |     0.01896 |            0.01255 |         0.04885 |                      0.03734 |                   0.0632  |                   0.08466 |        0.0502214 |           0.00671554 |                           0.13372 |                0.49079 |                  170 |              0.25797 |                   19 |              0.02883 |                    3 |              0.00455 |
| plain_global_weighted_mild_tail35_p05_tailstress_past | val     |   60468 |      659 |     0.00748 |            0.01252 |         0.04977 |                      0.03679 |                   0.06646 |                   0.07953 |        0.0502214 |           0.00427067 |                           0.08504 |                0.49168 |                  167 |              0.25341 |                   37 |              0.05615 |                    0 |              0       |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.