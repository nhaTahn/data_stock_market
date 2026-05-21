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
| plain_global_weighted_mild_tail35_p05_tailstress      | val     |   60468 |      659 |     0.03463 |            0.01247 |         0.04783 |                      0.03711 |                   0.06274 |                   0.12653 |        0.0502214 |           0.00984138 |                           0.19596 |                0.50541 |                  162 |              0.24583 |                   33 |              0.05008 |                   16 |              0.02428 |
| plain_global_weighted_mild_tail35_p05_tailstress_past | val     |   60468 |      659 |     0.02271 |            0.0126  |         0.04846 |                      0.03703 |                   0.06476 |                   0.11469 |        0.0502214 |           0.00708501 |                           0.14108 |                0.49752 |                  159 |              0.24127 |                   24 |              0.03642 |                    8 |              0.01214 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.