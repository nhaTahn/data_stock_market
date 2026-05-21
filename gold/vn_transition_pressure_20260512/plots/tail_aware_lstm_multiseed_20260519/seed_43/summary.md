# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `43`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                    | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:---------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_instance_rel  | val     |   60468 |      659 |     0.00448 |            0.01247 |         0.0501  |                      0.03669 |                   0.06571 |                   0.08183 |        0.0502214 |           0.00585045 |                           0.11649 |                0.49558 |                  171 |              0.25948 |                   36 |              0.05463 |                    2 |              0.00303 |
| plain_global_rel           | val     |   60468 |      659 |     0.01719 |            0.01245 |         0.04918 |                      0.03664 |                   0.06497 |                   0.09361 |        0.0502214 |           0.00713436 |                           0.14206 |                0.5047  |                  162 |              0.24583 |                   39 |              0.05918 |                   14 |              0.02124 |
| plain_global_weighted_mild | val     |   60468 |      659 |     0.02968 |            0.01242 |         0.04831 |                      0.03698 |                   0.0646  |                   0.10508 |        0.0502214 |           0.00750968 |                           0.14953 |                0.50921 |                  158 |              0.23976 |                   40 |              0.0607  |                   10 |              0.01517 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.