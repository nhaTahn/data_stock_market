# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                                   | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:----------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop10 | val     |   60468 |      659 |     0.02682 |            0.01265 |         0.04805 |                      0.03702 |                   0.06417 |                   0.11341 |        0.0502214 |           0.00888584 |                           0.17693 |                0.50748 |                  161 |              0.24431 |                   41 |              0.06222 |                   11 |              0.01669 |
| plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop20 | val     |   60468 |      659 |     0.02125 |            0.01261 |         0.04856 |                      0.03668 |                   0.0642  |                   0.0988  |        0.0502214 |           0.0068421  |                           0.13624 |                0.4998  |                  157 |              0.23824 |                   32 |              0.04856 |                   10 |              0.01517 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.