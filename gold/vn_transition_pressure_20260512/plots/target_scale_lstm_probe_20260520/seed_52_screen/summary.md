# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                                                       | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:--------------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_stressaux_w20               | val     |   60468 |      659 |     0.033   |            0.01247 |         0.04796 |                      0.03698 |                   0.06372 |                   0.09624 |        0.0502214 |           0.00858769 |                           0.171   |                0.51048 |                  166 |              0.2519  |                   29 |              0.04401 |                   12 |              0.01821 |
| plain_global_weighted_mild_tail35_stressaux_w20_floor50       | val     |   60468 |      659 |     0.02136 |            0.01242 |         0.04892 |                      0.03714 |                   0.06473 |                   0.10793 |        0.0502214 |           0.00657624 |                           0.13094 |                0.49916 |                  171 |              0.25948 |                   29 |              0.04401 |                   10 |              0.01517 |
| plain_global_weighted_mild_tail35_stressaux_w20_mktblendscale | val     |   60468 |      659 |     0.0275  |            0.01256 |         0.04818 |                      0.03704 |                   0.06503 |                   0.10366 |        0.0502214 |           0.00866036 |                           0.17244 |                0.50893 |                  157 |              0.23824 |                   39 |              0.05918 |                   15 |              0.02276 |
| plain_global_weighted_mild_tail35_stressaux_w20_mktmaxscale   | val     |   60468 |      659 |     0.02178 |            0.01246 |         0.04882 |                      0.03681 |                   0.06643 |                   0.10285 |        0.0502214 |           0.00613359 |                           0.12213 |                0.49939 |                  163 |              0.24734 |                   37 |              0.05615 |                   12 |              0.01821 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.