# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                    | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:---------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_instance_rel  | val     |   60468 |      659 |     0.00613 |            0.01251 |         0.04989 |                      0.03692 |                   0.0667  |                   0.08543 |        0.0502214 |           0.00476686 |                           0.09492 |                0.49254 |                  169 |              0.25645 |                   37 |              0.05615 |                    1 |              0.00152 |
| plain_global_rel           | val     |   60468 |      659 |     0.01539 |            0.01244 |         0.04934 |                      0.03724 |                   0.06585 |                   0.09624 |        0.0502214 |           0.00683322 |                           0.13606 |                0.49631 |                  162 |              0.24583 |                   38 |              0.05766 |                    9 |              0.01366 |
| plain_global_weighted_mild | val     |   60468 |      659 |     0.02705 |            0.0125  |         0.04835 |                      0.03685 |                   0.0639  |                   0.12188 |        0.0502214 |           0.0074696  |                           0.14873 |                0.50405 |                  160 |              0.24279 |                   41 |              0.06222 |                   17 |              0.0258  |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.