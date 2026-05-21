# Tail-Aware LSTM Probe

Scope: VN train/validation only. Holdout/test is not used.

Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.

- seed: `52`
- window_size: `15`
- lstm_units: `64,32`
- train_end_date: `2020-03-31`
- val_end_date: `2022-11-15`

## Validation Results

| variant                             | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_instance_rel           | val     |   60468 |      659 |     0.00613 |            0.01251 |         0.04989 |                      0.03692 |                   0.0667  |                   0.08543 |        0.0502214 |           0.00476686 |                           0.09492 |                0.49254 |                  169 |              0.25645 |                   37 |              0.05615 |                    1 |              0.00152 |
| plain_global_instance_weighted_mild | val     |   60468 |      659 |     0.00864 |            0.01258 |         0.04957 |                      0.03672 |                   0.06593 |                   0.08825 |        0.0502214 |           0.00639821 |                           0.1274  |                0.49309 |                  168 |              0.25493 |                   40 |              0.0607  |                    3 |              0.00455 |
| plain_global_rel                    | val     |   60468 |      659 |     0.01016 |            0.01245 |         0.04971 |                      0.03672 |                   0.06568 |                   0.08481 |        0.0502214 |           0.00576446 |                           0.11478 |                0.48985 |                  170 |              0.25797 |                   45 |              0.06829 |                   12 |              0.01821 |
| plain_global_weighted               | val     |   60468 |      659 |    -0.00848 |            0.01428 |         0.04745 |                      0.03794 |                   0.06277 |                   0.12443 |        0.0502214 |           0.0172432  |                           0.34334 |                0.49949 |                  159 |              0.24127 |                   38 |              0.05766 |                   18 |              0.02731 |
| plain_global_weighted_mild          | val     |   60468 |      659 |     0.01099 |            0.01261 |         0.04933 |                      0.03659 |                   0.06585 |                   0.09814 |        0.0502214 |           0.00563571 |                           0.11222 |                0.48705 |                  161 |              0.24431 |                   38 |              0.05766 |                    9 |              0.01366 |
| plain_multimarket_rel               | val     |   60799 |      659 |     0.00172 |            0.01241 |         0.05065 |                      0.03709 |                   0.06625 |                   0.07467 |        0.0504838 |           0.00409423 |                           0.0811  |                0.48302 |                  181 |              0.27466 |                   36 |              0.05463 |                    0 |              0       |
| plain_multimarket_weighted_mild     | val     |   60799 |      659 |    -0.00022 |            0.01251 |         0.05059 |                      0.03753 |                   0.06553 |                   0.08059 |        0.0504838 |           0.00608719 |                           0.12058 |                0.48146 |                  177 |              0.26859 |                   24 |              0.03642 |                    1 |              0.00152 |
| tailaware_multimarket_weighted      | val     |   60799 |      659 |    -0.05075 |            0.01485 |         0.04974 |                      0.03966 |                   0.06524 |                   0.10734 |        0.0504838 |           0.0200214  |                           0.39659 |                0.48343 |                  166 |              0.2519  |                   53 |              0.08042 |                   19 |              0.02883 |

## Read

- `plain_global_rel` mirrors the current broad portable setup most closely.
- `plain_global_instance_rel` tests per-window instance normalization.
- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.
- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.

A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.