# Two-Stream Probe Readout

Step 3 of input/target processing improvement plan.
Scope: VN train/validation only. Holdout/test is not used.

## Per-Seed Validation

|   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_p90 |   daily_max |   pred_actual_q90_ratio |   directional_accuracy |   market_pred_r2 |   market_pred_corr |   spike_days_ge_5pct |   spike_days_ge_7pct |   spike_days_ge_8pct | variant          |   seed | split   |
|------------:|-------------------:|----------------:|----------------:|------------:|------------------------:|-----------------------:|-----------------:|-------------------:|---------------------:|---------------------:|---------------------:|:-----------------|-------:|:--------|
|  0.0302354  |          0.0125295 |       0.0480435 |       0.0629219 |   0.116226  |                0.182555 |               0.50967  |       -0.0051695 |        nan         |                  155 |                   37 |                   15 | raw_baseline     |     43 | val     |
|  0.00290228 |          0.0126154 |       0.0499319 |       0.0654867 |   0.0821438 |                0.131991 |               0.484953 |      -35.4902    |         -0.0148538 |                  173 |                   35 |                    7 | two_stream_joint |     43 | val     |
|  0.0325011  |          0.0125124 |       0.0479069 |       0.0619075 |   0.111122  |                0.201271 |               0.511209 |       -0.0051695 |        nan         |                  163 |                   30 |                    8 | raw_baseline     |     52 | val     |
|  0.0209823  |          0.0126726 |       0.0484547 |       0.0647432 |   0.113016  |                0.213738 |               0.501381 |      -74.5427    |          0.0446314 |                  161 |                   39 |                   18 | two_stream_joint |     52 | val     |
|  0.0223735  |          0.0124528 |       0.0487894 |       0.0646811 |   0.106379  |                0.16841  |               0.507337 |       -0.0051695 |        nan         |                  159 |                   41 |                   12 | raw_baseline     |     71 | val     |
|  0.0110756  |          0.0127714 |       0.0490038 |       0.06479   |   0.109238  |                0.220497 |               0.495938 |      -91.3171    |          0.03798   |                  148 |                   48 |                   21 | two_stream_joint |     71 | val     |

## Aggregate

| variant          |   n_seeds |   rel_score_mean |   rel_score_std |   daily_max_mean |   daily_max_std |   spike_days_ge_8pct_mean |   spike_days_ge_8pct_std |   directional_accuracy_mean |   directional_accuracy_std |   market_pred_r2_mean |   market_pred_r2_std |   market_pred_corr_mean |   market_pred_corr_std |   pred_actual_q90_ratio_mean |   pred_actual_q90_ratio_std |
|:-----------------|----------:|-----------------:|----------------:|-----------------:|----------------:|--------------------------:|-------------------------:|----------------------------:|---------------------------:|----------------------:|---------------------:|------------------------:|-----------------------:|-----------------------------:|----------------------------:|
| raw_baseline     |         3 |        0.02837   |      0.00531528 |         0.111242 |      0.00492464 |                   11.6667 |                  3.51188 |                    0.509405 |                 0.00194917 |            -0.0051695 |               0      |             nan         |            nan         |                     0.184079 |                   0.0164833 |
| two_stream_joint |         3 |        0.0116534 |      0.00905384 |         0.101466 |      0.0168397  |                   15.3333 |                  7.37111 |                    0.494091 |                 0.00836844 |           -67.1166    |              28.6447 |               0.0225859 |              0.0325938 |                     0.188742 |                   0.0492637 |

## Decision

Two-stream passes if rel_score_mean > baseline AND spike_days_ge_8pct_mean < baseline.
market_pred_r2 and market_pred_corr show how well the market head learned.