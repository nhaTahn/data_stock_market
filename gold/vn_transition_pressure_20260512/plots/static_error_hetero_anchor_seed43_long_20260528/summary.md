# Static-Error Hetero Anchor Probe

Protocol: Stage-1 hetero anchor; Stage-2 attached static-error risk head. Holdout/test not used.

## Aggregate Validation

| variant                     |   rel_score_mean |   rel_score_std |   q90_abs_error_mean |   directional_accuracy_mean |   pred_actual_q90_ratio_mean |   n_seeds |
|:----------------------------|-----------------:|----------------:|---------------------:|----------------------------:|-----------------------------:|----------:|
| stage2_static_error_riskaux |         0.028764 |             nan |             0.047887 |                    0.507652 |                     0.229023 |         1 |
| stage1_hetero               |         0.019603 |             nan |             0.048906 |                    0.505418 |                     0.152464 |         1 |

## Per Seed

| variant                     |   seed | split   |   rel_score |   rel_score_vol_clipped |   median_abs_error |   q90_abs_error |   daily_q90_p90 |   daily_q90_max |   daily_q90_clipped_p90 |   daily_q90_clipped_max |   directional_accuracy |   spike_days_ge_5pct |   spike_days_ge_8pct |   spike_days_clipped_ge_8pct |   mean_sigma |   median_sigma |   pred_actual_q90_ratio |   mean_risk |   static_error_label_rate_train |   static_error_label_rate_val |
|:----------------------------|-------:|:--------|------------:|------------------------:|-------------------:|----------------:|----------------:|----------------:|------------------------:|------------------------:|-----------------------:|---------------------:|---------------------:|-----------------------------:|-------------:|---------------:|------------------------:|------------:|--------------------------------:|------------------------------:|
| stage1_hetero               |     43 | train   |    0.020332 |                0.020332 |           0.00959  |        0.035016 |        0.047764 |        0.082517 |                0.047764 |                0.082517 |               0.477797 |                  148 |                    2 |                            2 |     0.020994 |       0.019648 |                0.150631 |  nan        |                      nan        |                    nan        |
| stage1_hetero               |     43 | val     |    0.019603 |                0.019603 |           0.012499 |        0.048906 |        0.064457 |        0.116924 |                0.064457 |                0.116924 |               0.505418 |                  163 |                   13 |                           13 |     0.0244   |       0.02291  |                0.152464 |  nan        |                      nan        |                    nan        |
| stage2_static_error_riskaux |     43 | train   |    0.031966 |                0.031966 |           0.009625 |        0.034304 |        0.046664 |        0.088715 |                0.046664 |                0.088715 |               0.487271 |                  120 |                    1 |                            1 |     0.021131 |       0.019577 |                0.20915  |    0.435817 |                        0.100124 |                      0.168831 |
| stage2_static_error_riskaux |     43 | val     |    0.028764 |                0.028764 |           0.012663 |        0.047887 |        0.063207 |        0.117351 |                0.063207 |                0.117351 |               0.507652 |                  160 |                   10 |                           10 |     0.025305 |       0.023978 |                0.229023 |    0.533618 |                        0.100124 |                      0.168831 |

{
  "output_dir": "data/processed/assets/data_info_vn/history/training_runs/reports/static_error_hetero_anchor_seed43_long_20260528",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/static_error_hetero_anchor_seed43_long_20260528"
}