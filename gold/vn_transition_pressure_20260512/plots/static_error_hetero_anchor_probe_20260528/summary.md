# Static-Error Hetero Anchor Probe

Protocol: Stage-1 hetero anchor; Stage-2 attached static-error risk head. Holdout/test not used.

## Aggregate Validation

| variant                     |   rel_score_mean |   rel_score_std |   q90_abs_error_mean |   directional_accuracy_mean |   pred_actual_q90_ratio_mean |   n_seeds |
|:----------------------------|-----------------:|----------------:|---------------------:|----------------------------:|-----------------------------:|----------:|
| stage2_static_error_riskaux |         0.005705 |             nan |             0.049658 |                    0.492547 |                     0.115265 |         1 |
| stage1_hetero               |         0.00515  |             nan |             0.050004 |                    0.492911 |                     0.097426 |         1 |

## Per Seed

| variant                     |   seed | split   |   rel_score |   rel_score_vol_clipped |   median_abs_error |   q90_abs_error |   daily_q90_p90 |   daily_q90_max |   daily_q90_clipped_p90 |   daily_q90_clipped_max |   directional_accuracy |   spike_days_ge_5pct |   spike_days_ge_8pct |   spike_days_clipped_ge_8pct |   mean_sigma |   median_sigma |   pred_actual_q90_ratio |   mean_risk |   static_error_label_rate_train |   static_error_label_rate_val |
|:----------------------------|-------:|:--------|------------:|------------------------:|-------------------:|----------------:|----------------:|----------------:|------------------------:|------------------------:|-----------------------:|---------------------:|---------------------:|-----------------------------:|-------------:|---------------:|------------------------:|------------:|--------------------------------:|------------------------------:|
| stage1_hetero               |     43 | train   |    0.009879 |                0.009879 |           0.009706 |        0.035363 |        0.048719 |        0.075485 |                0.048719 |                0.075485 |               0.468566 |                  162 |                    0 |                            0 |     0.021642 |       0.019898 |                0.110585 |  nan        |                      nan        |                    nan        |
| stage1_hetero               |     43 | val     |    0.00515  |                0.00515  |           0.012495 |        0.050004 |        0.065901 |        0.085413 |                0.065901 |                0.085413 |               0.492911 |                  169 |                    5 |                            5 |     0.02542  |       0.02374  |                0.097426 |  nan        |                      nan        |                    nan        |
| stage2_static_error_riskaux |     43 | train   |    0.007742 |                0.007742 |           0.009846 |        0.035201 |        0.048362 |        0.077696 |                0.048362 |                0.077696 |               0.46251  |                  160 |                    0 |                            0 |     0.021594 |       0.019747 |                0.137136 |    0.427538 |                        0.101942 |                      0.172868 |
| stage2_static_error_riskaux |     43 | val     |    0.005705 |                0.005705 |           0.012647 |        0.049658 |        0.064438 |        0.093757 |                0.064438 |                0.093757 |               0.492547 |                  169 |                    5 |                            5 |     0.025658 |       0.024201 |                0.115265 |    0.522582 |                        0.101942 |                      0.172868 |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/static_error_hetero_anchor_probe_20260528",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/static_error_hetero_anchor_probe_20260528"
}