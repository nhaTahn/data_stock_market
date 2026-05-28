# Static-Error RiskAux Probe

Protocol: Stage-1 train return head; Stage-2 train detached risk head using static |y-p_stage1| labels. Holdout/test not used.

- seed: 52
- base_variant: `plain_global_weighted_mild_tail35_p05`
- risk_variant: `plain_global_weighted_mild_tail35_p05_static_error_riskaux_detached_w20`
- train static error label rate: 0.1098
- val static error label rate: 0.1743

## Summary

| variant                                                                 | split   |   n_obs |   n_days |   rel_score |   median_abs_error |   q90_abs_error |   daily_q90_abs_error_median |   daily_q90_abs_error_q90 |   daily_q90_abs_error_max |   actual_abs_q90 |   prediction_abs_q90 |   prediction_actual_abs_q90_ratio |   directional_accuracy |   spike_days_ge_5pct |   spike_rate_ge_5pct |   spike_days_ge_7pct |   spike_rate_ge_7pct |   spike_days_ge_8pct |   spike_rate_ge_8pct |
|:------------------------------------------------------------------------|:--------|--------:|---------:|------------:|-------------------:|----------------:|-----------------------------:|--------------------------:|--------------------------:|-----------------:|---------------------:|----------------------------------:|-----------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| plain_global_weighted_mild_tail35_p05_stage1                            | train   |  129720 |     1970 |    0.005306 |           0.009914 |        0.036774 |                     0.031459 |                  0.049273 |                  0.072906 |         0.037037 |             0.003099 |                          0.083664 |               0.449345 |                  183 |             0.092893 |                   15 |             0.007614 |                    0 |                    0 |
| plain_global_weighted_mild_tail35_p05_stage1                            | val     |   60468 |      659 |   -0.002534 |           0.012635 |        0.0503   |                     0.03704  |                  0.066048 |                  0.076947 |         0.050221 |             0.003877 |                          0.077206 |               0.474069 |                  173 |             0.262519 |                   23 |             0.034901 |                    0 |                    0 |
| plain_global_weighted_mild_tail35_p05_static_error_riskaux_detached_w20 | train   |  129720 |     1970 |    0.005306 |           0.009914 |        0.036774 |                     0.031459 |                  0.049273 |                  0.072906 |         0.037037 |             0.003099 |                          0.083664 |               0.449345 |                  183 |             0.092893 |                   15 |             0.007614 |                    0 |                    0 |
| plain_global_weighted_mild_tail35_p05_static_error_riskaux_detached_w20 | val     |   60468 |      659 |   -0.002534 |           0.012635 |        0.0503   |                     0.03704  |                  0.066048 |                  0.076947 |         0.050221 |             0.003877 |                          0.077206 |               0.474069 |                  173 |             0.262519 |                   23 |             0.034901 |                    0 |                    0 |

{
  "output_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/reports/static_error_riskaux_probe_20260528",
  "gold_dir": "/Users/lap15111/Documents/research-paper/data_stock_market/gold/vn_transition_pressure_20260512/plots/static_error_riskaux_probe_20260528"
}