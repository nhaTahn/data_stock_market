# Two-Head Raw + Alpha Probe

Holdout/test not used.

|   raw_rel_score |   raw_directional_accuracy |   raw_pred_actual_q90_ratio |   alpha_rel_score |   alpha_directional_accuracy |   alpha_pred_actual_q90_ratio |   alpha_q90_abs_error |   raw_daily_q90_max |   alpha_daily_q90_max |   seed | split   |   alpha_weight |
|----------------:|---------------------------:|----------------------------:|------------------:|-----------------------------:|------------------------------:|----------------------:|--------------------:|----------------------:|-------:|:--------|---------------:|
|       -0.002901 |                   0.515407 |                    0.133897 |         -0.001798 |                     0.497135 |                      0.0475   |              0.024175 |            0.129654 |              0.099715 |     43 | val     |           0.05 |
|       -0.000113 |                   0.515221 |                    0.143961 |         -0.002723 |                     0.500881 |                      0.03636  |              0.024223 |            0.12534  |              0.100401 |     52 | val     |           0.05 |
|       -0.000946 |                   0.497135 |                    0.067946 |         -0.005755 |                     0.499017 |                      0.063056 |              0.024293 |            0.127434 |              0.100225 |     62 | val     |           0.05 |

|      |   raw_rel_score |   alpha_rel_score |   raw_directional_accuracy |   alpha_directional_accuracy |   raw_pred_actual_q90_ratio |   alpha_pred_actual_q90_ratio |
|:-----|----------------:|------------------:|---------------------------:|-----------------------------:|----------------------------:|------------------------------:|
| mean |       -0.00132  |         -0.003425 |                   0.509255 |                     0.499011 |                    0.115268 |                      0.048972 |
| std  |        0.001431 |          0.00207  |                   0.010496 |                     0.001873 |                    0.04129  |                      0.013409 |

{
  "output_dir": "data/processed/assets/data_info_us/history/training_runs/reports/two_head_alpha_w005_smoke_20260526_us",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/two_head_alpha_w005_smoke_20260526_us"
}