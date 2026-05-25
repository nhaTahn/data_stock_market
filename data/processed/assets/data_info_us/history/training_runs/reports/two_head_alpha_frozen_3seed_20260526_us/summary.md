# Two-Head Raw + Alpha Probe

Holdout/test not used.

|   raw_rel_score |   raw_directional_accuracy |   raw_pred_actual_q90_ratio |   alpha_rel_score |   alpha_directional_accuracy |   alpha_pred_actual_q90_ratio |   alpha_q90_abs_error |   raw_daily_q90_max |   alpha_daily_q90_max |   seed | split   |   alpha_weight | train_mode      |
|----------------:|---------------------------:|----------------------------:|------------------:|-----------------------------:|------------------------------:|----------------------:|--------------------:|----------------------:|-------:|:--------|---------------:|:----------------|
|        0.003019 |                   0.516814 |                    0.086712 |         -0.002592 |                     0.497034 |                      0.033777 |              0.0242   |            0.126985 |              0.100142 |     43 | val     |              0 | frozen_backbone |
|        0.000416 |                   0.520102 |                    0.124279 |         -0.002667 |                     0.496966 |                      0.03902  |              0.024186 |            0.125504 |              0.100643 |     52 | val     |              0 | frozen_backbone |
|       -0.003814 |                   0.513339 |                    0.125381 |         -0.003582 |                     0.496644 |                      0.039767 |              0.024241 |            0.125601 |              0.100225 |     62 | val     |              0 | frozen_backbone |

|      |   raw_rel_score |   alpha_rel_score |   raw_directional_accuracy |   alpha_directional_accuracy |   raw_pred_actual_q90_ratio |   alpha_pred_actual_q90_ratio |
|:-----|----------------:|------------------:|---------------------------:|-----------------------------:|----------------------------:|------------------------------:|
| mean |       -0.000126 |         -0.002947 |                   0.516752 |                     0.496881 |                    0.112124 |                      0.037521 |
| std  |        0.003449 |          0.000551 |                   0.003382 |                     0.000208 |                    0.022014 |                      0.003264 |

{
  "output_dir": "data/processed/assets/data_info_us/history/training_runs/reports/two_head_alpha_frozen_3seed_20260526_us",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/two_head_alpha_frozen_3seed_20260526_us"
}