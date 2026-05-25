# Alpha Auxiliary Hetero Probe

Trains on date-demeaned returns (alpha), evaluates raw and alpha metrics. Holdout/test not used.

## Validation Per Seed

|   raw_rel_score |   raw_directional_accuracy |   raw_pred_actual_q90_ratio |   alpha_rel_score |   alpha_directional_accuracy |   alpha_pred_actual_q90_ratio |   alpha_median_abs_error |   alpha_q90_abs_error |   daily_q90_max |   spike_days_ge_8pct |   seed | split   |   mean_sigma |
|----------------:|---------------------------:|----------------------------:|------------------:|-----------------------------:|------------------------------:|-------------------------:|----------------------:|----------------:|---------------------:|-------:|:--------|-------------:|
|       -0.000365 |                   0.503441 |                    0.031992 |         -0.0018   |                     0.50178  |                      0.042108 |                 0.008054 |              0.024193 |        0.100977 |                    1 |     43 | val     |     0.015029 |
|        0.001823 |                   0.494254 |                    0.039996 |          0.000216 |                     0.514238 |                      0.052643 |                 0.00804  |              0.024139 |        0.099926 |                    1 |     52 | val     |     0.01451  |
|       -0.000747 |                   0.491305 |                    0.046036 |          0.000685 |                     0.511051 |                      0.060592 |                 0.008034 |              0.024132 |        0.09873  |                    1 |     62 | val     |     0.014436 |

## Aggregate

|      |   raw_rel_score |   alpha_rel_score |   raw_directional_accuracy |   alpha_directional_accuracy |   alpha_pred_actual_q90_ratio |   daily_q90_max |
|:-----|----------------:|------------------:|---------------------------:|-----------------------------:|------------------------------:|----------------:|
| mean |        0.000237 |          -0.0003  |                   0.496333 |                     0.509023 |                      0.051781 |        0.099878 |
| std  |        0.001386 |           0.00132 |                   0.00633  |                     0.006472 |                      0.009272 |        0.001124 |

{
  "output_dir": "data/processed/assets/data_info_us/history/training_runs/reports/alpha_aux_smoke_20260526_us",
  "gold_dir": "gold/vn_transition_pressure_20260512/plots/alpha_aux_smoke_20260526_us"
}