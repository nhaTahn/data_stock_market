# Heteroscedastic NLL Probe Readout

Priority 1 from advisor report §6: volatility forecasting head + NLL loss.
Scope: VN train/validation only. Holdout/test not used.

## Aggregate Validation (3 seeds)

| variant         |   n_seeds |   rel_score_mean |   rel_score_std |   rel_score_vol_clipped_mean |   rel_score_vol_clipped_std |   daily_q90_max_mean |   daily_q90_max_std |   daily_q90_clipped_max_mean |   daily_q90_clipped_max_std |   spike_days_ge_8pct_mean |   spike_days_ge_8pct_std |   spike_days_clipped_ge_8pct_mean |   spike_days_clipped_ge_8pct_std |   directional_accuracy_mean |   directional_accuracy_std |   pred_actual_q90_ratio_mean |   pred_actual_q90_ratio_std |   mean_sigma_mean |   mean_sigma_std |
|:----------------|----------:|-----------------:|----------------:|-----------------------------:|----------------------------:|---------------------:|--------------------:|-----------------------------:|----------------------------:|--------------------------:|-------------------------:|----------------------------------:|---------------------------------:|----------------------------:|---------------------------:|-----------------------------:|----------------------------:|------------------:|-----------------:|
| hetero_combined |         3 |       0.00101595 |      0.00289941 |                   0.00101595 |                  0.00289941 |            0.0839084 |         0.000877636 |                    0.0839084 |                 0.000877636 |                         1 |                        0 |                                 1 |                                0 |                    0.512625 |                 0.00197289 |                    0.0845124 |                   0.0401896 |         0.0188084 |      0.000336723 |

## Decision

Pass if rel_score_mean >= baseline AND (spike_days_ge_8pct_mean < baseline OR daily_q90_clipped_max < baseline).

If hetero_combined passes: promote as new candidate.
If only hetero_nll passes: use sigma for post-processing clip only.
If neither passes: move to Priority 2 (supervised gate).