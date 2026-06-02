# Academic Ablation + Baseline + Significance Report

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.

## Overall Ablation And Baseline Ranking

| model                                 |   rel_score |   absE_robust |   absE_q90 |       DA |   pred_actual_q90_ratio |
|:--------------------------------------|------------:|--------------:|-----------:|---------:|------------------------:|
| ensemble_mean_cal_each_train_cal_clip |    0.044775 |      0.036003 |   0.046932 | 0.518273 |                0.193216 |
| ensemble_mean_cal_each_train_cal      |    0.044759 |      0.036004 |   0.046954 | 0.518273 |                0.193216 |
| ensemble_mean_raw_train_cal           |    0.043911 |      0.036036 |   0.047043 | 0.517942 |                0.186029 |
| ensemble_mean_train_cal_each          |    0.039223 |      0.036212 |   0.047483 | 0.518273 |                0.154573 |
| ensemble_median_train_cal             |    0.038601 |      0.036236 |   0.047477 | 0.517479 |                0.156674 |
| ensemble_mean_raw                     |    0.037942 |      0.036261 |   0.04759  | 0.517942 |                0.148823 |
| single_seed43_train_cal               |    0.035966 |      0.036335 |   0.047701 | 0.512135 |                0.164976 |
| single_seed43_raw                     |    0.034983 |      0.036372 |   0.047758 | 0.512135 |                0.15712  |
| global_train_mean                     |    0.000536 |      0.037671 |   0.050112 | 0.474564 |                0.008776 |
| zero                                  |    0        |      0.037691 |   0.050224 | 0.074613 |                0        |
| stock_train_mean                      |   -0.001157 |      0.037734 |   0.050183 | 0.468128 |                0.029774 |
| ridge_last_step                       |   -0.003131 |      0.037809 |   0.049971 | 0.482604 |                0.089706 |
| lagged_stock_mean5_val_only           |   -0.072949 |      0.04044  |   0.050336 | 0.46156  |                0.424875 |

## 21-Day Fold Summary

| model                                 |   mean_fold_rel |   median_fold_rel |   min_fold_rel |   positive_folds |   folds |   mean_absE_robust |   mean_absE_q90 |
|:--------------------------------------|----------------:|------------------:|---------------:|-----------------:|--------:|-------------------:|----------------:|
| ensemble_mean_raw_train_cal           |        0.029875 |          0.018293 |      -0.024005 |               24 |      32 |           0.036371 |        0.045853 |
| ensemble_mean_cal_each_train_cal      |        0.029872 |          0.019495 |      -0.028232 |               24 |      32 |           0.03637  |        0.045857 |
| ensemble_mean_cal_each_train_cal_clip |        0.029796 |          0.020773 |      -0.028232 |               24 |      32 |           0.036373 |        0.045856 |
| ensemble_mean_raw                     |        0.029676 |          0.018852 |      -0.025482 |               24 |      32 |           0.036399 |        0.045957 |
| ensemble_mean_train_cal_each          |        0.029538 |          0.020217 |      -0.025876 |               23 |      32 |           0.036406 |        0.04595  |
| ensemble_median_train_cal             |        0.027895 |          0.018625 |      -0.022391 |               24 |      32 |           0.036459 |        0.04592  |
| single_seed43_train_cal               |        0.017452 |          0.010205 |      -0.144157 |               20 |      32 |           0.037068 |        0.046959 |
| single_seed43_raw                     |        0.016469 |          0.012102 |      -0.139165 |               19 |      32 |           0.037101 |        0.046991 |
| ridge_last_step                       |        0.000239 |         -0.002313 |      -0.029828 |               15 |      32 |           0.037715 |        0.047145 |
| zero                                  |        0        |          0        |       0        |                0 |      32 |           0.037778 |        0.047788 |
| global_train_mean                     |       -0.000681 |          0.000178 |      -0.013107 |               16 |      32 |           0.037807 |        0.047746 |
| stock_train_mean                      |       -0.000682 |         -0.000417 |      -0.016174 |               15 |      32 |           0.037804 |        0.047705 |
| lagged_stock_mean5_val_only           |       -0.09625  |         -0.114808 |      -0.188281 |                3 |      32 |           0.040861 |        0.048858 |

## Bootstrap Significance vs Candidate

| candidate                             | baseline                         |   n_folds |   mean_delta_rel_score |   median_delta_rel_score |   ci95_low |   ci95_high |   p_boot_delta_le_0 |   positive_delta_folds |
|:--------------------------------------|:---------------------------------|----------:|-----------------------:|-------------------------:|-----------:|------------:|--------------------:|-----------------------:|
| ensemble_mean_cal_each_train_cal_clip | lagged_stock_mean5_val_only      |        32 |               0.126046 |                 0.125164 |   0.109277 |    0.141802 |             0       |                     32 |
| ensemble_mean_cal_each_train_cal_clip | stock_train_mean                 |        32 |               0.030478 |                 0.021045 |   0.017311 |    0.044284 |             0       |                     25 |
| ensemble_mean_cal_each_train_cal_clip | global_train_mean                |        32 |               0.030477 |                 0.020606 |   0.017119 |    0.044965 |             0       |                     25 |
| ensemble_mean_cal_each_train_cal_clip | zero                             |        32 |               0.029796 |                 0.020773 |   0.016501 |    0.044093 |             0       |                     24 |
| ensemble_mean_cal_each_train_cal_clip | ridge_last_step                  |        32 |               0.029557 |                 0.029231 |   0.01722  |    0.042928 |             0       |                     23 |
| ensemble_mean_cal_each_train_cal_clip | single_seed43_raw                |        32 |               0.013327 |                 0.009806 |   0.00506  |    0.025051 |             0       |                     23 |
| ensemble_mean_cal_each_train_cal_clip | single_seed43_train_cal          |        32 |               0.012344 |                 0.007105 |   0.004231 |    0.024356 |             0.0001  |                     23 |
| ensemble_mean_cal_each_train_cal_clip | ensemble_median_train_cal        |        32 |               0.001901 |                 0.001093 |  -0.000814 |    0.004718 |             0.0866  |                     18 |
| ensemble_mean_cal_each_train_cal_clip | ensemble_mean_train_cal_each     |        32 |               0.000258 |                -0.000992 |  -0.001807 |    0.002501 |             0.42065 |                     12 |
| ensemble_mean_cal_each_train_cal_clip | ensemble_mean_raw                |        32 |               0.00012  |                -0.002197 |  -0.002243 |    0.002726 |             0.46705 |                      9 |
| ensemble_mean_cal_each_train_cal_clip | ensemble_mean_cal_each_train_cal |        32 |              -7.6e-05  |                 0        |  -0.000567 |    0.00036  |             0.62175 |                      2 |
| ensemble_mean_cal_each_train_cal_clip | ensemble_mean_raw_train_cal      |        32 |              -7.9e-05  |                 6.6e-05  |  -0.000953 |    0.000795 |             0.56715 |                     16 |

## Candidate Yearly Robustness

| model                                 |   year |   mean_rel_score |   positive_days |   days |   mean_absE_robust |
|:--------------------------------------|-------:|-----------------:|----------------:|-------:|-------------------:|
| ensemble_mean_cal_each_train_cal_clip |   2020 |         0.01254  |             103 |    193 |           0.029146 |
| ensemble_mean_cal_each_train_cal_clip |   2021 |         0.014476 |             143 |    250 |           0.033484 |
| ensemble_mean_cal_each_train_cal_clip |   2022 |         0.005864 |             116 |    216 |           0.041976 |

## Interpretation

- The frozen candidate is evaluated against zero, train-only mean baselines, a lagged validation-only baseline, ridge regression, and seed/ensemble ablations.
- Significance is paired by 21-day validation fold and uses bootstrap resampling of fold-level rel_score deltas.
- This package is validation-only evidence for paper development; holdout remains closed.