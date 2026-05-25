# Fixed Long-Train Ensemble Calibration

Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.

## Overall Metrics

|     n |   rel_score |   absE_robust |   base_robust |   absE_median |   absE_q90 |      DA |   pred_actual_q90_ratio | variant                                |
|------:|------------:|--------------:|--------------:|--------------:|-----------:|--------:|------------------------:|:---------------------------------------|
| 60445 |     0.04478 |       0.036   |       0.03769 |       0.01254 |    0.04693 | 0.51827 |                 0.19322 | ensemble_mean_cal_each_traincal_clip   |
| 60445 |     0.04476 |       0.036   |       0.03769 |       0.01253 |    0.04695 | 0.51827 |                 0.19322 | ensemble_mean_cal_each_traincal        |
| 60445 |     0.04391 |       0.03604 |       0.03769 |       0.01251 |    0.04704 | 0.51794 |                 0.18603 | ensemble_mean_raw_traincal             |
| 60445 |     0.04171 |       0.03612 |       0.03769 |       0.01251 |    0.04722 | 0.51748 |                 0.17234 | ensemble_median_cal_each_traincal_clip |
| 60445 |     0.04137 |       0.03613 |       0.03769 |       0.0125  |    0.04726 | 0.51748 |                 0.17234 | ensemble_median_cal_each_traincal      |
| 60445 |     0.03922 |       0.03621 |       0.03769 |       0.01247 |    0.04748 | 0.51827 |                 0.15457 | ensemble_mean_cal_each                 |
| 60445 |     0.0386  |       0.03624 |       0.03769 |       0.0125  |    0.04748 | 0.51748 |                 0.15667 | ensemble_median_cal_each               |
| 60445 |     0.03794 |       0.03626 |       0.03769 |       0.01247 |    0.04759 | 0.51794 |                 0.14882 | ensemble_mean_raw                      |

## 21-Day Fold Metrics

| variant                                |   mean_fold_rel |   median_fold_rel |   min_fold_rel |   positive_folds |   folds |   mean_absE_robust |   p90_absE_robust |   mean_absE_q90 |
|:---------------------------------------|----------------:|------------------:|---------------:|-----------------:|--------:|-------------------:|------------------:|----------------:|
| ensemble_mean_raw_traincal             |         0.02987 |           0.01829 |       -0.024   |               24 |      32 |            0.03637 |           0.05    |         0.04585 |
| ensemble_mean_cal_each_traincal        |         0.02987 |           0.01949 |       -0.02823 |               24 |      32 |            0.03637 |           0.05007 |         0.04586 |
| ensemble_mean_cal_each_traincal_clip   |         0.0298  |           0.02077 |       -0.02823 |               24 |      32 |            0.03637 |           0.05012 |         0.04586 |
| ensemble_mean_raw                      |         0.02968 |           0.01885 |       -0.02548 |               24 |      32 |            0.0364  |           0.05054 |         0.04596 |
| ensemble_mean_cal_each                 |         0.02954 |           0.02022 |       -0.02588 |               23 |      32 |            0.03641 |           0.05047 |         0.04595 |
| ensemble_median_cal_each_traincal      |         0.02827 |           0.01752 |       -0.0286  |               25 |      32 |            0.03642 |           0.05033 |         0.04587 |
| ensemble_median_cal_each_traincal_clip |         0.02803 |           0.01752 |       -0.0286  |               25 |      32 |            0.03644 |           0.05036 |         0.04586 |
| ensemble_median_cal_each               |         0.02789 |           0.01863 |       -0.02239 |               24 |      32 |            0.03646 |           0.0506  |         0.04592 |