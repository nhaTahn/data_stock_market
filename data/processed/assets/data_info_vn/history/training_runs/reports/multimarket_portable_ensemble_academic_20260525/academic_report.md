# Multi-Market Portable Ensemble Academic Report

Protocol: VN/US/JP, common portable features, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.

## Top Ensemble/Ablation Metrics

| market   | model                                 |   rel_score |   alpha_rel_score |   absE_robust |   alpha_absE_robust |       DA |   pred_actual_q90_ratio |
|:---------|:--------------------------------------|------------:|------------------:|--------------:|--------------------:|---------:|------------------------:|
| JP       | ensemble_median_raw                   |    0.004504 |         -0.000885 |      0.026006 |            0.020321 | 0.517707 |                0.04983  |
| JP       | seed43_raw                            |    0.002776 |         -0.001081 |      0.026051 |            0.020325 | 0.510504 |                0.06455  |
| JP       | seed62_raw                            |    0.002602 |         -0.002701 |      0.026055 |            0.020358 | 0.512965 |                0.058213 |
| JP       | seed62_train_cal                      |    0.002184 |         -0.001428 |      0.026066 |            0.020332 | 0.512965 |                0.052391 |
| US       | ensemble_mean_raw                     |    0.006037 |         -0.002223 |      0.026035 |            0.020159 | 0.519458 |                0.070582 |
| US       | ensemble_median_raw                   |    0.004934 |         -0.002197 |      0.026064 |            0.020158 | 0.518526 |                0.065509 |
| US       | seed82_train_cal                      |    0.004148 |         -0.001321 |      0.026084 |            0.020141 | 0.513678 |                0.074524 |
| US       | ensemble_median_train_cal             |    0.003477 |         -0.003149 |      0.026102 |            0.020177 | 0.518526 |                0.088437 |
| VN       | seed43_train_cal                      |    0.007486 |          0.000769 |      0.037525 |            0.029946 | 0.476515 |                0.090913 |
| VN       | seed43_raw                            |    0.006228 |          0.001852 |      0.037573 |            0.029914 | 0.476515 |                0.079055 |
| VN       | ensemble_mean_train_cal               |    0.006132 |          0.002853 |      0.037576 |            0.029884 | 0.476037 |                0.075566 |
| VN       | selected_train_only_portable_ensemble |    0.006132 |          0.002853 |      0.037576 |            0.029884 | 0.476037 |                0.075566 |

## Candidate Fold Summary

| market   |   mean_fold_rel |   positive_folds |   folds |   mean_alpha_fold_rel |   positive_alpha_folds |
|:---------|----------------:|-----------------:|--------:|----------------------:|-----------------------:|
| JP       |        7.9e-05  |               15 |      31 |             -0.001049 |                     11 |
| US       |       -0.001637 |               16 |      32 |             -0.005689 |                      8 |
| VN       |        0.005393 |               23 |      32 |              0.000846 |                     21 |

## Bootstrap Significance Focus

| market   | candidate                             | baseline                 | metric          |   n_folds |   mean_delta |   ci95_low |   ci95_high |   p_boot_delta_le_0 |   positive_delta_folds |
|:---------|:--------------------------------------|:-------------------------|:----------------|----------:|-------------:|-----------:|------------:|--------------------:|-----------------------:|
| JP       | selected_train_only_portable_ensemble | simple:global_train_mean | rel_score       |        31 |     0.001187 |  -0.001867 |    0.004239 |             0.2258  |                     16 |
| JP       | selected_train_only_portable_ensemble | simple:zero              | rel_score       |        31 |     7.9e-05  |  -0.001422 |    0.001585 |             0.45095 |                     15 |
| JP       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | rel_score       |        31 |    -0.000649 |  -0.001932 |    0.00063  |             0.83725 |                     13 |
| JP       | selected_train_only_portable_ensemble | ensemble_median_raw      | rel_score       |        31 |    -0.000654 |  -0.003332 |    0.002016 |             0.68995 |                     14 |
| JP       | selected_train_only_portable_ensemble | simple:ridge_portable    | rel_score       |        31 |    -0.000894 |  -0.00371  |    0.002041 |             0.73245 |                     13 |
| JP       | selected_train_only_portable_ensemble | ensemble_mean_raw        | rel_score       |        31 |    -0.002307 |  -0.005067 |    0.000519 |             0.9454  |                     11 |
| US       | selected_train_only_portable_ensemble | simple:ridge_portable    | rel_score       |        32 |     0.002086 |  -0.003662 |    0.00787  |             0.2419  |                     19 |
| US       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | rel_score       |        32 |     0        |   0        |    0        |             1       |                      0 |
| US       | selected_train_only_portable_ensemble | simple:zero              | rel_score       |        32 |    -0.001637 |  -0.008161 |    0.004679 |             0.68545 |                     16 |
| US       | selected_train_only_portable_ensemble | ensemble_mean_raw        | rel_score       |        32 |    -0.001743 |  -0.004879 |    0.001158 |             0.87515 |                     14 |
| US       | selected_train_only_portable_ensemble | ensemble_median_raw      | rel_score       |        32 |    -0.002343 |  -0.005712 |    0.000824 |             0.92185 |                     14 |
| US       | selected_train_only_portable_ensemble | simple:global_train_mean | rel_score       |        32 |    -0.003327 |  -0.008928 |    0.001741 |             0.8934  |                     16 |
| VN       | selected_train_only_portable_ensemble | simple:global_train_mean | rel_score       |        32 |     0.005905 |  -0.000131 |    0.01167  |             0.02745 |                     21 |
| VN       | selected_train_only_portable_ensemble | simple:zero              | rel_score       |        32 |     0.005393 |  -0.000505 |    0.011131 |             0.03705 |                     23 |
| VN       | selected_train_only_portable_ensemble | simple:ridge_portable    | rel_score       |        32 |     0.005207 |  -0.000871 |    0.010734 |             0.0443  |                     21 |
| VN       | selected_train_only_portable_ensemble | ensemble_median_raw      | rel_score       |        32 |     0.000929 |  -0.000846 |    0.002583 |             0.142   |                     21 |
| VN       | selected_train_only_portable_ensemble | ensemble_mean_raw        | rel_score       |        32 |     0.000408 |  -0.0013   |    0.001938 |             0.30285 |                     20 |
| VN       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | rel_score       |        32 |     0        |   0        |    0        |             1       |                      0 |
| JP       | selected_train_only_portable_ensemble | ensemble_mean_raw        | alpha_rel_score |        31 |     0.001215 |  -0.001398 |    0.003918 |             0.1833  |                     15 |
| JP       | selected_train_only_portable_ensemble | ensemble_median_raw      | alpha_rel_score |        31 |     0.000496 |  -0.001723 |    0.002744 |             0.3289  |                     16 |
| JP       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | alpha_rel_score |        31 |     5.8e-05  |  -0.001317 |    0.001481 |             0.4754  |                     14 |
| US       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | alpha_rel_score |        32 |     0        |   0        |    0        |             1       |                      0 |
| US       | selected_train_only_portable_ensemble | ensemble_median_raw      | alpha_rel_score |        32 |    -0.002625 |  -0.004632 |   -0.000671 |             0.9961  |                     11 |
| US       | selected_train_only_portable_ensemble | ensemble_mean_raw        | alpha_rel_score |        32 |    -0.002803 |  -0.004493 |   -0.00118  |             0.9995  |                      9 |
| VN       | selected_train_only_portable_ensemble | ensemble_median_raw      | alpha_rel_score |        32 |     6.2e-05  |  -0.001637 |    0.001761 |             0.46255 |                     15 |
| VN       | selected_train_only_portable_ensemble | ensemble_mean_train_cal  | alpha_rel_score |        32 |     0        |   0        |    0        |             1       |                      0 |
| VN       | selected_train_only_portable_ensemble | ensemble_mean_raw        | alpha_rel_score |        32 |    -0.000615 |  -0.001701 |    0.000418 |             0.8716  |                     15 |

## Interpretation

- `selected_train_only_portable_ensemble` is selected separately per market using train rel_score only among raw/calibrated mean/median ensembles.
- The regular rel_score includes market drift; `alpha_rel_score` demeans each date cross-section and is a stricter stock-selection metric.
- This remains validation-only evidence. Holdout/test remains closed.