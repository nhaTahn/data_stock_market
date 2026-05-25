# Multi-Market Portable Baseline Significance Smoke Test

Protocol: common VN/US/JP portable features, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.

## Overall Metrics

|     n |   rel_score |   absE_robust |   base_robust |   absE_q90 |       DA |   pred_actual_q90_ratio | market   | model                       |   train_rows |   val_rows |   n_codes |
|------:|------------:|--------------:|--------------:|-----------:|---------:|------------------------:|:---------|:----------------------------|-------------:|-----------:|----------:|
| 16660 |    0.000683 |      0.026106 |      0.026123 |   0.030712 | 0.515066 |                0.042538 | JP       | ridge_portable              |        64446 |      16660 |        26 |
| 16660 |    0        |      0.026123 |      0.026123 |   0.030739 | 0.007263 |                0        | JP       | zero                        |        64446 |      16660 |        26 |
| 16660 |   -0.001082 |      0.026152 |      0.026123 |   0.030754 | 0.509424 |                0.017354 | JP       | global_train_mean           |        64446 |      16660 |        26 |
| 16660 |   -0.001488 |      0.026162 |      0.026123 |   0.030761 | 0.509424 |                0.029851 | JP       | stock_train_mean            |        64446 |      16660 |        26 |
| 16660 |   -0.111869 |      0.029046 |      0.026123 |   0.033806 | 0.484394 |                0.451606 | JP       | lagged_stock_mean5_val_only |        64446 |      16660 |        26 |
| 58998 |    0.00235  |      0.026131 |      0.026193 |   0.031715 | 0.520323 |                0.018391 | US       | global_train_mean           |       225295 |      58998 |        89 |
| 58998 |    0.001243 |      0.02616  |      0.026193 |   0.031726 | 0.519729 |                0.032793 | US       | stock_train_mean            |       225295 |      58998 |        89 |
| 58998 |    9.4e-05  |      0.02619  |      0.026193 |   0.03178  | 0.514255 |                0.083797 | US       | ridge_portable              |       225295 |      58998 |        89 |
| 58998 |    0        |      0.026193 |      0.026193 |   0.031778 | 0.003051 |                0        | US       | zero                        |       225295 |      58998 |        89 |
| 58998 |   -0.121099 |      0.029365 |      0.026193 |   0.035415 | 0.496152 |                0.453095 | US       | lagged_stock_mean5_val_only |       225295 |      58998 |        89 |
| 60655 |    0        |      0.037808 |      0.037808 |   0.050459 | 0.075262 |                0        | VN       | zero                        |       134618 |      60655 |        93 |
| 60655 |   -0.000286 |      0.037819 |      0.037808 |   0.050392 | 0.47424  |                0.00777  | VN       | global_train_mean           |       134618 |      60655 |        93 |
| 60655 |   -0.001426 |      0.037862 |      0.037808 |   0.050454 | 0.467365 |                0.02354  | VN       | stock_train_mean            |       134618 |      60655 |        93 |
| 60655 |   -0.002041 |      0.037885 |      0.037808 |   0.050401 | 0.472937 |                0.034818 | VN       | ridge_portable              |       134618 |      60655 |        93 |
| 60655 |   -0.06985  |      0.040449 |      0.037808 |   0.050364 | 0.461611 |                0.425073 | VN       | lagged_stock_mean5_val_only |       134618 |      60655 |        93 |

## Best Two Baselines Per Market

|     n |   rel_score |   absE_robust |   base_robust |   absE_q90 |       DA |   pred_actual_q90_ratio | market   | model             |   train_rows |   val_rows |   n_codes |
|------:|------------:|--------------:|--------------:|-----------:|---------:|------------------------:|:---------|:------------------|-------------:|-----------:|----------:|
| 16660 |    0.000683 |      0.026106 |      0.026123 |   0.030712 | 0.515066 |                0.042538 | JP       | ridge_portable    |        64446 |      16660 |        26 |
| 16660 |    0        |      0.026123 |      0.026123 |   0.030739 | 0.007263 |                0        | JP       | zero              |        64446 |      16660 |        26 |
| 58998 |    0.00235  |      0.026131 |      0.026193 |   0.031715 | 0.520323 |                0.018391 | US       | global_train_mean |       225295 |      58998 |        89 |
| 58998 |    0.001243 |      0.02616  |      0.026193 |   0.031726 | 0.519729 |                0.032793 | US       | stock_train_mean  |       225295 |      58998 |        89 |
| 60655 |    0        |      0.037808 |      0.037808 |   0.050459 | 0.075262 |                0        | VN       | zero              |       134618 |      60655 |        93 |
| 60655 |   -0.000286 |      0.037819 |      0.037808 |   0.050392 | 0.47424  |                0.00777  | VN       | global_train_mean |       134618 |      60655 |        93 |

## Fold Summary

| market   | model                       |   mean_fold_rel |   median_fold_rel |   min_fold_rel |   positive_folds |   folds |
|:---------|:----------------------------|----------------:|------------------:|---------------:|-----------------:|--------:|
| JP       | global_train_mean           |       -0.001107 |         -0.00157  |      -0.015046 |               12 |      31 |
| JP       | lagged_stock_mean5_val_only |       -0.111545 |         -0.110206 |      -0.209452 |                0 |      31 |
| JP       | ridge_portable              |        0.000974 |          0.002181 |      -0.02655  |               19 |      31 |
| JP       | stock_train_mean            |       -0.001212 |         -0.000578 |      -0.019875 |               13 |      31 |
| JP       | zero                        |        0        |          0        |       0        |                0 |      31 |
| US       | global_train_mean           |        0.00169  |          0.002936 |      -0.014799 |               19 |      32 |
| US       | lagged_stock_mean5_val_only |       -0.11975  |         -0.117091 |      -0.185342 |                0 |      32 |
| US       | ridge_portable              |       -0.003723 |         -0.002464 |      -0.027904 |               15 |      32 |
| US       | stock_train_mean            |        0.000539 |          0.001208 |      -0.01907  |               21 |      32 |
| US       | zero                        |        0        |          0        |       0        |                0 |      32 |
| VN       | global_train_mean           |       -0.000512 |         -9.4e-05  |      -0.011525 |               15 |      32 |
| VN       | lagged_stock_mean5_val_only |       -0.093474 |         -0.11348  |      -0.187829 |                3 |      32 |
| VN       | ridge_portable              |        0.000186 |          0.002028 |      -0.023263 |               20 |      32 |
| VN       | stock_train_mean            |       -0.000759 |         -0.001134 |      -0.015116 |               13 |      32 |
| VN       | zero                        |        0        |          0        |       0        |                0 |      32 |

## Bootstrap vs Zero

| market   | model                       |   n_folds |   mean_delta_vs_zero |   ci95_low |   ci95_high |   p_boot_delta_le_0 |   positive_delta_folds |
|:---------|:----------------------------|----------:|---------------------:|-----------:|------------:|--------------------:|-----------------------:|
| JP       | ridge_portable              |        31 |             0.000974 |  -0.002032 |    0.003708 |             0.24375 |                     19 |
| JP       | global_train_mean           |        31 |            -0.001107 |  -0.004019 |    0.001875 |             0.7667  |                     12 |
| JP       | stock_train_mean            |        31 |            -0.001212 |  -0.004091 |    0.001591 |             0.7957  |                     13 |
| JP       | lagged_stock_mean5_val_only |        31 |            -0.111545 |  -0.127567 |   -0.095533 |             1       |                      0 |
| US       | global_train_mean           |        32 |             0.00169  |  -0.000668 |    0.003983 |             0.0782  |                     19 |
| US       | stock_train_mean            |        32 |             0.000539 |  -0.001932 |    0.00291  |             0.3283  |                     21 |
| US       | ridge_portable              |        32 |            -0.003723 |  -0.007267 |   -0.000365 |             0.98565 |                     15 |
| US       | lagged_stock_mean5_val_only |        32 |            -0.11975  |  -0.130464 |   -0.10856  |             1       |                      0 |
| VN       | ridge_portable              |        32 |             0.000186 |  -0.002449 |    0.002656 |             0.4334  |                     20 |
| VN       | global_train_mean           |        32 |            -0.000512 |  -0.002144 |    0.001075 |             0.73315 |                     15 |
| VN       | stock_train_mean            |        32 |            -0.000759 |  -0.002589 |    0.001034 |             0.79605 |                     13 |
| VN       | lagged_stock_mean5_val_only |        32 |            -0.093474 |  -0.115369 |   -0.070004 |             1       |                      3 |

## Interpretation

- This is a pre-model smoke test for the multi-market paper protocol.
- The goal is to confirm schema/date/feature compatibility and identify how hard each market is before expensive LSTM/heteroscedastic ensemble training.
- If ridge/simple baselines are weak but stable, the next step is a portable heteroscedastic ensemble run on US and JP with the same academic report template.