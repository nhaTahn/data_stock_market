# Multi-Market Context Adapter Probe

Protocol: common portable features + market context adapter, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.

## Overall Metrics

|     n |   rel_score |   absE_robust |   base_robust |   absE_q90 |       DA |   pred_actual_q90_ratio | market   | model                 |   alpha_rel_score |   alpha_absE_robust |   train_rows |   val_rows |   n_codes |
|------:|------------:|--------------:|--------------:|-----------:|---------:|------------------------:|:---------|:----------------------|------------------:|--------------------:|-------------:|-----------:|----------:|
| 16660 |    0.000683 |      0.026106 |      0.026123 |   0.030712 | 0.515066 |                0.042538 | JP       | ridge_portable        |         -0.002763 |            0.02036  |        64446 |      16660 |        26 |
| 16660 |    0        |      0.026123 |      0.026123 |   0.030739 | 0.007263 |                0        | JP       | zero                  |          0        |            0.020303 |        64446 |      16660 |        26 |
| 16660 |   -0.001491 |      0.026162 |      0.026123 |   0.030743 | 0.516387 |                0.047221 | JP       | ridge_context_adapter |         -0.002525 |            0.020355 |        64446 |      16660 |        26 |
| 58998 |    9.4e-05  |      0.02619  |      0.026193 |   0.03178  | 0.514255 |                0.083797 | US       | ridge_portable        |         -0.003166 |            0.020178 |       225295 |      58998 |        89 |
| 58998 |    0        |      0.026193 |      0.026193 |   0.031778 | 0.003051 |                0        | US       | zero                  |          0        |            0.020114 |       225295 |      58998 |        89 |
| 58998 |   -0.00342  |      0.026283 |      0.026193 |   0.031863 | 0.507339 |                0.117757 | US       | ridge_context_adapter |         -0.001117 |            0.020136 |       225295 |      58998 |        89 |
| 60655 |    0        |      0.037808 |      0.037808 |   0.050459 | 0.075262 |                0        | VN       | zero                  |          0        |            0.029969 |       134618 |      60655 |        93 |
| 60655 |   -0.002041 |      0.037885 |      0.037808 |   0.050401 | 0.472937 |                0.034818 | VN       | ridge_portable        |         -9.2e-05  |            0.029972 |       134618 |      60655 |        93 |
| 60655 |   -0.003576 |      0.037943 |      0.037808 |   0.050338 | 0.475344 |                0.070055 | VN       | ridge_context_adapter |          0.000989 |            0.029939 |       134618 |      60655 |        93 |

## Paired Bootstrap: Context Adapter vs Portable Ridge

| market   | metric          |   n_folds |   mean_delta_context_vs_portable |   ci95_low |   ci95_high |   p_boot_delta_le_0 |   positive_delta_folds |
|:---------|:----------------|----------:|---------------------------------:|-----------:|------------:|--------------------:|-----------------------:|
| JP       | rel_score       |        31 |                        -0.001272 |  -0.002896 |    0.000474 |             0.92535 |                     10 |
| US       | rel_score       |        32 |                        -0.003584 |  -0.008982 |    0.00138  |             0.91675 |                     14 |
| VN       | rel_score       |        32 |                         0.000369 |  -0.002607 |    0.003316 |             0.4044  |                     16 |
| JP       | alpha_rel_score |        31 |                         0.000521 |  -0.001639 |    0.002659 |             0.3111  |                     16 |
| US       | alpha_rel_score |        32 |                         0.003479 |   0.000674 |    0.006214 |             0.00755 |                     21 |
| VN       | alpha_rel_score |        32 |                         0.000376 |  -0.001842 |    0.002624 |             0.36395 |                     15 |

## Interpretation

- This is a lightweight adapter probe, not the final LSTM model.
- Positive regular rel_score with negative alpha_rel_score indicates market-drift capture, not stock-selection skill.
- Use this to decide which context features should enter the next heteroscedastic ensemble run.