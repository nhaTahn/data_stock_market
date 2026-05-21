# MarketAux LSTM Screen

Scope: VN train/validation only. Holdout/test is not used.

Hypothesis: keep LSTM return output, add soft auxiliary regression heads for next-day market mean return and next-day cross-sectional q90 absolute return.

This was run as a quick screen because the full run was abnormally slow before writing the first model artifact.

| candidate | rel_score | q90 error | daily q90 p90 | daily max | days >=7% | days >=8% | pred/actual q90 | dir_acc |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `plain_global_weighted_mild_tail35_marketaux_w10` | -0.00045 | 5.032% | 6.676% | 7.481% | 36 | 0 | 0.088 | 0.477 |
| `plain_global_weighted_mild_tail35_marketaux_w20` | -0.00625 | 5.021% | 6.608% | 8.279% | 43 | 7 | 0.150 | 0.473 |

## Decision

- MarketAux does not improve validation rel_score in the screen.
- The lower daily max comes from under-amplified predictions, not better return forecasting.
- Do not continue full MarketAux multiseed until input/target processing is improved.