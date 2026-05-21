# Daily LSTM High-Error Filter Probe

Purpose: test whether a small LSTM over daily regime features can detect days where the frozen base LSTM has high q90 prediction error.
Target label: `q90(|actual_return - predicted_return|) > 3.5%`. Holdout/test is not used.

## Result

- Train AUC: `0.815`; validation AUC: `0.792`.
- Validation baseline spike rate: `54.69%`.
- Validation top-20% LSTM-risk spike rate: `88.10%`.
- 2017 segment top-20% LSTM-risk spike rate: `60.00%`.
- 2017 segment top-risk median q90(|E|): `3.68%` vs rest `3.18%`.

Interpretation: this filter should be used as a risk/no-trade or position-sizing layer. It does not change base next-day return predictions; it tests whether the timing of large errors is learnable from regime features.

## Metrics

| model      | split                 |   n_days | spike_rate   |      auc |   average_precision |   top20_days | top20_spike_rate   | rest_spike_rate   | top20_median_q90_abs_error   | rest_median_q90_abs_error   |
|:-----------|:----------------------|---------:|:-------------|---------:|--------------------:|-------------:|:-------------------|:------------------|:-----------------------------|:----------------------------|
| daily_lstm | train                 |     1940 | 36.80%       | 0.814995 |            0.754581 |          388 | 79.90%             | 26.03%            | 4.19%                        | 2.91%                       |
| daily_lstm | val                   |      629 | 54.69%       | 0.791952 |            0.816309 |          126 | 88.10%             | 46.32%            | 5.71%                        | 3.37%                       |
| daily_lstm | segment_2017_d200_250 |       50 | 40.00%       | 0.585    |            0.555608 |           10 | 60.00%             | 35.00%            | 3.68%                        | 3.18%                       |

## Next

If this is promoted, the next full experiment should add these regime features to the existing per-stock LSTM filter and evaluate trading metrics after cost, not only q90 error timing.