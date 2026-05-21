# Daily LSTM High-Error Filter Probe

Purpose: test whether a small LSTM over daily regime features can detect days where the frozen base LSTM has high q90 prediction error.
Target label: `q90(|actual_return - predicted_return|) > 3.5%`. Holdout/test is not used.

## Result

- Train AUC: `0.791`; validation AUC: `0.779`.
- Validation baseline spike rate: `54.93%`.
- Validation top-20% LSTM-risk spike rate: `85.16%`.
- 2017 segment top-20% LSTM-risk spike rate: `50.00%`.
- 2017 segment top-risk median q90(|E|): `3.43%` vs rest `3.18%`.

Interpretation: this filter should be used as a risk/no-trade or position-sizing layer. It does not change base next-day return predictions; it tests whether the timing of large errors is learnable from regime features.

## Metrics

| model      | split                 |   n_days | spike_rate   |      auc |   average_precision |   top20_days | top20_spike_rate   | rest_spike_rate   | top20_median_q90_abs_error   | rest_median_q90_abs_error   |
|:-----------|:----------------------|---------:|:-------------|---------:|--------------------:|-------------:|:-------------------|:------------------|:-----------------------------|:----------------------------|
| daily_lstm | train                 |     1950 | 36.92%       | 0.790597 |            0.702108 |          390 | 74.62%             | 27.50%            | 4.11%                        | 2.92%                       |
| daily_lstm | val                   |      639 | 54.93%       | 0.778629 |            0.794645 |          128 | 85.16%             | 47.36%            | 5.69%                        | 3.38%                       |
| daily_lstm | segment_2017_d200_250 |       50 | 40.00%       | 0.595    |            0.564535 |           10 | 50.00%             | 37.50%            | 3.43%                        | 3.18%                       |

## Next

If this is promoted, the next full experiment should add these regime features to the existing per-stock LSTM filter and evaluate trading metrics after cost, not only q90 error timing.