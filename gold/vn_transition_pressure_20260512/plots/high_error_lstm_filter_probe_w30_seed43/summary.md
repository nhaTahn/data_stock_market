# Daily LSTM High-Error Filter Probe

Purpose: test whether a small LSTM over daily regime features can detect days where the frozen base LSTM has high q90 prediction error.
Target label: `q90(|actual_return - predicted_return|) > 3.5%`. Holdout/test is not used.

## Result

- Train AUC: `0.798`; validation AUC: `0.785`.
- Validation baseline spike rate: `54.77%`.
- Validation top-20% LSTM-risk spike rate: `85.48%`.
- 2017 segment top-20% LSTM-risk spike rate: `60.00%`.
- 2017 segment top-risk median q90(|E|): `3.67%` vs rest `3.17%`.

Interpretation: this filter should be used as a risk/no-trade or position-sizing layer. It does not change base next-day return predictions; it tests whether the timing of large errors is learnable from regime features.

## Metrics

| model      | split                 |   n_days | spike_rate   |      auc |   average_precision |   top20_days | top20_spike_rate   | rest_spike_rate   | top20_median_q90_abs_error   | rest_median_q90_abs_error   |
|:-----------|:----------------------|---------:|:-------------|---------:|--------------------:|-------------:|:-------------------|:------------------|:-----------------------------|:----------------------------|
| daily_lstm | train                 |     1930 | 36.53%       | 0.797872 |            0.712724 |          386 | 75.65%             | 26.75%            | 4.13%                        | 2.91%                       |
| daily_lstm | val                   |      619 | 54.77%       | 0.785324 |            0.795483 |          124 | 85.48%             | 47.07%            | 5.71%                        | 3.37%                       |
| daily_lstm | segment_2017_d200_250 |       50 | 40.00%       | 0.635    |            0.581846 |           10 | 60.00%             | 35.00%            | 3.67%                        | 3.17%                       |

## Next

If this is promoted, the next full experiment should add these regime features to the existing per-stock LSTM filter and evaluate trading metrics after cost, not only q90 error timing.