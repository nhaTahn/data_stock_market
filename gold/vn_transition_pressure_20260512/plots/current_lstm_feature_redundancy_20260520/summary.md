# Current LSTM Feature Redundancy Readout 2026-05-20

Scope: current default LSTM feature set, train-only diagnostics. Validation/test are not used for feature decisions.

- features checked: `26`
- train_end_date: `2020-03-31`
- target: `target_next_return`
- redundant threshold: `|corr| >= 0.75`
- strict threshold: `|corr| >= 0.85`

## Redundant Pairs

- pairs >= threshold: `1`
- strict pairs: `1`

| feature_a | feature_b | corr |
|:--|:--|--:|
| `market_leader_return` | `vnindex_return` | 0.868 |

## Train-Only Signal/Quality Top Features

| feature | daily IC mean | daily IC t-stat | overall Spearman | missing | quality |
|:--|--:|--:|--:|--:|--:|
| `close_position` | -0.0818 | -23.01 | -0.0504 | 3.9% | 0.2324 |
| `lower_shadow` | -0.0762 | -23.63 | -0.0602 | 0.0% | 0.2226 |
| `selling_pressure` | 0.0707 | 19.06 | 0.0470 | 3.9% | 0.1975 |
| `volume_ratio_20` | 0.0486 | 14.95 | 0.0487 | 1.6% | 0.1644 |
| `upper_shadow` | 0.0485 | 13.55 | 0.0420 | 0.0% | 0.1490 |
| `effort_result_ratio` | 0.0402 | 12.42 | 0.0406 | 3.9% | 0.1303 |
| `momentum_5` | -0.0401 | -9.36 | -0.0166 | 0.3% | 0.1148 |
| `buying_pressure` | -0.0373 | -10.76 | -0.0128 | 3.9% | 0.0967 |
| `intraday_return` | -0.0322 | -8.47 | -0.0041 | 0.0% | 0.0886 |
| `macd_hist` | -0.0221 | -5.89 | -0.0069 | 0.0% | 0.0687 |
| `rolling_max_20_gap` | -0.0218 | -4.97 | -0.0013 | 1.2% | 0.0602 |
| `vnindex_return` | nan | nan | 0.0381 | 0.0% | 0.0529 |

## Suggested Correlation-Pruned Candidate

- keep `25` features
- drop `1` features

Drop list:

```text
market_leader_return
```

Feature columns for next LSTM probe:

```text
volume_ratio_20,intraday_return,gap_open,close_position,upper_shadow,lower_shadow,momentum_5,momentum_20,volatility_20,ma_200_gap,rolling_max_20_gap,bb_width,vwap_gap,obv_change,macd_hist,effort_result_ratio,buying_pressure,selling_pressure,wyckoff_phase_60d,a_d_ratio,vnindex_return,rsi_14,day_of_week,sector_momentum_rank,is_top_2_sector
```

## Read

- This is a diagnostic, not proof of model improvement.
- The pruning rule uses train-only feature correlation and train-only feature-target IC.
- A feature should only be removed permanently if the LSTM probe improves or stays neutral on validation rel_score and daily error stability.