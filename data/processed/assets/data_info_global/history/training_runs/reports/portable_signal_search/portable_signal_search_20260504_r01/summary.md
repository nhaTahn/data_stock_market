# Portable Signal Search

Signals are oriented using train-only mean daily Spearman IC. Validation columns are not holdout/test.

## Top Signals

| Rank | Feature | Direction | Val IC | Min market IC | Positive markets | t-stat | Positive days |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `volume_delta_1` | `+1` | +0.01413 | +0.00132 | 3 | +3.69 | 54.2% |
| 2 | `momentum_5` | `-1` | +0.01072 | +0.00905 | 3 | +1.77 | 52.2% |
| 3 | `macd_hist` | `-1` | +0.00531 | +0.00357 | 3 | +1.03 | 51.7% |
| 4 | `adjust_return` | `-1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 5 | `alpha_market` | `-1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 6 | `alpha_sector` | `-1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 7 | `close_return` | `-1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 8 | `market_return` | `+1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 9 | `sector_return` | `+1` | +0.01996 | +0.01504 | 2 | +2.64 | 53.1% |
| 10 | `ma_5_gap` | `-1` | +0.01845 | +0.01744 | 2 | +2.38 | 52.3% |
| 11 | `ma_20_gap` | `-1` | +0.01027 | +0.00810 | 2 | +1.31 | 49.9% |
| 12 | `volume_level_20` | `+1` | +0.01264 | -0.00220 | 2 | +3.21 | 53.7% |
| 13 | `volume_ratio_20` | `+1` | +0.01264 | -0.00220 | 2 | +3.21 | 53.7% |
| 14 | `close_position` | `-1` | +0.01173 | -0.00571 | 2 | +2.40 | 52.5% |
| 15 | `open_delta_1` | `-1` | +0.01174 | -0.01107 | 2 | +2.15 | 52.8% |
| 16 | `volatility_20` | `-1` | +0.00678 | -0.00171 | 2 | +1.07 | 50.5% |
| 17 | `close_level_20` | `-1` | +0.00550 | -0.00385 | 2 | +0.90 | 50.0% |
| 18 | `intraday_return` | `-1` | +0.00752 | -0.01495 | 2 | +1.37 | 52.2% |
| 19 | `relative_sector_momentum_20` | `-1` | +0.00219 | -0.00416 | 2 | +0.39 | 49.5% |
| 20 | `open_level_20` | `-1` | +0.00176 | -0.00392 | 2 | +0.29 | 49.4% |
| 21 | `high_delta_1` | `-1` | +0.00416 | -0.01362 | 2 | +0.73 | 50.3% |
| 22 | `high_level_20` | `-1` | +0.00121 | -0.00750 | 2 | +0.20 | 49.4% |
| 23 | `gap_open` | `+1` | +0.00347 | -0.01375 | 2 | +0.65 | 50.3% |
| 24 | `bb_width` | `-1` | +0.00163 | -0.00941 | 2 | +0.29 | 49.6% |
| 25 | `low_level_20` | `-1` | +0.00078 | -0.00919 | 2 | +0.13 | 49.6% |
| 26 | `momentum_20` | `-1` | +0.00049 | -0.00923 | 2 | +0.08 | 48.7% |
| 27 | `close_delta_1` | `-1` | +0.00396 | -0.02780 | 2 | +0.69 | 50.3% |
| 28 | `sector_momentum_20` | `+1` | +0.01430 | +0.01430 | 1 | +1.78 | 53.4% |
| 29 | `sector_momentum_rank` | `-1` | +0.01430 | +0.01430 | 1 | +1.78 | 53.4% |
| 30 | `sector_momentum_rank_pct` | `-1` | +0.01430 | +0.01430 | 1 | +1.78 | 53.4% |
| 31 | `low_delta_1` | `-1` | +0.00891 | -0.00278 | 1 | +1.60 | 51.3% |
| 32 | `sector_ad_ratio` | `+1` | +0.00677 | -0.00927 | 1 | +0.51 | 50.6% |

## By Market For Top 10

### `volume_delta_1`
- `VN`: signed mean IC `+0.03804`, t-stat `+7.04`, days `659`
- `JP`: signed mean IC `+0.00132`, t-stat `+0.15`, days `641`
- `US`: signed mean IC `+0.00274`, t-stat `+0.50`, days `663`

### `momentum_5`
- `VN`: signed mean IC `+0.00905`, t-stat `+0.99`, days `659`
- `JP`: signed mean IC `+0.01336`, t-stat `+1.22`, days `641`
- `US`: signed mean IC `+0.00981`, t-stat `+0.87`, days `663`

### `macd_hist`
- `VN`: signed mean IC `+0.00634`, t-stat `+0.87`, days `659`
- `JP`: signed mean IC `+0.00357`, t-stat `+0.35`, days `641`
- `US`: signed mean IC `+0.00597`, t-stat `+0.65`, days `663`

### `adjust_return`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `alpha_market`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `alpha_sector`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `close_return`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `market_return`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `sector_return`
- `JP`: signed mean IC `+0.02504`, t-stat `+2.45`, days `641`
- `US`: signed mean IC `+0.01504`, t-stat `+1.36`, days `663`

### `ma_5_gap`
- `JP`: signed mean IC `+0.01950`, t-stat `+1.80`, days `641`
- `US`: signed mean IC `+0.01744`, t-stat `+1.57`, days `663`
