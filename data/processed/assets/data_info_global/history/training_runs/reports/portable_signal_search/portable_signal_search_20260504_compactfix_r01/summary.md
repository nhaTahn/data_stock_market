# Portable Signal Search

Signals are oriented using train-only mean daily Spearman IC. Validation columns are not holdout/test.

## Top Signals

| Rank | Feature | Direction | Val IC | Min market IC | Positive markets | t-stat | Positive days |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `sector_return` | `+1` | +0.02148 | +0.01507 | 3 | +3.95 | 54.4% |
| 2 | `ma_5_gap` | `-1` | +0.01569 | +0.01047 | 3 | +2.62 | 53.2% |
| 3 | `volume_delta_1` | `+1` | +0.01413 | +0.00132 | 3 | +3.69 | 54.2% |
| 4 | `momentum_5` | `-1` | +0.01052 | +0.00798 | 3 | +1.74 | 52.0% |
| 5 | `macd_hist` | `-1` | +0.00531 | +0.00357 | 3 | +1.03 | 51.7% |
| 6 | `volume_level_20` | `+1` | +0.01264 | -0.00220 | 2 | +3.21 | 53.7% |
| 7 | `volume_ratio_20` | `+1` | +0.01264 | -0.00220 | 2 | +3.21 | 53.7% |
| 8 | `sector_ad_ratio` | `+1` | +0.01598 | -0.00955 | 2 | +2.74 | 52.6% |
| 9 | `close_position` | `-1` | +0.01173 | -0.00571 | 2 | +2.40 | 52.5% |
| 10 | `open_delta_1` | `-1` | +0.01174 | -0.01107 | 2 | +2.15 | 52.8% |
| 11 | `alpha_sector` | `-1` | +0.01040 | -0.00860 | 2 | +1.91 | 51.3% |
| 12 | `volatility_20` | `-1` | +0.00651 | -0.00233 | 2 | +1.02 | 50.4% |
| 13 | `close_level_20` | `-1` | +0.00550 | -0.00385 | 2 | +0.90 | 50.0% |
| 14 | `ma_20_gap` | `-1` | +0.00526 | -0.00459 | 2 | +0.86 | 50.1% |
| 15 | `intraday_return` | `-1` | +0.00752 | -0.01495 | 2 | +1.37 | 52.2% |
| 16 | `relative_sector_momentum_20` | `-1` | +0.00223 | -0.00408 | 2 | +0.39 | 49.3% |
| 17 | `open_level_20` | `-1` | +0.00176 | -0.00392 | 2 | +0.29 | 49.4% |
| 18 | `high_delta_1` | `-1` | +0.00416 | -0.01362 | 2 | +0.73 | 50.3% |
| 19 | `high_level_20` | `-1` | +0.00121 | -0.00750 | 2 | +0.20 | 49.4% |
| 20 | `gap_open` | `+1` | +0.00347 | -0.01375 | 2 | +0.65 | 50.3% |
| 21 | `bb_width` | `-1` | +0.00163 | -0.00941 | 2 | +0.29 | 49.6% |
| 22 | `low_level_20` | `-1` | +0.00078 | -0.00919 | 2 | +0.13 | 49.6% |
| 23 | `momentum_20` | `-1` | +0.00026 | -0.00995 | 2 | +0.04 | 48.6% |
| 24 | `adjust_return` | `-1` | +0.00436 | -0.02660 | 2 | +0.76 | 50.5% |
| 25 | `alpha_market` | `-1` | +0.00436 | -0.02660 | 2 | +0.76 | 50.5% |
| 26 | `market_return` | `+1` | +0.00436 | -0.02660 | 2 | +0.76 | 50.5% |
| 27 | `close_delta_1` | `-1` | +0.00396 | -0.02780 | 2 | +0.69 | 50.3% |
| 28 | `close_return` | `-1` | +0.00396 | -0.02780 | 2 | +0.69 | 50.3% |
| 29 | `sector_positive_ratio` | `+1` | +0.01699 | +0.01699 | 1 | +2.88 | 54.9% |
| 30 | `sector_momentum_20` | `+1` | +0.01419 | +0.01419 | 1 | +1.77 | 53.0% |
| 31 | `sector_momentum_rank` | `-1` | +0.01419 | +0.01419 | 1 | +1.77 | 53.0% |
| 32 | `sector_momentum_rank_pct` | `-1` | +0.01419 | +0.01419 | 1 | +1.77 | 53.0% |
| 33 | `low_delta_1` | `-1` | +0.00891 | -0.00278 | 1 | +1.60 | 51.3% |

## By Market For Top 10

### `sector_return`
- `VN`: signed mean IC `+0.02440`, t-stat `+3.87`, days `659`
- `JP`: signed mean IC `+0.02511`, t-stat `+2.46`, days `641`
- `US`: signed mean IC `+0.01507`, t-stat `+1.36`, days `663`

### `ma_5_gap`
- `VN`: signed mean IC `+0.01047`, t-stat `+1.15`, days `659`
- `JP`: signed mean IC `+0.01957`, t-stat `+1.81`, days `641`
- `US`: signed mean IC `+0.01713`, t-stat `+1.54`, days `663`

### `volume_delta_1`
- `VN`: signed mean IC `+0.03804`, t-stat `+7.04`, days `659`
- `JP`: signed mean IC `+0.00132`, t-stat `+0.15`, days `641`
- `US`: signed mean IC `+0.00274`, t-stat `+0.50`, days `663`

### `momentum_5`
- `VN`: signed mean IC `+0.00798`, t-stat `+0.87`, days `659`
- `JP`: signed mean IC `+0.01372`, t-stat `+1.25`, days `641`
- `US`: signed mean IC `+0.00995`, t-stat `+0.88`, days `663`

### `macd_hist`
- `VN`: signed mean IC `+0.00634`, t-stat `+0.87`, days `659`
- `JP`: signed mean IC `+0.00357`, t-stat `+0.35`, days `641`
- `US`: signed mean IC `+0.00597`, t-stat `+0.65`, days `663`

### `volume_level_20`
- `VN`: signed mean IC `+0.03774`, t-stat `+6.48`, days `659`
- `JP`: signed mean IC `-0.00220`, t-stat `-0.26`, days `641`
- `US`: signed mean IC `+0.00204`, t-stat `+0.36`, days `663`

### `volume_ratio_20`
- `VN`: signed mean IC `+0.03774`, t-stat `+6.48`, days `659`
- `JP`: signed mean IC `-0.00220`, t-stat `-0.26`, days `641`
- `US`: signed mean IC `+0.00204`, t-stat `+0.36`, days `663`

### `sector_ad_ratio`
- `VN`: signed mean IC `+0.02022`, t-stat `+3.30`, days `652`
- `JP`: signed mean IC `+0.02697`, t-stat `+1.26`, days `113`
- `US`: signed mean IC `-0.00955`, t-stat `-0.56`, days `157`

### `close_position`
- `VN`: signed mean IC `-0.00571`, t-stat `-0.80`, days `659`
- `JP`: signed mean IC `+0.01964`, t-stat `+2.13`, days `641`
- `US`: signed mean IC `+0.02142`, t-stat `+2.41`, days `663`

### `open_delta_1`
- `VN`: signed mean IC `+0.02163`, t-stat `+3.00`, days `659`
- `JP`: signed mean IC `+0.02515`, t-stat `+2.43`, days `641`
- `US`: signed mean IC `-0.01107`, t-stat `-1.06`, days `663`
