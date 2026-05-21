# Multi-Market Prediction Router

Selection uses train split only. Validation metrics are reported for comparison and do not drive the routers.

## Validation Overall

| Candidate | rel_score | Direction | Daily IC | IC t | Quartile equity | Panels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `marketplus_signmag_top2` | +0.00355 | 49.82% | +0.0372 | +3.90 | 2.090 | 208 |
| `marketplus_signmag_ensemble` | +0.00341 | 49.81% | +0.0403 | +4.18 | 1.843 | 208 |
| `compact_signmag_top2` | +0.00335 | 49.99% | +0.0307 | +3.15 | 1.480 | 208 |
| `router_train_rel_score` | +0.00283 | 49.56% | +0.0212 | +2.12 | 0.798 | 208 |
| `signal_signmag_ensemble` | +0.00262 | 49.78% | +0.0322 | +3.21 | 1.120 | 208 |
| `signal_signmag_top2` | +0.00252 | 49.71% | +0.0331 | +3.19 | 1.020 | 208 |
| `router_train_daily_ic` | +0.00221 | 49.65% | +0.0352 | +3.75 | 1.772 | 208 |
| `router_train_quartile_equity` | +0.00221 | 49.65% | +0.0352 | +3.75 | 1.772 | 208 |
| `compact_lstm_top2` | +0.00188 | 49.49% | +0.0225 | +2.43 | 0.907 | 208 |
| `signal_lstm_top2` | +0.00161 | 49.41% | +0.0324 | +3.59 | 1.474 | 208 |

## Train-Selected Router Choices

- `router_train_daily_ic`: JP: signal_lstm_top2, US: compact_lstm_top2, VN: signal_lstm_top2
- `router_train_quartile_equity`: JP: signal_lstm_top2, US: compact_lstm_top2, VN: signal_lstm_top2
- `router_train_rel_score`: JP: compact_lstm_top2, US: compact_lstm_top2, VN: signal_signmag_ensemble

## Best Validation By Market

### VN
- `router_train_daily_ic`: rel_score `+0.00124`, daily IC `+0.0480`, quartile equity `2.174`
- `router_train_quartile_equity`: rel_score `+0.00124`, daily IC `+0.0480`, quartile equity `2.174`
- `signal_lstm_top2`: rel_score `+0.00124`, daily IC `+0.0480`, quartile equity `2.174`
- `compact_lstm_top2`: rel_score `+0.00258`, daily IC `+0.0337`, quartile equity `0.885`
- `marketplus_signmag_top2`: rel_score `+0.00484`, daily IC `+0.0281`, quartile equity `1.100`

### JP
- `signal_signmag_top2`: rel_score `+0.00072`, daily IC `+0.0225`, quartile equity `1.025`
- `signal_signmag_ensemble`: rel_score `-0.00005`, daily IC `+0.0222`, quartile equity `1.163`
- `marketplus_signmag_ensemble`: rel_score `+0.00035`, daily IC `+0.0198`, quartile equity `1.231`
- `compact_lstm_top2`: rel_score `-0.00135`, daily IC `+0.0161`, quartile equity `1.425`
- `router_train_rel_score`: rel_score `-0.00135`, daily IC `+0.0161`, quartile equity `1.425`

### US
- `signal_signmag_top2`: rel_score `+0.00234`, daily IC `+0.0226`, quartile equity `1.551`
- `signal_signmag_ensemble`: rel_score `+0.00246`, daily IC `+0.0206`, quartile equity `1.667`
- `marketplus_signmag_ensemble`: rel_score `+0.00331`, daily IC `+0.0199`, quartile equity `1.825`
- `marketplus_signmag_top2`: rel_score `+0.00285`, daily IC `+0.0179`, quartile equity `1.748`
- `compact_lstm_top2`: rel_score `+0.00294`, daily IC `+0.0136`, quartile equity `1.821`
