# VN30 Proxy vs q90 Absolute Error Diagnostic

Scope: train/in-sample only. Holdout/test is not used.

Formula:

```text
E_d = actual_return_{i,d} - predicted_return_{i,d}
ts_error(d) = Q_0.90(|E_d|)
```

Read: the raw time-series plot is noisy, so the key question is whether `ts_error(d)` is higher in unstable market states.

## Main Findings

- Direction alone is weak: corr(`index_proxy_return`, `q90(|E|)`) = `-0.060`.
- Market movement size matters: corr(`|index_proxy_return|`, `q90(|E|)`) = `+0.308`.
- Market volatility matters more: corr(`vol20`, `q90(|E|)`) = `+0.444`.
- Drawdown has a weaker negative relation: corr(`drawdown`, `q90(|E|)`) = `-0.109`.

## Regime Summary

| Regime | Days | Median q90(|E|) | Mean q90(|E|) | P90 q90(|E|) |
|---|---:|---:|---:|---:|
| Down 20% return days | 392 | 2.67% | 3.08% | 5.19% |
| Middle 60% return days | 1175 | 2.27% | 2.54% | 4.06% |
| Up 20% return days | 392 | 2.76% | 3.04% | 4.97% |
| High vol20 top 20% | 390 | 3.35% | 3.70% | 6.26% |
| Not high vol20 | 1569 | 2.26% | 2.51% | 4.02% |
| High absolute market-return top 20% | 392 | 3.03% | 3.38% | 5.46% |
| Not high absolute market-return | 1567 | 2.32% | 2.59% | 4.11% |
| Drawdown greater than 10% | 775 | 2.49% | 2.92% | 4.89% |
| Not drawdown greater than 10% | 1184 | 2.40% | 2.63% | 4.38% |

## Interpretation

The Base LSTM error is regime-dependent. The model is not only worse on down days; it is worse when market movement is large or market volatility is high. This supports keeping the Base LSTM as the forecasting layer, while adding a filter/risk layer before using the signal for trading.

