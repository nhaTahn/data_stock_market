# Portable LSTM Filter Signal Probe

Scope: train/validation only. Test/out-sample data is not used.

Architecture: base LSTM prediction -> small LSTM confidence filter -> generic market risk gate.

Market/risk context uses portable names such as `market_proxy_return_20`, `market_breadth_20`, and `market_proxy_drawdown_60`; market-index source columns are treated as aliases behind `market_proxy_return_1`.

Market proxy source used in this run: `vnindex_return`.
Train-selected gate threshold: `0.40`.
Train-selected daily top coverage by filter probability: `5.0%`.
Train-selected daily top coverage by expected move / rel_score: `5.0%`.
Train-selected daily top coverage by expected move / mean IC: `40.0%`.
Tradeable label rate: train `21.2%`, val `24.7%`.

## Metrics

| Split | Candidate | rel_score | Direction | Mean IC | IC t | Quartile equity | Hit rate | Obs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `gate` | +0.0008 | 17.0% | +0.0413 | +3.32 | 1.679 | 51.4% | 128605 |
| `train` | `gate_risk_scaled` | +0.0008 | 17.0% | +0.0414 | +3.32 | 1.679 | 51.4% | 128605 |
| `train` | `probability_scaled` | +0.0002 | 42.6% | +0.0444 | +11.89 | 112.468 | 59.5% | 128605 |
| `train` | `daily_top_train_selected_risk_scaled` | -0.0001 | 18.1% | +0.0173 | +5.04 | 3.347 | 52.9% | 128605 |
| `train` | `probability_risk_scaled` | -0.0004 | 42.6% | +0.0444 | +11.90 | 113.293 | 59.4% | 128605 |
| `train` | `daily_top_train_selected` | -0.0005 | 18.1% | +0.0173 | +5.04 | 3.347 | 52.9% | 128605 |
| `train` | `move_top_train_selected_risk_scaled` | -0.0017 | 18.0% | +0.0216 | +6.34 | 3.937 | 53.8% | 128605 |
| `train` | `daily_top_10` | -0.0021 | 19.5% | +0.0208 | +5.84 | 6.278 | 55.9% | 128605 |
| `train` | `daily_top_20` | -0.0026 | 22.5% | +0.0243 | +6.72 | 13.749 | 56.5% | 128605 |
| `train` | `move_top_train_selected` | -0.0029 | 18.0% | +0.0216 | +6.34 | 3.937 | 53.8% | 128605 |
| `train` | `move_top_10` | -0.0040 | 19.4% | +0.0277 | +8.03 | 7.822 | 55.4% | 128605 |
| `train` | `move_top_train_ic_selected_risk_scaled` | -0.0043 | 27.4% | +0.0392 | +11.00 | 52.829 | 59.0% | 128605 |
| `train` | `move_top_20` | -0.0047 | 22.1% | +0.0311 | +9.12 | 17.178 | 58.3% | 128605 |
| `train` | `move_top_train_ic_selected` | -0.0072 | 27.4% | +0.0392 | +11.00 | 53.247 | 59.0% | 128605 |
| `train` | `base` | -0.0085 | 42.6% | +0.0439 | +11.84 | 120.688 | 60.8% | 128605 |
| `val` | `daily_top_20` | +0.0056 | 15.9% | +0.0113 | +1.69 | 0.917 | 52.3% | 59445 |
| `val` | `gate` | +0.0056 | 8.7% | +0.0293 | +1.21 | 1.111 | 51.9% | 59445 |
| `val` | `move_top_20` | +0.0043 | 15.9% | +0.0194 | +2.92 | 1.175 | 52.8% | 59445 |
| `val` | `move_top_train_ic_selected` | +0.0034 | 23.9% | +0.0267 | +3.60 | 1.596 | 53.5% | 59445 |
| `val` | `daily_top_10` | +0.0033 | 11.9% | +0.0054 | +0.91 | 0.835 | 48.3% | 59445 |
| `val` | `probability_scaled` | +0.0032 | 46.7% | +0.0294 | +3.77 | 2.420 | 56.8% | 59445 |
| `val` | `move_top_10` | +0.0025 | 11.9% | +0.0192 | +3.19 | 1.067 | 51.2% | 59445 |
| `val` | `probability_risk_scaled` | +0.0018 | 46.7% | +0.0294 | +3.77 | 2.430 | 56.8% | 59445 |
| `val` | `daily_top_train_selected` | +0.0015 | 9.7% | +0.0064 | +1.19 | 0.773 | 49.1% | 59445 |
| `val` | `gate_risk_scaled` | +0.0014 | 8.7% | +0.0293 | +1.21 | 1.111 | 51.9% | 59445 |
| `val` | `move_top_train_selected` | +0.0013 | 9.7% | +0.0163 | +2.90 | 0.937 | 49.7% | 59445 |
| `val` | `base` | +0.0011 | 46.7% | +0.0297 | +3.82 | 2.290 | 56.0% | 59445 |
| `val` | `move_top_train_ic_selected_risk_scaled` | +0.0002 | 23.9% | +0.0267 | +3.61 | 1.599 | 53.5% | 59445 |
| `val` | `daily_top_train_selected_risk_scaled` | -0.0009 | 9.7% | +0.0064 | +1.19 | 0.773 | 49.1% | 59445 |
| `val` | `move_top_train_selected_risk_scaled` | -0.0009 | 9.7% | +0.0163 | +2.90 | 0.937 | 49.7% | 59445 |

## Gate Coverage

| Split | Gate coverage | Base hit rate | Active gate hit rate | Mean probability | P90 probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 1.5% | 44.2% | 52.7% | 0.220 | 0.303 |
| `val` | 2.8% | 48.0% | 50.8% | 0.251 | 0.333 |

## Candidate Coverage

| Split | Candidate | Coverage | Active hit rate |
| --- | --- | ---: | ---: |
| `train` | `gate` | 1.5% | 52.7% |
| `train` | `gate_risk_scaled` | 1.5% | 52.7% |
| `train` | `daily_top_train_selected` | 5.7% | 45.6% |
| `train` | `move_top_train_selected` | 5.7% | 45.8% |
| `train` | `daily_top_train_selected_risk_scaled` | 5.7% | 45.6% |
| `train` | `move_top_train_selected_risk_scaled` | 5.7% | 45.8% |
| `train` | `daily_top_10` | 10.7% | 45.5% |
| `train` | `move_top_10` | 10.7% | 45.7% |
| `train` | `daily_top_20` | 20.6% | 45.2% |
| `train` | `move_top_20` | 20.6% | 45.6% |
| `train` | `move_top_train_ic_selected` | 40.6% | 45.4% |
| `train` | `move_top_train_ic_selected_risk_scaled` | 40.6% | 45.4% |
| `train` | `base` | 100.0% | 44.2% |
| `train` | `probability_scaled` | 100.0% | 44.2% |
| `train` | `probability_risk_scaled` | 100.0% | 44.2% |
| `val` | `gate` | 2.8% | 50.8% |
| `val` | `gate_risk_scaled` | 2.8% | 50.8% |
| `val` | `daily_top_train_selected` | 5.5% | 49.9% |
| `val` | `move_top_train_selected` | 5.5% | 51.7% |
| `val` | `daily_top_train_selected_risk_scaled` | 5.5% | 49.9% |
| `val` | `move_top_train_selected_risk_scaled` | 5.5% | 51.7% |
| `val` | `daily_top_10` | 10.7% | 49.3% |
| `val` | `move_top_10` | 10.7% | 50.9% |
| `val` | `daily_top_20` | 20.5% | 49.3% |
| `val` | `move_top_20` | 20.5% | 49.6% |
| `val` | `move_top_train_ic_selected` | 40.5% | 49.3% |
| `val` | `move_top_train_ic_selected_risk_scaled` | 40.5% | 49.3% |
| `val` | `base` | 100.0% | 48.0% |
| `val` | `probability_scaled` | 100.0% | 48.0% |
| `val` | `probability_risk_scaled` | 100.0% | 48.0% |

## Read

- If validation improves only after hard gating, keep the filter as a post-model selection layer rather than retraining the base model.
- If expected-move daily selection beats hard gating with materially higher coverage, prefer it as the next router candidate.
- If probability scaling hurts `rel_score`, the filter is learning tradeability but not calibration; use it for position sizing or no-trade gates only.
- If naive risk scaling hurts, keep market regime context as input to the filter before using it as an output multiplier.
- This is a first probe, not final model selection. Threshold is selected on train only.
