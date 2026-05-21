# Portable LSTM Filter Signal Probe

Scope: train/validation only. Test/out-sample data is not used.

Architecture: base LSTM prediction -> small LSTM confidence filter -> generic market risk gate.

Market/risk context uses portable names such as `market_proxy_return_20`, `market_breadth_20`, and `market_proxy_drawdown_60`; market-index source columns are treated as aliases behind `market_proxy_return_1`.

Market proxy source used in this run: `vnindex_return`.
Train-selected gate threshold: `0.35`.
Train-selected daily top coverage by filter probability: `3.0%`.
Train-selected daily top coverage by expected move / rel_score: `3.0%`.
Train-selected daily top coverage by expected move / mean IC: `40.0%`.
Tradeable label rate: train `20.7%`, val `24.3%`.

## Metrics

| Split | Candidate | rel_score | Direction | Mean IC | IC t | Quartile equity | Hit rate | Obs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `gate` | +0.0018 | 17.4% | +0.0533 | +4.70 | 2.202 | 56.9% | 128605 |
| `train` | `gate_risk_scaled` | +0.0008 | 17.4% | +0.0533 | +4.70 | 2.202 | 56.9% | 128605 |
| `train` | `daily_top_train_selected` | +0.0008 | 17.6% | +0.0241 | +6.75 | 2.398 | 51.6% | 128605 |
| `train` | `daily_top_train_selected_risk_scaled` | +0.0007 | 17.6% | +0.0241 | +6.76 | 2.398 | 51.6% | 128605 |
| `train` | `probability_scaled` | +0.0006 | 42.7% | +0.0386 | +9.94 | 80.297 | 58.7% | 128605 |
| `train` | `probability_risk_scaled` | +0.0005 | 42.7% | +0.0385 | +9.93 | 78.822 | 58.7% | 128605 |
| `train` | `daily_top_10` | +0.0004 | 19.8% | +0.0263 | +7.58 | 6.575 | 55.0% | 128605 |
| `train` | `move_top_train_selected_risk_scaled` | -0.0000 | 17.5% | +0.0172 | +4.78 | 2.086 | 52.3% | 128605 |
| `train` | `move_top_train_selected` | -0.0001 | 17.5% | +0.0172 | +4.78 | 2.086 | 52.3% | 128605 |
| `train` | `daily_top_20` | -0.0003 | 22.9% | +0.0277 | +7.42 | 16.705 | 56.8% | 128605 |
| `train` | `move_top_20` | -0.0004 | 22.8% | +0.0289 | +7.90 | 20.220 | 57.0% | 128605 |
| `train` | `move_top_10` | -0.0007 | 19.7% | +0.0200 | +5.54 | 4.316 | 54.0% | 128605 |
| `train` | `move_top_train_ic_selected_risk_scaled` | -0.0019 | 28.0% | +0.0361 | +9.50 | 59.492 | 59.2% | 128605 |
| `train` | `move_top_train_ic_selected` | -0.0028 | 28.0% | +0.0361 | +9.50 | 59.707 | 59.2% | 128605 |
| `train` | `base` | -0.0045 | 42.7% | +0.0373 | +9.67 | 73.983 | 58.6% | 128605 |
| `val` | `move_top_train_ic_selected` | +0.0077 | 24.4% | +0.0220 | +2.86 | 1.958 | 56.0% | 59445 |
| `val` | `move_top_20` | +0.0052 | 16.0% | +0.0200 | +2.91 | 1.370 | 54.0% | 59445 |
| `val` | `gate` | +0.0050 | 9.4% | +0.0218 | +1.35 | 1.202 | 50.3% | 59445 |
| `val` | `base` | +0.0049 | 47.1% | +0.0242 | +2.97 | 2.084 | 58.2% | 59445 |
| `val` | `daily_top_20` | +0.0044 | 16.2% | +0.0181 | +2.73 | 1.568 | 51.2% | 59445 |
| `val` | `move_top_10` | +0.0037 | 12.0% | +0.0161 | +2.65 | 1.063 | 51.5% | 59445 |
| `val` | `daily_top_10` | +0.0035 | 12.0% | +0.0109 | +1.85 | 1.009 | 50.3% | 59445 |
| `val` | `move_top_train_ic_selected_risk_scaled` | +0.0029 | 24.4% | +0.0220 | +2.86 | 1.957 | 56.0% | 59445 |
| `val` | `probability_risk_scaled` | +0.0020 | 47.1% | +0.0244 | +2.97 | 2.126 | 57.6% | 59445 |
| `val` | `probability_scaled` | +0.0017 | 47.1% | +0.0244 | +2.97 | 2.127 | 57.6% | 59445 |
| `val` | `gate_risk_scaled` | +0.0012 | 9.4% | +0.0218 | +1.35 | 1.202 | 50.3% | 59445 |
| `val` | `daily_top_train_selected` | +0.0010 | 8.8% | -0.0003 | -0.07 | 0.702 | 47.7% | 59445 |
| `val` | `move_top_train_selected` | +0.0009 | 8.8% | +0.0065 | +1.23 | 0.789 | 49.4% | 59445 |
| `val` | `daily_top_train_selected_risk_scaled` | +0.0003 | 8.8% | -0.0004 | -0.07 | 0.702 | 47.7% | 59445 |
| `val` | `move_top_train_selected_risk_scaled` | -0.0003 | 8.8% | +0.0065 | +1.22 | 0.789 | 49.4% | 59445 |

## Gate Coverage

| Split | Gate coverage | Base hit rate | Active gate hit rate | Mean probability | P90 probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 2.3% | 43.7% | 53.9% | 0.206 | 0.288 |
| `val` | 4.3% | 47.7% | 53.6% | 0.241 | 0.317 |

## Candidate Coverage

| Split | Candidate | Coverage | Active hit rate |
| --- | --- | ---: | ---: |
| `train` | `gate` | 2.3% | 53.9% |
| `train` | `gate_risk_scaled` | 2.3% | 53.9% |
| `train` | `daily_top_train_selected` | 3.6% | 49.2% |
| `train` | `move_top_train_selected` | 3.6% | 46.8% |
| `train` | `daily_top_train_selected_risk_scaled` | 3.6% | 49.2% |
| `train` | `move_top_train_selected_risk_scaled` | 3.6% | 46.8% |
| `train` | `daily_top_10` | 10.7% | 46.9% |
| `train` | `move_top_10` | 10.7% | 46.2% |
| `train` | `daily_top_20` | 20.6% | 46.0% |
| `train` | `move_top_20` | 20.6% | 45.9% |
| `train` | `move_top_train_ic_selected` | 40.6% | 45.5% |
| `train` | `move_top_train_ic_selected_risk_scaled` | 40.6% | 45.5% |
| `train` | `base` | 100.0% | 43.7% |
| `train` | `probability_scaled` | 100.0% | 43.7% |
| `train` | `probability_risk_scaled` | 100.0% | 43.7% |
| `val` | `daily_top_train_selected` | 3.3% | 48.1% |
| `val` | `move_top_train_selected` | 3.3% | 49.9% |
| `val` | `daily_top_train_selected_risk_scaled` | 3.3% | 48.1% |
| `val` | `move_top_train_selected_risk_scaled` | 3.3% | 49.9% |
| `val` | `gate` | 4.3% | 53.6% |
| `val` | `gate_risk_scaled` | 4.3% | 53.6% |
| `val` | `daily_top_10` | 10.7% | 49.5% |
| `val` | `move_top_10` | 10.7% | 49.8% |
| `val` | `daily_top_20` | 20.5% | 49.8% |
| `val` | `move_top_20` | 20.5% | 49.7% |
| `val` | `move_top_train_ic_selected` | 40.5% | 48.8% |
| `val` | `move_top_train_ic_selected_risk_scaled` | 40.5% | 48.8% |
| `val` | `base` | 100.0% | 47.7% |
| `val` | `probability_scaled` | 100.0% | 47.7% |
| `val` | `probability_risk_scaled` | 100.0% | 47.7% |

## Read

- If validation improves only after hard gating, keep the filter as a post-model selection layer rather than retraining the base model.
- If expected-move daily selection beats hard gating with materially higher coverage, prefer it as the next router candidate.
- If probability scaling hurts `rel_score`, the filter is learning tradeability but not calibration; use it for position sizing or no-trade gates only.
- If naive risk scaling hurts, keep market regime context as input to the filter before using it as an output multiplier.
- This is a first probe, not final model selection. Threshold is selected on train only.
