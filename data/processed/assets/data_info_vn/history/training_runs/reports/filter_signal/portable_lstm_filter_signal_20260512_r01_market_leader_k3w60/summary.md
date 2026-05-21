# Portable LSTM Filter Signal Probe

Scope: train/validation only. Test/out-sample data is not used.

Architecture: base LSTM prediction -> small LSTM confidence filter -> generic market risk gate.

Market/risk context uses portable names such as `market_proxy_return_20`, `market_breadth_20`, and `market_proxy_drawdown_60`; market-index source columns are treated as aliases behind `market_proxy_return_1`.

Market proxy source used in this run: `vnindex_return`.
Train-selected gate threshold: `0.35`.
Train-selected daily top coverage by filter probability: `5.0%`.
Train-selected daily top coverage by expected move / rel_score: `15.0%`.
Train-selected daily top coverage by expected move / mean IC: `40.0%`.
Tradeable label rate: train `20.7%`, val `24.3%`.

## Metrics

| Split | Candidate | rel_score | Direction | Mean IC | IC t | Quartile equity | Hit rate | Obs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `gate` | +0.0016 | 17.8% | +0.0547 | +5.99 | 2.850 | 57.6% | 128605 |
| `train` | `gate_risk_scaled` | +0.0008 | 17.8% | +0.0547 | +6.00 | 2.836 | 57.6% | 128605 |
| `train` | `daily_top_train_selected` | +0.0008 | 18.3% | +0.0231 | +6.60 | 3.057 | 52.8% | 128605 |
| `train` | `daily_top_train_selected_risk_scaled` | +0.0005 | 18.3% | +0.0231 | +6.60 | 3.057 | 52.8% | 128605 |
| `train` | `probability_risk_scaled` | +0.0005 | 42.7% | +0.0382 | +9.87 | 90.041 | 59.5% | 128605 |
| `train` | `probability_scaled` | +0.0003 | 42.7% | +0.0383 | +9.88 | 91.795 | 59.6% | 128605 |
| `train` | `daily_top_20` | -0.0001 | 22.9% | +0.0287 | +7.83 | 21.970 | 57.6% | 128605 |
| `train` | `move_top_train_selected` | -0.0001 | 21.3% | +0.0256 | +7.05 | 9.890 | 55.9% | 128605 |
| `train` | `daily_top_10` | -0.0001 | 19.8% | +0.0245 | +6.82 | 6.290 | 54.6% | 128605 |
| `train` | `move_top_train_selected_risk_scaled` | -0.0004 | 21.3% | +0.0256 | +7.06 | 9.890 | 55.9% | 128605 |
| `train` | `move_top_10` | -0.0008 | 19.7% | +0.0219 | +6.11 | 5.831 | 54.3% | 128605 |
| `train` | `move_top_20` | -0.0010 | 22.7% | +0.0296 | +8.01 | 20.260 | 56.5% | 128605 |
| `train` | `move_top_train_ic_selected_risk_scaled` | -0.0019 | 28.0% | +0.0352 | +9.29 | 58.691 | 59.5% | 128605 |
| `train` | `move_top_train_ic_selected` | -0.0029 | 28.0% | +0.0352 | +9.29 | 58.402 | 59.5% | 128605 |
| `train` | `base` | -0.0045 | 42.7% | +0.0373 | +9.67 | 73.983 | 58.6% | 128605 |
| `val` | `move_top_train_ic_selected` | +0.0073 | 24.4% | +0.0233 | +3.01 | 2.030 | 56.2% | 59445 |
| `val` | `move_top_20` | +0.0058 | 16.1% | +0.0205 | +3.03 | 1.360 | 53.2% | 59445 |
| `val` | `gate` | +0.0056 | 11.6% | +0.0042 | +0.22 | 0.906 | 48.6% | 59445 |
| `val` | `base` | +0.0049 | 47.1% | +0.0242 | +2.97 | 2.084 | 58.2% | 59445 |
| `val` | `move_top_train_selected` | +0.0046 | 13.9% | +0.0209 | +3.21 | 1.422 | 54.5% | 59445 |
| `val` | `move_top_10` | +0.0040 | 12.1% | +0.0205 | +3.34 | 1.142 | 53.2% | 59445 |
| `val` | `daily_top_20` | +0.0037 | 16.2% | +0.0163 | +2.46 | 1.300 | 49.2% | 59445 |
| `val` | `daily_top_10` | +0.0021 | 12.1% | +0.0119 | +1.95 | 1.080 | 48.6% | 59445 |
| `val` | `probability_risk_scaled` | +0.0020 | 47.1% | +0.0241 | +2.95 | 1.992 | 57.7% | 59445 |
| `val` | `move_top_train_ic_selected_risk_scaled` | +0.0018 | 24.4% | +0.0233 | +3.01 | 2.029 | 56.2% | 59445 |
| `val` | `daily_top_train_selected` | +0.0015 | 9.8% | +0.0121 | +2.14 | 0.923 | 47.7% | 59445 |
| `val` | `gate_risk_scaled` | +0.0014 | 11.6% | +0.0042 | +0.22 | 0.906 | 48.6% | 59445 |
| `val` | `probability_scaled` | +0.0012 | 47.1% | +0.0241 | +2.95 | 2.005 | 57.9% | 59445 |
| `val` | `move_top_train_selected_risk_scaled` | +0.0007 | 13.9% | +0.0209 | +3.21 | 1.422 | 54.5% | 59445 |
| `val` | `daily_top_train_selected_risk_scaled` | +0.0004 | 9.8% | +0.0121 | +2.13 | 0.923 | 47.7% | 59445 |

## Gate Coverage

| Split | Gate coverage | Base hit rate | Active gate hit rate | Mean probability | P90 probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 4.0% | 43.7% | 53.2% | 0.218 | 0.301 |
| `val` | 9.8% | 47.7% | 50.2% | 0.254 | 0.348 |

## Candidate Coverage

| Split | Candidate | Coverage | Active hit rate |
| --- | --- | ---: | ---: |
| `train` | `gate` | 4.0% | 53.2% |
| `train` | `gate_risk_scaled` | 4.0% | 53.2% |
| `train` | `daily_top_train_selected` | 5.7% | 47.3% |
| `train` | `daily_top_train_selected_risk_scaled` | 5.7% | 47.3% |
| `train` | `daily_top_10` | 10.7% | 46.7% |
| `train` | `move_top_10` | 10.7% | 46.2% |
| `train` | `move_top_train_selected` | 15.8% | 46.0% |
| `train` | `move_top_train_selected_risk_scaled` | 15.8% | 46.0% |
| `train` | `daily_top_20` | 20.6% | 46.1% |
| `train` | `move_top_20` | 20.6% | 46.0% |
| `train` | `move_top_train_ic_selected` | 40.6% | 45.4% |
| `train` | `move_top_train_ic_selected_risk_scaled` | 40.6% | 45.4% |
| `train` | `base` | 100.0% | 43.7% |
| `train` | `probability_scaled` | 100.0% | 43.7% |
| `train` | `probability_risk_scaled` | 100.0% | 43.7% |
| `val` | `daily_top_train_selected` | 5.5% | 50.0% |
| `val` | `daily_top_train_selected_risk_scaled` | 5.5% | 50.0% |
| `val` | `gate` | 9.8% | 50.2% |
| `val` | `gate_risk_scaled` | 9.8% | 50.2% |
| `val` | `daily_top_10` | 10.7% | 49.3% |
| `val` | `move_top_10` | 10.7% | 50.2% |
| `val` | `move_top_train_selected` | 15.2% | 50.1% |
| `val` | `move_top_train_selected_risk_scaled` | 15.2% | 50.1% |
| `val` | `daily_top_20` | 20.5% | 49.4% |
| `val` | `move_top_20` | 20.5% | 49.6% |
| `val` | `move_top_train_ic_selected` | 40.5% | 48.9% |
| `val` | `move_top_train_ic_selected_risk_scaled` | 40.5% | 48.9% |
| `val` | `base` | 100.0% | 47.7% |
| `val` | `probability_scaled` | 100.0% | 47.7% |
| `val` | `probability_risk_scaled` | 100.0% | 47.7% |

## Read

- If validation improves only after hard gating, keep the filter as a post-model selection layer rather than retraining the base model.
- If expected-move daily selection beats hard gating with materially higher coverage, prefer it as the next router candidate.
- If probability scaling hurts `rel_score`, the filter is learning tradeability but not calibration; use it for position sizing or no-trade gates only.
- If naive risk scaling hurts, keep market regime context as input to the filter before using it as an output multiplier.
- This is a first probe, not final model selection. Threshold is selected on train only.
