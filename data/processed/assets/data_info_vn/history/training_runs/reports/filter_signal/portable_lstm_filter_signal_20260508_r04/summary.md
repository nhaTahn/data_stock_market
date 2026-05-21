# Portable LSTM Filter Signal Probe

Scope: train/validation only. Test/out-sample data is not used.

Architecture: base LSTM prediction -> small LSTM confidence filter -> generic market risk gate.

Market/risk context uses portable names such as `market_proxy_return_20`, `market_breadth_20`, and `market_proxy_drawdown_60`; market-index source columns are treated as aliases behind `market_proxy_return_1`.

Market proxy source used in this run: `vnindex_return`.
Train-selected gate threshold: `0.35`.
Train-selected daily top coverage by filter probability: `10.0%`.
Train-selected daily top coverage by expected move / rel_score: `5.0%`.
Train-selected daily top coverage by expected move / mean IC: `40.0%`.
Tradeable label rate: train `20.7%`, val `24.3%`.

## Metrics

| Split | Candidate | rel_score | Direction | Mean IC | IC t | Quartile equity | Hit rate | Obs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train` | `gate` | +0.0012 | 17.8% | +0.0289 | +3.41 | 2.724 | 54.7% | 128605 |
| `train` | `gate_risk_scaled` | +0.0008 | 17.8% | +0.0289 | +3.42 | 2.724 | 54.7% | 128605 |
| `train` | `probability_risk_scaled` | +0.0006 | 42.7% | +0.0379 | +9.78 | 76.503 | 59.0% | 128605 |
| `train` | `daily_top_train_selected_risk_scaled` | +0.0006 | 19.7% | +0.0185 | +5.16 | 4.738 | 54.3% | 128605 |
| `train` | `probability_scaled` | +0.0003 | 42.7% | +0.0379 | +9.78 | 76.764 | 59.0% | 128605 |
| `train` | `daily_top_10` | +0.0002 | 19.7% | +0.0185 | +5.16 | 4.738 | 54.3% | 128605 |
| `train` | `daily_top_train_selected` | +0.0002 | 19.7% | +0.0185 | +5.16 | 4.738 | 54.3% | 128605 |
| `train` | `move_top_train_selected` | +0.0000 | 18.2% | +0.0164 | +4.73 | 2.348 | 52.8% | 128605 |
| `train` | `move_top_train_selected_risk_scaled` | -0.0002 | 18.2% | +0.0164 | +4.72 | 2.348 | 52.8% | 128605 |
| `train` | `daily_top_20` | -0.0004 | 22.7% | +0.0225 | +6.08 | 12.322 | 56.2% | 128605 |
| `train` | `move_top_20` | -0.0006 | 22.7% | +0.0270 | +7.32 | 16.364 | 55.9% | 128605 |
| `train` | `move_top_10` | -0.0007 | 19.7% | +0.0209 | +5.79 | 4.885 | 53.4% | 128605 |
| `train` | `move_top_train_ic_selected_risk_scaled` | -0.0022 | 28.0% | +0.0345 | +9.10 | 59.985 | 59.5% | 128605 |
| `train` | `move_top_train_ic_selected` | -0.0032 | 28.0% | +0.0345 | +9.10 | 60.029 | 59.5% | 128605 |
| `train` | `base` | -0.0045 | 42.7% | +0.0373 | +9.67 | 74.074 | 58.6% | 128605 |
| `val` | `move_top_train_ic_selected` | +0.0072 | 24.2% | +0.0238 | +3.05 | 2.142 | 56.6% | 59445 |
| `val` | `gate` | +0.0068 | 10.6% | +0.0231 | +1.39 | 1.078 | 53.2% | 59445 |
| `val` | `move_top_20` | +0.0066 | 16.1% | +0.0201 | +2.94 | 1.409 | 53.2% | 59445 |
| `val` | `base` | +0.0049 | 47.1% | +0.0242 | +2.97 | 2.084 | 58.2% | 59445 |
| `val` | `daily_top_20` | +0.0039 | 16.1% | +0.0109 | +1.55 | 1.128 | 51.2% | 59445 |
| `val` | `move_top_10` | +0.0038 | 12.0% | +0.0180 | +2.94 | 1.171 | 52.3% | 59445 |
| `val` | `daily_top_10` | +0.0033 | 12.1% | +0.0119 | +1.97 | 0.921 | 49.5% | 59445 |
| `val` | `daily_top_train_selected` | +0.0033 | 12.1% | +0.0119 | +1.97 | 0.921 | 49.5% | 59445 |
| `val` | `move_top_train_ic_selected_risk_scaled` | +0.0017 | 24.2% | +0.0237 | +3.05 | 2.142 | 56.6% | 59445 |
| `val` | `probability_scaled` | +0.0017 | 47.1% | +0.0242 | +2.95 | 2.093 | 57.1% | 59445 |
| `val` | `probability_risk_scaled` | +0.0017 | 47.1% | +0.0242 | +2.95 | 2.092 | 57.1% | 59445 |
| `val` | `gate_risk_scaled` | +0.0014 | 10.6% | +0.0231 | +1.39 | 1.078 | 53.2% | 59445 |
| `val` | `daily_top_train_selected_risk_scaled` | +0.0011 | 12.1% | +0.0119 | +1.96 | 0.921 | 49.5% | 59445 |
| `val` | `move_top_train_selected` | +0.0011 | 9.8% | +0.0131 | +2.31 | 0.972 | 51.5% | 59445 |
| `val` | `move_top_train_selected_risk_scaled` | +0.0001 | 9.8% | +0.0131 | +2.31 | 0.972 | 51.5% | 59445 |

## Gate Coverage

| Split | Gate coverage | Base hit rate | Active gate hit rate | Mean probability | P90 probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 4.1% | 43.7% | 49.0% | 0.222 | 0.308 |
| `val` | 6.9% | 47.7% | 50.9% | 0.252 | 0.331 |

## Candidate Coverage

| Split | Candidate | Coverage | Active hit rate |
| --- | --- | ---: | ---: |
| `train` | `gate` | 4.1% | 49.0% |
| `train` | `gate_risk_scaled` | 4.1% | 49.0% |
| `train` | `move_top_train_selected` | 5.7% | 45.9% |
| `train` | `move_top_train_selected_risk_scaled` | 5.7% | 45.9% |
| `train` | `daily_top_10` | 10.7% | 45.7% |
| `train` | `daily_top_train_selected` | 10.7% | 45.7% |
| `train` | `move_top_10` | 10.7% | 46.1% |
| `train` | `daily_top_train_selected_risk_scaled` | 10.7% | 45.7% |
| `train` | `daily_top_20` | 20.6% | 45.3% |
| `train` | `move_top_20` | 20.6% | 45.8% |
| `train` | `move_top_train_ic_selected` | 40.6% | 45.3% |
| `train` | `move_top_train_ic_selected_risk_scaled` | 40.6% | 45.3% |
| `train` | `base` | 100.0% | 43.7% |
| `train` | `probability_scaled` | 100.0% | 43.7% |
| `train` | `probability_risk_scaled` | 100.0% | 43.7% |
| `val` | `move_top_train_selected` | 5.5% | 50.2% |
| `val` | `move_top_train_selected_risk_scaled` | 5.5% | 50.2% |
| `val` | `gate` | 6.9% | 50.9% |
| `val` | `gate_risk_scaled` | 6.9% | 50.9% |
| `val` | `daily_top_10` | 10.7% | 49.4% |
| `val` | `daily_top_train_selected` | 10.7% | 49.4% |
| `val` | `move_top_10` | 10.7% | 49.9% |
| `val` | `daily_top_train_selected_risk_scaled` | 10.7% | 49.4% |
| `val` | `daily_top_20` | 20.5% | 49.0% |
| `val` | `move_top_20` | 20.5% | 49.8% |
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
