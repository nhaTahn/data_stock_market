# Cross-Artifact Stability Summary

This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.
Risk-control pass requires worst-artifact max drawdown no worse than `25%` and max average turnover no higher than `0.20`.

## Risk-Controlled Stable Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_abstain_m10_dd20_t25` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_abstain_m10_dd15_t20` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_abstain_m10_dd10_t15` | 2 | 1.032 | 1.072 | +0.21 | -24.7% | 0.18 | Passes both artifacts and the risk-control screen. |

## Stable Non-Oracle Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_abstain_m10_dd20_t25` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m10_dd15_t20` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m10_dd10_t15` | 2 | 1.032 | 1.072 | +0.21 | -24.7% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25` | 2 | 1.174 | 1.266 | +0.57 | -29.5% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd15_t20` | 2 | 1.174 | 1.266 | +0.57 | -29.5% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m05_dd20_t25` | 2 | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best` | 2 | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m02_dd20_t25` | 2 | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m05_dd15_t20` | 2 | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m02_dd15_t20` | 2 | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only` | 2 | 1.010 | 1.212 | +0.16 | -36.3% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m05_dd10_t15` | 2 | 1.001 | 1.017 | +0.10 | -30.9% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m05_dd20_t25` | 2 | 1.001 | 1.195 | +0.10 | -30.9% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m05_dd15_t20` | 2 | 1.001 | 1.195 | +0.10 | -30.9% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m02_dd20_t25` | 2 | 1.001 | 1.107 | +0.10 | -30.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m02_dd15_t20` | 2 | 1.001 | 1.107 | +0.10 | -30.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_abstain_m10_dd20_t25` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | 2 | False |
| gate | `train_phase_abstain_m10_dd15_t20` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | 2 | False |
| gate | `train_phase_abstain_m10_dd10_t15` | 1.032 | 1.072 | +0.21 | -24.7% | 0.18 | 2 | False |
| gate | `train_phase_regularized_m10_dd20_t25` | 1.174 | 1.266 | +0.57 | -29.5% | 0.22 | 2 | False |
| gate | `train_phase_regularized_m10_dd15_t20` | 1.174 | 1.266 | +0.57 | -29.5% | 0.22 | 2 | False |
| gate | `train_phase_regularized_m05_dd20_t25` | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | 2 | False |
| gate | `train_phase_best` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | 2 | False |
| gate | `train_phase_regularized_m02_dd20_t25` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | 2 | False |
| gate | `train_phase_regularized_m05_dd15_t20` | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | 2 | False |
| gate | `train_phase_regularized_m02_dd15_t20` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | 2 | False |
| gate | `vn_h1_distribution_only` | 1.010 | 1.212 | +0.16 | -36.3% | 0.24 | 2 | False |
| gate | `train_phase_abstain_m05_dd10_t15` | 1.001 | 1.017 | +0.10 | -30.9% | 0.19 | 2 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.873 | -0.16 | -35.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 0.838 | -0.50 | -32.9% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 0.816 | -0.47 | -30.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 0.712 | -0.73 | -35.6% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 0.712 | -0.73 | -35.6% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 0.704 | -0.69 | -42.9% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 1.240 | +0.76 | -23.3% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd10_t15` | 1.174 | +0.57 | -29.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 1.174 | +0.57 | -29.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd15_t20` | 1.174 | +0.57 | -29.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m10_dd10_t15` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m10_dd20_t25` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m10_dd15_t20` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m02_dd20_t25` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m05_dd20_t25` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m02_dd10_t15` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m05_dd10_t15` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m02_dd15_t20` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m05_dd15_t20` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 1.010 | +0.16 | -36.3% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m02_dd10_t15` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m05_dd10_t15` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m02_dd20_t25` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m05_dd20_t25` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m02_dd15_t20` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m05_dd15_t20` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 0.943 | -0.04 | -25.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 0.838 | -0.50 | -32.9% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_transition` | 0.740 | -0.65 | -32.7% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 0.739 | -0.55 | -41.8% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 0.739 | -0.55 | -41.8% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 0.712 | -0.73 | -35.6% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee` | 0.704 | -0.69 | -42.9% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `aggressive_prior` | 0.698 | -0.81 | -36.2% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_markup_distribution_transition` | 0.698 | -0.81 | -36.2% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else` | 0.676 | -0.77 | -45.2% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 1.401 | +0.91 | -31.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 1.401 | +0.91 | -31.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 1.306 | +0.73 | -25.3% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 1.266 | +0.73 | -24.8% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 1.211 | +0.62 | -28.7% | 0.13 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.082 | +0.33 | -31.7% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 1.807 | +1.47 | -24.8% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 1.807 | +1.47 | -24.8% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 1.586 | +1.18 | -26.0% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.551 | +1.12 | -23.4% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.488 | +1.03 | -23.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.488 | +1.03 | -23.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 1.462 | +1.03 | -26.8% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.414 | +0.92 | -23.9% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 1.401 | +0.91 | -31.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m05_dd20_t25` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m10_dd20_t25` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m05_dd15_t20` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m10_dd15_t20` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m05_dd20_t25` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m05_dd15_t20` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd15_t20` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 1.306 | +0.73 | -25.3% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 1.266 | +0.73 | -24.8% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 1.245 | +0.64 | -25.2% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 1.213 | +0.58 | -28.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m02_dd20_t25` | 1.213 | +0.58 | -28.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m02_dd15_t20` | 1.213 | +0.58 | -28.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m02_dd20_t25` | 1.213 | +0.58 | -28.6% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m02_dd15_t20` | 1.213 | +0.58 | -28.6% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m05_dd10_t15` | 1.032 | +0.21 | -24.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m10_dd10_t15` | 1.032 | +0.21 | -24.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m05_dd10_t15` | 0.968 | +0.06 | -32.3% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd10_t15` | 0.968 | +0.06 | -32.3% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m02_dd10_t15` | 0.901 | -0.22 | -34.5% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m02_dd10_t15` | 0.862 | -0.20 | -37.1% | 0.28 |
