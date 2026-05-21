# Cross-Artifact Stability Summary

This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.
Risk-control pass requires worst-artifact max drawdown no worse than `25%` and max average turnover no higher than `0.20`.

## Risk-Controlled Stable Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.344 | 1.569 | +1.73 | -22.6% | 0.15 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.344 | 1.423 | +2.04 | -5.3% | 0.13 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.257 | 1.470 | +1.58 | -22.6% | 0.14 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.257 | 1.333 | +1.73 | -5.3% | 0.11 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 2 | 1.156 | 1.269 | +0.91 | -24.6% | 0.14 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 2 | 1.156 | 1.156 | +0.81 | -16.1% | 0.13 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 2 | 1.111 | 1.144 | +0.55 | -23.9% | 0.15 | Passes both artifacts and the risk-control screen. |
| gate | `train_phase_abstain_m10_dd20_t25` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts and the risk-control screen. |
| gate | `conservative_prior_transition_pressure_nonneg` | 2 | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | Passes both artifacts and the risk-control screen. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 2 | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | Passes both artifacts and the risk-control screen. |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 2 | 1.028 | 1.374 | +0.20 | -23.4% | 0.19 | Passes both artifacts and the risk-control screen. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 2 | 1.028 | 1.374 | +0.20 | -23.4% | 0.19 | Passes both artifacts and the risk-control screen. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 2 | 1.017 | 1.328 | +0.16 | -24.4% | 0.15 | Passes both artifacts and the risk-control screen. |

## Stable Non-Oracle Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.344 | 1.569 | +1.73 | -22.6% | 0.15 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.344 | 1.423 | +2.04 | -5.3% | 0.13 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.257 | 1.470 | +1.58 | -22.6% | 0.14 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.257 | 1.333 | +1.73 | -5.3% | 0.11 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 2 | 1.156 | 1.269 | +0.91 | -24.6% | 0.14 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 2 | 1.156 | 1.156 | +0.81 | -16.1% | 0.13 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 2 | 1.111 | 1.144 | +0.55 | -23.9% | 0.15 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m10_dd20_t25` | 2 | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 2 | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 2 | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 2 | 1.028 | 1.374 | +0.20 | -23.4% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 2 | 1.028 | 1.374 | +0.20 | -23.4% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 2 | 1.017 | 1.328 | +0.16 | -24.4% | 0.15 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 2 | 1.417 | 1.584 | +1.21 | -22.6% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 2 | 1.401 | 1.409 | +1.06 | -24.7% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 2 | 1.322 | 1.480 | +1.01 | -22.6% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 2 | 1.310 | 1.316 | +0.89 | -24.7% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best_transition_pressure_nonneg` | 2 | 1.301 | 1.434 | +0.91 | -22.6% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 2 | 1.289 | 1.557 | +0.87 | -22.6% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best_transition_mr20_pressure_nonneg` | 2 | 1.215 | 1.340 | +0.71 | -22.6% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 2 | 1.215 | 1.281 | +0.71 | -24.5% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 2 | 1.202 | 1.453 | +0.67 | -22.6% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25` | 2 | 1.174 | 1.266 | +0.57 | -29.5% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best_transition_mr20_nonneg` | 2 | 1.117 | 1.162 | +0.44 | -24.9% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 2 | 1.105 | 1.254 | +0.40 | -24.5% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25` | 2 | 1.103 | 1.138 | +0.38 | -30.7% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m05_dd20_t25` | 2 | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best` | 2 | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m02_dd20_t25` | 2 | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 2 | 1.077 | 1.146 | +0.33 | -32.3% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only` | 2 | 1.010 | 1.212 | +0.16 | -36.3% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m05_dd20_t25` | 2 | 1.001 | 1.195 | +0.10 | -30.9% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_abstain_m02_dd20_t25` | 2 | 1.001 | 1.107 | +0.10 | -30.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.344 | 1.569 | +1.73 | -22.6% | 0.15 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.344 | 1.423 | +2.04 | -5.3% | 0.13 | 2 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.257 | 1.470 | +1.58 | -22.6% | 0.14 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.257 | 1.333 | +1.73 | -5.3% | 0.11 | 2 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.156 | 1.269 | +0.91 | -24.6% | 0.14 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.156 | 1.156 | +0.81 | -16.1% | 0.13 | 2 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.111 | 1.144 | +0.55 | -23.9% | 0.15 | 2 | False |
| gate | `train_phase_abstain_m10_dd20_t25` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | 2 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | 2 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.067 | 1.468 | +0.30 | -22.6% | 0.18 | 2 | False |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.028 | 1.374 | +0.20 | -23.4% | 0.19 | 2 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.873 | -0.16 | -35.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 0.838 | -0.50 | -32.9% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 0.816 | -0.47 | -30.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 0.712 | -0.73 | -35.6% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 0.712 | -0.73 | -35.6% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 0.704 | -0.69 | -42.9% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.417 | +1.21 | -13.9% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.417 | +1.21 | -13.9% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.344 | +2.04 | -4.6% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.344 | +2.04 | -4.6% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.322 | +1.01 | -13.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.322 | +1.01 | -13.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_pressure_nonneg` | 1.301 | +0.91 | -17.5% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.289 | +0.87 | -19.4% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.257 | +1.73 | -4.6% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.257 | +1.73 | -4.6% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 1.240 | +0.76 | -23.3% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.215 | +0.71 | -17.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.215 | +0.71 | -20.3% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 1.215 | +0.71 | -20.3% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.202 | +0.67 | -19.4% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25` | 1.174 | +0.57 | -29.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 1.174 | +0.57 | -29.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.156 | +0.91 | -16.1% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.156 | +0.91 | -16.1% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_nonneg` | 1.117 | +0.44 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m10_dd20_t25` | 1.111 | +0.55 | -23.9% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.105 | +0.40 | -22.0% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m02_dd20_t25` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m05_dd20_t25` | 1.078 | +0.33 | -31.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_pressure_nonneg` | 1.067 | +0.30 | -19.4% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.067 | +0.30 | -19.4% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.028 | +0.20 | -23.4% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.028 | +0.20 | -23.4% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.017 | +0.16 | -20.7% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 1.010 | +0.16 | -36.3% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m02_dd20_t25` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_abstain_m05_dd20_t25` | 1.001 | +0.10 | -30.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 0.980 | +0.05 | -26.5% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 0.976 | +0.04 | -24.9% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 0.946 | -0.04 | -29.6% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 0.946 | -0.04 | -29.6% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 0.943 | -0.04 | -25.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 0.941 | -0.07 | -30.5% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.901 | -0.19 | -32.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 0.865 | -0.30 | -36.1% | 0.16 |
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
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.947 | +1.79 | -22.6% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 1.869 | +1.70 | -22.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.869 | +1.70 | -22.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.824 | +1.62 | -22.6% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 1.807 | +1.47 | -24.8% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 1.807 | +1.47 | -24.8% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.795 | +1.73 | -22.6% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.756 | +1.54 | -22.6% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.752 | +1.57 | -22.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.721 | +1.48 | -22.6% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.721 | +1.48 | -22.6% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.705 | +1.47 | -22.6% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.683 | +1.58 | -22.6% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.640 | +1.31 | -24.4% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.638 | +1.41 | -22.6% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 1.586 | +1.18 | -26.0% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_pressure_nonneg` | 1.566 | +1.25 | -22.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.551 | +1.12 | -23.4% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.503 | +1.12 | -25.3% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.502 | +2.36 | -5.3% | 0.10 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.488 | +1.03 | -23.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.488 | +1.03 | -23.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.479 | +1.07 | -24.4% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 1.473 | +1.07 | -25.3% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.473 | +1.07 | -25.3% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.466 | +1.09 | -22.6% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 1.462 | +1.03 | -26.8% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.414 | +0.92 | -23.9% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.409 | +2.17 | -5.3% | 0.10 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.403 | +0.96 | -24.5% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 1.401 | +0.91 | -31.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.401 | +1.06 | -24.7% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m05_dd20_t25` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m10_dd20_t25` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.388 | +0.93 | -23.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.383 | +0.99 | -24.6% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m05_dd20_t25` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 1.358 | +0.84 | -23.9% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.347 | +0.88 | -24.5% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.310 | +0.89 | -24.7% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 1.306 | +0.73 | -25.3% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 1.266 | +0.73 | -24.8% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.266 | +0.69 | -27.1% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 1.245 | +0.64 | -25.2% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 1.213 | +0.58 | -28.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m02_dd20_t25` | 1.213 | +0.58 | -28.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_abstain_m02_dd20_t25` | 1.213 | +0.58 | -28.6% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_nonneg` | 1.206 | +0.59 | -24.9% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.177 | +0.69 | -19.6% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.157 | +0.81 | -14.7% | 0.10 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25` | 1.103 | +0.38 | -30.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 1.077 | +0.33 | -32.3% | 0.21 |
