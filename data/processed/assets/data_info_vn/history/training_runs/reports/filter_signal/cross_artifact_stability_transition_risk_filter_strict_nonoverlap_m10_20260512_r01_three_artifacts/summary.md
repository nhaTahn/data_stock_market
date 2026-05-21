# Cross-Artifact Stability Summary

This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.
Risk-control pass requires worst-artifact max drawdown no worse than `25%` and max average turnover no higher than `0.20`.

## Risk-Controlled Stable Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3 | 1.285 | 1.341 | +0.76 | -21.7% | 0.17 | Passes both artifacts and the risk-control screen. |
| gate | `conservative_prior_transition_pressure_nonneg` | 3 | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | Passes both artifacts and the risk-control screen. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 3 | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | Passes both artifacts and the risk-control screen. |

## Stable Non-Oracle Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3 | 1.285 | 1.341 | +0.76 | -21.7% | 0.17 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 3 | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 3 | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 3 | 1.069 | 1.160 | +0.26 | -29.0% | 0.15 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.285 | 1.341 | +0.76 | -21.7% | 0.17 | 3 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | 3 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.100 | 1.359 | +0.35 | -21.7% | 0.19 | 3 | False |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 1.069 | 1.160 | +0.26 | -29.0% | 0.15 | 3 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.095 | 1.205 | +0.56 | -21.7% | 0.15 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.095 | 1.166 | +0.56 | -11.8% | 0.12 | 2 | False |
| gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.040 | 1.040 | +0.20 | -19.8% | 0.21 | 1 | True |
| gate | `train_phase_best_transition_pressure_nonneg` | 1.029 | 1.280 | +0.17 | -21.7% | 0.23 | 2 | False |
| gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 0.973 | 1.212 | +0.03 | -24.4% | 0.25 | 2 | False |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 0.953 | 1.077 | -0.04 | -29.0% | 0.16 | 2 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.952 | 1.136 | -0.28 | -21.7% | 0.14 | 1 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.952 | 1.097 | -0.28 | -14.1% | 0.11 | 1 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 1.335 | +0.69 | -41.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 1.269 | +0.57 | -48.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 1.117 | +0.37 | -27.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.738 | -0.44 | -61.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.317 | +0.83 | -17.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.285 | +0.77 | -19.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `aggressive_prior` | 1.105 | +0.34 | -30.1% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_markup_distribution_transition` | 1.105 | +0.34 | -30.1% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_pressure_nonneg` | 1.100 | +0.35 | -18.8% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.100 | +0.35 | -18.8% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.095 | +0.56 | -10.5% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.095 | +0.56 | -10.5% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.040 | +0.20 | -19.8% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_pressure_nonneg` | 1.029 | +0.17 | -21.0% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_transition` | 1.025 | +0.17 | -27.3% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 0.998 | +0.10 | -34.4% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 0.975 | +0.03 | -36.0% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 0.973 | +0.03 | -24.4% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 0.953 | -0.04 | -24.3% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.952 | -0.28 | -14.1% | 0.08 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.952 | -0.28 | -14.1% | 0.08 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee` | 0.952 | -0.00 | -43.0% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 0.942 | -0.07 | -21.8% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 0.942 | -0.07 | -21.8% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 0.930 | -0.12 | -26.2% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else` | 0.929 | -0.06 | -44.4% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 0.915 | -0.11 | -30.9% | 0.28 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_pressure_nonneg` | 0.902 | -0.21 | -29.5% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 0.894 | -0.23 | -31.4% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 0.888 | -0.23 | -30.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 0.888 | -0.23 | -30.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.886 | -0.60 | -18.8% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.886 | -0.60 | -18.8% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 0.845 | -0.34 | -34.4% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_nonneg` | 0.843 | -0.39 | -35.4% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.839 | -0.57 | -30.6% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.839 | -0.57 | -30.6% | 0.11 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_nonneg` | 0.834 | -0.40 | -37.1% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.819 | -0.44 | -33.9% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 0.818 | -0.47 | -32.1% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 0.818 | -0.47 | -32.1% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 0.806 | -0.42 | -39.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 0.801 | -0.39 | -41.2% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 0.801 | -0.39 | -41.2% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 0.800 | -0.51 | -35.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 0.796 | -0.44 | -39.4% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 0.789 | -0.49 | -39.8% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 0.773 | -0.57 | -38.5% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 0.773 | -0.57 | -38.5% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 0.762 | -0.64 | -37.8% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.762 | -0.64 | -37.8% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 0.759 | -0.50 | -42.1% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 0.728 | -0.65 | -40.6% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25` | 0.728 | -0.65 | -40.6% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 0.687 | -0.50 | -52.2% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 0.635 | -0.80 | -46.3% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.548 | +1.12 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.548 | +1.12 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.533 | +1.07 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_pressure_nonneg` | 1.532 | +1.06 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 1.528 | +1.09 | -21.7% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.528 | +1.09 | -21.7% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.522 | +1.05 | -21.7% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.438 | +0.93 | -21.7% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.402 | +0.88 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.402 | +0.88 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.333 | +0.76 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.332 | +0.76 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 1.322 | +0.69 | -23.6% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.320 | +0.78 | -21.7% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.315 | +0.77 | -21.7% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_nonneg` | 1.278 | +0.63 | -24.6% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.242 | +1.10 | -11.2% | 0.11 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.237 | +1.05 | -11.8% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 1.210 | +0.51 | -23.6% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.209 | +0.52 | -23.6% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.209 | +0.52 | -23.6% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 1.173 | +0.46 | -25.5% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.173 | +0.46 | -25.5% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.169 | +0.45 | -24.6% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.159 | +0.43 | -29.0% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.154 | +0.42 | -23.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.153 | +0.42 | -27.7% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.150 | +0.44 | -27.4% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.150 | +0.44 | -27.4% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.129 | +0.38 | -23.6% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.114 | +0.35 | -25.5% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.096 | +0.32 | -24.6% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.095 | +0.31 | -29.0% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.083 | +0.34 | -17.6% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.052 | +0.23 | -28.4% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.031 | +0.18 | -17.7% | 0.13 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25` | 1.013 | +0.15 | -27.4% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 0.992 | +0.11 | -36.7% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 0.967 | +0.05 | -27.8% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.958 | +0.02 | -27.4% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.878 | -0.13 | -34.4% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 0.818 | -0.34 | -32.7% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 0.801 | -0.30 | -35.0% | 0.13 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 0.761 | -0.44 | -37.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 0.751 | -0.46 | -34.8% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 0.751 | -0.46 | -34.8% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 0.536 | -1.13 | -53.5% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `all_committee_candidates` | 1.273 | +0.58 | -32.9% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h3_rank_committee` | 0.815 | -0.19 | -49.1% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h1_tradeability_filter` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h2_risk_conditioned` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `legacy_filter_shortlist` | 0.667 | -0.78 | -44.7% | 0.13 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.474 | +1.12 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.474 | +1.12 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_pressure_nonneg` | 1.450 | +1.07 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.450 | +1.07 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.361 | +0.88 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_nonneg` | 1.339 | +0.85 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.339 | +0.85 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.330 | +0.82 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.300 | +0.76 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.299 | +0.76 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.295 | +0.76 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior` | 1.260 | +0.61 | -22.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else` | 1.260 | +0.61 | -22.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `oracle_eval_phase_best` | 1.248 | +0.62 | -25.3% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.237 | +0.63 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only` | 1.231 | +0.56 | -23.8% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else` | 1.131 | +0.38 | -22.8% | 0.15 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.119 | +0.37 | -25.8% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_transition_only` | 1.117 | +0.36 | -35.4% | 0.26 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.069 | +0.26 | -25.8% | 0.15 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_transition` | 1.022 | +0.16 | -39.4% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.017 | +0.15 | -25.8% | 0.15 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee` | 0.930 | -0.02 | -30.8% | 0.13 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `aggressive_prior` | 0.926 | -0.07 | -38.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_markup_distribution_transition` | 0.926 | -0.07 | -38.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m10_dd20_t25` | 0.892 | -0.12 | -34.2% | 0.26 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m15_dd20_t25` | 0.892 | -0.12 | -34.2% | 0.26 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_best` | 0.879 | -0.15 | -35.2% | 0.26 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_h1` | 0.669 | -0.71 | -54.6% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_legacy` | 0.660 | -0.81 | -45.1% | 0.13 |
