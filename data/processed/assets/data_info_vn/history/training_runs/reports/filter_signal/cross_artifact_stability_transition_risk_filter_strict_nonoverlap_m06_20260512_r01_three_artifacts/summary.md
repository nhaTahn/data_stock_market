# Cross-Artifact Stability Summary

This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.
Risk-control pass requires worst-artifact max drawdown no worse than `25%` and max average turnover no higher than `0.20`.

## Risk-Controlled Stable Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3 | 1.356 | 1.551 | +0.92 | -21.7% | 0.18 | Passes both artifacts and the risk-control screen. |

## Stable Non-Oracle Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3 | 1.356 | 1.551 | +0.92 | -21.7% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 3 | 1.288 | 1.323 | +0.63 | -29.4% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 3 | 1.068 | 1.470 | +0.27 | -21.7% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 3 | 1.068 | 1.470 | +0.27 | -21.7% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_transition` | 3 | 1.057 | 1.071 | +0.24 | -39.4% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 3 | 1.006 | 1.120 | +0.11 | -29.4% | 0.17 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.356 | 1.551 | +0.92 | -21.7% | 0.18 | 3 | False |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 1.288 | 1.323 | +0.63 | -29.4% | 0.16 | 3 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 1.068 | 1.470 | +0.27 | -21.7% | 0.21 | 3 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.068 | 1.470 | +0.27 | -21.7% | 0.21 | 3 | False |
| gate | `vn_h1_distribution_transition` | 1.057 | 1.071 | +0.24 | -39.4% | 0.22 | 3 | False |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.006 | 1.120 | +0.11 | -29.4% | 0.17 | 3 | False |
| gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.094 | 1.094 | +0.34 | -21.6% | 0.22 | 1 | True |
| gate | `train_phase_best_transition_pressure_nonneg` | 1.085 | 1.323 | +0.31 | -22.8% | 0.25 | 2 | False |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.060 | 1.238 | +0.37 | -21.7% | 0.17 | 2 | False |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.060 | 1.196 | +0.37 | -13.4% | 0.14 | 2 | False |
| gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 0.994 | 1.247 | +0.08 | -21.7% | 0.24 | 1 | False |
| gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 0.994 | 1.101 | +0.08 | -28.0% | 0.23 | 1 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 1.335 | +0.69 | -41.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 1.269 | +0.57 | -48.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 1.117 | +0.37 | -27.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.738 | -0.44 | -61.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.390 | +0.98 | -17.5% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.356 | +0.92 | -19.5% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `aggressive_prior` | 1.179 | +0.49 | -30.1% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_markup_distribution_transition` | 1.179 | +0.49 | -30.1% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.094 | +0.34 | -21.6% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_pressure_nonneg` | 1.085 | +0.31 | -22.8% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_pressure_nonneg` | 1.068 | +0.27 | -20.6% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.068 | +0.27 | -20.6% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 1.066 | +0.26 | -30.8% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 1.064 | +0.25 | -34.4% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_transition` | 1.061 | +0.25 | -27.3% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.060 | +0.37 | -13.4% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.060 | +0.37 | -13.4% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.006 | +0.11 | -24.3% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee` | 1.005 | +0.12 | -43.0% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 0.994 | +0.08 | -21.6% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 0.994 | +0.08 | -21.6% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 0.982 | +0.04 | -26.1% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else` | 0.981 | +0.06 | -44.4% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_pressure_nonneg` | 0.949 | -0.06 | -29.8% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 0.945 | -0.04 | -26.1% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 0.943 | -0.07 | -32.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 0.939 | -0.04 | -30.9% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.922 | -0.47 | -17.4% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.922 | -0.47 | -17.4% | 0.09 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_nonneg` | 0.886 | -0.23 | -35.7% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_nonneg` | 0.880 | -0.24 | -38.5% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.865 | -0.29 | -33.9% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 0.863 | -0.30 | -31.9% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 0.863 | -0.30 | -31.9% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 0.862 | -0.31 | -32.0% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 0.862 | -0.31 | -32.0% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.858 | -0.76 | -21.9% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.858 | -0.76 | -21.9% | 0.10 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 0.844 | -0.36 | -35.5% | 0.13 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 0.820 | -0.41 | -35.9% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 0.818 | -0.36 | -37.2% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 0.810 | -0.38 | -38.0% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 0.804 | -0.47 | -37.6% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.804 | -0.47 | -37.6% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.783 | -0.80 | -33.1% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.783 | -0.80 | -33.1% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 0.778 | -0.46 | -41.2% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 0.778 | -0.46 | -41.2% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 0.766 | -0.56 | -41.2% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 0.750 | -0.65 | -39.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 0.750 | -0.65 | -39.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 0.741 | -0.58 | -39.1% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25` | 0.741 | -0.58 | -39.1% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 0.711 | -0.64 | -45.4% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 0.687 | -0.50 | -52.2% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 0.635 | -0.80 | -46.3% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 1.742 | +1.33 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.742 | +1.33 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.724 | +1.27 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.594 | +1.15 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.594 | +1.15 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_pressure_nonneg` | 1.561 | +1.05 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.561 | +1.07 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 1.547 | +0.94 | -23.6% | 0.27 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.501 | +1.00 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 1.487 | +0.89 | -23.6% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.464 | +0.92 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.458 | +0.83 | -23.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.457 | +0.91 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.415 | +0.92 | -21.7% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.400 | +0.86 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.392 | +0.79 | -23.6% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.365 | +0.78 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.335 | +0.89 | -17.6% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.331 | +1.26 | -11.8% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.325 | +0.78 | -21.7% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.288 | +0.63 | -29.4% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.287 | +0.60 | -29.5% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.287 | +0.60 | -29.5% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_nonneg` | 1.261 | +0.59 | -24.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.247 | +1.03 | -11.4% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 1.232 | +0.55 | -25.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.232 | +0.55 | -25.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25` | 1.221 | +0.52 | -28.0% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 1.215 | +0.51 | -28.8% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.211 | +0.52 | -24.6% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.208 | +0.54 | -28.0% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.207 | +0.50 | -27.7% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.186 | +0.46 | -25.5% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.166 | +0.44 | -29.4% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.141 | +0.40 | -24.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.126 | +0.38 | -28.0% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.073 | +0.32 | -17.7% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 1.057 | +0.24 | -36.5% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.047 | +0.22 | -29.5% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 1.024 | +0.18 | -36.7% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.974 | +0.06 | -28.0% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 0.936 | -0.00 | -32.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 0.936 | -0.00 | -32.9% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.901 | -0.07 | -34.8% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 0.782 | -0.29 | -43.0% | 0.14 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 0.655 | -0.69 | -53.1% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 0.635 | -0.80 | -46.3% | 0.12 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `all_committee_candidates` | 1.273 | +0.58 | -32.9% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h3_rank_committee` | 0.815 | -0.19 | -49.1% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h1_tradeability_filter` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h2_risk_conditioned` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `legacy_filter_shortlist` | 0.667 | -0.78 | -44.7% | 0.13 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_pressure_nonneg` | 1.601 | +1.28 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.601 | +1.28 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.572 | +1.18 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `oracle_eval_phase_best` | 1.559 | +1.05 | -25.3% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.508 | +1.15 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.508 | +1.15 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.446 | +1.01 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior` | 1.426 | +0.86 | -22.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else` | 1.426 | +0.86 | -22.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else` | 1.403 | +0.80 | -22.8% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.361 | +0.82 | -21.7% | 0.27 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_nonneg` | 1.319 | +0.79 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.319 | +0.79 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.293 | +0.69 | -25.8% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only` | 1.292 | +0.63 | -23.8% | 0.28 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.264 | +0.67 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.228 | +0.59 | -21.7% | 0.27 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.189 | +0.51 | -25.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.186 | +0.50 | -23.9% | 0.28 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee` | 1.154 | +0.41 | -30.8% | 0.15 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_transition_only` | 1.127 | +0.37 | -35.4% | 0.28 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m10_dd20_t25` | 1.112 | +0.34 | -34.2% | 0.28 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m15_dd20_t25` | 1.112 | +0.34 | -34.2% | 0.28 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_best` | 1.096 | +0.31 | -35.2% | 0.29 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_transition` | 1.093 | +0.31 | -39.4% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.040 | +0.20 | -25.8% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `aggressive_prior` | 1.020 | +0.16 | -38.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_markup_distribution_transition` | 1.020 | +0.16 | -38.7% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_h1` | 0.736 | -0.50 | -54.6% | 0.16 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_legacy` | 0.667 | -0.78 | -44.7% | 0.13 |
