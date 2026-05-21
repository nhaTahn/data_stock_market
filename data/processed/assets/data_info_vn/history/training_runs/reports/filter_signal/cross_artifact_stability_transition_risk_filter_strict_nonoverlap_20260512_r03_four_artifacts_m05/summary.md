# Cross-Artifact Stability Summary

This report compares committee hypotheses and execution-level Wyckoff gates across VN filter-signal artifacts. Holdout/test data is not used.
Risk-control pass requires worst-artifact max drawdown no worse than `25%` and max average turnover no higher than `0.20`.

## Risk-Controlled Stable Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| - | - | 0 | - | - | - | - | - | No policy passed the risk-control screen. |

## Stable Non-Oracle Policies

| Kind | Policy | Artifacts | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 4 | 1.253 | 1.475 | +0.64 | -25.0% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 4 | 1.243 | 1.531 | +0.60 | -21.7% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 4 | 1.243 | 1.531 | +0.60 | -21.7% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 4 | 1.240 | 1.679 | +0.59 | -24.3% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 4 | 1.144 | 1.380 | +0.42 | -24.7% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 4 | 1.144 | 1.380 | +0.42 | -24.7% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 4 | 1.084 | 1.252 | +0.29 | -29.8% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 4 | 1.052 | 1.437 | +0.23 | -29.8% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 4 | 1.044 | 1.217 | +0.21 | -30.8% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `aggressive_prior` | 4 | 1.042 | 1.133 | +0.21 | -39.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_markup_distribution_transition` | 4 | 1.042 | 1.133 | +0.21 | -39.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_transition` | 4 | 1.011 | 1.132 | +0.15 | -38.9% | 0.25 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.253 | 1.475 | +0.64 | -25.0% | 0.21 | 4 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 1.243 | 1.531 | +0.60 | -21.7% | 0.24 | 4 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.243 | 1.531 | +0.60 | -21.7% | 0.24 | 4 | False |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.240 | 1.679 | +0.59 | -24.3% | 0.20 | 4 | False |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.144 | 1.380 | +0.42 | -24.7% | 0.25 | 4 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.144 | 1.380 | +0.42 | -24.7% | 0.25 | 4 | False |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.084 | 1.252 | +0.29 | -29.8% | 0.20 | 4 | False |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 1.052 | 1.437 | +0.23 | -29.8% | 0.19 | 4 | False |
| gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.044 | 1.217 | +0.21 | -30.8% | 0.21 | 4 | False |
| gate | `aggressive_prior` | 1.042 | 1.133 | +0.21 | -39.9% | 0.20 | 4 | False |
| gate | `vn_h1_markup_distribution_transition` | 1.042 | 1.133 | +0.21 | -39.9% | 0.20 | 4 | False |
| gate | `vn_h1_distribution_transition` | 1.011 | 1.132 | +0.15 | -38.9% | 0.25 | 4 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 1.335 | +0.69 | -41.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 1.269 | +0.57 | -48.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 1.117 | +0.37 | -27.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.738 | -0.44 | -61.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.723 | +1.34 | -24.3% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.681 | +1.29 | -24.3% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_pressure_nonneg` | 1.376 | +0.82 | -21.6% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.376 | +0.82 | -21.6% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee` | 1.335 | +0.69 | -41.9% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else` | 1.302 | +0.64 | -43.3% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.284 | +0.69 | -25.0% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `aggressive_prior` | 1.271 | +0.61 | -39.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_markup_distribution_transition` | 1.271 | +0.61 | -39.9% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.253 | +0.64 | -25.0% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_transition` | 1.148 | +0.40 | -38.9% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.144 | +0.42 | -24.7% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.144 | +0.42 | -24.7% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_pressure_nonneg` | 1.133 | +0.39 | -21.6% | 0.32 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.124 | +0.37 | -30.8% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 1.117 | +0.37 | -27.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.106 | +0.44 | -13.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.106 | +0.44 | -13.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.097 | +0.32 | -30.8% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.092 | +0.30 | -25.5% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.088 | +0.30 | -22.6% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.088 | +0.30 | -22.6% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 1.074 | +0.27 | -41.2% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 1.074 | +0.27 | -41.2% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 1.013 | +0.15 | -38.3% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 1.013 | +0.15 | -28.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.013 | +0.15 | -28.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 0.993 | +0.10 | -27.1% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 0.988 | +0.08 | -31.6% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.968 | -0.07 | -17.4% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 0.968 | -0.07 | -17.4% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_pressure_nonneg` | 0.953 | +0.01 | -35.0% | 0.33 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 0.950 | -0.02 | -32.4% | 0.28 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 0.950 | -0.02 | -32.4% | 0.28 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 0.927 | -0.01 | -40.1% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_nonneg` | 0.917 | -0.07 | -36.6% | 0.33 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 0.895 | -0.06 | -40.2% | 0.35 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.895 | -0.20 | -36.3% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.895 | -0.20 | -36.3% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.894 | -0.30 | -22.0% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.894 | -0.30 | -22.0% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 0.889 | -0.10 | -37.6% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25` | 0.889 | -0.10 | -37.6% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_nonneg` | 0.884 | -0.13 | -39.8% | 0.35 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 0.879 | -0.17 | -37.4% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.879 | -0.17 | -37.4% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 0.867 | -0.21 | -36.4% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 0.820 | -0.24 | -45.2% | 0.32 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 0.804 | -0.34 | -41.1% | 0.32 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 0.687 | -0.50 | -52.2% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 0.635 | -0.80 | -46.3% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.923 | +1.45 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 1.889 | +1.46 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.889 | +1.46 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.718 | +1.23 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.709 | +1.26 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.709 | +1.26 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.436 | +0.83 | -29.4% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.433 | +0.83 | -24.7% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.427 | +0.77 | -26.7% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_pressure_nonneg` | 1.426 | +0.83 | -23.4% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.425 | +0.84 | -22.3% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.420 | +0.84 | -21.7% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 1.413 | +0.76 | -25.5% | 0.30 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.396 | +0.73 | -29.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.396 | +0.73 | -29.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 1.357 | +0.76 | -24.6% | 0.27 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.351 | +0.77 | -22.2% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 1.344 | +0.68 | -26.6% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.329 | +0.68 | -27.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 1.321 | +0.68 | -25.5% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.321 | +0.68 | -25.5% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 1.299 | +0.61 | -34.2% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.284 | +0.62 | -29.4% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.275 | +0.65 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.275 | +0.67 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 1.254 | +0.57 | -23.6% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.238 | +0.54 | -25.5% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_nonneg` | 1.230 | +0.53 | -24.6% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 1.223 | +0.50 | -30.3% | 0.30 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.203 | +0.58 | -19.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.200 | +0.72 | -17.6% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.199 | +0.80 | -12.6% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 1.169 | +0.42 | -24.3% | 0.35 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.168 | +0.44 | -24.6% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.168 | +0.42 | -29.5% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 1.144 | +0.38 | -30.0% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 1.144 | +0.38 | -30.0% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25` | 1.104 | +0.32 | -28.0% | 0.27 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.098 | +0.32 | -24.6% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 1.092 | +0.31 | -28.0% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.087 | +0.30 | -28.0% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.033 | +0.18 | -17.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.992 | +0.12 | -34.8% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 0.940 | -0.01 | -28.0% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 0.635 | -0.80 | -46.3% | 0.12 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `all_committee_candidates` | 1.273 | +0.58 | -32.9% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h3_rank_committee` | 0.815 | -0.19 | -49.1% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h1_tradeability_filter` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `h2_risk_conditioned` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | committee | `legacy_filter_shortlist` | 0.667 | -0.78 | -44.7% | 0.13 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.870 | +1.55 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.649 | +1.28 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_pressure_nonneg` | 1.615 | +1.27 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.615 | +1.27 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else` | 1.548 | +0.95 | -25.2% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.537 | +1.06 | -25.8% | 0.18 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.468 | +1.05 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.468 | +1.05 | -21.7% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `oracle_eval_phase_best` | 1.450 | +0.81 | -21.7% | 0.30 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.398 | +0.85 | -25.4% | 0.21 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.356 | +0.79 | -25.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior` | 1.335 | +0.69 | -23.8% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else` | 1.335 | +0.69 | -23.8% | 0.22 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee` | 1.273 | +0.58 | -32.9% | 0.17 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `conservative_prior_transition_mr20_nonneg` | 1.245 | +0.61 | -23.9% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.245 | +0.61 | -23.9% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.161 | +0.44 | -23.7% | 0.31 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.150 | +0.41 | -29.2% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only` | 1.105 | +0.32 | -24.7% | 0.32 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.092 | +0.31 | -22.8% | 0.30 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m10_dd20_t25` | 1.046 | +0.22 | -30.8% | 0.33 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m15_dd20_t25` | 1.046 | +0.22 | -30.8% | 0.33 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `aggressive_prior` | 1.042 | +0.21 | -37.6% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_markup_distribution_transition` | 1.042 | +0.21 | -37.6% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_best` | 1.031 | +0.20 | -31.8% | 0.34 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.021 | +0.16 | -28.3% | 0.32 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_transition` | 1.011 | +0.15 | -38.3% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_transition_only` | 0.875 | -0.12 | -35.9% | 0.32 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_h1` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_legacy` | 0.667 | -0.78 | -44.7% | 0.13 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `h1_tradeability_filter` | 0.867 | -0.10 | -43.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `h2_risk_conditioned` | 0.867 | -0.10 | -43.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `legacy_filter_shortlist` | 0.786 | -0.39 | -42.4% | 0.16 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `all_committee_candidates` | 0.769 | -0.34 | -35.1% | 0.15 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | committee | `h3_rank_committee` | 0.621 | -0.69 | -49.6% | 0.17 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `oracle_eval_phase_best` | 1.318 | +0.64 | -25.1% | 0.34 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.279 | +0.65 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `conservative_prior_transition_pressure_nonneg` | 1.243 | +0.60 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.243 | +0.60 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.240 | +0.59 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_transition_only` | 1.210 | +0.48 | -24.1% | 0.32 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.199 | +0.52 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.199 | +0.52 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.115 | +0.34 | -26.5% | 0.30 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.086 | +0.29 | -26.5% | 0.31 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.084 | +0.29 | -29.8% | 0.18 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `aggressive_prior` | 1.074 | +0.27 | -34.7% | 0.19 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_markup_distribution_transition` | 1.074 | +0.27 | -34.7% | 0.19 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_distribution_transition` | 1.069 | +0.26 | -38.1% | 0.24 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.052 | +0.23 | -29.8% | 0.16 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.044 | +0.21 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.012 | +0.16 | -26.5% | 0.32 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `conservative_prior_transition_mr20_nonneg` | 0.973 | +0.07 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 0.973 | +0.07 | -21.7% | 0.24 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_h1_distribution_only` | 0.933 | +0.03 | -32.8% | 0.32 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `conservative_prior` | 0.922 | -0.02 | -23.5% | 0.22 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_markup_all_else` | 0.922 | -0.02 | -23.5% | 0.22 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.918 | -0.17 | -21.7% | 0.04 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.918 | -0.17 | -21.7% | 0.04 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `train_phase_regularized_m10_dd20_t25` | 0.912 | -0.02 | -36.8% | 0.34 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `train_phase_regularized_m15_dd20_t25` | 0.912 | -0.02 | -36.8% | 0.34 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `vn_legacy_acc_all_else` | 0.907 | -0.05 | -23.5% | 0.17 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.885 | -0.10 | -29.8% | 0.18 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_h1` | 0.867 | -0.10 | -43.8% | 0.17 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `train_phase_best` | 0.811 | -0.20 | -43.8% | 0.33 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_legacy` | 0.786 | -0.39 | -42.4% | 0.16 |
| `portable_lstm_filter_signal_20260512_r02_no_leader_seed43` | gate | `baseline_all_committee` | 0.769 | -0.34 | -35.1% | 0.15 |
