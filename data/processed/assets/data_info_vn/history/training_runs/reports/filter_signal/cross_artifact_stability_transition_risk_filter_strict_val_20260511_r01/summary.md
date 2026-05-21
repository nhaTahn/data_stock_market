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
| gate | `baseline_all_committee_transition_pressure_nonneg` | 2 | 3.450 | 3.594 | +1.52 | -29.7% | 0.17 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 2 | 3.171 | 4.260 | +1.44 | -29.7% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 2 | 2.599 | 3.947 | +1.20 | -28.5% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 2 | 2.599 | 3.947 | +1.20 | -28.5% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 2 | 2.309 | 2.637 | +1.07 | -30.5% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 2 | 2.184 | 2.810 | +1.04 | -30.2% | 0.28 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 2 | 2.184 | 2.268 | +1.04 | -34.6% | 0.28 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 2 | 2.123 | 3.182 | +0.98 | -30.5% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best_transition_pressure_nonneg` | 2 | 1.827 | 2.717 | +0.80 | -30.3% | 0.30 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 2 | 1.788 | 3.141 | +0.79 | -37.8% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 2 | 1.788 | 3.141 | +0.79 | -37.8% | 0.23 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else` | 2 | 1.765 | 2.023 | +0.69 | -56.1% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| committee | `all_committee_candidates` | 2 | 1.594 | 1.757 | +0.59 | -56.1% | 0.15 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee` | 2 | 1.594 | 1.757 | +0.59 | -56.1% | 0.15 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 2 | 1.594 | 2.471 | +0.68 | -36.9% | 0.27 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 2 | 1.594 | 1.943 | +0.68 | -36.9% | 0.27 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_mr20_nonneg` | 2 | 1.578 | 1.619 | +0.59 | -48.4% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 2 | 1.577 | 2.604 | +0.62 | -32.0% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior` | 2 | 1.540 | 2.405 | +0.56 | -54.1% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else` | 2 | 1.540 | 2.405 | +0.56 | -54.1% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 2 | 1.526 | 1.892 | +0.57 | -48.4% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.471 | 2.169 | +0.79 | -27.7% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2 | 1.471 | 1.583 | +0.79 | -27.7% | 0.16 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best_transition_mr20_pressure_nonneg` | 2 | 1.333 | 2.426 | +0.45 | -37.0% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_nonneg` | 2 | 1.329 | 1.860 | +0.43 | -44.7% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 2 | 1.329 | 1.860 | +0.43 | -44.7% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25` | 2 | 1.312 | 2.009 | +0.40 | -49.8% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25` | 2 | 1.312 | 1.654 | +0.40 | -49.8% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_transition_only` | 2 | 1.240 | 1.957 | +0.35 | -45.1% | 0.30 | Passes both artifacts, but still needs stricter validation. |
| committee | `legacy_filter_shortlist` | 2 | 1.230 | 1.324 | +0.37 | -41.0% | 0.12 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_legacy` | 2 | 1.230 | 1.324 | +0.37 | -41.0% | 0.12 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 2 | 1.149 | 2.338 | +0.27 | -38.6% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 2 | 1.145 | 1.597 | +0.27 | -44.5% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 2 | 1.145 | 1.273 | +0.27 | -44.5% | 0.29 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_best` | 2 | 1.101 | 1.973 | +0.23 | -50.6% | 0.31 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_transition` | 2 | 1.095 | 1.783 | +0.22 | -43.7% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `aggressive_prior` | 2 | 1.088 | 1.497 | +0.22 | -47.3% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_markup_distribution_transition` | 2 | 1.088 | 1.497 | +0.22 | -47.3% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.080 | 1.947 | +0.22 | -26.8% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2 | 1.080 | 1.372 | +0.22 | -26.8% | 0.16 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 3.450 | 3.594 | +1.52 | -29.7% | 0.17 | 2 | False |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3.171 | 4.260 | +1.44 | -29.7% | 0.18 | 2 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 2.599 | 3.947 | +1.20 | -28.5% | 0.21 | 2 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 2.599 | 3.947 | +1.20 | -28.5% | 0.21 | 2 | False |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 2.309 | 2.637 | +1.07 | -30.5% | 0.18 | 2 | False |
| gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 2.184 | 2.810 | +1.04 | -30.2% | 0.28 | 2 | False |
| gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 2.184 | 2.268 | +1.04 | -34.6% | 0.28 | 2 | False |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 2.123 | 3.182 | +0.98 | -30.5% | 0.19 | 2 | False |
| gate | `train_phase_best_transition_pressure_nonneg` | 1.827 | 2.717 | +0.80 | -30.3% | 0.30 | 2 | False |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.788 | 3.141 | +0.79 | -37.8% | 0.23 | 2 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.788 | 3.141 | +0.79 | -37.8% | 0.23 | 2 | False |
| gate | `vn_legacy_acc_all_else` | 1.765 | 2.023 | +0.69 | -56.1% | 0.16 | 2 | False |

## Artifact Details

| Artifact | Kind | Policy | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `all_committee_candidates` | 1.920 | +0.77 | -56.1% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `legacy_filter_shortlist` | 1.230 | +0.37 | -41.0% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h3_rank_committee` | 1.005 | +0.15 | -68.6% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h1_tradeability_filter` | 0.712 | -0.21 | -61.5% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h2_risk_conditioned` | 0.712 | -0.21 | -61.5% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | committee | `h0_forecast_abs` | 0.629 | -0.34 | -73.5% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_pressure_nonneg` | 3.450 | +1.52 | -29.7% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3.171 | +1.44 | -29.7% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 3.005 | +1.36 | -29.7% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_pressure_nonneg` | 2.599 | +1.20 | -28.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 2.599 | +1.20 | -28.5% | 0.21 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 2.309 | +1.07 | -30.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 2.184 | +1.04 | -30.2% | 0.28 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 2.184 | +1.04 | -30.2% | 0.28 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 2.123 | +0.98 | -30.5% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_pressure_nonneg` | 2.070 | +0.95 | -37.0% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 1.974 | +0.80 | -51.4% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee` | 1.920 | +0.77 | -56.1% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_pressure_nonneg` | 1.827 | +0.80 | -30.3% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.788 | +0.79 | -37.8% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.788 | +0.79 | -37.8% | 0.23 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else` | 1.765 | +0.69 | -56.1% | 0.14 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.660 | +0.66 | -48.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 1.594 | +0.68 | -36.9% | 0.27 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 1.594 | +0.68 | -36.9% | 0.27 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.577 | +0.62 | -32.0% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 1.540 | +0.56 | -54.1% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 1.540 | +0.56 | -54.1% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_nonneg` | 1.539 | +0.58 | -44.4% | 0.24 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.526 | +0.57 | -48.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.471 | +0.79 | -20.1% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.471 | +0.79 | -20.1% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 1.333 | +0.45 | -37.0% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 1.329 | +0.43 | -44.7% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.329 | +0.43 | -44.7% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25` | 1.312 | +0.40 | -49.8% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25` | 1.312 | +0.40 | -49.8% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 1.240 | +0.35 | -45.1% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 1.230 | +0.37 | -41.0% | 0.12 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.149 | +0.27 | -38.6% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 1.145 | +0.27 | -44.5% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 1.145 | +0.27 | -44.5% | 0.29 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 1.101 | +0.23 | -50.6% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_transition` | 1.095 | +0.22 | -43.7% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `aggressive_prior` | 1.088 | +0.22 | -47.3% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_markup_distribution_transition` | 1.088 | +0.22 | -47.3% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.080 | +0.22 | -26.8% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.080 | +0.22 | -26.8% | 0.16 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best_transition_mr20_nonneg` | 0.960 | +0.08 | -51.8% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.873 | -0.08 | -51.8% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.873 | -0.08 | -51.8% | 0.19 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 0.860 | -0.00 | -54.8% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 0.830 | -0.07 | -53.8% | 0.30 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.773 | -0.34 | -41.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 0.773 | -0.34 | -41.4% | 0.17 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 0.712 | -0.21 | -61.5% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 1.594 | +0.59 | -30.5% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 1.418 | +0.54 | -37.7% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.049 | +0.18 | -55.5% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 0.833 | -0.03 | -65.3% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 0.833 | -0.03 | -65.3% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 0.611 | -0.30 | -66.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 5.349 | +2.02 | -21.7% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 5.295 | +2.05 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 5.295 | +2.05 | -21.7% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 4.493 | +1.85 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 4.493 | +1.85 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 4.242 | +1.75 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 3.738 | +1.57 | -28.0% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 3.630 | +1.57 | -25.0% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_pressure_nonneg` | 3.608 | +1.58 | -24.3% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 3.527 | +1.57 | -25.0% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_pressure_nonneg` | 3.518 | +1.59 | -23.4% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_pressure_nonneg` | 3.437 | +1.55 | -28.4% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_pressure_nonneg` | 3.349 | +1.56 | -22.4% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 3.324 | +1.49 | -21.7% | 0.30 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 3.271 | +1.29 | -29.3% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 3.271 | +1.29 | -29.3% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 2.964 | +1.31 | -28.0% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 2.882 | +1.15 | -29.0% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 2.867 | +1.44 | -27.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 2.845 | +1.14 | -27.8% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 2.814 | +1.46 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25` | 2.706 | +1.11 | -27.1% | 0.27 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 2.673 | +1.08 | -37.6% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 2.471 | +1.01 | -39.2% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 2.391 | +1.03 | -36.4% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 2.391 | +1.03 | -36.4% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_pressure_nonneg` | 2.352 | +1.11 | -34.6% | 0.25 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_pressure_nonneg` | 2.291 | +1.11 | -34.6% | 0.24 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 2.282 | +0.95 | -29.3% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 2.258 | +0.96 | -37.3% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 2.228 | +0.96 | -28.0% | 0.21 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 2.172 | +0.94 | -36.4% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best_transition_mr20_nonneg` | 2.156 | +0.94 | -35.4% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t25_transition_mr20_nonneg` | 2.048 | +0.90 | -34.7% | 0.27 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25` | 1.996 | +0.81 | -34.6% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 1.907 | +0.77 | -36.8% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 1.907 | +0.77 | -36.8% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.712 | +0.74 | -35.4% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_pressure_nonneg` | 1.694 | +1.04 | -27.7% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_pressure_nonneg` | 1.663 | +1.08 | -20.7% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 1.594 | +0.59 | -30.5% | 0.15 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.578 | +0.59 | -42.3% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 1.418 | +0.54 | -37.7% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 1.416 | +0.55 | -28.0% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t25_transition_mr20_nonneg` | 1.402 | +0.49 | -40.9% | 0.26 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60_transition_mr20_nonneg` | 1.012 | +0.10 | -35.4% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 0.833 | -0.03 | -65.3% | 0.17 |
