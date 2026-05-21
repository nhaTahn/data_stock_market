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
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 3 | 1.681 | 1.825 | +1.29 | -24.3% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 3 | 1.436 | 1.566 | +0.83 | -29.4% | 0.19 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_pressure_nonneg` | 3 | 1.376 | 1.627 | +0.82 | -21.7% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 3 | 1.376 | 1.627 | +0.82 | -21.7% | 0.24 | Passes both artifacts, but still needs stricter validation. |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 3 | 1.284 | 1.308 | +0.62 | -29.4% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 3 | 1.253 | 1.540 | +0.64 | -25.0% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else` | 3 | 1.168 | 1.339 | +0.42 | -43.3% | 0.18 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 3 | 1.144 | 1.440 | +0.42 | -24.7% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 3 | 1.144 | 1.440 | +0.42 | -24.7% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 3 | 1.097 | 1.275 | +0.32 | -30.8% | 0.21 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior` | 3 | 1.074 | 1.268 | +0.27 | -41.2% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else` | 3 | 1.074 | 1.268 | +0.27 | -41.2% | 0.22 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m10_dd20_t20` | 3 | 1.046 | 1.071 | +0.22 | -32.8% | 0.34 | Passes both artifacts, but still needs stricter validation. |
| gate | `train_phase_regularized_m15_dd20_t20` | 3 | 1.046 | 1.071 | +0.22 | -32.8% | 0.34 | Passes both artifacts, but still needs stricter validation. |
| gate | `aggressive_prior` | 3 | 1.042 | 1.152 | +0.21 | -39.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_markup_distribution_transition` | 3 | 1.042 | 1.152 | +0.21 | -39.9% | 0.20 | Passes both artifacts, but still needs stricter validation. |
| gate | `conservative_prior_transition_mr20_nonneg` | 3 | 1.013 | 1.193 | +0.15 | -28.4% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 3 | 1.013 | 1.193 | +0.15 | -28.4% | 0.25 | Passes both artifacts, but still needs stricter validation. |
| gate | `vn_h1_distribution_transition` | 3 | 1.011 | 1.153 | +0.15 | -38.9% | 0.25 | Passes both artifacts, but still needs stricter validation. |

## Top Policies By Worst Artifact

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Positive artifacts | Oracle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.681 | 1.825 | +1.29 | -24.3% | 0.20 | 3 | False |
| gate | `baseline_all_committee_transition_pressure_nonneg` | 1.436 | 1.566 | +0.83 | -29.4% | 0.19 | 3 | False |
| gate | `conservative_prior_transition_pressure_nonneg` | 1.376 | 1.627 | +0.82 | -21.7% | 0.24 | 3 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.376 | 1.627 | +0.82 | -21.7% | 0.24 | 3 | False |
| gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.284 | 1.308 | +0.62 | -29.4% | 0.20 | 3 | False |
| gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.253 | 1.540 | +0.64 | -25.0% | 0.21 | 3 | False |
| gate | `vn_legacy_acc_all_else` | 1.168 | 1.339 | +0.42 | -43.3% | 0.18 | 3 | False |
| gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.144 | 1.440 | +0.42 | -24.7% | 0.25 | 3 | False |
| gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.144 | 1.440 | +0.42 | -24.7% | 0.25 | 3 | False |
| gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.097 | 1.275 | +0.32 | -30.8% | 0.21 | 3 | False |
| gate | `conservative_prior` | 1.074 | 1.268 | +0.27 | -41.2% | 0.22 | 3 | False |
| gate | `vn_legacy_acc_markup_all_else` | 1.074 | 1.268 | +0.27 | -41.2% | 0.22 | 3 | False |

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
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_all_committee_transition_mr20_nonneg` | 1.124 | +0.37 | -30.8% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_legacy` | 1.117 | +0.37 | -27.0% | 0.15 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m10_dd20_t20` | 1.105 | +0.33 | -32.8% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_regularized_m15_dd20_t20` | 1.105 | +0.33 | -32.8% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.097 | +0.32 | -30.8% | 0.20 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_pressure_nonneg` | 1.092 | +0.30 | -25.5% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior` | 1.074 | +0.27 | -41.2% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else` | 1.074 | +0.27 | -41.2% | 0.22 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `baseline_h1` | 1.051 | +0.23 | -48.7% | 0.18 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_transition_only` | 1.013 | +0.15 | -38.3% | 0.34 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `conservative_prior_transition_mr20_nonneg` | 1.013 | +0.15 | -28.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.013 | +0.15 | -28.4% | 0.25 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m10_dd20_t20_ts050_pf60` | 1.000 | +nan | 0.0% | 0.00 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_robust_abstain_m15_dd20_t20_ts050_pf60` | 1.000 | +nan | 0.0% | 0.00 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 0.993 | +0.10 | -27.1% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_pressure_nonneg` | 0.953 | +0.01 | -35.0% | 0.33 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best` | 0.895 | -0.06 | -40.2% | 0.35 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `oracle_eval_phase_best_transition_mr20_nonneg` | 0.884 | -0.13 | -39.8% | 0.35 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 0.867 | -0.21 | -36.4% | 0.31 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `train_phase_best` | 0.860 | -0.17 | -46.2% | 0.37 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only` | 0.820 | -0.24 | -45.2% | 0.32 |
| `portable_lstm_filter_signal_20260508_r05_signmag` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 0.804 | -0.34 | -41.1% | 0.32 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h0_forecast_abs` | 1.008 | +0.15 | -41.6% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `all_committee_candidates` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h1_tradeability_filter` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h2_risk_conditioned` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `h3_rank_committee` | 0.687 | -0.50 | -52.2% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | committee | `legacy_filter_shortlist` | 0.637 | -0.79 | -46.2% | 0.12 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.923 | +1.45 | -21.7% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_pressure_nonneg` | 1.889 | +1.46 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.889 | +1.46 | -21.7% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.718 | +1.23 | -21.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_pressure_nonneg` | 1.709 | +1.26 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.709 | +1.26 | -21.7% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_pressure_nonneg` | 1.436 | +0.83 | -29.4% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_pressure_nonneg` | 1.433 | +0.83 | -24.7% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only` | 1.427 | +0.77 | -26.7% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.425 | +0.84 | -22.3% | 0.28 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior` | 1.396 | +0.73 | -29.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else` | 1.396 | +0.73 | -29.5% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.329 | +0.68 | -27.7% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `conservative_prior_transition_mr20_nonneg` | 1.321 | +0.68 | -25.5% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.321 | +0.68 | -25.5% | 0.22 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_transition` | 1.299 | +0.61 | -34.2% | 0.23 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.284 | +0.62 | -29.4% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.238 | +0.54 | -25.5% | 0.29 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_transition_only` | 1.223 | +0.50 | -30.3% | 0.30 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `oracle_eval_phase_best` | 1.169 | +0.42 | -24.3% | 0.35 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_legacy_acc_all_else` | 1.168 | +0.42 | -29.5% | 0.17 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `aggressive_prior` | 1.144 | +0.38 | -30.0% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `vn_h1_markup_distribution_transition` | 1.144 | +0.38 | -30.0% | 0.19 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m10_dd20_t20` | 1.062 | +0.25 | -27.0% | 0.32 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_regularized_m15_dd20_t20` | 1.062 | +0.25 | -27.0% | 0.32 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee_transition_mr20_nonneg` | 0.992 | +0.12 | -34.8% | 0.18 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_best` | 0.964 | +0.08 | -39.4% | 0.34 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m10_dd20_t20_ts050_pf60` | 0.942 | -0.10 | -21.7% | 0.05 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `train_phase_robust_abstain_m15_dd20_t20_ts050_pf60` | 0.942 | -0.10 | -21.7% | 0.05 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_all_committee` | 0.872 | -0.09 | -43.0% | 0.16 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_h1` | 0.801 | -0.26 | -50.8% | 0.20 |
| `portable_lstm_filter_signal_20260509_r06_selector_module` | gate | `baseline_legacy` | 0.637 | -0.79 | -46.2% | 0.12 |
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
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m10_dd20_t20` | 1.046 | +0.22 | -30.8% | 0.33 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_regularized_m15_dd20_t20` | 1.046 | +0.22 | -30.8% | 0.33 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `aggressive_prior` | 1.042 | +0.21 | -37.6% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_markup_distribution_transition` | 1.042 | +0.21 | -37.6% | 0.20 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_best` | 1.031 | +0.20 | -31.8% | 0.34 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_only_transition_mr20_nonneg` | 1.021 | +0.16 | -28.3% | 0.32 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_distribution_transition` | 1.011 | +0.15 | -38.3% | 0.25 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m10_dd20_t20_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `train_phase_robust_abstain_m15_dd20_t20_ts050_pf60` | 0.892 | -0.26 | -23.0% | 0.05 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `vn_h1_transition_only` | 0.875 | -0.12 | -35.9% | 0.32 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_h1` | 0.752 | -0.39 | -53.8% | 0.19 |
| `portable_lstm_filter_signal_20260512_r01_market_leader_k3w60` | gate | `baseline_legacy` | 0.667 | -0.78 | -44.7% | 0.13 |
