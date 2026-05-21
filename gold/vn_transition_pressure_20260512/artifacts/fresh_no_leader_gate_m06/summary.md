# Wyckoff Phase Gate Execution Simulation

This simulation recomputes positions, turnover, and cost when the gate changes policy by Wyckoff phase.
`oracle_eval_phase_best` remains an upper-bound diagnostic because it uses validation phase outcomes.

## Gate Maps

| Gate | Accumulation | Markup | Distribution | Markdown | Transition |
| --- | --- | --- | --- | --- | --- |
| `train_phase_best` | `h1_tradeability_filter` | `legacy_filter_shortlist` | `h3_rank_committee` | `h1_tradeability_filter` | `h0_forecast_abs` |
| `conservative_prior` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` |
| `aggressive_prior` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `h1_tradeability_filter` | `all_committee_candidates` | `h1_tradeability_filter` |
| `oracle_eval_phase_best` | `h1_tradeability_filter` | `h1_tradeability_filter` | `legacy_filter_shortlist` | `all_committee_candidates` | `h0_forecast_abs` |
| `baseline_h1` | `h1_tradeability_filter` | `h1_tradeability_filter` | `h1_tradeability_filter` | `h1_tradeability_filter` | `h1_tradeability_filter` |
| `baseline_all_committee` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` |
| `baseline_legacy` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `legacy_filter_shortlist` |
| `train_phase_regularized_m10_dd20_t25` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h3_rank_committee` | `h1_tradeability_filter` | `h0_forecast_abs` |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `__cash__` | `__cash__` | `__cash__` | `h0_forecast_abs` | `__cash__` |
| `train_phase_regularized_m15_dd20_t25` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h3_rank_committee` | `h1_tradeability_filter` | `h0_forecast_abs` |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `__cash__` | `__cash__` | `__cash__` | `h0_forecast_abs` | `__cash__` |
| `vn_legacy_acc_all_else` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` |
| `vn_legacy_acc_markup_all_else` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` |
| `vn_h1_distribution_only` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `all_committee_candidates` | `all_committee_candidates` |
| `vn_h1_transition_only` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `h1_tradeability_filter` |
| `vn_h1_distribution_transition` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `all_committee_candidates` | `h1_tradeability_filter` |
| `vn_h1_markup_distribution_transition` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `h1_tradeability_filter` | `all_committee_candidates` | `h1_tradeability_filter` |
| `conservative_prior_transition_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_pressure_delta_20_gte_0` |
| `conservative_prior_transition_mr20_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_gte_0` |
| `conservative_prior_transition_mr20_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0` |
| `baseline_all_committee_transition_pressure_nonneg` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_pressure_delta_20_gte_0` |
| `baseline_all_committee_transition_mr20_nonneg` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_gte_0` |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0` |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_pressure_delta_20_gte_0` |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_gte_0` |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0` |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_pressure_delta_20_gte_0` |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_gte_0` |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `all_committee_candidates` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0` |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `all_committee_candidates` | `all_committee_candidates_if_pressure_delta_20_gte_0` |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_gte_0` |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `legacy_filter_shortlist` | `legacy_filter_shortlist` | `h1_tradeability_filter` | `all_committee_candidates` | `all_committee_candidates_if_market_return_20_and_pressure_delta_20_gte_0` |

## Results

| Gate | Net equity | Sharpe | Max DD | Avg turnover | Hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_eval_phase_best` | 1.385 | +0.80 | -30.7% | 0.28 | 41.9% |
| `vn_h1_transition_only` | 1.255 | +0.59 | -21.7% | 0.26 | 35.9% |
| `conservative_prior_transition_mr20_pressure_nonneg` | 1.224 | +0.62 | -21.7% | 0.18 | 26.0% |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | 1.224 | +0.62 | -21.7% | 0.18 | 26.0% |
| `conservative_prior_transition_mr20_nonneg` | 1.215 | +0.59 | -21.7% | 0.18 | 28.6% |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | 1.215 | +0.59 | -21.7% | 0.18 | 28.6% |
| `conservative_prior_transition_pressure_nonneg` | 1.213 | +0.60 | -21.7% | 0.17 | 26.4% |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | 1.213 | +0.60 | -21.7% | 0.17 | 26.4% |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | 1.198 | +0.58 | -21.7% | 0.14 | 26.4% |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | 1.196 | +0.56 | -21.7% | 0.14 | 29.2% |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | 1.186 | +0.55 | -21.7% | 0.13 | 26.8% |
| `vn_h1_distribution_only_transition_mr20_nonneg` | 1.168 | +0.46 | -26.5% | 0.24 | 30.6% |
| `conservative_prior` | 1.123 | +0.38 | -23.5% | 0.16 | 32.1% |
| `vn_legacy_acc_markup_all_else` | 1.123 | +0.38 | -23.5% | 0.16 | 32.1% |
| `vn_h1_distribution_only_transition_pressure_nonneg` | 1.122 | +0.38 | -26.5% | 0.23 | 27.6% |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | 1.117 | +0.37 | -26.5% | 0.23 | 27.0% |
| `vn_h1_distribution_transition` | 1.109 | +0.33 | -38.1% | 0.20 | 38.3% |
| `vn_legacy_acc_all_else` | 1.083 | +0.30 | -23.5% | 0.12 | 32.7% |
| `vn_h1_distribution_only` | 1.074 | +0.27 | -32.8% | 0.24 | 34.1% |
| `aggressive_prior` | 1.024 | +0.17 | -34.7% | 0.15 | 38.3% |
| `vn_h1_markup_distribution_transition` | 1.024 | +0.17 | -34.7% | 0.15 | 38.3% |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | 1.016 | +0.14 | -29.8% | 0.12 | 25.8% |
| `baseline_all_committee_transition_mr20_nonneg` | 1.014 | +0.14 | -29.8% | 0.12 | 28.6% |
| `baseline_all_committee_transition_pressure_nonneg` | 1.006 | +0.12 | -29.8% | 0.11 | 26.2% |
| `baseline_all_committee` | 0.919 | -0.07 | -35.1% | 0.10 | 32.1% |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | 0.918 | -0.17 | -21.7% | 0.04 | 6.3% |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | 0.918 | -0.17 | -21.7% | 0.04 | 6.3% |
| `train_phase_regularized_m10_dd20_t25` | 0.860 | -0.18 | -39.8% | 0.26 | 36.5% |
| `train_phase_regularized_m15_dd20_t25` | 0.860 | -0.18 | -39.8% | 0.26 | 36.5% |
| `baseline_h1` | 0.826 | -0.26 | -43.8% | 0.12 | 38.9% |
| `train_phase_best` | 0.765 | -0.38 | -46.5% | 0.25 | 36.9% |
| `baseline_legacy` | 0.764 | -0.45 | -42.4% | 0.14 | 41.5% |

## By Phase

| Gate | Phase | Days | Net equity | Sharpe | Max DD | Avg turnover |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `aggressive_prior` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `aggressive_prior` | `markup` | 55 | 1.067 | +2.63 | -4.9% | 0.13 |
| `aggressive_prior` | `distribution` | 94 | 0.984 | -0.18 | -11.4% | 0.14 |
| `aggressive_prior` | `markdown` | 56 | 0.999 | +0.23 | -21.7% | 0.26 |
| `aggressive_prior` | `transition` | 269 | 1.013 | +0.16 | -20.0% | 0.12 |
| `baseline_all_committee` | `accumulation` | 30 | 0.819 | -4.78 | -22.9% | 0.05 |
| `baseline_all_committee` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `baseline_all_committee` | `distribution` | 94 | 1.049 | +1.65 | -4.2% | 0.08 |
| `baseline_all_committee` | `markdown` | 56 | 0.969 | -0.04 | -22.8% | 0.15 |
| `baseline_all_committee` | `transition` | 269 | 1.123 | +0.75 | -13.4% | 0.10 |
| `baseline_all_committee_transition_mr20_nonneg` | `accumulation` | 30 | 0.819 | -4.78 | -22.9% | 0.05 |
| `baseline_all_committee_transition_mr20_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `baseline_all_committee_transition_mr20_nonneg` | `distribution` | 94 | 1.057 | +1.57 | -4.2% | 0.12 |
| `baseline_all_committee_transition_mr20_nonneg` | `markdown` | 56 | 0.996 | +0.21 | -21.7% | 0.20 |
| `baseline_all_committee_transition_mr20_nonneg` | `transition` | 269 | 1.197 | +1.74 | -4.7% | 0.12 |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `accumulation` | 30 | 0.819 | -4.78 | -22.9% | 0.05 |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `distribution` | 94 | 1.140 | +3.42 | -3.4% | 0.13 |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `markdown` | 56 | 0.996 | +0.21 | -21.7% | 0.20 |
| `baseline_all_committee_transition_mr20_pressure_nonneg` | `transition` | 269 | 1.112 | +1.23 | -4.9% | 0.11 |
| `baseline_all_committee_transition_pressure_nonneg` | `accumulation` | 30 | 0.819 | -4.78 | -22.9% | 0.05 |
| `baseline_all_committee_transition_pressure_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `baseline_all_committee_transition_pressure_nonneg` | `distribution` | 94 | 1.156 | +3.97 | -3.4% | 0.11 |
| `baseline_all_committee_transition_pressure_nonneg` | `markdown` | 56 | 0.996 | +0.21 | -21.7% | 0.20 |
| `baseline_all_committee_transition_pressure_nonneg` | `transition` | 269 | 1.086 | +0.96 | -4.9% | 0.10 |
| `baseline_h1` | `accumulation` | 30 | 0.863 | -3.52 | -21.6% | 0.14 |
| `baseline_h1` | `markup` | 55 | 1.071 | +2.78 | -4.9% | 0.13 |
| `baseline_h1` | `distribution` | 94 | 0.984 | -0.18 | -11.4% | 0.14 |
| `baseline_h1` | `markdown` | 56 | 0.894 | -0.72 | -23.3% | 0.13 |
| `baseline_h1` | `transition` | 269 | 1.016 | +0.18 | -19.8% | 0.12 |
| `baseline_legacy` | `accumulation` | 30 | 0.841 | -4.44 | -21.1% | 0.05 |
| `baseline_legacy` | `markup` | 55 | 0.981 | -0.67 | -8.0% | 0.18 |
| `baseline_legacy` | `distribution` | 94 | 1.141 | +3.55 | -2.4% | 0.14 |
| `baseline_legacy` | `markdown` | 56 | 0.912 | -0.64 | -22.6% | 0.14 |
| `baseline_legacy` | `transition` | 269 | 0.891 | -0.45 | -20.4% | 0.15 |
| `conservative_prior` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `conservative_prior` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `conservative_prior` | `distribution` | 94 | 1.067 | +2.15 | -4.2% | 0.08 |
| `conservative_prior` | `markdown` | 56 | 0.971 | -0.01 | -22.8% | 0.21 |
| `conservative_prior` | `transition` | 269 | 1.173 | +0.94 | -13.4% | 0.14 |
| `conservative_prior_transition_mr20_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `conservative_prior_transition_mr20_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `conservative_prior_transition_mr20_nonneg` | `distribution` | 94 | 1.057 | +1.57 | -4.2% | 0.12 |
| `conservative_prior_transition_mr20_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `conservative_prior_transition_mr20_nonneg` | `transition` | 269 | 1.245 | +1.80 | -4.3% | 0.15 |
| `conservative_prior_transition_mr20_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `conservative_prior_transition_mr20_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `conservative_prior_transition_mr20_pressure_nonneg` | `distribution` | 94 | 1.140 | +3.42 | -3.4% | 0.13 |
| `conservative_prior_transition_mr20_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `conservative_prior_transition_mr20_pressure_nonneg` | `transition` | 269 | 1.163 | +1.47 | -3.9% | 0.15 |
| `conservative_prior_transition_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `conservative_prior_transition_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `conservative_prior_transition_pressure_nonneg` | `distribution` | 94 | 1.156 | +3.97 | -3.4% | 0.11 |
| `conservative_prior_transition_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `conservative_prior_transition_pressure_nonneg` | `transition` | 269 | 1.136 | +1.25 | -5.8% | 0.13 |
| `oracle_eval_phase_best` | `accumulation` | 30 | 0.959 | -2.10 | -8.9% | 0.30 |
| `oracle_eval_phase_best` | `markup` | 55 | 1.046 | +1.49 | -4.2% | 0.28 |
| `oracle_eval_phase_best` | `distribution` | 94 | 1.201 | +4.16 | -4.3% | 0.40 |
| `oracle_eval_phase_best` | `markdown` | 56 | 1.000 | +0.25 | -21.7% | 0.24 |
| `oracle_eval_phase_best` | `transition` | 269 | 1.151 | +0.73 | -23.5% | 0.24 |
| `train_phase_best` | `accumulation` | 30 | 0.863 | -3.52 | -21.6% | 0.14 |
| `train_phase_best` | `markup` | 55 | 0.956 | -1.51 | -9.1% | 0.35 |
| `train_phase_best` | `distribution` | 94 | 0.873 | -1.84 | -16.8% | 0.33 |
| `train_phase_best` | `markdown` | 56 | 0.919 | -0.49 | -22.1% | 0.15 |
| `train_phase_best` | `transition` | 269 | 1.157 | +0.75 | -23.3% | 0.23 |
| `train_phase_regularized_m10_dd20_t25` | `accumulation` | 30 | 0.964 | -3.10 | -6.1% | 0.22 |
| `train_phase_regularized_m10_dd20_t25` | `markup` | 55 | 0.956 | -1.51 | -9.1% | 0.35 |
| `train_phase_regularized_m10_dd20_t25` | `distribution` | 94 | 0.873 | -1.84 | -16.8% | 0.33 |
| `train_phase_regularized_m10_dd20_t25` | `markdown` | 56 | 0.924 | -0.43 | -22.1% | 0.21 |
| `train_phase_regularized_m10_dd20_t25` | `transition` | 269 | 1.156 | +0.75 | -23.3% | 0.23 |
| `train_phase_regularized_m15_dd20_t25` | `accumulation` | 30 | 0.964 | -3.10 | -6.1% | 0.22 |
| `train_phase_regularized_m15_dd20_t25` | `markup` | 55 | 0.956 | -1.51 | -9.1% | 0.35 |
| `train_phase_regularized_m15_dd20_t25` | `distribution` | 94 | 0.873 | -1.84 | -16.8% | 0.33 |
| `train_phase_regularized_m15_dd20_t25` | `markdown` | 56 | 0.924 | -0.43 | -22.1% | 0.21 |
| `train_phase_regularized_m15_dd20_t25` | `transition` | 269 | 1.156 | +0.75 | -23.3% | 0.23 |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `accumulation` | 30 | 0.994 | -6.12 | -0.4% | 0.13 |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `markup` | 55 | 1.000 | +nan | 0.0% | 0.00 |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `distribution` | 94 | 1.000 | +nan | 0.0% | 0.00 |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `markdown` | 56 | 0.924 | -0.46 | -21.7% | 0.29 |
| `train_phase_robust_abstain_m10_dd20_t25_ts050_pf60` | `transition` | 269 | 1.000 | +nan | 0.0% | 0.00 |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `accumulation` | 30 | 0.994 | -6.12 | -0.4% | 0.13 |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `markup` | 55 | 1.000 | +nan | 0.0% | 0.00 |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `distribution` | 94 | 1.000 | +nan | 0.0% | 0.00 |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `markdown` | 56 | 0.924 | -0.46 | -21.7% | 0.29 |
| `train_phase_robust_abstain_m15_dd20_t25_ts050_pf60` | `transition` | 269 | 1.000 | +nan | 0.0% | 0.00 |
| `vn_h1_distribution_only` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_distribution_only` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_h1_distribution_only` | `distribution` | 94 | 0.962 | -0.36 | -15.2% | 0.35 |
| `vn_h1_distribution_only` | `markdown` | 56 | 0.971 | -0.01 | -22.8% | 0.21 |
| `vn_h1_distribution_only` | `transition` | 269 | 1.243 | +1.18 | -13.5% | 0.19 |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `distribution` | 94 | 0.966 | -0.31 | -15.1% | 0.32 |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_h1_distribution_only_transition_mr20_nonneg` | `transition` | 269 | 1.310 | +1.99 | -8.1% | 0.19 |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `distribution` | 94 | 0.967 | -0.30 | -15.0% | 0.31 |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_h1_distribution_only_transition_mr20_pressure_nonneg` | `transition` | 269 | 1.252 | +1.89 | -5.4% | 0.18 |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `distribution` | 94 | 0.965 | -0.32 | -15.1% | 0.33 |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_h1_distribution_only_transition_pressure_nonneg` | `transition` | 269 | 1.259 | +1.89 | -5.4% | 0.18 |
| `vn_h1_distribution_transition` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_distribution_transition` | `markup` | 55 | 0.956 | -1.53 | -9.0% | 0.36 |
| `vn_h1_distribution_transition` | `distribution` | 94 | 1.083 | +1.17 | -11.8% | 0.13 |
| `vn_h1_distribution_transition` | `markdown` | 56 | 0.999 | +0.23 | -21.7% | 0.26 |
| `vn_h1_distribution_transition` | `transition` | 269 | 1.113 | +0.58 | -21.6% | 0.17 |
| `vn_h1_markup_distribution_transition` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_markup_distribution_transition` | `markup` | 55 | 1.067 | +2.63 | -4.9% | 0.13 |
| `vn_h1_markup_distribution_transition` | `distribution` | 94 | 0.984 | -0.18 | -11.4% | 0.14 |
| `vn_h1_markup_distribution_transition` | `markdown` | 56 | 0.999 | +0.23 | -21.7% | 0.26 |
| `vn_h1_markup_distribution_transition` | `transition` | 269 | 1.013 | +0.16 | -20.0% | 0.12 |
| `vn_h1_transition_only` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_h1_transition_only` | `markup` | 55 | 0.956 | -1.53 | -9.0% | 0.36 |
| `vn_h1_transition_only` | `distribution` | 94 | 1.005 | +0.17 | -12.1% | 0.30 |
| `vn_h1_transition_only` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_h1_transition_only` | `transition` | 269 | 1.359 | +1.52 | -13.3% | 0.22 |
| `vn_legacy_acc_all_else` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_all_else` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `vn_legacy_acc_all_else` | `distribution` | 94 | 1.049 | +1.65 | -4.2% | 0.08 |
| `vn_legacy_acc_all_else` | `markdown` | 56 | 0.971 | -0.01 | -22.8% | 0.21 |
| `vn_legacy_acc_all_else` | `transition` | 269 | 1.123 | +0.75 | -13.4% | 0.10 |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `distribution` | 94 | 1.057 | +1.57 | -4.2% | 0.12 |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_all_else_transition_mr20_nonneg` | `transition` | 269 | 1.197 | +1.74 | -4.7% | 0.12 |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `distribution` | 94 | 1.140 | +3.42 | -3.4% | 0.13 |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_all_else_transition_mr20_pressure_nonneg` | `transition` | 269 | 1.112 | +1.23 | -4.9% | 0.11 |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `markup` | 55 | 0.983 | -0.59 | -7.5% | 0.11 |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `distribution` | 94 | 1.156 | +3.97 | -3.4% | 0.11 |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_all_else_transition_pressure_nonneg` | `transition` | 269 | 1.086 | +0.96 | -4.9% | 0.10 |
| `vn_legacy_acc_markup_all_else` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_markup_all_else` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_legacy_acc_markup_all_else` | `distribution` | 94 | 1.067 | +2.15 | -4.2% | 0.08 |
| `vn_legacy_acc_markup_all_else` | `markdown` | 56 | 0.971 | -0.01 | -22.8% | 0.21 |
| `vn_legacy_acc_markup_all_else` | `transition` | 269 | 1.173 | +0.94 | -13.4% | 0.14 |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `distribution` | 94 | 1.057 | +1.57 | -4.2% | 0.12 |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_markup_all_else_transition_mr20_nonneg` | `transition` | 269 | 1.245 | +1.80 | -4.3% | 0.15 |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `distribution` | 94 | 1.140 | +3.42 | -3.4% | 0.13 |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_markup_all_else_transition_mr20_pressure_nonneg` | `transition` | 269 | 1.163 | +1.47 | -3.9% | 0.15 |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `accumulation` | 30 | 0.963 | -3.13 | -6.1% | 0.23 |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `markup` | 55 | 0.960 | -1.37 | -9.0% | 0.31 |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `distribution` | 94 | 1.156 | +3.97 | -3.4% | 0.11 |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `markdown` | 56 | 0.998 | +0.23 | -21.7% | 0.26 |
| `vn_legacy_acc_markup_all_else_transition_pressure_nonneg` | `transition` | 269 | 1.136 | +1.25 | -5.8% | 0.13 |
