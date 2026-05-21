# Current Best Path

Updated: 2026-05-09

Full Vietnamese status note: `docs/current_research_status.md`.
All-VN finalization note: `docs/allvn_model_selection_20260428.md`.

## All-VN Base Model

For the current clean broad VN universe, the finalized base model is:

- run: `broad_signmag_portable_no_identity_20260428_allvn_r01`
- model alias: `lstm_best_by_val`
- checkpoint: `model_seed_52.keras`

Why this is the current base:

- it is not the highest validation run, but it is the strongest completed all-VN candidate on out-sample `rel_score`
- `portable_no_identity_units48_24` wins validation but does not hold up on out-sample

Guardrail:

- this out-sample read was opened once on `2026-04-28` to finalize the all-VN base model
- do not keep tuning future variants against this same holdout

## Active Candidates

Keep these in the active research path:

| Role | Run / Report | Notes |
| --- | --- | --- |
| Prediction anchor | `broad_signmag_prune_general_sector_full_20260424_r04` | Best current validation `rel_score` among trusted portable sector-general models. |
| Trade challenger | `broad_signmag_prune_phase_ic_sector19_20260425_r09` | Lower `rel_score`, but strong quartile trade behavior; useful for routing/ensemble tests. |
| Prediction-safe ensemble | `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01` | `w_challenger=0.10` gives the best validation `rel_score` in the current grid and slightly improves trade equity versus anchor. |
| Trade router | `reports/router_weight_grid/anchor_sector19_weight_grid_20260426_r01` | `w_challenger=0.75` gives the best validation quartile trade equity in the current grid. |
| Objective/window smoke | `reports/feature_pruning/broad_signmag_prune_20260426_r02` | Negative result: window 10/20, `rel_score_sharp`, and `rel_score_weighted` did not beat the window-15 `rel_score` anchor on validation `rel_score`. |
| Cross-sectional IC router | `reports/cross_sectional_ic/anchor_sector19_cross_sectional_ic_20260426_r01` | `sector19_down_up_anchor_else` has the best validation mean daily IC: `+0.0538`, t-stat `+5.51`. |
| Downtrend follow-up audit | `docs/downtrend_expert_findings.md` | Negative result: hard-filter LSTM, downtrend sidecar, and soft downtrend weighting did not beat the current anchor/router enough to keep as active paths. |
| Rank-objective proof | `reports/rank_objective_offline/anchor_sector19_rank_objective_20260427_r01` | `sector19_down_up_anchor_else` is best on validation Spearman IC `+0.0539` and top-bottom equity `4.026`, supporting rank/portfolio objective work. |
| Train-selected rank router | `reports/rank_router_train_selected/anchor_sector19_rank_router_20260427_r01` | New trade-side improvement: `train_rank_regime_ic_weight` raises validation quartile equity to `4.250` and worst-year equity to `1.472`, but does not beat `sector19_down_up_anchor_else` on IC. |
| Stock reliability/bias checks | `reports/stock_reliability_filter/anchor_sector19_stock_reliability_20260427_r01`, `reports/stock_bias_calibration/anchor_sector19_stock_bias_calibration_20260427_r01` | Negative/limited result: dropping weak stocks hurts ranking/trade; stock-bias calibration improves some rel_score variants but reduces IC/equity versus current router candidates. |
| Direct rank-loss sidecar smoke | `reports/portability_ablation/broad_signmag_allvn_20260506_rank_smoke_r01`, `reports/portability_ablation/broad_signmag_allvn_20260506_rank_smoke_r02` | Negative result: all-VN portable no-identity signmag with pairwise rank sidecar weights `0.05` and `0.03` reduced validation `rel_score` and quartile equity versus the base signmag. |
| Ichimoku cycle signal smoke | `reports/signal_search/ichimoku_cycle/vn_ichimoku_8_22_44_signal_search_20260506_r03`, `reports/portability_ablation/broad_signmag_allvn_20260506_ichi_smoke_r01` | Mixed/negative result: `8/21/42` and `8/22/44` Tenkan-Kijun gaps have positive validation IC, but adding either feature directly to the all-VN portable no-identity signmag LSTM reduced smoke validation `rel_score`. |
| Portable LSTM filter signal probe | `reports/filter_signal/portable_lstm_filter_signal_20260508_r04`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module`, `reports/filter_signal/portable_lstm_filter_signal_20260508_r05_signmag`, `reports/filter_signal/filter_signal_stability_20260508.md` | Positive result: base LSTM -> small LSTM tradeability filter -> expected-move daily selector. Plain LSTM `move_top_train_ic_selected` improved validation `rel_score` from base `+0.0049` to `+0.0072` with `40.5%` coverage. Full rerun after extracting `src/models/selection/filter_signal.py` matches pre-refactor `r04` exactly. Signmag confirms the filter layer helps, but its best validation result is weaker (`+0.0056`). |
| Filter selector cost/turnover read | `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/cost_turnover_summary`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/rebalance_cost_summary`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/walk_forward_cost_summary`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/walk_forward_cost_summary_val_split`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/constraint_grid_train_split_t005_040_dd010_025`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/constraint_grid_val_split_t005_040_dd010_025` | Improved result: daily rebalance is not trade-ready after 15 bps cost, but holding-period-aware selection has positive expectation. Train split walk-forward has stitched net equity `1.78` for full-train Sharpe selector and `1.65` for worst-year selector. Validation split rolling check is weaker but still positive for worst-year selector with stitched net equity `1.08`. The expanded constrained worst-year grid finds a better risk-control region: validation stitched net equity `1.27`, Sharpe `+0.73`, max DD `-24.8%`, avg turnover `0.14` with turnover cap `0.25+` and train max DD cap `15-25%`. |
| Modular committee hypothesis grid | `docs/research_design_committee.md`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/committee_hypothesis_grid_train_split_t025_dd15`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/committee_hypothesis_grid_val_split_t025_dd15`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_architecture_eval`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_eval`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_execution`, `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_execution_vn_grid`, `reports/filter_signal/portable_lstm_filter_signal_20260508_r05_signmag/committee_hypothesis_grid_val_split_t025_dd15`, `reports/filter_signal/portable_lstm_filter_signal_20260508_r05_signmag/wyckoff_phase_gate_execution_vn_grid`, `reports/filter_signal/cross_artifact_stability_20260509_r01` | Positive in `r06` but not stable enough for promotion. In `r06`, H1 reaches validation net equity `1.40`, all-candidates reaches `1.31`, the clean VN gate `vn_legacy_acc_all_else` reaches `1.55`, and the aggressive gate reaches `1.81`. The fresh `r05_signmag` retest does not confirm this: all committee hypotheses have validation net equity below `1.0`, and `vn_legacy_acc_all_else` falls to `0.68`. Cross-artifact stability leaves only `train_phase_best` and `vn_h1_distribution_only` positive across both artifacts, but both are too weak/risky for promotion. Treat the committee/Wyckoff layer as a research direction, not a final VN policy. |

## Current Validation Read

- `general_sector_full` remains the main standalone model to trust for prediction quality.
- `phase_ic_sector19` should not replace the anchor as a standalone model, but it adds useful trade signal diversity.
- For prediction scoring, use anchor or the conservative `w_challenger=0.10` ensemble; do not jump straight to the high-trade weight.
- For trade ranking experiments, `w_challenger=0.75` is currently strongest, but it is a router policy on top of the two models rather than a replacement LSTM.
- Objective/window smoke rejects shorter/longer windows for now: `w10` and `w20` both reduce validation `rel_score` and quartile equity.
- `rel_score_sharp` slightly improves signmag trade equity versus anchor but damages `rel_score` too much, so it is only a trade-side clue, not a replacement objective.
- Cross-sectional IC confirms the useful edge is ranking, not raw return calibration: regime router improves validation IC from anchor `+0.0495` to `+0.0538`.
- Anchor is weak in downtrend (`IC=-0.0066`), while sector19 fixes downtrend to `IC=+0.0243`; this is the clearest regime-specific direction found so far.
- Downtrend-specific training has not worked yet: hard-filter LSTM overfits, sidecar variants do not beat sector19 downtrend routing, and soft downtrend weighting reduces overall `rel_score`/equity.
- Offline rank-objective analysis confirms the next useful direction: keep anchor for raw prediction, but optimize/evaluate router candidates with daily ranking and top-bottom portfolio metrics.
- Train-selected rank routing gives a modest trade improvement without using validation for parameter selection; keep it as a trade candidate, not a prediction replacement.
- Stock-level reliability filters should not be used as trade filters yet; they reduce breadth and degrade validation IC/equity.
- Stock-level bias calibration is not worth adding to active routing yet: it improves `train_rank_regime_ic_weight` rel_score from `+0.0037` to `+0.0046`, but lowers IC/equity/worst-year equity.
- Direct pairwise rank loss inside the current signmag LSTM should not be continued for now: `rank003` and `rank005` both damaged validation `rel_score` and quartile equity. Keep rank work outside the model as router/portfolio selection until a better training design is defined.
- Ichimoku-style trading-month windows are useful as standalone diagnostics, not as direct LSTM features yet. The stable signal is the Tenkan-Kijun gap, and `8/21/42`/`8/22/44` both beat standard `9/26/52` in validation IC screening, but the direct all-VN LSTM ablation reduced `rel_score` versus the same 8-epoch anchor smoke.
- A two-stage path is worth continuing: base LSTM prediction -> small LSTM tradeability filter -> train-selected output selector. The best current selector is `move_top_train_ic_selected`: rank by `abs(base_prediction) * filter_probability`, choose daily coverage on train mean IC, and apply that coverage to validation. It improved validation `rel_score` to `+0.0072` with `40.5%` coverage, while preserving validation mean daily IC near the base (`+0.0238` versus base `+0.0242`).
- The reusable selector now lives in `src/models/selection/filter_signal.py`. Use `fit_filter_signal_selection()` on train-only scored frames, then apply the resulting `FilterSignalSelectionParams` with `apply_filter_signal_selection()` on validation/new-market frames.
- Cost/turnover read changes the next priority: daily rebalance versions are not production-ready after a 15 bps cost assumption. The useful direction is now holding-period-aware selection, not a larger neural architecture. Walk-forward checks support positive expectation after cost, but not enough for production: train split walk-forward is strong, while validation split rolling is only mildly positive and still has large drawdown.
- The reusable holding-period logic now lives in `src/models/selection/holding_period.py`. Both rebalance-cost and walk-forward reports should use this module so turnover, cost, Sharpe, drawdown, and train-only selector rules stay consistent across experiments and future inference wiring.
- The expanded constrained holding-period grid is promising. Versus the unconstrained worst-year selector on validation rolling, the best region improves stitched net equity from `1.081` to `1.266` (`+17.1%` relative), Sharpe from `+0.33` to `+0.73`, max drawdown from `-33.8%` to `-24.8%`, and avg turnover from `0.28` to `0.14`. On train rolling, the comparable region keeps net equity near the baseline (`1.61-1.63` versus `1.65`) while reducing turnover and drawdown. Treat this as a risk-control candidate, not a confirmed final policy, because the grid itself is another hypothesis search.
- The first modular committee ablation supports H1: selecting by `abs(base_prediction) * filter_probability` materially improves validation net equity, but the direct H1 policy is not clean enough because it relaxes constraints in one fold and has larger validation drawdown. The all-candidates committee is the better immediate candidate because it improves over the legacy shortlist while keeping turnover/drawdown close and avoiding relaxed folds.
- H2 does not add value yet: multiplying by `market_risk_scale` produced the same selections as H1 in the current artifact. Risk/regime needs a feature or rule that changes ranking/selection, not only output scaling.
- Wyckoff-style regime read confirms the next weakness: the architecture is phase-dependent. On validation, H1 does well in distribution (`1.162`) and transition (`1.246`) but loses in markdown (`0.940`). The all-candidates committee wins markdown (`1.155`) and distribution (`1.103`) but loses badly in accumulation (`0.863`). A Wyckoff/regime gate is now a higher-priority improvement than adding more base LSTM variants.
- VN-first Wyckoff gate is promising only on the plain LSTM `r06` artifact: `vn_legacy_acc_all_else` gives net equity `1.551`, Sharpe `+1.12`, max DD `-23.4%`, and avg turnover `0.15` after recomputed turnover/cost. The `r05_signmag` retest rejects promotion because the same gate drops to net equity `0.676`, Sharpe `-0.77`, and max DD `-45.2%`.
- Aggressive VN Wyckoff gate has the best raw `r06` validation result after execution-level simulation (`1.807`, Sharpe `+1.47`) but also fails the `r05_signmag` confirmation (`0.698`, Sharpe `-0.81`). Treat it as an overfit-risk diagnostic, not a candidate for market transfer.
- Cross-artifact stability summary (`reports/filter_signal/cross_artifact_stability_20260509_r01`) leaves only two non-oracle execution gates positive on both checked VN artifacts: `train_phase_best` with worst-artifact net equity `1.078`, Sharpe `+0.33`, max DD `-31.9%`; and `vn_h1_distribution_only` with worst-artifact net equity `1.010`, Sharpe `+0.16`, max DD `-36.3%`. This is not enough for production, but it points to a narrower next search: phase-conditioned selection may help, while broad fixed gates are brittle.
- Abstain-capable train-phase gate is the best current VN-first improvement (`reports/filter_signal/cross_artifact_stability_abstain_grid_20260509_r01`). `train_phase_abstain_m10_dd20_t25` allows cash when no phase expert clears a 10% train-equity margin versus fallback. It passes both checked VN artifacts and the risk-control screen: worst-artifact net equity `1.111`, mean equity `1.250`, min Sharpe `+0.55`, worst max DD `-23.9%`, max avg turnover `0.16`. This improves materially over raw `train_phase_best` (`1.078`, `+0.33`, `-31.9%`, `0.23`) and over the previous regularized gate's drawdown.
- Suitability check is still not strong enough for promotion. `train_phase_abstain_m10_dd20_t25` is active on `54.5%` of `r06` validation days and `43.4%` of `r05` validation days, so it is not a trivial all-cash rule; however, daily-return t-stats remain low (`+1.13` on `r06`, `+0.67` on `r05`) and `r05` has only `50%` positive folds. Treat it as the current best research candidate, not a deployable or market-transfer policy.
- Transition risk-filter diagnostic is the best current VN validation result. `pressure_delta_20 >= 0` was retested with a strict non-overlap rolling check (`w126/t21/s21`, 24 folds), then checked on a market-leader filter artifact and one fresh no-leader seed-43 artifact. The leading research policy remains `vn_legacy_acc_all_else_transition_pressure_nonneg` with `min_positions=5`: across four artifacts it stays positive with worst-artifact equity `1.240`, mean equity `1.679`, min Sharpe `+0.59`, worst max DD `-24.3%`, max turnover `0.201`. A conservative execution variant with the same policy and `min_positions=6` passes the strict turnover/drawdown screen across four artifacts: worst-artifact equity `1.186`, mean equity `1.460`, min Sharpe `+0.55`, worst max DD `-21.7%`, max turnover `0.179`. This is strong enough for advisor reporting as a VN-first candidate, but not final deployment because the rule is still validation-derived and holdout/test must remain closed until the policy is frozen.
- Stability check against `lstm_signmag_seed_52` supports the filter concept but not a model switch: signmag base validation `rel_score` was only `+0.0011`; the filter improved it to about `+0.0056`, still below the plain LSTM filter candidate.
- Use generic market context names for portability: `market_proxy_return_*`, `market_proxy_volatility_ratio_20`, `market_breadth_20`, `market_ad_ratio_20`, `market_proxy_drawdown_60`, and `market_liquidity_zscore_20`. `vnindex_return` should remain only a source alias behind `market_proxy_return_1`; for new markets prefer `market_index_return`, `index_return`, `benchmark_return`, or fallback equal-weight cross-sectional return.
- Do not apply the current hand-built risk multiplier as final output scaling: `gate_risk_scaled` reduced validation `rel_score` from the hard-gate `+0.0068` to `+0.0014`. Risk/regime features should feed the filter first; output throttling needs separate training-selected calibration.
- Do not use out-sample/test until the model/router is finalized.

## Archived As Ineffective Or Superseded

Moved to `data/processed/assets/data_info_vn/history/training_runs/_archive/ineffective_20260425_phase_research/`:

- r06 hparam variants: lower validation `rel_score` than anchor.
- r07 market-regime feature variants: extra market features hurt or failed to improve.
- r08 compact smoke variants: superseded by r09 or weaker than anchor.
- r09 `phase_ic_sector17_no_level` and `phase_ic_sector20_vol`: weak trade performance despite `sector20_vol` improving `rel_score`.

Moved to `data/processed/assets/data_info_vn/history/training_runs/_archive/ineffective_20260426_objective_window_smoke/`:

- r02 objective/window smoke run dirs: kept the compact report under `reports/feature_pruning/broad_signmag_prune_20260426_r02`, archived duplicate/ineffective run directories.

Deleted after compact reporting:

- 20260427 soft downtrend weighting smoke run dirs: kept compact summaries under `reports/feature_pruning/broad_signmag_prune_20260427_r02` and `reports/regime_analysis/downtrend_focus_smoke_regime_20260427_r02`; removed heavy ineffective run artifacts.

## Next Step

Stop adding window/loss/regime-weight variants until the signal is diagnosed cross-sectionally. The immediate benchmark is:

- anchor `general_sector_full` with `skip_downtrend`
- conservative weighted ensemble `w_challenger=0.10`
- trade weighted ensemble `w_challenger=0.75`
- regime router `sector19_down_up_anchor_else`
- next model-development step should target ranking/portfolio behavior at the router/selection layer, with `sector19_down_up_anchor_else` as the benchmark; do not train another downtrend expert unless it is a clean A/B confirmation run
- current trade-side candidate to compare next: `train_rank_regime_ic_weight` from `anchor_sector19_rank_router_20260427_r01`
- avoid stock-dropping filters for now; if stock-aware work continues, prefer a softer penalty/position sizing rule rather than removing stocks from the cross-section
- keep Ichimoku cycle signals outside the core LSTM for now; if reused, test them as router/regime/portfolio filters instead of adding them to the sequence feature set
- continue the portable LSTM-filter path by validating `move_top_train_ic_selected` through rolling validation and then wiring `src/models/selection/filter_signal.py` plus `src/models/selection/holding_period.py` into inference/reporting
- before training a larger model, retest the constrained holding-period candidate around turnover cap `0.25-0.35` and train max drawdown cap `15-25%` on a fresh run or new market proxy; optimize for net metrics after cost, not only `rel_score`
- continue VN-first committee work by freezing `vn_legacy_acc_all_else_transition_pressure_nonneg` with `pressure_delta_20 >= 0`; the overlapping strict split was discarded, and the non-overlap strict split is positive across four validation artifacts but still validation-derived, so do not touch holdout/test or other markets yet
