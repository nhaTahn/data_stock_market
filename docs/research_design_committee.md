# Modular Expert Committee Research Design

Updated: 2026-05-09

## Objective

Evaluate whether a modular expert committee can improve cost-aware trading performance over a single LSTM forecast signal without opening holdout/test data.

The committee is treated as a selection and execution layer, not as a larger monolithic forecasting model.

## Hypotheses

- H0 forecast-only: rank names by raw forecast magnitude.
- H1 tradeability filter: rank names by `abs(base_prediction) * filter_probability`.
- H2 risk-conditioned filter: rank names by `abs(base_prediction) * filter_probability * market_risk_scale`.
- H3 rank committee: rank names by the average daily percentile rank of forecast magnitude, filter probability, and market risk scale.
- Legacy shortlist: keep the current hand-selected filter candidates for continuity.
- All-candidates committee: allow the train-window selector to choose across all committee candidates.

## Validation Protocol

- Data source: `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/filter_predictions.csv.gz`.
- Confirmation source: `reports/filter_signal/portable_lstm_filter_signal_20260508_r05_signmag/filter_predictions.csv.gz`.
- Holdout/test data is not used.
- Transaction cost: 15 bps per unit turnover.
- Train-window selector: constrained worst-year net equity.
- Selector constraints:
  - train average turnover <= 0.25
  - train max drawdown no worse than 15%
- Candidate coverage grid: 5%, 10%, 15%, 20%, 25%, 30%, 40%.
- Validation split walk-forward:
  - train window: 252 trading days
  - test window: 63 trading days
  - step: 63 trading days
- Train split walk-forward:
  - train window: 756 trading days
  - test window: 126 trading days
  - step: 126 trading days

## Current Results

Primary artifact: `portable_lstm_filter_signal_20260509_r06_selector_module`.

Validation split:

| Hypothesis | Net equity | Sharpe | Max DD | Avg turnover | Positive folds | Relaxed folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 tradeability filter | 1.401 | +0.91 | -31.6% | 0.26 | 66.7% | 1 |
| H2 risk-conditioned | 1.401 | +0.91 | -31.6% | 0.26 | 66.7% | 1 |
| All-candidates committee | 1.306 | +0.73 | -25.3% | 0.14 | 83.3% | 0 |
| Legacy shortlist | 1.266 | +0.73 | -24.8% | 0.14 | 66.7% | 0 |
| H3 rank committee | 1.211 | +0.62 | -28.7% | 0.13 | 83.3% | 1 |
| H0 forecast-only | 1.082 | +0.33 | -31.7% | 0.14 | 83.3% | 1 |

Train split:

| Hypothesis | Net equity | Sharpe | Max DD | Avg turnover | Positive folds | Relaxed folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-candidates committee | 1.776 | +0.99 | -15.2% | 0.13 | 100.0% | 0 |
| Legacy shortlist | 1.614 | +0.93 | -14.8% | 0.10 | 66.7% | 0 |
| H0 forecast-only | 1.604 | +0.87 | -23.2% | 0.12 | 88.9% | 0 |
| H1 tradeability filter | 1.484 | +0.75 | -14.4% | 0.10 | 77.8% | 0 |
| H2 risk-conditioned | 1.484 | +0.75 | -14.4% | 0.10 | 77.8% | 0 |
| H3 rank committee | 1.335 | +0.54 | -14.7% | 0.11 | 55.6% | 0 |

## Interpretation

- H1 is the strongest validation equity result and supports the tradeability-filter hypothesis.
- H2 currently adds no measurable value over H1; `market_risk_scale` is not differentiating selections enough in this artifact.
- H3 improves over H0 but does not beat H1 or the legacy shortlist.
- The all-candidates committee is the best balanced candidate: it improves validation net equity over the legacy shortlist while keeping turnover and drawdown close to the current constrained policy.
- H1 should not be promoted directly yet because one validation fold relaxed constraints and validation drawdown is worse than the legacy shortlist.

## Wyckoff-Style Regime Read

Report: `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_architecture_eval`.

Phase assignment is point-in-time on the signal date, using VN quality-dataset market breadth, market return, pressure features, and `wyckoff_phase_60d`.

Validation split phase distribution:

| Phase | Days | Share |
| --- | ---: | ---: |
| Accumulation | 30 | 7.9% |
| Markup | 32 | 8.5% |
| Distribution | 64 | 16.9% |
| Markdown | 56 | 14.8% |
| Transition | 196 | 51.9% |

Key validation results:

| Hypothesis | Accumulation | Markup | Distribution | Markdown | Transition |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 tradeability filter | 0.991 | 1.039 | 1.162 | 0.940 | 1.246 |
| All-candidates committee | 0.863 | 1.012 | 1.103 | 1.155 | 1.173 |
| Legacy shortlist | 1.024 | 1.031 | 1.077 | 0.986 | 1.129 |

Wyckoff interpretation:

- The architecture is useful, but not uniformly robust by market phase.
- H1 is strongest overall but fails markdown and has higher validation drawdown; it needs a markdown risk gate before promotion.
- The all-candidates committee is more balanced in markdown/distribution, but it performs poorly in accumulation.
- The legacy shortlist is more conservative in accumulation, but weaker overall.
- H2 still adds no value over H1, which means the current risk scale is not a true regime expert.
- A production-grade committee should add a Wyckoff/regime gate that can switch away from H1 in markdown and away from all-candidates committee in accumulation.

## Wyckoff Phase Gate Diagnostic

Report: `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_eval`.

This is a return-splice diagnostic: it switches between already simulated hypothesis daily returns by Wyckoff phase. It is useful for measuring regime-gate potential, but it is not yet an execution-perfect backtest because additional turnover from switching between policy portfolios is not fully recomputed.

Validation results:

| Gate | Net equity | Sharpe | Max DD | Avg turnover | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| Conservative prior | 1.579 | +1.16 | -22.3% | 0.14 | Best clean candidate so far; improves equity and drawdown versus H1/all/legacy. |
| Aggressive prior | 1.779 | +1.44 | -24.6% | 0.22 | Strong but relies heavily on H1 and should be treated as higher overfit risk. |
| Train phase best | 1.252 | +0.65 | -28.3% | 0.13 | Fails because train selected H0 for distribution, which did not transfer to validation. |
| Oracle eval phase best | 1.824 | +1.55 | -25.3% | 0.17 | Upper bound only; not a tradable policy. |

Conservative prior map:

| Phase | Policy |
| --- | --- |
| Accumulation | `legacy_filter_shortlist` |
| Markup | `legacy_filter_shortlist` |
| Distribution | `all_committee_candidates` |
| Markdown | `all_committee_candidates` |
| Transition | `all_committee_candidates` |

Gate interpretation:

- The Wyckoff gate can materially improve the architecture if it avoids high-turnover H1 in fragile phases.
- Pure train-selected phase best is not reliable yet because phase samples are small and phase performance is unstable.
- The conservative prior is the next candidate to convert into an execution-level backtest.
- The aggressive prior and oracle result show upside potential, but they should not be used as final policy without fresh-market confirmation.

## Wyckoff Phase Gate Execution Simulation

Report: `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_execution`.

This simulation recomputes positions, turnover, and transaction cost when the gate switches between policies by Wyckoff phase.

Validation results:

| Policy | Net equity | Sharpe | Max DD | Avg turnover | Policy changes | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Aggressive prior | 1.807 | +1.47 | -24.8% | 0.25 | 16 | Best equity, but higher turnover and more H1 exposure. Treat as high-upside candidate. |
| Conservative prior | 1.488 | +1.03 | -23.4% | 0.17 | 30 | Best clean candidate after execution-level costs; improves over H1/all/legacy. |
| Baseline H1 | 1.401 | +0.91 | -31.6% | 0.26 | 6 | Strong, but worse drawdown and markdown weakness. |
| Baseline all-committee | 1.306 | +0.73 | -25.3% | 0.14 | 6 | Balanced baseline, but weaker than conservative gate. |
| Baseline legacy | 1.266 | +0.73 | -24.8% | 0.14 | 6 | Conservative baseline. |
| Train phase best | 1.213 | +0.58 | -28.6% | 0.22 | 47 | Not robust; train phase-specific selection overfits. |

Execution interpretation:

- The conservative Wyckoff gate survives recomputed turnover/cost and remains the strongest clean candidate.
- The return-splice diagnostic overstated conservative prior net equity (`1.579` vs execution `1.488`), but the edge did not disappear.
- The aggressive prior is now the best raw equity result, but it uses H1 in markup/distribution/transition and should be retested on a fresh artifact before promotion.
- Train-selected phase best remains weak; do not select a different expert per phase using only phase-local train equity until more data or regularization is added.

## VN-First Gate Grid

Report: `reports/filter_signal/portable_lstm_filter_signal_20260509_r06_selector_module/wyckoff_phase_gate_execution_vn_grid`.

The VN-first grid keeps the number of tested gates small and uses the observed VN phase weaknesses:

- `legacy_filter_shortlist` is safer in accumulation;
- `all_committee_candidates` is stronger in markdown/distribution than legacy;
- H1 adds upside in markup/distribution/transition but increases turnover and overfit risk.

Validation execution-level results:

| Gate | Net equity | Sharpe | Max DD | Avg turnover | Policy changes | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `vn_h1_markup_distribution_transition` | 1.807 | +1.47 | -24.8% | 0.25 | 16 | Same as aggressive prior; best equity but higher risk. |
| `vn_h1_distribution_transition` | 1.586 | +1.18 | -26.0% | 0.26 | 32 | Strong equity but turnover/drawdown worse than the clean gate. |
| `vn_legacy_acc_all_else` | 1.551 | +1.12 | -23.4% | 0.15 | 14 | Best clean VN-first candidate. |
| `conservative_prior` | 1.488 | +1.03 | -23.4% | 0.17 | 30 | Previous clean candidate; superseded by `vn_legacy_acc_all_else`. |
| `baseline_h1` | 1.401 | +0.91 | -31.6% | 0.26 | 6 | Good but drawdown risk remains. |
| `baseline_all_committee` | 1.306 | +0.73 | -25.3% | 0.14 | 6 | Balanced baseline. |
| `baseline_legacy` | 1.266 | +0.73 | -24.8% | 0.14 | 6 | Conservative baseline. |

Current VN-first recommendation:

- Primary clean candidate inside `r06` only: `vn_legacy_acc_all_else`.
- High-upside challenger inside `r06` only: `vn_h1_markup_distribution_transition`.
- Do not promote either policy until it survives a fresh VN confirmation.

## Fresh VN Confirmation

Confirmation artifact: `reports/filter_signal/portable_lstm_filter_signal_20260508_r05_signmag`.

The same committee and execution-gate protocol was rerun on the signmag filter artifact. This is still VN validation data and does not open holdout/test.

Committee validation split:

| Hypothesis | Net equity | Sharpe | Max DD | Avg turnover | Positive folds | Relaxed folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 forecast-only | 0.873 | -0.16 | -35.5% | 0.13 | 16.7% | 1 |
| Legacy shortlist | 0.838 | -0.50 | -32.9% | 0.10 | 33.3% | 0 |
| H3 rank committee | 0.816 | -0.47 | -30.0% | 0.15 | 33.3% | 1 |
| H1 tradeability filter | 0.712 | -0.73 | -35.6% | 0.14 | 33.3% | 0 |
| H2 risk-conditioned | 0.712 | -0.73 | -35.6% | 0.14 | 33.3% | 0 |
| All-candidates committee | 0.704 | -0.69 | -42.9% | 0.11 | 16.7% | 0 |

Execution-level VN gate confirmation:

| Gate | Net equity | Sharpe | Max DD | Avg turnover | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `train_phase_best` | 1.078 | +0.33 | -31.9% | 0.23 | Positive but phase-local and unstable. |
| `vn_h1_distribution_only` | 1.010 | +0.16 | -36.3% | 0.24 | Barely positive, not enough to promote. |
| `baseline_legacy` | 0.838 | -0.50 | -32.9% | 0.10 | Weak baseline on this artifact. |
| `baseline_all_committee` | 0.704 | -0.69 | -42.9% | 0.11 | Fails confirmation. |
| `vn_h1_markup_distribution_transition` | 0.698 | -0.81 | -36.2% | 0.16 | Rejects aggressive `r06` gate. |
| `vn_legacy_acc_all_else` | 0.676 | -0.77 | -45.2% | 0.14 | Rejects clean `r06` gate. |

Confirmation interpretation:

- The `r06` VN-first gate is artifact-specific so far.
- Signmag confirms that a filter layer can improve prediction metrics, but it does not confirm the cost-aware committee/gate execution layer.
- The current problem is not lack of more Wyckoff rules; it is unstable expert quality across base model families.
- Do not transfer this architecture to other markets yet. First stabilize the VN base filter/selector or require a gate rule to pass both plain LSTM and signmag VN artifacts.

## Cross-Artifact Stability

Report: `reports/filter_signal/cross_artifact_stability_20260509_r01`.

Worst-artifact ranking across `r06` plain LSTM and `r05_signmag`:

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gate | `train_phase_best` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Only robust positive gate with a real margin, but still weak. |
| Gate | `vn_h1_distribution_only` | 1.010 | 1.212 | +0.16 | -36.3% | 0.24 | Barely positive and high drawdown. |
| Gate | `vn_h1_transition_only` | 0.943 | 1.094 | -0.04 | -25.4% | 0.28 | Does not pass both artifacts. |
| Committee | `h0_forecast_abs` | 0.873 | 0.977 | -0.16 | -35.5% | 0.14 | Best committee by worst artifact, but still below breakeven. |

Stability interpretation:

- No fixed committee hypothesis is robust across both checked VN artifacts.
- The only positive non-oracle results are execution gates, not standalone committee policies.
- `train_phase_best` deserves a stricter follow-up because it adapts the policy map by train evidence, but it needs more regularization: fewer phase switches, minimum phase sample size, and a hard drawdown/turnover filter.
- `vn_h1_distribution_only` suggests H1 exposure may be most defensible only in distribution, not broadly in markup/transition.

Regularized follow-up report: `reports/filter_signal/cross_artifact_stability_regularized_train_20260509_r01`.

| Kind | Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gate | `train_phase_regularized_m05_dd20_t25` | 1.078 | 1.218 | +0.33 | -31.9% | 0.23 | Best current cross-artifact gate, but still not deployment-grade. |
| Gate | `train_phase_best` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Lower mean equity than the regularized 5% margin gate. |
| Gate | `train_phase_regularized_m02_dd20_t25` | 1.078 | 1.146 | +0.33 | -31.9% | 0.23 | Same realized mapping as raw train phase best in this retest. |
| Gate | `vn_h1_distribution_only` | 1.010 | 1.212 | +0.16 | -36.3% | 0.24 | Positive on both artifacts but too fragile. |

Regularized gate interpretation:

- A 5% train-equity margin avoids the weak `r06` distribution H0 switch and improves `r06` from net equity `1.213` to `1.358`.
- The same rule does not improve the weak `r05` worst artifact, so the bottleneck is still base-family signal quality.
- This is the best current direction for VN-first work, but not enough to move to holdout/test or other markets.

## Abstain-Capable Phase Gate

Report: `reports/filter_signal/cross_artifact_stability_abstain_grid_20260509_r01`.

The next risk-control variant lets a phase go to cash (`__cash__`) when no train-phase expert clears the margin/drawdown/turnover hurdle. This tests whether the committee should abstain in weak VN regimes instead of forcing fallback exposure.

Risk-control screen:

- positive net equity on both checked VN artifacts;
- worst-artifact max drawdown no worse than 25%;
- max average turnover no higher than 0.20.

Best risk-controlled cross-artifact policies:

| Policy | Min equity | Mean equity | Min Sharpe | Worst DD | Max turnover | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `train_phase_abstain_m10_dd20_t25` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Best current VN-first candidate. |
| `train_phase_abstain_m10_dd15_t20` | 1.111 | 1.250 | +0.55 | -23.9% | 0.16 | Same realized mapping in this grid. |
| `train_phase_abstain_m10_dd10_t15` | 1.032 | 1.072 | +0.21 | -24.7% | 0.18 | Safer but too much opportunity loss. |

Artifact-level read for `train_phase_abstain_m10_dd20_t25`:

| Artifact | Net equity | Sharpe | Daily t-stat | Positive folds | Worst fold | Max DD | Active days | Avg turnover | Avg positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r06_selector_module` | 1.388 | +0.93 | +1.13 | 83.3% | 0.821 | -23.9% | 54.5% | 0.16 | 8.1 |
| `r05_signmag` | 1.111 | +0.55 | +0.67 | 50.0% | 0.934 | -23.9% | 43.4% | 0.15 | 7.0 |

Policy maps:

| Artifact | Accumulation | Markup | Distribution | Markdown | Transition |
| --- | --- | --- | --- | --- | --- |
| `r06_selector_module` | `__cash__` | `__cash__` | `__cash__` | `all_committee_candidates` | `all_committee_candidates` |
| `r05_signmag` | `__cash__` | `__cash__` | `__cash__` | `__cash__` | `all_committee_candidates` |

Abstain-gate interpretation:

- This is the first VN-first candidate that passes both checked artifacts and a drawdown/turnover screen.
- The improvement comes from not forcing exposure in weak or low-confidence phases, especially accumulation/markup/distribution.
- It is not a trivial all-cash policy because it remains active on `54.5%` of `r06` days and `43.4%` of `r05` days.
- It is still not deployment-grade because daily t-stats are low and `r05` has only `50%` positive folds.
- The policy is still validation-only and should not be promoted to holdout/test yet.
- The next test should confirm whether the cash/transition-heavy behavior survives a fresh plain LSTM run, stricter walk-forward, or a different VN period.

## Transition Risk Filter Diagnostic

Report: `reports/filter_signal/cross_artifact_stability_transition_risk_filter_20260511_r01`.

After the suitability check, the weak area was late transition exposure. A point-in-time pressure filter was tested as a diagnostic: keep the robust-abstain gate, but trade transition only when `pressure_delta_20 >= 0`; otherwise go to cash.

Best diagnostic policy:

`train_phase_robust_abstain_m10_dd20_t25_ts050_pf60_transition_pressure_nonneg`

Cross-artifact read:

| Metric | Value |
| --- | ---: |
| Worst-artifact net equity | 1.344 |
| Mean net equity | 1.570 |
| Min Sharpe | +1.73 |
| Worst max DD | -22.6% |
| Max avg turnover | 0.15 |

Artifact-level diagnostics:

| Artifact | Net equity | Sharpe | Daily t-stat | Positive folds | Worst fold | Max DD | Active days | Avg turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r06_selector_module` | 1.795 | +1.73 | +2.12 | 66.7% | 0.919 | -22.6% | 40.5% | 0.15 |
| `r05_signmag` | 1.344 | +2.04 | +2.50 | 50.0% | 0.986 | -4.6% | 25.1% | 0.13 |

Phase read:

| Artifact | Main active phases | Transition equity | Transition active days |
| --- | --- | ---: | ---: |
| `r06_selector_module` | Markdown + filtered transition | 1.525 | 97 / 197 |
| `r05_signmag` | Filtered transition only | 1.373 | 95 / 197 |

Interpretation:

- The pressure filter is the strongest diagnostic found so far for VN risk control.
- It materially improves t-stat, drawdown, and worst-fold damage versus the plain abstain gate.
- It was discovered after inspecting validation transition behavior, so it must be treated as a new hypothesis, not a confirmed policy.
- Promotion requires a fresh VN artifact, stricter rolling split, or a pre-registered rerun before holdout/test or market transfer.

## Strict Non-Overlap Rolling Check

Report: `reports/filter_signal/cross_artifact_stability_transition_risk_filter_strict_nonoverlap_20260511_r01`.

The first strict run used `test=42, step=21`, which overlaps test windows and therefore double-counts some validation days in stitched equity. That overlap run is diagnostic only and should not be used for promotion.

The corrected strict check uses non-overlapping validation folds:

- train window: 126 trading days;
- test window: 21 trading days;
- step: 21 trading days;
- folds: 24 per artifact;
- holdout/test data is not used.

Best risk-controlled policy:

`vn_legacy_acc_all_else_transition_pressure_nonneg`

Policy map:

| Phase | Policy |
| --- | --- |
| Accumulation | `legacy_filter_shortlist` |
| Markup | `all_committee_candidates` |
| Distribution | `all_committee_candidates` |
| Markdown | `all_committee_candidates` |
| Transition | `all_committee_candidates` only when `pressure_delta_20 >= 0`, otherwise `__cash__` |

Cross-artifact strict non-overlap read:

| Metric | Value |
| --- | ---: |
| Worst-artifact net equity | 1.681 |
| Mean net equity | 1.802 |
| Min Sharpe | +1.29 |
| Worst max DD | -24.3% |
| Max avg turnover | 0.19 |

Artifact diagnostics:

| Artifact | Net equity | Sharpe | Daily t-stat | Positive folds | Worst fold | Max DD | Active days | Avg turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r06_selector_module` | 1.923 | +1.45 | +2.06 | 62.5% | 0.866 | -21.7% | 67.7% | 0.19 |
| `r05_signmag` | 1.681 | +1.29 | +1.83 | 50.0% | 0.886 | -24.3% | 68.1% | 0.19 |

Strict interpretation:

- The pressure filter survives a stricter non-overlap rolling validation and improves the simple VN gate more than the robust-abstain gate.
- The rule is still validation-derived, so it is not a final policy.
- Positive-fold rate is still not high enough on `r05` (`50.0%`), so the next confirmation should use a fresh VN artifact or a pre-registered rerun.
- If it survives fresh confirmation, this simple pressure-filtered gate is a better candidate than the more complex robust-abstain rule.

## Next Optimization

The next useful experiments are:

- Add a strict no-relax mode and compare H1 against the all-candidates committee.
- Add a second risk feature or regime proxy that changes ranking, not only output scale.
- Add a Wyckoff phase gate:
  - avoid H1 during markdown unless the train-window evidence is strong;
  - avoid all-candidates committee during accumulation;
  - prefer legacy shortlist or low-turnover rules in accumulation/early recovery.
- Replace the current fixed VN gate with a stability-constrained rule that must pass both `r06` plain LSTM and `r05_signmag`, or explicitly route away from weak base families.
- Next concrete test: validate `vn_legacy_acc_all_else_transition_pressure_nonneg` on a fresh VN artifact or pre-registered rerun, with the threshold fixed as `pressure_delta_20 >= 0`.
- Retest H1 and all-candidates committee on a fresh VN run before any new-market proxy.
- Keep the objective cost-aware: net equity, Sharpe, max drawdown, turnover, positive-fold rate, and worst-fold equity.
