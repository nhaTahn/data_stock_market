# VN Regime-Risk Paper Blueprint

Updated: 2026-04-28

## Goal

Draft a paper direction that matches the current repo better than a generic
"LSTM beats ARIMA" story.

This blueprint is designed to reuse the strongest parts of the repo:

- `rel_score` as the main prediction metric
- cross-sectional IC and quartile equity as economic-read metrics
- regime-aware analysis already present in the VN research path
- uncertainty-aware and event-aware model heads already present in code
- backtest/reporting pipeline already present in the repo

## Recommended Paper Direction

Recommended main direction:

**Regime-Aware and Risk-Calibrated Return Predictability in the Vietnamese Stock Market**

This direction is stronger than a plain LSTM forecasting paper because it lets
the work make three claims at once:

1. the VN market is not fully weak-form efficient in all regimes;
2. deep sequence models extract predictive edge unevenly across regimes;
3. adding risk-aware filtering or selective prediction improves trading utility
   beyond raw point prediction.

## Why This Direction Fits The Attached Papers

### Compared with `vnstock.pdf`

`vnstock.pdf` studies weak-form efficiency and return predictability in Vietnam.
The natural extension here is:

- move from classical random-walk tests to sequence models;
- evaluate predictability by regime, not only by full-sample average;
- show whether ML edge is concentrated in `downtrend`, `recovery`, or
  `distribution` periods.

### Compared with `risk.pdf`

`risk.pdf` frames LSTM as a financial risk predictor. The repo can go further by
linking risk outputs to trading decisions:

- event probability for large adverse move or downside risk;
- quantile spread as prediction uncertainty;
- abstain or size down when model uncertainty is high.

That creates a paper with economic value, not just classification accuracy.

### Compared with `signal.pdf`

`signal.pdf` motivates multi-step and tokenized representations. The repo
already has:

- `signal` architecture
- `pcie_lite` architecture
- `return`, `return_3d`, and `return_5d` target modes

So a second-stage paper can extend the main VN regime-risk paper into a
multi-step sequence paper.

### Compared with `lstm.pdf`

A generic "LSTM + sentiment + web app" story is too weak for this repo.
The current codebase is already richer in:

- robust metrics
- cross-sectional ranking diagnostics
- risk-aware outputs
- trade evaluation

So the paper should emphasize market structure, regime dependence, and selective
decision quality instead.

## Candidate Titles

Use one of these as a working paper title:

1. `Regime-Aware and Risk-Calibrated Return Predictability in the Vietnamese Stock Market`
2. `Beyond Point Forecasting: Regime-Specific and Risk-Aware Deep Learning for Vietnamese Equities`
3. `Weak-Form Predictability, Regime Shifts, and Selective Trading Signals in the Vietnamese Stock Market`

## Draft Abstract

This study investigates whether return predictability in the Vietnamese stock
market can be extracted more effectively by deep sequence models when market
regimes and prediction risk are modeled explicitly. While prior work on Vietnam
has mainly focused on weak-form efficiency tests and full-sample return
predictability, we examine whether predictive edge is stable across market
states such as uptrend, downtrend, recovery, and distribution. We build on a
regime-aware LSTM research pipeline with robust return calibration via
`rel_score`, cross-sectional ranking evaluation via daily Spearman information
coefficient, and trade-side evaluation via quartile long-short equity and
drawdown diagnostics. In addition to point prediction, we study risk-calibrated
extensions including sign-magnitude decomposition, event-gated outputs for large
move detection, and uncertainty-aware trade filtering. Our results are designed
to answer three questions: whether the Vietnamese market exhibits regime-varying
weak-form predictability, whether sequence models improve forecasting quality
over classical baselines, and whether risk-aware abstention improves realized
trade utility beyond raw prediction accuracy. The proposed framework aims to
bridge market efficiency analysis, deep sequence modeling, and decision-aware
evaluation in an emerging market setting.

## Main Research Questions

The paper should be built around these questions:

1. Does the Vietnamese stock market show different levels of return
   predictability across market regimes?
2. Does the current LSTM family improve robust return calibration and
   cross-sectional ranking versus classical baselines?
3. Can risk-aware outputs improve trade quality by filtering or reweighting
   predictions with high downside risk or high uncertainty?

## Proposed Contributions

Keep the contribution list narrow and testable.

### Contribution 1

**Regime-aware return predictability benchmark for VN equities.**

This contribution formalizes that return predictability should not be reported
only at full-sample level. Instead, it should be decomposed by regime:

- `uptrend`
- `downtrend`
- `recovery`
- `distribution`
- `sideways`

Deliverables:

- regime-specific `rel_score`
- regime-specific daily Spearman IC
- regime-specific quartile equity and drawdown

### Contribution 2

**Risk-calibrated selective prediction rather than pure point prediction.**

This contribution reframes the task from:

- "predict next return"

to:

- "predict next return and know when not to trust the model"

Deliverables:

- `event_prob` head for downside or large-move risk
- `prediction_uncertainty` from quantile spread
- uncertainty-gated or event-gated threshold backtests

### Contribution 3

**Decision-aware evaluation for an emerging market.**

This contribution says a useful forecasting paper for finance should not end at
MSE or MAE. The repo already supports a stronger evaluation stack:

- `rel_score`
- directional accuracy
- cross-sectional IC / IC t-stat
- quartile long-short equity
- drawdown / hit rate

This is a genuine paper-level improvement over generic forecasting studies.

## Strongest Repo Assets To Reuse

### Modeling

- [src/models/architectures/signmag.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/signmag.py)
- [src/models/architectures/event.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/event.py)
- [src/models/architectures/quantile.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/quantile.py)
- [src/models/architectures/signal.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/signal.py)
- [src/models/architectures/pcie_lite.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures/pcie_lite.py)

### Training / orchestration

- [src/models/training/pipeline.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/pipeline.py)
- [src/models/training/targets.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/targets.py)
- [src/models/training/sample_weights.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training/sample_weights.py)

### Evaluation / reporting

- [src/evaluation/metric.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/evaluation/metric.py)
- [src/reporting/standard_report.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/reporting/standard_report.py)
- [src/backtesting/threshold_backtest.py](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting/threshold_backtest.py)

### Existing research evidence

- [docs/current_best_path.md](/Users/lap15111/Documents/research-paper/data_stock_market/docs/current_best_path.md)
- [docs/current_research_status.md](/Users/lap15111/Documents/research-paper/data_stock_market/docs/current_research_status.md)
- [docs/optimization_and_ood_readiness_20260428.md](/Users/lap15111/Documents/research-paper/data_stock_market/docs/optimization_and_ood_readiness_20260428.md)
- [docs/downtrend_expert_findings.md](/Users/lap15111/Documents/research-paper/data_stock_market/docs/downtrend_expert_findings.md)

## Minimal Paper Story

If the goal is to publish something realistic from the current repo, the
smallest defensible story is:

1. establish that VN predictability is regime-dependent;
2. show that the current sign-magnitude family is a stronger prediction model
   than classical baselines;
3. show that uncertainty-aware or event-aware filtering improves trade utility;
4. validate with walk-forward or rolling splits, not only one static split.

## Recommended Experiment Matrix

The paper should not try to compare every architecture in the repo.

### Baselines

- `ARIMA`
- `linear_regression`
- `plain LSTM`
- `LSTM sign-magnitude`

### Risk-aware candidates

- `event-gated LSTM`
- `quantile LSTM` with uncertainty-gated threshold backtest

### Optional architecture section

Only if time allows:

- `signal`
- `pcie_lite`

These should be framed as secondary architecture probes, not the paper's main
claim.

## Core Tables

Prepare the paper around a small set of tables.

### Table A

**Full-sample validation / test metrics**

Columns:

- model
- rel_score
- directional accuracy
- mean daily IC
- IC t-stat
- quartile equity
- max drawdown

### Table B

**Metrics by regime**

Rows:

- anchor
- signmag
- event-gated
- quantile + uncertainty filter

Columns:

- regime
- rel_score
- IC
- quartile equity
- drawdown

### Table C

**Selective prediction / risk calibration**

Rows:

- no filter
- low-uncertainty filter
- high-event-risk reject
- combined filter

Columns:

- coverage
- rel_score on traded rows
- avg trade return
- final equity
- max drawdown

## Core Figures

1. Regime timeline on VN data
2. IC by regime
3. Equity curves with and without uncertainty filter
4. Error / `rel_score` histograms
5. Coverage versus performance curve for uncertainty gating

## What Is Missing Today

These are the most important missing pieces if this is to become a clean paper.

### Missing Piece 1

**Walk-forward evaluation**

Current repo is strong on train/val/test, but a paper should add rolling or
walk-forward evaluation to reduce overfitting risk.

### Missing Piece 2

**Formal significance reporting**

The paper should report:

- IC t-stat
- positive IC days
- bootstrap or rolling confidence interval for trade equity
- paired comparison where possible

### Missing Piece 3

**Transaction cost sensitivity**

Even a simple friction grid would make the paper much stronger:

- 0 bps
- 10 bps
- 20 bps
- 30 bps

### Missing Piece 4

**Clear risk target definition**

If using `event-gated`, define exactly what risk event means:

- large absolute move
- downside return below threshold
- future drawdown over next `k` days

For paper quality, downside-risk or drawdown-event is better than a generic
"large move" event.

## Best Immediate Implementation Path

Do not try to build everything at once. The most practical path is:

1. keep `signmag` as prediction anchor;
2. add a clean downside-risk event target;
3. add uncertainty-gated backtest packaging;
4. add walk-forward evaluation script;
5. produce regime table + risk table + equity plots.

## Suggested New Work Packages

### Work Package 1

**Regime-aware evaluation package**

Needed outputs:

- per-regime metrics CSV
- per-regime backtest summary
- figure-ready summary table

### Work Package 2

**Risk-aware selective trading package**

Needed outputs:

- event-target definition options
- uncertainty-gated backtests
- coverage/performance tradeoff report

### Work Package 3

**Paper-grade rolling validation package**

Needed outputs:

- walk-forward run launcher
- rolling IC / equity summaries
- train/test degradation summary

## What Not To Do

Avoid these as the main paper direction:

- generic sentiment integration without a strong point-in-time pipeline
- adding many features without regime-level diagnosis
- claiming generalization to `JP/US` before the portable branch is fully stable
- optimizing only for one holdout split and then writing a paper around it

## Recommended Thesis Statement

Use this as the one-sentence center of the paper:

**In the Vietnamese equity market, predictive edge is regime-dependent, and a
deep sequence model becomes more practically useful when its return forecast is
paired with explicit risk calibration and selective decision rules.**

## Recommended Next Build

If turning this blueprint into actual research code, the next concrete step
should be:

1. implement `downside-risk` event target mode;
2. generate `per-regime` and `uncertainty-gated` reports for the current VN
   anchor and challenger models;
3. add a walk-forward experiment driver for the final paper tables.

