# RelScore To Quantile Roadmap

## Why This Roadmap Exists

The repo has grown enough that it is easy to add model families faster than we can understand them. The current priority is not to build the biggest framework. The priority is:

- reduce moving parts
- keep the main objective aligned with `rel_score`
- improve stability on VN runs
- add only one meaningful modeling idea at a time

This roadmap is the narrow path the repo should follow unless a specific experiment justifies broader scope.

## Current Baseline To Keep

Keep these assumptions fixed unless there is evidence they are the bottleneck:

- market focus: VN
- task focus: next-return forecasting
- main metric: `test rel_score`
- default training loss: `rel_score`
- main data unit: sector-wide pools or mini-groups
- framework: TensorFlow / Keras

## What Not To Add Yet

Do not expand into these areas yet:

- sentiment or news pipelines
- multi-modal text alignment
- TRA or router-based multi-head dispatch
- full long-short portfolio framework rewrite
- large object-model refactor for its own sake
- too many new hyperparameters at once

These may become useful later, but they will slow diagnosis right now.

## Near-Term Development Plan

Status note:

- Phase 2 now has a first implementation in repo via `--enable-quantile-family`.
- It should still be treated as experimental until it beats the plain `lstm` baseline on repeatable VN runs.
- Phase 3 now has a first practical backtest-side implementation in [`src/models/backtest_threshold.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/backtest_threshold.py).
- On the current strongest VN mini-group run, `q90 - q50` worked better as a `high-spread` sidecar gate than as a `low-uncertainty` rejection filter.

### Phase 1: Stabilize The RelScore Baseline

Goal:

- make `rel_score` the default objective everywhere
- reduce day-to-day ambiguity about which model family to trust

Scope:

- treat plain `lstm` as the primary benchmark
- keep `signmag` as a secondary branch, not the first thing to tune
- leave attention and event models off by default
- compare sector-wide vs mini-group only where search summaries support it

Success criteria:

- reproducible runs with clear `config.json`
- sector/mini-group comparisons tracked in summary CSVs
- more runs with positive `test rel_score`

### Phase 2: Add Minimal Quantile Forecasting

Goal:

- upgrade from one point forecast to two quantiles: `q50` and `q90`

Scope:

- keep the current sequence pipeline and dataset build flow
- replace the scalar output head with a 2-value output head
- train with:

```text
pinball(q50) + 0.5 * pinball(q90)
```

- evaluate `q50` with the existing `rel_score`
- keep backtests driven by `q50`

Why this is the next step:

- it stays close to the current `rel_score` design because the metric already emphasizes median and tail scale
- it adds useful uncertainty information without introducing a large architecture jump

Success criteria:

- `q50` matches or beats the current point-forecast `rel_score`
- `q90` is stable enough to use as a risk or confidence signal

### Phase 3: Use Quantile Spread As A Filter

Goal:

- convert `q90 - q50` into a practical uncertainty measure

Scope:

- keep the model unchanged
- add optional filters in backtest or post-processing
- test whether wide-spread predictions should be down-weighted, skipped, or explicitly favored

Why this matters:

- it adds a risk-aware control without needing a more complex architecture
- it lets the repo learn whether `q90 - q50` behaves like uncertainty or like upside-spread on VN data

Success criteria:

- improved threshold backtests without degrading core `rel_score`

Current reading of the first real run:

- `q90 - q50` should not automatically be treated as "smaller is better"
- for the F&B mini-group winner, the best backtest kept rows with `high` spread rather than `low` spread
- this means the current two-quantile head is more useful as a sidecar scoring signal than as a replacement model family

### Phase 4: Revisit Advanced Modules Only If Needed

Only revisit the following after Phase 2 or Phase 3 is stable:

- attention as a small optional upgrade
- mini-group heuristics for sectors that already show alpha
- sentiment features if a clean VN-aligned dataset exists
- TRA-like routing only if simpler alternatives clearly plateau

## Parameters To Care About Now

These are the parameters that matter most for current research:

- `target_mode`
- `loss`
- `window_size`
- `feature_selection_mode`
- `feature_columns`
- `sector` or explicit `stocks`
- `target_normalizer`
- `lstm_seeds`

## Parameters To Ignore Until Needed

These should not drive the research loop yet:

- attention heads
- event thresholds
- multi-family loss weighting details
- benchmark-specific settings
- large search spaces over architecture variants

## Code Touch Points For The Next Upgrade

When Phase 2 starts, the expected files to touch are:

- [`src/models/dl_architectures/lstm_builder.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/dl_architectures/lstm_builder.py)
- [`src/models/components/losses.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/components/losses.py)
- [`src/models/trainer_wrapper.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/trainer_wrapper.py)
- [`scripts/run_train.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_train.py)
- [`src/evaluation/metric.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/evaluation/metric.py) only if evaluation needs an explicit quantile-aware adapter
- [`src/models/backtest_threshold.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/backtest_threshold.py) if uncertainty filtering is added

## Decision Rule

Before adding any new module, ask:

1. Does it directly improve `rel_score` or the reliability of `rel_score`?
2. Can it be added without changing the whole training/data/backtest stack?
3. Can its effect be isolated in one or two experiments?

If the answer is not clearly yes, it should wait.
