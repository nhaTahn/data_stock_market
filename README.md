# Data Stock Market

This repository manages stock-market data collection, feature engineering, LSTM-based forecasting, backtests, and run reporting for VN-focused research.

## Current Direction

The active research direction is intentionally narrower than a full "advanced quant framework":

- primary target: next-return forecasting for VN equities
- primary decision metric: `rel_score`
- current training default: `--loss rel_score`
- current research unit: sector-wide pools and mini-groups, not one universal market model
- active expansion branch: `shared_vn30` context model combined with expert runs through a simple committee
- optional experimental branch: `--enable-quantile-family` adds a minimal `q50/q90` head without rewriting the framework
- current use of quantiles is primarily as a backtest sidecar via `q90 - q50`, not as the main model family

If you are starting work in this repo, read [`docs/models_code_map.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/models_code_map.md) to navigate the code layout, [`docs/feature_correlation_report.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/feature_correlation_report.md) for the pruned feature baseline, [`docs/vn30_paper_phase1_20260412.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/vn30_paper_phase1_20260412.md) for the latest VN30 representation experiment, and [`docs/lstm_model_glossary.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/lstm_model_glossary.md) to decode run/model names.

## Main Flow

1. Fetch raw market data.
2. Build quality datasets by market.
3. Generate technical and context features.
4. Train the LSTM family plus simple baselines.
5. Evaluate with `rel_score`, directional accuracy, and threshold backtests.
6. Mirror clean report artifacts under each run folder.

## Key Paths

- [`run_fetch.py`](/Users/lap15111/Documents/research-paper/data_stock_market/run_fetch.py): data download entrypoint.
- [`scripts/run_train.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_train.py): main training entrypoint.
- [`scripts/run_sector_group_batch.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_sector_group_batch.py): sector-wide batch runner from search summary.
- [`scripts/run_sector_mini_group_batch.py`](/Users/lap15111/Documents/research-paper/data_stock_market/scripts/run_sector_mini_group_batch.py): mini-group batch runner.
- [`src/data_pipeline/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/data_pipeline): dataset build pipeline.
- [`src/utils/features.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/utils/features.py): feature engineering.
- [`src/models/config.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/config.py): default training configuration.
- [`src/models/training/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training): sequence prep, scalers, targets, seeds, and family fitters.
- [`src/models/architectures/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/architectures): model builders split by family.
- [`src/models/reporting/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/reporting): report layout helpers.
- [`src/backtesting/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/backtesting): threshold and multi-strategy backtests plus performance summaries.
- [`src/reporting/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/reporting): report rebuild entrypoints.
- [`src/research/`](/Users/lap15111/Documents/research-paper/data_stock_market/src/research): feature-search and run-comparison utilities.
- [`src/models/sequence_utils.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/sequence_utils.py): legacy compatibility shim to the new training package.
- [`src/models/trainer_wrapper.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/trainer_wrapper.py): legacy compatibility shim to the new training fitters.
- [`src/models/dl_architectures/lstm_builder.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/dl_architectures/lstm_builder.py): legacy compatibility shim to the new architecture package.
- [`src/models/components/losses.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/components/losses.py): `mse`, `huber`, `directional_huber`, and differentiable `rel_score` surrogate.
- [`src/models/training_recipe.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/models/training_recipe.py): recipe builder from stock search summaries.
- [`src/evaluation/metric.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/evaluation/metric.py): final evaluation metric.
- [`src/visualization/model_plots.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/visualization/model_plots.py): prediction and histogram plots.
- [`data/processed/assets/data_info_vn/history/training_runs/`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs): saved experiments.

## Core Metric: `rel_score`

For return targets, the repository evaluates models with a robust relative skill score.

### Step 1: Align prediction and target

Predictions and targets are aligned one step forward within each stock code so the metric does not leak information across time or across tickers.

### Step 2: Define base and error

```text
base_i  = aligned actual return
error_i = base_i - prediction_i
```

### Step 3: Robust loss used by the metric

```text
loss(x) = q50(|x|) + 0.5 * q90(|x|)
```

### Step 4: Relative score

```text
base_loss = loss(base)
abs_loss  = loss(error)
rel_score = 1 - abs_loss / base_loss
```

### Interpretation

- `rel_score > 0`: model error is smaller than the typical move scale.
- `rel_score = 0`: model error is about as large as the move scale.
- `rel_score < 0`: model error is larger than the move scale.
- A practical milestone used in this repo is `test rel_score > 0.03`.

## Important Training Note

The repo now supports `--loss rel_score` in training. This is a differentiable batch-level surrogate of the final evaluation metric, not a bit-for-bit copy of the evaluator in [`src/evaluation/metric.py`](/Users/lap15111/Documents/research-paper/data_stock_market/src/evaluation/metric.py). It is still much closer to the actual objective than training with `huber` and only selecting checkpoints by `rel_score`.

## Recommended Scope Right Now

If your goal is to make progress without adding more complexity, focus on these knobs first:

- `--target-mode return`
- `--loss rel_score`
- `--window-size`
- `--feature-selection-mode`
- `--sector` or `--stocks`
- `--target-normalizer volatility_20`
- `--lstm-seeds`

Current best reading order:

- first: [`docs/feature_correlation_report.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/feature_correlation_report.md)
- second: [`docs/models_code_map.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/models_code_map.md)
- third: [`data/processed/assets/data_info_vn/history/training_runs/README.md`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/README.md)
- fourth: the shortlisted run folders named in that document

Treat these as experimental or second-order for now:

- quantile family
- attention family
- event-gated family
- Fischer-Krauss benchmark
- aggressive sample weighting
- text or sentiment inputs
- one-model-for-the-whole-market designs

One controlled exception is the new `shared_vn30 + expert committee` branch: it is still experimental, but it is the preferred next step if you want a broader market-aware model without jumping straight into a single giant VN100 predictor.

## Quantile Sidecar Note

The repo can now store these extra columns in [`predictions.csv`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs):

- `prediction_q50`
- `prediction_q90`
- `prediction_uncertainty`

For the current VN mini-group winner, the useful interpretation was not "keep the smallest uncertainty". The better interpretation was:

- use plain `lstm` predictions as the trading signal
- use quantile `q90 - q50` as a sidecar spread score
- test both `low` and `high` filters before assuming which direction is better

The first strong result so far came from keeping `high` spread rows on top of the plain `lstm_best_by_val` signal, not from replacing the signal with the quantile model itself.

## Run Outputs

Each run folder keeps both legacy top-level artifacts and mirrored report folders:

- `reports/core/`: `config.json`, `metrics.json`, `metric_details.json`, `predictions.csv`, histories
- `reports/plots/`: actual-vs-prediction plots and `rel_score` histograms
- `reports/metric_series/`: aligned error/base series used for diagnostics
- `reports/backtests/`: threshold or strategy backtests
- `reports/benchmark/`: Fischer-Krauss benchmark artifacts when enabled

Open [`data/processed/assets/data_info_vn/history/training_runs/README.md`](/Users/lap15111/Documents/research-paper/data_stock_market/data/processed/assets/data_info_vn/history/training_runs/README.md) for a quick guide to reading saved runs.

## Example Commands

### Train a single sector run with the current recommended objective

```bash
venv/bin/python scripts/run_train.py \
  --target-mode return \
  --sector "Bất động sản" \
  --feature-selection-mode search_summary \
  --loss rel_score \
  --target-normalizer volatility_20 \
  --run-name demo_bds_relscore
```

### Run sector-wide batches

```bash
venv/bin/python scripts/run_sector_group_batch.py \
  --target-mode return \
  --loss rel_score \
  --run-name-suffix relscore
```

### Run the first shared market-context committee experiment

```bash
venv/bin/python scripts/run_shared_vn30_committee.py
```

### Run mini-group batches

```bash
venv/bin/python scripts/run_sector_mini_group_batch.py \
  --target-mode return \
  --loss rel_score \
  --run-name-suffix relscore
```

### Rebuild report artifacts for an existing run

```bash
venv/bin/python src/reporting/update_run_reports.py \
  data/processed/assets/data_info_vn/history/training_runs/demo_bds_relscore
```

### Run threshold backtest

```bash
venv/bin/python src/backtesting/threshold_backtest.py \
  data/processed/assets/data_info_vn/history/training_runs/demo_bds_relscore
```

### Run threshold backtest with quantile sidecar

```bash
venv/bin/python src/backtesting/threshold_backtest.py \
  data/processed/assets/data_info_vn/history/training_runs/demo_bds_relscore \
  --models lstm_best_by_val \
  --uncertainty-model lstm_quantile_best_by_val \
  --uncertainty-side high \
  --uncertainty-quantiles 0.25,0.5,0.75
```
