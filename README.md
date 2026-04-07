# Data Stock Market

This repository manages stock-market data collection, feature engineering, LSTM training, baseline comparison, backtests, and run reporting for VN, US, and JP datasets.

## Main Flow

1. Fetch raw market data.
2. Build quality datasets by market.
3. Generate technical and macro features.
4. Train LSTM plus baseline models.
5. Evaluate with `rel_score`, directional accuracy, and backtests.
6. Save plots and summaries under `training_runs/`.

## Key Paths

- `run_fetch.py`: data download entrypoint.
- `scripts/run_train.py`: main training entrypoint.
- `scripts/run_overnight.sh`: heavier batch jobs for overnight experiments.
- `src/data_pipeline/`: dataset build pipeline.
- `src/utils/features.py`: feature engineering.
- `src/models/lstm.py`: LSTM model, scaling, and sequence utilities.
- `src/models/baseline.py`: linear-regression and ARIMA-style baselines.
- `src/evaluation/metric.py`: `rel_score` and directional evaluation.
- `src/visualization/model_plots.py`: prediction, histogram, and equity plots.
- `data/processed/assets/data_info_vn/history/training_runs/`: saved experiments.

## Core Metric: `rel_score`

The repository does not optimize only `mse` or `mae`. The main target is `rel_score`, which compares model error against the typical magnitude of the market move being predicted.

### Step 1: Align prediction and target

For each stock code, prediction and target are aligned one step forward before evaluation so the metric does not leak information across time or across tickers.

### Step 2: Define base and error

```text
base_i  = aligned actual return
error_i = base_i - prediction_i
```

### Step 3: Robust loss used by the metric

```text
loss(x) = q50(|x|) + 0.5 * q90(|x|)
```

where:

- `q50` is the median
- `q90` is the 90th percentile
- `|x|` is absolute value

This makes the metric more robust than plain mean absolute error.

### Step 4: Relative score

```text
base_loss = loss(base)
abs_loss  = loss(error)
rel_score = 1 - abs_loss / base_loss
```

### Interpretation

- `rel_score > 0`: model error is smaller than the typical move scale. This is good.
- `rel_score = 0`: model error is about as large as the move scale.
- `rel_score < 0`: model error is larger than the move scale.
- Higher is better. A practical milestone used in this repo is `test rel_score > 0.03`.

## Why The Histogram Can Show A Large `-3` Bar

Each run also saves `rel_score_hist_<model>.png`. This plot is not the metric itself. It is a per-row proxy used to inspect the distribution of local outcomes.

The aggregate `rel_score` above is the main truth. The histogram is only a diagnostic view.

### Raw Local Proxy

```text
raw_proxy_i = 1 - |error_i| / max(|base_i|, 1e-6)
raw_proxy_i is clipped to [-3.0, 1.0]
```

This proxy is very sensitive when `|base_i|` is close to `0`.

If `base_i` is tiny but `error_i` is ordinary, the ratio explodes and many rows get clipped to `-3`. That can create a large left-edge bar even when the model is not globally collapsing.

### Stabilized Local Proxy

To make the histogram easier to read, the plot also shows a stabilized version:

```text
proxy_floor = max(base_loss, 1e-4)
stabilized_proxy_i = 1 - |error_i| / max(|base_i|, proxy_floor)
stabilized_proxy_i is clipped to [-1.5, 1.0]
```

This keeps the proxy in the same scale family as the aggregate `rel_score` by using the robust split-level `base_loss` as the denominator floor.

## How To Read `rel_score_hist_<model>.png`

Each split now shows two histograms side by side:

- `Raw Proxy`: shows sensitivity to tiny-base rows and makes the old `-3` pile-up visible.
- `Stabilized Proxy`: shows the bulk shape of the distribution without letting near-zero targets dominate the chart.

The plot also shows:

- red line: mean local proxy
- green dashed line: aggregate `rel_score`
- `share(proxy>0)`: fraction of rows on the positive side
- `near_zero_base`: fraction of rows where `|base|` is very small

### Practical Reading Rule

Use the following order:

1. Read `metrics.json` first. `aggregate rel_score` is the main decision metric.
2. Look at `Stabilized Proxy` to judge whether the distribution is mostly right of `0`.
3. Look at `Raw Proxy` to check whether extreme left bars come from tiny-base rows.
4. If `share(raw<=-2.9)` is high but `near_zero_base` is also high, the left edge is often a plotting artifact rather than proof of total model failure.

## Training Outputs

Each run folder typically contains:

- `config.json`
- `history.csv`
- `metrics.json`
- `metric_details.json`
- `predictions.csv`
- `metric_series_<model>_<split>.csv`
- `actual_vs_prediction_<model>.png`
- `rel_score_hist_<model>.png`
- `threshold_backtest_summary_non_overlap.json`

## Example Commands

### Train

```bash
venv/bin/python scripts/run_train.py \
  --target-mode return_3d \
  --stocks VHM,KDH,NLG,DIG,DXG,PDR,NVL \
  --run-name demo_return3d
```

### Update plots and metric details

```bash
venv/bin/python src/models/update_run_reports.py \
  data/processed/assets/data_info_vn/history/training_runs/demo_return3d
```

### Threshold backtest

```bash
venv/bin/python src/models/backtest_threshold.py \
  data/processed/assets/data_info_vn/history/training_runs/demo_return3d \
  --non-overlap
```
