`training_runs` stores saved experiments, comparisons, and representative candidates.

## Structure

- `active/`
  Current manual experiments.
- `active/reports/`
  Comparison CSV and PNG files for active runs.
- `search_runs/`
  Feature-search folders such as `search_*`.
- `representative_runs/`
  Curated runs worth keeping.
- `reports/`
  Aggregated summaries and shared plots.
- `overnight_logs/`
  Timestamped overnight batch logs and summaries.

## Files To Open First In A Run Folder

- `config.json`
  Exact hyperparameters and feature set.
- `metrics.json`
  Main evaluation summary. Use this first.
- `predictions.csv`
  Raw predictions by split and model.
- `actual_vs_prediction_<model>.png`
  Actual vs predicted values.
- `rel_score_hist_<model>.png`
  Raw vs stabilized local proxy histogram.
- `threshold_backtest_summary_non_overlap.json`
  Threshold backtest results.

## `rel_score` Formula

The repository scores models with a robust relative metric:

```text
base_i  = aligned actual return
error_i = base_i - prediction_i

loss(x) = q50(|x|) + 0.5 * q90(|x|)

base_loss = loss(base)
abs_loss  = loss(error)
rel_score = 1 - abs_loss / base_loss
```

Interpretation:

- `rel_score > 0`: useful signal
- `rel_score < 0`: model error is still too large
- higher is better

## Why `rel_score_hist` Has Two Panels

Each split shows two local proxy views:

### Raw Proxy

```text
raw_proxy_i = 1 - |error_i| / max(|base_i|, 1e-6)
clip to [-3.0, 1.0]
```

This can produce a big left-edge bar when many rows have tiny `|base|`.

### Stabilized Proxy

```text
proxy_floor = max(base_loss, 1e-4)
stabilized_proxy_i = 1 - |error_i| / max(|base_i|, proxy_floor)
clip to [-1.5, 1.0]
```

This is easier to read and better aligned with the aggregate metric.

## Quick Reading Rule

1. Trust `metrics.json` first.
2. Use `Stabilized Proxy` to judge whether the distribution leans right of `0`.
3. Use `Raw Proxy` to see whether the far-left tail is caused by near-zero targets.
4. Do not treat the `-3` bar by itself as proof that the whole model has collapsed.
