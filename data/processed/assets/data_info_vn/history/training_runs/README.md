`training_runs` stores saved experiments, comparisons, and batch logs for VN research.

## Git Hygiene

Only lightweight artifacts should be committed from `training_runs`:

- keep: `config.json`, `metrics.json`, `family_selection_summary.json`, backtest summary JSON, feature-correlation summaries, top-level README/summary CSV files
- keep local only: `predictions.csv`, `history*.csv`, `metric_details.json`, `reports/metric_series/*`, plots, and other generated debug artifacts

The repo `.gitignore` is set up to keep the heavy files local and out of future pushes.

## Open These First

If you want the shortest path to the current best results, start with these run folders:

- `mini_tpdouong_g06_uncertainty_sidecar`
  Best current run where `best by val` and `best by test` line up on the same plain-LSTM seed.
- `mini_bat_ong_san_g01_return_w20_pruned_v2`
  Best BĐS ceiling by test, but validation selection is still unstable.
- `mini_bat_ong_san_g01_return_w15_pruned_v1`
  Good comparison point for a more stable BĐS selection.
- `mini_ngan_hang_g02_return_w20_pruned_v2`
  Current best bank mini-group.
- `mini_thuc_pham_va_o_uong_g03_return_w20_pruned_v2`
  Smaller F&B pruned run without the quantile sidecar.

For the repo-level summary of why these are the current shortlist, read:

- [`docs/current_best_path.md`](/Users/lap15111/Documents/research-paper/data_stock_market/docs/current_best_path.md)

## Structure

- `active/`
  Manual experiments worth keeping close at hand.
- `search_runs/`
  Feature-search folders such as `search_*`.
- `representative_runs/`
  Curated runs that should remain readable after cleanup.
- `reports/`
  Aggregated summaries and shared plots across runs.
- `overnight_logs/`
  Timestamped overnight batch logs and summaries.
- `sector_logs/`
  Sector-wide batch logs and `sector_batch_summary.csv`.
- `mini_group_logs/`
  Mini-group batch logs and `mini_group_batch_summary.csv`.
- `sector_*` and `mini_*`
  Actual run folders produced by the batch runners.

## Files To Open First In A Run Folder

Start in `reports/` when it exists:

- `reports/core/config.json`
  Exact hyperparameters, features, sector, stocks, and loss.
- `reports/core/metrics.json`
  Main evaluation summary. Open this first.
- `reports/core/predictions.csv`
  Raw predictions by split and model.
- `reports/plots/actual_vs_prediction_<model>.png`
  Prediction-vs-actual visual check.
- `reports/plots/rel_score_hist_<model>.png`
  Raw vs stabilized local proxy histogram.
- `reports/backtests/threshold_backtest_summary.json`
  Main backtest summary for `return`.
- `reports/backtests/threshold_backtest_summary_non_overlap.json`
  Main backtest summary for multi-day return horizons.

If `reports/` is missing, the same artifacts may still exist at the run root for backward compatibility.

## How To Read A Run Quickly

1. Open `reports/core/metrics.json`.
2. Check the best `test rel_score` among `lstm*` models.
3. Check whether validation is also positive or at least materially better than baseline.
4. Only then read `reports/backtests/*`.
5. Use `reports/plots/rel_score_hist_<model>.png` as a diagnostic, not as the primary truth.

## `rel_score` Formula

The repository scores return forecasts with a robust relative metric:

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

## Batch Summaries

For current VN research, these summary files are the fastest entrypoints:

- `sector_logs/<timestamp>/sector_batch_summary.csv`
- `mini_group_logs/<timestamp>/mini_group_batch_summary.csv`

Read the summary first, then open the best few run folders behind it.
