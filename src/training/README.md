# Training Guide

This directory contains both reusable packages and runnable scripts.

## Packages

- `tf_lstm/`: reusable TensorFlow regression pipeline for next-day return style experiments.
- `fk_lstm_classifier/`: Fischer-Krauss style binary ranking model with optional temporal attention.
- `model_benchmark/`: benchmark helpers for classical baselines and transformer models.

## Script Entry Points

The runnable entrypoints now live under `src/training/scripts/`.

- `train_lstm_next_price_tf.py`: run the original TensorFlow regression training flow.
- `scripts/run_fischer_krauss_lstm_tf.py`: run the binary cross-sectional return classifier.
- `scripts/run_fk_batch.py`: run multiple FK classifier configs and render one comparison index.
- `scripts/render_fk_dashboard.py`: render an HTML dashboard for an existing FK classifier run.
- `scripts/render_tf_lstm_dashboard.py`: render an HTML dashboard for an existing tf_lstm regression run.
- `scripts/render_benchmark_dashboard.py`: render an HTML dashboard for an existing benchmark run.
- `scripts/render_run_index.py`: build one comparison page across multiple run directories.
- `scripts/run_lstm_experiments_tf.py`: compare multiple TensorFlow LSTM regression configurations.
- `scripts/run_lstm_feature_ablation_tf.py`: feature-group ablation runs.
- `scripts/run_sequence_model_sweep.py`: broader sequence-model sweep runner.
- `scripts/run_transformer_focus_sweep.py`: transformer-focused sweep runner.
- `scripts/run_sequence_ensemble.py`: sequence ensemble experiments.
- `scripts/run_model_benchmarks.py`: classical and transformer benchmark comparison.

## Navigation Order

If you are trying to understand a training flow, read in this order:

1. the relevant `run_*.py` entrypoint
2. the package `config.py`
3. the package `data.py`
4. the package `model.py`
5. the reporting and metrics helpers

This keeps the mental model small and avoids jumping straight into helper modules.

## FK Research Workflow

For the Fischer-Krauss classifier, the repo now supports config-driven experiments:

1. Start with `configs/fk_lstm/*.json`.
2. Run one config with `scripts/run_fischer_krauss_lstm_tf.py --config ...`.
3. Run a suite with `scripts/run_fk_batch.py --config-dir configs/fk_lstm`.
4. Inspect `artifacts/fk_lstm_classifier/<run_name>/` for:
   - `config.json`
   - `market_rules.json`
   - `fit_history.csv`
   - `validation_predictions.csv`
   - `portfolio_holdings.csv`
   - `validation_long_short_returns.csv`
   - `fold_summary.csv`
   - `dashboard.html`

For Colab or faster iteration:

1. Start with `configs/fk_lstm_fast/*.json`.
2. Use `scripts/run_fk_batch.py --config-dir configs/fk_lstm_fast --skip-existing`.
3. Optionally override artifact storage with `--output-root`, for example a mounted Google Drive path.
4. Use [fk_lstm_colab_batch_runner.ipynb](/Users/lap15111/Documents/research-paper/data_stock_market/notebook/fk_lstm_colab_batch_runner.ipynb) as the Colab entrypoint.

For the VN-only walk-forward pack:

1. Use `scripts/run_vn_fischer_krauss_pack.py --preset fast` for a lighter Colab-friendly pass.
2. Switch to `--preset full` for the full VN walk-forward suite.
3. Use [fk_lstm_vn_walkforward_colab.ipynb](/Users/lap15111/Documents/research-paper/data_stock_market/notebook/fk_lstm_vn_walkforward_colab.ipynb) when you want a Drive-backed Colab workflow.
