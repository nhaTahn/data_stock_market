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
