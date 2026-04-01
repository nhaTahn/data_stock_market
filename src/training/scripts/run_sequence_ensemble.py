from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from model_benchmark.data import prepare_sequence_data
from model_benchmark.reporting import save_model_metrics
from model_benchmark.sequence_predict import evaluate_return_predictions, load_keras_model, predict_returns
from tf_lstm.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TRAIN_END_DATE, VAL_END_DATE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple average ensemble between the best Transformer and LSTM models.")
    parser.add_argument(
        "--transformer-model",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sequence_model_sweep" / "transformer_bliq_w30_ff64_h16_do20_lr5e4" / "transformer_model.keras",
        help="Path to the trained Transformer Keras model.",
    )
    parser.add_argument(
        "--lstm-model",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sequence_model_sweep" / "lstm_bliq_w30_u96_do20_lr4e4" / "best_model.keras",
        help="Path to the trained LSTM Keras model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sequence_ensemble",
        help="Directory for ensemble outputs.",
    )
    args = parser.parse_args()

    bundle = prepare_sequence_data(
        data_path=DEFAULT_DATA_PATH,
        train_end=TRAIN_END_DATE,
        val_end=VAL_END_DATE,
        window_size=30,
        feature_groups=["base", "liquidity"],
    )

    transformer = load_keras_model(args.transformer_model)
    lstm = load_keras_model(args.lstm_model)

    splits = {
        "train": (bundle.train_seq, bundle.train_targets),
        "val": (bundle.val_seq, bundle.val_targets),
        "test": (bundle.test_seq, bundle.test_targets),
    }

    ensemble_metrics: dict[str, dict[str, float]] = {}
    transformer_metrics: dict[str, dict[str, float]] = {}
    lstm_metrics: dict[str, dict[str, float]] = {}

    for split_name, (seq_batch, y_true) in splits.items():
        transformer_pred = predict_returns(transformer, seq_batch, bundle.target_mean, bundle.target_std)
        lstm_pred = predict_returns(lstm, seq_batch, bundle.target_mean, bundle.target_std)
        ensemble_pred = (transformer_pred + lstm_pred) / 2.0

        transformer_metrics[split_name] = evaluate_return_predictions(y_true, transformer_pred)
        lstm_metrics[split_name] = evaluate_return_predictions(y_true, lstm_pred)
        ensemble_metrics[split_name] = evaluate_return_predictions(y_true, ensemble_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_model_metrics(
        args.output_dir,
        "ensemble_average",
        ensemble_metrics,
        {
            "feature_groups": bundle.feature_groups,
            "window_size": 30,
            "ensemble": "simple average of transformer and lstm return predictions",
            "transformer_model": str(args.transformer_model),
            "lstm_model": str(args.lstm_model),
        },
    )

    summary_df = pd.DataFrame(
        [
            {"model": "baseline", "split": split, **metrics}
            for split, metrics in bundle.baseline_metrics.items()
        ]
        + [
            {"model": "transformer", "split": split, **metrics}
            for split, metrics in transformer_metrics.items()
        ]
        + [
            {"model": "lstm", "split": split, **metrics}
            for split, metrics in lstm_metrics.items()
        ]
        + [
            {"model": "ensemble_average", "split": split, **metrics}
            for split, metrics in ensemble_metrics.items()
        ]
    )
    summary_df.to_csv(args.output_dir / "ensemble_summary.csv", index=False)
    print(f"Saved ensemble outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
