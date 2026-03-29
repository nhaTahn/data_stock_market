from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tf_lstm.config import FEATURE_COLUMNS, TARGET_COLUMN
from tf_lstm.data import (
    build_sequences,
    fit_feature_scaler,
    fit_target_scaler,
    load_dataset,
    scale_split,
    split_dataset,
)
from tf_lstm.metrics import baseline_predict, compute_metrics, invert_target_scale
from tf_lstm.model import build_callbacks, build_model, set_seed
from tf_lstm.reporting import save_results


def evaluate_scaled_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    return compute_metrics(
        invert_target_scale(y_true_scaled, target_mean, target_std),
        invert_target_scale(y_pred_scaled, target_mean, target_std),
    )


def run_experiment(args: SimpleNamespace, run_name: str | None = None) -> tuple[Path, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    np.random.seed(args.seed)
    set_seed(args.seed)

    df = load_dataset(Path(args.data_path))
    train_df, val_df, test_df = split_dataset(df, train_end=args.train_end, val_end=args.val_end)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the splits is empty. Adjust --train-end / --val-end.")

    feature_mean, feature_std = fit_feature_scaler(train_df)
    target_mean, target_std = fit_target_scaler(train_df)

    scaled_train = scale_split(train_df, feature_mean, feature_std, target_mean, target_std)
    scaled_val = scale_split(val_df, feature_mean, feature_std, target_mean, target_std)
    scaled_test = scale_split(test_df, feature_mean, feature_std, target_mean, target_std)

    train_seq = build_sequences(scaled_train, window_size=args.window_size)
    val_seq = build_sequences(scaled_val, window_size=args.window_size)
    test_seq = build_sequences(scaled_test, window_size=args.window_size)
    if len(train_seq.targets) == 0 or len(val_seq.targets) == 0 or len(test_seq.targets) == 0:
        raise ValueError("Sequence generation returned an empty split. Reduce --window-size or adjust split dates.")

    baseline_metrics = {
        "train": compute_metrics(
            invert_target_scale(train_seq.targets, target_mean, target_std),
            baseline_predict(train_seq.targets, target_mean, target_std),
        ),
        "val": compute_metrics(
            invert_target_scale(val_seq.targets, target_mean, target_std),
            baseline_predict(val_seq.targets, target_mean, target_std),
        ),
        "test": compute_metrics(
            invert_target_scale(test_seq.targets, target_mean, target_std),
            baseline_predict(test_seq.targets, target_mean, target_std),
        ),
    }

    model = build_model(
        window_size=args.window_size,
        num_features=len(FEATURE_COLUMNS),
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
    )
    history = model.fit(
        train_seq.features,
        train_seq.targets,
        validation_data=(val_seq.features, val_seq.targets),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=build_callbacks(),
    )

    train_pred_scaled = model.predict(train_seq.features, verbose=0).reshape(-1)
    val_pred_scaled = model.predict(val_seq.features, verbose=0).reshape(-1)
    test_pred_scaled = model.predict(test_seq.features, verbose=0).reshape(-1)

    lstm_metrics = {
        "train": evaluate_scaled_predictions(train_seq.targets, train_pred_scaled, target_mean, target_std),
        "val": evaluate_scaled_predictions(val_seq.targets, val_pred_scaled, target_mean, target_std),
        "test": evaluate_scaled_predictions(test_seq.targets, test_pred_scaled, target_mean, target_std),
    }

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = run_name or f"tf_lstm_next_return_{timestamp}"
    run_dir = Path(args.output_dir) / run_dir_name
    history_df = pd.DataFrame(history.history)
    config = {
        "framework": "tensorflow",
        "data_path": str(args.data_path),
        "train_end": args.train_end,
        "val_end": args.val_end,
        "window_size": args.window_size,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "lstm_units": args.lstm_units,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "train_sequences": len(train_seq.targets),
        "val_sequences": len(val_seq.targets),
        "test_sequences": len(test_seq.targets),
        "target_scaling": "z-score using train split",
        "baseline_1": "predict next-day return = 0",
    }
    save_results(run_dir, baseline_metrics, lstm_metrics, config, history_df, model)
    return run_dir, baseline_metrics, lstm_metrics
