"""Two-head raw-return + alpha-return heteroscedastic probe.

Goal:
- preserve raw-return rel_score via a heteroscedastic raw head,
- add an auxiliary date-demeaned alpha head for stock-selection skill,
- evaluate both raw rel_score and alpha_rel_score.

Holdout/test is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_hetero_nll_probe import (  # noqa: E402
    MARKET_CONTEXT_FEATURES,
    CombinedRelScoreNLLLoss,
    PreparedData,
    evaluate_predictions,
    load_and_prepare,
    parse_lstm_units,
    parse_seeds,
    predict_raw,
    train_model,
)
from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)

DEFAULT_FEATURES = (
    "close_return",
    "adjust_return",
    "range_pct",
    "body_pct",
    "volume_change",
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "ma_5_gap",
    "ma_20_gap",
    *MARKET_CONTEXT_FEATURES,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", default="43,52,62")
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--add-market-context-features", action="store_true", default=True)
    parser.add_argument("--alpha-weight", type=float, default=0.25)
    parser.add_argument(
        "--train-mode",
        choices=("joint", "frozen_backbone"),
        default="joint",
        help="joint trains both heads together; frozen_backbone trains raw first, then alpha with backbone/raw frozen.",
    )
    return parser.parse_args(argv)


def date_demean(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates), "value": values})
    return (frame["value"] - frame.groupby("Date")["value"].transform("mean")).to_numpy(dtype=np.float32)


def build_two_head_targets(data: PreparedData) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    alpha_train = date_demean(data.y_train_raw, data.meta_train["Date"])
    alpha_val = date_demean(data.y_val_raw, data.meta_val["Date"])
    alpha_train_local = apply_local_target_normalizer(alpha_train, data.train_scale, data.local_normalizer)
    alpha_val_local = apply_local_target_normalizer(alpha_val, data.val_scale, data.local_normalizer)
    alpha_scaler = fit_target_scaler(alpha_train_local)
    alpha_train_scaled = apply_target_scaler(alpha_train_local, alpha_scaler).reshape(-1, 1)
    alpha_val_scaled = apply_target_scaler(alpha_val_local, alpha_scaler).reshape(-1, 1)
    y_train = np.concatenate([data.y_train_model, alpha_train_scaled], axis=1).astype(np.float32)
    y_val = np.concatenate([data.y_val_model, alpha_val_scaled], axis=1).astype(np.float32)
    return y_train, y_val, alpha_val.astype(np.float32), alpha_scaler


@keras.utils.register_keras_serializable(package="two_head")
class TwoHeadRawAlphaLoss(keras.losses.Loss):
    def __init__(
        self,
        raw_loss: keras.losses.Loss,
        alpha_weight: float = 0.25,
        name: str = "two_head_raw_alpha_loss",
    ):
        super().__init__(name=name)
        self.raw_loss = raw_loss
        self.alpha_weight = float(alpha_weight)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        raw_true = y_true[:, 0:2]
        alpha_true = y_true[:, 2:3]
        raw_pred = y_pred[:, 0:2]
        alpha_pred = y_pred[:, 2:3]
        raw_component = self.raw_loss(raw_true, raw_pred)
        alpha_component = tf.reduce_mean(tf.square(alpha_true - alpha_pred))
        return raw_component + self.alpha_weight * alpha_component


@keras.utils.register_keras_serializable(package="two_head")
class RawOnlyLoss(keras.losses.Loss):
    def __init__(self, raw_loss: keras.losses.Loss, name: str = "raw_only_loss"):
        super().__init__(name=name)
        self.raw_loss = raw_loss

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return self.raw_loss(y_true[:, 0:2], y_pred[:, 0:2])


@keras.utils.register_keras_serializable(package="two_head")
class AlphaOnlyLoss(keras.losses.Loss):
    def __init__(self, name: str = "alpha_only_loss"):
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square(y_true[:, 2:3] - y_pred[:, 2:3]))


def build_raw_loss(data: PreparedData) -> keras.losses.Loss:
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=data.target_scaler.mean,
        target_std=data.target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=data.local_normalizer.floor,
        high_quantile=0.85,
        high_weight=1.75,
        base_weight=1.0,
        tail_error_threshold=0.035,
        tail_penalty_weight=0.05,
    )
    return CombinedRelScoreNLLLoss(rel_loss, w_rel=0.7, w_nll=0.3)


def build_two_head_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
    )
    raw_mu = layers.Dense(1, name="raw_mu")(encoded)
    raw_log_sigma = layers.Dense(1, name="raw_log_sigma")(encoded)
    alpha_mu = layers.Dense(1, name="alpha_mu")(encoded)
    output = layers.Concatenate(name="raw_mu_logsigma_alpha_mu")([raw_mu, raw_log_sigma, alpha_mu])
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=TwoHeadRawAlphaLoss(build_raw_loss(data), alpha_weight=args.alpha_weight),
    )
    return model


def compile_train_stage(model: keras.Model, data: PreparedData, args: argparse.Namespace, stage: str) -> None:
    if stage == "joint":
        loss = TwoHeadRawAlphaLoss(build_raw_loss(data), alpha_weight=args.alpha_weight)
    elif stage == "raw_only":
        loss = RawOnlyLoss(build_raw_loss(data))
    elif stage == "alpha_only":
        loss = AlphaOnlyLoss()
    else:
        raise ValueError(f"Unknown train stage: {stage}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0), loss=loss)


def freeze_for_alpha_stage(model: keras.Model) -> None:
    for layer in model.layers:
        layer.trainable = layer.name == "alpha_mu"


def predict_two_head(
    model: keras.Model,
    data: PreparedData,
    alpha_scaler: object,
    x: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    raw_scaled = out[:, :2]
    alpha_scaled = out[:, 2]
    raw_mu, raw_sigma = predict_raw(keras.Model(model.inputs, model.output[:, :2]), data, x, scale)
    alpha_local = inverse_target_scaler_values(alpha_scaled, alpha_scaler)
    alpha_raw = inverse_local_target_normalizer(alpha_local, scale, data.local_normalizer).reshape(-1)
    return raw_mu, raw_sigma, alpha_raw.astype(np.float32)


def metrics(raw_actual: np.ndarray, alpha_actual: np.ndarray, raw_pred: np.ndarray, alpha_pred: np.ndarray, sigma: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    raw = evaluate_predictions(raw_actual, raw_pred, sigma, meta)
    alpha = evaluate_predictions(alpha_actual, alpha_pred, np.zeros_like(alpha_pred), meta)
    return {
        "raw_rel_score": raw["rel_score"],
        "raw_directional_accuracy": raw["directional_accuracy"],
        "raw_pred_actual_q90_ratio": raw["pred_actual_q90_ratio"],
        "alpha_rel_score": alpha["rel_score"],
        "alpha_directional_accuracy": alpha["directional_accuracy"],
        "alpha_pred_actual_q90_ratio": alpha["pred_actual_q90_ratio"],
        "alpha_q90_abs_error": alpha["q90_abs_error"],
        "raw_daily_q90_max": raw["daily_q90_max"],
        "alpha_daily_q90_max": alpha["daily_q90_max"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare(args)
    y_train, y_val, alpha_val, alpha_scaler = build_two_head_targets(data)
    train_data = PreparedData(
        feature_columns=data.feature_columns,
        x_train=data.x_train,
        x_val=data.x_val,
        y_train_raw=data.y_train_raw,
        y_val_raw=data.y_val_raw,
        y_train_model=y_train,
        y_val_model=y_val,
        meta_train=data.meta_train,
        meta_val=data.meta_val,
        train_scale=data.train_scale,
        val_scale=data.val_scale,
        target_scaler=data.target_scaler,
        local_normalizer=data.local_normalizer,
    )
    rows: list[dict[str, float | int | str]] = []
    for seed in parse_seeds(args.seeds):
        model = build_two_head_model(data, args)
        if args.train_mode == "joint":
            compile_train_stage(model, data, args, "joint")
            train_model(model, train_data, args, seed)
        else:
            compile_train_stage(model, data, args, "raw_only")
            train_model(model, train_data, args, seed)
            freeze_for_alpha_stage(model)
            compile_train_stage(model, data, args, "alpha_only")
            train_model(model, train_data, args, seed)
        raw_mu, raw_sigma, alpha_mu = predict_two_head(model, data, alpha_scaler, data.x_val, data.val_scale)
        row = metrics(data.y_val_raw, alpha_val, raw_mu, alpha_mu, raw_sigma, data.meta_val)
        row.update({"seed": seed, "split": "val", "alpha_weight": args.alpha_weight, "train_mode": args.train_mode})
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    result.to_csv(args.gold_dir / "results_per_seed.csv", index=False)
    agg = result.agg(
        {
            "raw_rel_score": ["mean", "std"],
            "alpha_rel_score": ["mean", "std"],
            "raw_directional_accuracy": ["mean", "std"],
            "alpha_directional_accuracy": ["mean", "std"],
            "raw_pred_actual_q90_ratio": ["mean", "std"],
            "alpha_pred_actual_q90_ratio": ["mean", "std"],
        }
    )
    agg.to_csv(args.output_dir / "aggregate.csv")
    agg.to_csv(args.gold_dir / "aggregate.csv")
    text = "\n".join(
        [
            "# Two-Head Raw + Alpha Probe",
            "",
            "Holdout/test not used.",
            "",
            result.round(6).to_markdown(index=False),
            "",
            agg.round(6).to_markdown(),
            "",
            json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2),
        ]
    )
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
