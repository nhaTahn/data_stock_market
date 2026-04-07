from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.evaluation.metric import evaluate

LSTM_L2_FACTOR = 1e-5


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: tuple[str, ...]


@dataclass
class TargetScaler:
    mean: float
    std: float


class ValidationRelScoreCallback(keras.callbacks.Callback):
    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        group_ids: np.ndarray | None = None,
        target_scaler: TargetScaler | None = None,
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.group_ids = group_ids
        self.target_scaler = target_scaler

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if len(self.x_val) < 3 or len(self.y_val) < 3:
            logs["val_rel_score"] = np.nan
            return
        prediction = self.model.predict(self.x_val, verbose=0).reshape(-1)
        try:
            metric_prediction = inverse_target_scaler_values(prediction, self.target_scaler)
            metric_target = inverse_target_scaler_values(self.y_val, self.target_scaler)
            logs["val_rel_score"] = float(
                evaluate(metric_prediction, metric_target, group_ids=self.group_ids)["rel_score"]
            )
        except Exception:
            logs["val_rel_score"] = np.nan


class DirectionalHuberLoss(keras.losses.Loss):
    def __init__(self, delta: float = 0.01, penalty_weight: float = 20.0, name: str = "directional_huber_loss"):
        super().__init__(name=name)
        self.delta = delta
        self.penalty_weight = penalty_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return directional_huber_loss(
            y_true,
            y_pred,
            delta=self.delta,
            penalty_weight=self.penalty_weight,
        )

    def get_config(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "delta": self.delta,
            "penalty_weight": self.penalty_weight,
        }


def directional_huber_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    delta: float = 0.01,
    penalty_weight: float = 20.0,
) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * tf.square(quadratic) + delta * linear
    sign_penalty = tf.maximum(0.0, -(y_true * y_pred))
    return tf.reduce_mean(huber + penalty_weight * sign_penalty)


def resolve_loss(loss: str, huber_delta: float):
    if loss == "mse":
        return "mse"
    if loss == "huber":
        return keras.losses.Huber(delta=huber_delta)
    if loss == "directional_huber":
        return DirectionalHuberLoss(delta=huber_delta)
    raise ValueError("loss must be one of: mse, huber, directional_huber")


def build_training_callbacks(
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_group_ids: np.ndarray | None,
    patience: int,
    monitor_metric: str,
    target_scaler: TargetScaler | None = None,
) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = []
    if monitor_metric == "val_rel_score":
        callbacks.append(
            ValidationRelScoreCallback(
                x_val,
                y_val,
                group_ids=val_group_ids,
                target_scaler=target_scaler,
            )
        )
        mode = "max"
    else:
        mode = "min"
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
        )
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=mode,
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-5,
            verbose=0,
        )
    )
    return callbacks


def build_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "mse",
    huber_delta: float = 0.01,
) -> keras.Model:
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")

    model_layers: list[keras.layers.Layer] = [layers.Input(shape=(window_size, num_features))]
    for layer_idx, units in enumerate(unit_stack):
        return_sequences = layer_idx < len(unit_stack) - 1
        model_layers.append(
            layers.LSTM(
                units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(LSTM_L2_FACTOR),
                recurrent_regularizer=regularizers.l2(LSTM_L2_FACTOR),
            )
        )
        if dropout > 0 and return_sequences:
            model_layers.append(layers.Dropout(dropout))
    model_layers.append(layers.Dense(1))

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=resolve_loss(loss, huber_delta),
    )
    return model


def fit_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    loss: str = "mse",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
):
    model = build_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
        loss=loss,
        huber_delta=huber_delta,
    )
    callbacks = build_training_callbacks(
        x_val,
        y_val,
        val_group_ids,
        patience,
        monitor_metric,
        target_scaler=target_scaler,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def predict(model: keras.Model, x):
    return model.predict(x, verbose=0).reshape(-1)


def split_frame_by_date(df: pd.DataFrame, train_end_date: str, val_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    train_df = work[work["Date"] <= pd.Timestamp(train_end_date)].copy()
    val_df = work[(work["Date"] > pd.Timestamp(train_end_date)) & (work["Date"] <= pd.Timestamp(val_end_date))].copy()
    test_df = work[work["Date"] > pd.Timestamp(val_end_date)].copy()
    return train_df, val_df, test_df


def fit_feature_scaler(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> FeatureScaler:
    features = df.loc[:, feature_columns].astype(float)
    mean = features.mean(axis=0).to_numpy()
    std = features.std(axis=0).replace(0, 1).fillna(1).to_numpy()
    return FeatureScaler(mean=mean, std=std, feature_columns=feature_columns)


def fit_target_scaler(y: np.ndarray) -> TargetScaler:
    y = np.asarray(y, dtype=np.float32)
    std = float(np.std(y))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return TargetScaler(mean=float(np.mean(y)), std=std)


def apply_target_scaler(y: np.ndarray, scaler: TargetScaler | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if scaler is None:
        return y
    return ((y - scaler.mean) / scaler.std).astype(np.float32)


def inverse_target_scaler_values(y: np.ndarray, scaler: TargetScaler | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if scaler is None:
        return y
    return (y * scaler.std + scaler.mean).astype(np.float32)


def apply_feature_scaler(df: pd.DataFrame, scaler: FeatureScaler) -> pd.DataFrame:
    work = df.copy()
    work = work.astype({col: float for col in scaler.feature_columns})
    values = work.loc[:, scaler.feature_columns].to_numpy()
    work.loc[:, scaler.feature_columns] = (values - scaler.mean) / scaler.std
    return work


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list = []
    y_list = []
    meta_rows = []
    required_cols = list(feature_columns) + [target_column, "Date", "code"]

    for code, group in df.sort_values(["code", "Date"]).groupby("code"):
        group = group.dropna(subset=required_cols).reset_index(drop=True)
        if len(group) < window_size:
            continue
        feature_values = group.loc[:, feature_columns].to_numpy(dtype=float)
        target_values = group.loc[:, target_column].to_numpy(dtype=float)
        dates = pd.to_datetime(group["Date"])

        for end_idx in range(window_size - 1, len(group)):
            start_idx = end_idx - window_size + 1
            x_list.append(feature_values[start_idx : end_idx + 1])
            y_list.append(target_values[end_idx])
            meta_rows.append(
                {
                    "code": code,
                    "Date": dates.iloc[end_idx],
                    "target": target_values[end_idx],
                }
            )

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def split_sequence_dataset(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    dates = pd.to_datetime(meta["Date"])
    train_mask = dates <= pd.Timestamp(train_end_date)
    val_mask = (dates > pd.Timestamp(train_end_date)) & (dates <= pd.Timestamp(val_end_date))
    test_mask = dates > pd.Timestamp(val_end_date)

    return {
        "train": (x[train_mask], y[train_mask], meta.loc[train_mask].reset_index(drop=True)),
        "val": (x[val_mask], y[val_mask], meta.loc[val_mask].reset_index(drop=True)),
        "test": (x[test_mask], y[test_mask], meta.loc[test_mask].reset_index(drop=True)),
    }
