from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from src.evaluation.metric import evaluate


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: tuple[str, ...]


class ValidationRelScoreCallback(keras.callbacks.Callback):
    def __init__(self, x_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if len(self.x_val) < 3 or len(self.y_val) < 3:
            logs["val_rel_score"] = np.nan
            return
        prediction = self.model.predict(self.x_val, verbose=0).reshape(-1)
        try:
            logs["val_rel_score"] = float(evaluate(prediction, self.y_val)["rel_score"])
        except Exception:
            logs["val_rel_score"] = np.nan


def resolve_loss(loss: str, huber_delta: float):
    if loss == "mse":
        return "mse"
    if loss == "huber":
        return keras.losses.Huber(delta=huber_delta)
    raise ValueError("loss must be one of: mse, huber")


def build_training_callbacks(
    x_val: np.ndarray,
    y_val: np.ndarray,
    patience: int,
    monitor_metric: str,
) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = []
    if monitor_metric == "val_rel_score":
        callbacks.append(ValidationRelScoreCallback(x_val, y_val))
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
    return callbacks


def build_model(
    window_size: int,
    num_features: int,
    lstm_units: int,
    dropout: float,
    lr: float,
    loss: str = "mse",
    huber_delta: float = 0.01,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(window_size, num_features)),
            layers.LSTM(
                lstm_units, 
                kernel_regularizer=regularizers.l2(1e-4),
                recurrent_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization(),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
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
    lstm_units: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    loss: str = "mse",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
):
    model = build_model(window_size, num_features, lstm_units, dropout, lr, loss=loss, huber_delta=huber_delta)
    callbacks = build_training_callbacks(x_val, y_val, patience, monitor_metric)
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
