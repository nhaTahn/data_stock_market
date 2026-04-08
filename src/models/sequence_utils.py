from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: tuple[str, ...]


@dataclass
class TargetScaler:
    mean: float
    std: float


@dataclass
class LocalTargetNormalizer:
    column: str
    floor: float


def build_magnitude_sample_weights(
    y: np.ndarray,
    strength: float = 1.5,
    reference_quantile: float = 0.75,
    clip_multiple: float = 3.0,
) -> np.ndarray:
    abs_y = np.abs(np.asarray(y, dtype=np.float32).reshape(-1))
    if len(abs_y) == 0:
        return np.ones(0, dtype=np.float32)

    valid = abs_y[np.isfinite(abs_y)]
    if len(valid) == 0:
        return np.ones_like(abs_y, dtype=np.float32)

    reference = float(np.quantile(valid, reference_quantile))
    reference = max(reference, 1e-4)
    normalized = np.clip(abs_y / reference, 0.0, clip_multiple)
    weights = 1.0 + strength * np.tanh(normalized)
    return weights.astype(np.float32)


def build_sign_magnitude_sample_weights(sample_weight: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    return {
        "signed_prediction": sample_weight,
        "magnitude": sample_weight,
        "sign_prob": np.sqrt(sample_weight).astype(np.float32),
    }


def build_event_gated_sample_weights(
    sample_weight: np.ndarray | None,
    event_target: np.ndarray,
) -> dict[str, np.ndarray]:
    event_target = np.asarray(event_target, dtype=np.float32).reshape(-1)
    if sample_weight is None:
        base = np.ones_like(event_target, dtype=np.float32)
    else:
        base = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    event_focus = 0.25 + 0.75 * event_target
    magnitude_focus = 0.1 + 0.9 * event_target
    return {
        "signed_prediction": base,
        "event_prob": base,
        "sign_prob": (base * event_focus).astype(np.float32),
        "magnitude": (base * magnitude_focus).astype(np.float32),
    }


def extract_prediction_array(raw_prediction, prediction_key: str | int | None = None) -> np.ndarray:
    if isinstance(raw_prediction, dict):
        if prediction_key is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output dict predictions.")
            raw_prediction = next(iter(raw_prediction.values()))
        else:
            raw_prediction = raw_prediction[prediction_key]
    elif isinstance(raw_prediction, (list, tuple)):
        if prediction_key is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output list predictions.")
            raw_prediction = raw_prediction[0]
        else:
            raw_prediction = raw_prediction[int(prediction_key)]
    array = np.asarray(raw_prediction, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(-1)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.shape[-1] == 1:
        return array.reshape(-1)
    if prediction_key is None:
        raise ValueError("prediction_key is required for multi-column predictions.")
    return np.asarray(array[:, int(prediction_key)], dtype=np.float32).reshape(-1)


def set_global_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def predict(model: keras.Model, x, prediction_key: str | int | None = None):
    raw_prediction = model.predict(x, verbose=0)
    return extract_prediction_array(raw_prediction, prediction_key)


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


def fit_local_target_normalizer(
    scale_values: np.ndarray,
    column: str,
) -> LocalTargetNormalizer:
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.abs(scale_values)
    valid = scale_values[np.isfinite(scale_values) & (scale_values > 0)]
    if len(valid) == 0:
        floor = 1.0
    else:
        floor = max(float(np.quantile(valid, 0.25)), 1e-4)
    return LocalTargetNormalizer(column=column, floor=floor)


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


def apply_local_target_normalizer(
    y: np.ndarray,
    scale_values: np.ndarray | None,
    normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if normalizer is None or scale_values is None:
        return y
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.where(np.isfinite(scale_values), np.abs(scale_values), normalizer.floor)
    denom = np.maximum(scale_values, normalizer.floor)
    return (y / denom).astype(np.float32)


def inverse_local_target_normalizer(
    y: np.ndarray,
    scale_values: np.ndarray | None,
    normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if normalizer is None or scale_values is None:
        return y
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.where(np.isfinite(scale_values), np.abs(scale_values), normalizer.floor)
    denom = np.maximum(scale_values, normalizer.floor)
    return (y * denom).astype(np.float32)


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
    extra_meta_columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list = []
    y_list = []
    meta_rows = []
    required_cols = list(feature_columns) + [target_column, "Date", "code", *extra_meta_columns]

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
                    **{col: group.iloc[end_idx][col] for col in extra_meta_columns},
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
