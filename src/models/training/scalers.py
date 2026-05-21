from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


def _broadcast_scale_values(scale_values: np.ndarray, target_ndim: int) -> np.ndarray:
    scale_values = np.asarray(scale_values, dtype=np.float32)
    while scale_values.ndim < target_ndim:
        scale_values = scale_values[..., None]
    return scale_values


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
    scale_values = _broadcast_scale_values(scale_values, y.ndim)
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
    scale_values = _broadcast_scale_values(scale_values, y.ndim)
    scale_values = np.where(np.isfinite(scale_values), np.abs(scale_values), normalizer.floor)
    denom = np.maximum(scale_values, normalizer.floor)
    return (y * denom).astype(np.float32)


def apply_feature_scaler(df: pd.DataFrame, scaler: FeatureScaler) -> pd.DataFrame:
    work = df.copy()
    work = work.astype({col: float for col in scaler.feature_columns})
    values = work.loc[:, scaler.feature_columns].to_numpy()
    work.loc[:, scaler.feature_columns] = (values - scaler.mean) / scaler.std
    return work
