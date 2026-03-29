from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tf_lstm.config import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class SequenceBatch:
    features: np.ndarray
    targets: np.ndarray
    dates: list[str]
    codes: list[str]


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["code", "Date"]).reset_index(drop=True)

    required_cols = ["Date", "code", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")

    return df.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN]).copy()


def split_dataset(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train_df = df[df["Date"] <= train_end_ts].copy()
    val_df = df[(df["Date"] > train_end_ts) & (df["Date"] <= val_end_ts)].copy()
    test_df = df[df["Date"] > val_end_ts].copy()
    return train_df, val_df, test_df


def fit_feature_scaler(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train_df[FEATURE_COLUMNS].mean()
    std = train_df[FEATURE_COLUMNS].std(ddof=0).replace(0, 1.0)
    return mean, std


def fit_target_scaler(train_df: pd.DataFrame) -> tuple[float, float]:
    mean = float(train_df[TARGET_COLUMN].mean())
    std = float(train_df[TARGET_COLUMN].std(ddof=0))
    return mean, (std if std != 0 else 1.0)


def apply_feature_scaler(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    scaled = df.copy()
    scaled[FEATURE_COLUMNS] = (scaled[FEATURE_COLUMNS] - mean) / std
    return scaled


def apply_target_scaler(df: pd.DataFrame, target_mean: float, target_std: float) -> pd.DataFrame:
    scaled = df.copy()
    scaled[TARGET_COLUMN] = (scaled[TARGET_COLUMN] - target_mean) / target_std
    return scaled


def scale_split(df: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series, target_mean: float, target_std: float) -> pd.DataFrame:
    return apply_target_scaler(apply_feature_scaler(df, feature_mean, feature_std), target_mean, target_std)


def build_sequences(df: pd.DataFrame, window_size: int) -> SequenceBatch:
    features_list: list[np.ndarray] = []
    targets_list: list[float] = []
    dates: list[str] = []
    codes: list[str] = []

    for code, group in df.groupby("code"):
        group = group.sort_values("Date").reset_index(drop=True)
        feature_values = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target_values = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
        date_values = group["Date"].dt.strftime("%Y-%m-%d").tolist()

        for end_idx in range(window_size - 1, len(group)):
            start_idx = end_idx - window_size + 1
            features_list.append(feature_values[start_idx : end_idx + 1])
            targets_list.append(target_values[end_idx])
            dates.append(date_values[end_idx])
            codes.append(code)

    if not features_list:
        return SequenceBatch(
            features=np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32),
            targets=np.empty((0,), dtype=np.float32),
            dates=[],
            codes=[],
        )

    return SequenceBatch(
        features=np.stack(features_list),
        targets=np.array(targets_list, dtype=np.float32),
        dates=dates,
        codes=codes,
    )
