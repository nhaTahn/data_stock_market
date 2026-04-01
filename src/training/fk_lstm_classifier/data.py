from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RETURN_COLUMN = "daily_return"
STANDARDIZED_RETURN_COLUMN = "daily_return_z"
TARGET_COLUMN = "target_class"
NEXT_RETURN_COLUMN = "next_return"
REALIZED_DATE_COLUMN = "realized_date"
ANCHOR_DATE_COLUMN = "Date"
CODE_COLUMN = "code"
PRICE_COLUMN = "adjust"


@dataclass
class PreparedPanel:
    frame: pd.DataFrame
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    scaler_mean: float
    scaler_std: float


@dataclass
class SequenceDataset:
    features: np.ndarray
    labels: np.ndarray
    labels_one_hot: np.ndarray
    anchor_dates: np.ndarray
    realized_dates: np.ndarray
    codes: np.ndarray
    next_returns: np.ndarray


def load_market_data(data_dir: Path, markets: Iterable[str]) -> pd.DataFrame:
    """Placeholder loader that reads all ticker CSVs under the selected market folders.

    Expected schema per file: Date, code, adjust.
    Replace or extend this function if you later source data from a database or parquet lake.
    """

    frames: list[pd.DataFrame] = []
    for market in markets:
        market_dir = Path(data_dir) / market
        if not market_dir.exists():
            raise FileNotFoundError(f"Market directory does not exist: {market_dir}")

        for csv_path in sorted(market_dir.glob("*.csv")):
            frame = pd.read_csv(csv_path, usecols=["Date", "code", PRICE_COLUMN])
            frame["market"] = market
            frames.append(frame)

    if not frames:
        raise ValueError(f"No CSV files were found under {data_dir} for markets={tuple(markets)}")

    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna(subset=["Date", CODE_COLUMN, PRICE_COLUMN]).copy()
    data[CODE_COLUMN] = data["market"].astype(str) + ":" + data[CODE_COLUMN].astype(str)
    data = data.sort_values([CODE_COLUMN, ANCHOR_DATE_COLUMN]).reset_index(drop=True)
    return data


def compute_daily_returns(price_frame: pd.DataFrame, price_column: str = PRICE_COLUMN) -> pd.DataFrame:
    frame = price_frame.sort_values([CODE_COLUMN, ANCHOR_DATE_COLUMN]).copy()
    frame[RETURN_COLUMN] = frame.groupby(CODE_COLUMN)[price_column].pct_change()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_classification_panel(
    price_frame: pd.DataFrame,
    min_cross_sectional_count: int = 20,
) -> pd.DataFrame:
    """Build the one-day-ahead cross-sectional target used in Fischer-Krauss style ranking."""

    frame = compute_daily_returns(price_frame)
    frame[NEXT_RETURN_COLUMN] = frame.groupby(CODE_COLUMN)[RETURN_COLUMN].shift(-1)
    frame[REALIZED_DATE_COLUMN] = frame.groupby(CODE_COLUMN)[ANCHOR_DATE_COLUMN].shift(-1)

    valid_counts = frame.groupby(ANCHOR_DATE_COLUMN)[NEXT_RETURN_COLUMN].transform("count")
    frame = frame[valid_counts >= min_cross_sectional_count].copy()
    cross_sectional_median = frame.groupby(ANCHOR_DATE_COLUMN)[NEXT_RETURN_COLUMN].transform("median")
    frame["cross_sectional_median_next_return"] = cross_sectional_median
    frame[TARGET_COLUMN] = (frame[NEXT_RETURN_COLUMN] >= cross_sectional_median).astype(np.int32)

    required = [RETURN_COLUMN, NEXT_RETURN_COLUMN, REALIZED_DATE_COLUMN, TARGET_COLUMN]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    return frame


def split_dates(frame: pd.DataFrame, train_ratio: float) -> tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]:
    unique_dates = tuple(sorted(frame[ANCHOR_DATE_COLUMN].drop_duplicates().tolist()))
    if len(unique_dates) < 2:
        raise ValueError("Need at least two distinct dates to create train and validation splits.")

    split_idx = int(len(unique_dates) * train_ratio)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    return unique_dates[:split_idx], unique_dates[split_idx:]


def standardize_returns(
    frame: pd.DataFrame,
    train_dates: tuple[pd.Timestamp, ...],
) -> tuple[pd.DataFrame, float, float]:
    train_mask = frame[ANCHOR_DATE_COLUMN].isin(train_dates)
    train_returns = frame.loc[train_mask, RETURN_COLUMN]
    mean = float(train_returns.mean())
    std = float(train_returns.std(ddof=0))
    std = std if std > 0 else 1.0

    standardized = frame.copy()
    standardized[STANDARDIZED_RETURN_COLUMN] = (standardized[RETURN_COLUMN] - mean) / std
    return standardized, mean, std


def _to_one_hot(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[labels.astype(np.int64)]


def build_sequences(
    frame: pd.DataFrame,
    lookback: int = 240,
) -> SequenceDataset:
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    anchor_dates: list[np.datetime64] = []
    realized_dates: list[np.datetime64] = []
    codes: list[str] = []
    next_returns: list[float] = []

    usable = frame.sort_values([CODE_COLUMN, ANCHOR_DATE_COLUMN]).copy()
    usable = usable.dropna(
        subset=[
            STANDARDIZED_RETURN_COLUMN,
            TARGET_COLUMN,
            NEXT_RETURN_COLUMN,
            REALIZED_DATE_COLUMN,
        ]
    )

    for code, group in usable.groupby(CODE_COLUMN, sort=False):
        group = group.reset_index(drop=True)
        signal = group[STANDARDIZED_RETURN_COLUMN].to_numpy(dtype=np.float32)
        labels = group[TARGET_COLUMN].to_numpy(dtype=np.int32)
        realized = pd.to_datetime(group[REALIZED_DATE_COLUMN]).to_numpy(dtype="datetime64[ns]")
        anchors = pd.to_datetime(group[ANCHOR_DATE_COLUMN]).to_numpy(dtype="datetime64[ns]")
        next_ret = group[NEXT_RETURN_COLUMN].to_numpy(dtype=np.float32)

        for end_idx in range(lookback - 1, len(group)):
            start_idx = end_idx - lookback + 1
            features_list.append(signal[start_idx : end_idx + 1].reshape(lookback, 1))
            labels_list.append(int(labels[end_idx]))
            anchor_dates.append(anchors[end_idx])
            realized_dates.append(realized[end_idx])
            codes.append(code)
            next_returns.append(float(next_ret[end_idx]))

    if not features_list:
        raise ValueError(
            "No sequences were generated. Reduce the lookback window or load a longer history."
        )

    labels_array = np.asarray(labels_list, dtype=np.int32)
    return SequenceDataset(
        features=np.asarray(features_list, dtype=np.float32),
        labels=labels_array,
        labels_one_hot=_to_one_hot(labels_array),
        anchor_dates=np.asarray(anchor_dates, dtype="datetime64[ns]"),
        realized_dates=np.asarray(realized_dates, dtype="datetime64[ns]"),
        codes=np.asarray(codes),
        next_returns=np.asarray(next_returns, dtype=np.float32),
    )


def slice_dataset(dataset: SequenceDataset, mask: np.ndarray) -> SequenceDataset:
    return SequenceDataset(
        features=dataset.features[mask],
        labels=dataset.labels[mask],
        labels_one_hot=dataset.labels_one_hot[mask],
        anchor_dates=dataset.anchor_dates[mask],
        realized_dates=dataset.realized_dates[mask],
        codes=dataset.codes[mask],
        next_returns=dataset.next_returns[mask],
    )


def prepare_datasets(
    price_frame: pd.DataFrame,
    lookback: int = 240,
    train_ratio: float = 0.8,
    min_cross_sectional_count: int = 20,
) -> tuple[PreparedPanel, SequenceDataset, SequenceDataset]:
    panel = build_classification_panel(
        price_frame=price_frame,
        min_cross_sectional_count=min_cross_sectional_count,
    )
    train_dates, validation_dates = split_dates(panel, train_ratio=train_ratio)
    standardized_panel, mean, std = standardize_returns(panel, train_dates=train_dates)
    full_dataset = build_sequences(standardized_panel, lookback=lookback)

    train_mask = np.isin(full_dataset.anchor_dates, np.asarray(train_dates, dtype="datetime64[ns]"))
    validation_mask = np.isin(
        full_dataset.anchor_dates,
        np.asarray(validation_dates, dtype="datetime64[ns]"),
    )
    train_dataset = slice_dataset(full_dataset, train_mask)
    validation_dataset = slice_dataset(full_dataset, validation_mask)

    if train_dataset.features.size == 0 or validation_dataset.features.size == 0:
        raise ValueError(
            "Train or validation sequence set is empty. Increase the history or lower the lookback."
        )

    prepared = PreparedPanel(
        frame=standardized_panel,
        train_dates=train_dates,
        validation_dates=validation_dates,
        scaler_mean=mean,
        scaler_std=std,
    )
    return prepared, train_dataset, validation_dataset
