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
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume_match"
VALUE_COLUMN = "value_match"
VALUE_EST_COLUMN = "value_match_est"
VALUE_TRADED_COLUMN = "value_traded"
ADV20_VALUE_COLUMN = "adv20_value_traded"
LIMIT_UP_LIKE_COLUMN = "entry_limit_up_like"
LIMIT_DOWN_LIKE_COLUMN = "entry_limit_down_like"
HARD_ISSUE_COLUMN = "has_hard_issue"
PROFILE_EVENT_COLUMN = "profile_event"
CLEAN_PROFILE_COLUMN = "clean_profile"
PANEL_ROW_ID_COLUMN = "panel_row_id"


@dataclass
class PreparedPanel:
    frame: pd.DataFrame
    train_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    scaler_mean: float
    scaler_std: float
    test_dates: tuple[pd.Timestamp, ...] = ()


@dataclass
class SequenceDataset:
    features: np.ndarray
    labels: np.ndarray
    labels_one_hot: np.ndarray
    anchor_dates: np.ndarray
    realized_dates: np.ndarray
    codes: np.ndarray
    next_returns: np.ndarray
    panel_row_ids: np.ndarray


def _read_market_csv(csv_path: Path) -> pd.DataFrame:
    preferred_columns = [
        "Date",
        "code",
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        PRICE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
        VALUE_EST_COLUMN,
        "value_match_imputed",
        HARD_ISSUE_COLUMN,
        PROFILE_EVENT_COLUMN,
        "drop_due_to_event_buffer",
        CLEAN_PROFILE_COLUMN,
    ]
    header = pd.read_csv(csv_path, nrows=0).columns
    usecols = [column for column in preferred_columns if column in header]
    return pd.read_csv(csv_path, usecols=usecols)


def _load_vn_profile_data(data_dir: Path, profile: str) -> pd.DataFrame:
    history_dir = Path(data_dir) / "assets" / "data_info_vn" / "history"
    profile_path = history_dir / f"vn_gold_{profile}.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"VN curated profile does not exist: {profile_path}")
    return _read_market_csv(profile_path)


def load_market_data(
    data_dir: Path,
    markets: Iterable[str],
    vn_data_profile: str | None = None,
) -> pd.DataFrame:
    """Placeholder loader that reads all ticker CSVs under the selected market folders.

    Expected schema per file: Date, code, adjust.
    Replace or extend this function if you later source data from a database or parquet lake.
    """

    frames: list[pd.DataFrame] = []
    for market in markets:
        if market == "VN" and vn_data_profile:
            frame = _load_vn_profile_data(data_dir, vn_data_profile)
            frame["market"] = market
            frames.append(frame)
            continue

        market_dir = Path(data_dir) / market
        if not market_dir.exists():
            raise FileNotFoundError(f"Market directory does not exist: {market_dir}")

        for csv_path in sorted(market_dir.glob("*.csv")):
            frame = _read_market_csv(csv_path)
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
    forward_horizon_days: int = 1,
) -> pd.DataFrame:
    """Build the forward-horizon cross-sectional target used in Fischer-Krauss style ranking."""

    if forward_horizon_days <= 0:
        raise ValueError("forward_horizon_days must be positive.")

    frame = compute_daily_returns(price_frame)
    for column in [OPEN_COLUMN, HIGH_COLUMN, LOW_COLUMN, CLOSE_COLUMN, VOLUME_COLUMN, VALUE_COLUMN, VALUE_EST_COLUMN]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    traded_value = None
    if VALUE_EST_COLUMN in frame.columns:
        traded_value = frame[VALUE_EST_COLUMN]
    elif {CLOSE_COLUMN, VOLUME_COLUMN}.issubset(frame.columns):
        traded_value = frame[CLOSE_COLUMN] * frame[VOLUME_COLUMN]
    elif VALUE_COLUMN in frame.columns:
        traded_value = frame[VALUE_COLUMN]
    if VALUE_COLUMN in frame.columns:
        observed_value = frame[VALUE_COLUMN]
        if traded_value is None:
            traded_value = observed_value
        else:
            traded_value = observed_value.where(observed_value.notna() & observed_value.gt(0), traded_value)
    if traded_value is None:
        traded_value = pd.Series(np.nan, index=frame.index)
    frame[VALUE_TRADED_COLUMN] = pd.to_numeric(traded_value, errors="coerce")
    frame[ADV20_VALUE_COLUMN] = frame.groupby(CODE_COLUMN)[VALUE_TRADED_COLUMN].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )

    close_returns = pd.to_numeric(frame[RETURN_COLUMN], errors="coerce")
    if {CLOSE_COLUMN, HIGH_COLUMN}.issubset(frame.columns):
        frame[LIMIT_UP_LIKE_COLUMN] = frame[CLOSE_COLUMN].eq(frame[HIGH_COLUMN]) & close_returns.ge(0.068)
    else:
        frame[LIMIT_UP_LIKE_COLUMN] = False
    if {CLOSE_COLUMN, LOW_COLUMN}.issubset(frame.columns):
        frame[LIMIT_DOWN_LIKE_COLUMN] = frame[CLOSE_COLUMN].eq(frame[LOW_COLUMN]) & close_returns.le(-0.068)
    else:
        frame[LIMIT_DOWN_LIKE_COLUMN] = False
    if HARD_ISSUE_COLUMN not in frame.columns:
        frame[HARD_ISSUE_COLUMN] = False
    else:
        frame[HARD_ISSUE_COLUMN] = frame[HARD_ISSUE_COLUMN].fillna(False).astype(bool)
    if PROFILE_EVENT_COLUMN not in frame.columns:
        frame[PROFILE_EVENT_COLUMN] = False
    else:
        frame[PROFILE_EVENT_COLUMN] = frame[PROFILE_EVENT_COLUMN].fillna(False).astype(bool)
    if CLEAN_PROFILE_COLUMN not in frame.columns:
        frame[CLEAN_PROFILE_COLUMN] = pd.NA

    def _forward_return(series: pd.Series) -> pd.Series:
        future_leg = (1.0 + series).shift(-1)
        rolled = future_leg.rolling(window=forward_horizon_days, min_periods=forward_horizon_days).apply(
            np.prod,
            raw=True,
        )
        return rolled.shift(-(forward_horizon_days - 1)) - 1.0

    frame[NEXT_RETURN_COLUMN] = frame.groupby(CODE_COLUMN, group_keys=False)[RETURN_COLUMN].apply(_forward_return)
    frame[REALIZED_DATE_COLUMN] = frame.groupby(CODE_COLUMN)[ANCHOR_DATE_COLUMN].shift(-forward_horizon_days)

    valid_counts = frame.groupby(ANCHOR_DATE_COLUMN)[NEXT_RETURN_COLUMN].transform("count")
    frame = frame[valid_counts >= min_cross_sectional_count].copy()
    cross_sectional_median = frame.groupby(ANCHOR_DATE_COLUMN)[NEXT_RETURN_COLUMN].transform("median")
    frame["cross_sectional_median_next_return"] = cross_sectional_median
    frame[TARGET_COLUMN] = (frame[NEXT_RETURN_COLUMN] >= cross_sectional_median).astype(np.int32)

    required = [RETURN_COLUMN, NEXT_RETURN_COLUMN, REALIZED_DATE_COLUMN, TARGET_COLUMN]
    frame = frame.dropna(subset=required).reset_index(drop=True)
    frame[PANEL_ROW_ID_COLUMN] = np.arange(len(frame), dtype=np.int64)
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
    panel_row_ids: list[int] = []

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
        row_ids = group[PANEL_ROW_ID_COLUMN].to_numpy(dtype=np.int64)

        for end_idx in range(lookback - 1, len(group)):
            start_idx = end_idx - lookback + 1
            features_list.append(signal[start_idx : end_idx + 1].reshape(lookback, 1))
            labels_list.append(int(labels[end_idx]))
            anchor_dates.append(anchors[end_idx])
            realized_dates.append(realized[end_idx])
            codes.append(code)
            next_returns.append(float(next_ret[end_idx]))
            panel_row_ids.append(int(row_ids[end_idx]))

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
        panel_row_ids=np.asarray(panel_row_ids, dtype=np.int64),
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
        panel_row_ids=dataset.panel_row_ids[mask],
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
    prepared, train_dataset, validation_dataset, _ = build_datasets_for_date_splits(
        panel=panel,
        lookback=lookback,
        train_dates=train_dates,
        validation_dates=validation_dates,
    )
    return prepared, train_dataset, validation_dataset


def build_datasets_for_date_splits(
    panel: pd.DataFrame,
    lookback: int,
    train_dates: tuple[pd.Timestamp, ...],
    validation_dates: tuple[pd.Timestamp, ...],
    test_dates: tuple[pd.Timestamp, ...] | None = None,
) -> tuple[PreparedPanel, SequenceDataset, SequenceDataset, SequenceDataset | None]:
    standardized_panel, mean, std = standardize_returns(panel, train_dates=train_dates)
    full_dataset = build_sequences(standardized_panel, lookback=lookback)

    train_mask = np.isin(full_dataset.anchor_dates, np.asarray(train_dates, dtype="datetime64[ns]"))
    validation_mask = np.isin(full_dataset.anchor_dates, np.asarray(validation_dates, dtype="datetime64[ns]"))
    test_mask = (
        np.isin(full_dataset.anchor_dates, np.asarray(test_dates, dtype="datetime64[ns]"))
        if test_dates is not None
        else None
    )

    train_dataset = slice_dataset(full_dataset, train_mask)
    validation_dataset = slice_dataset(full_dataset, validation_mask)
    test_dataset = slice_dataset(full_dataset, test_mask) if test_mask is not None else None

    if train_dataset.features.size == 0 or validation_dataset.features.size == 0:
        raise ValueError(
            "Train or validation sequence set is empty. Increase the history or lower the lookback."
        )
    if test_dates is not None and (test_dataset is None or test_dataset.features.size == 0):
        raise ValueError(
            "Test sequence set is empty. Increase history, widen the test window, or lower the lookback."
        )

    prepared = PreparedPanel(
        frame=standardized_panel,
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=tuple(test_dates or ()),
        scaler_mean=mean,
        scaler_std=std,
    )
    return prepared, train_dataset, validation_dataset, test_dataset
