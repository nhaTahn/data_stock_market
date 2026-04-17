from __future__ import annotations

import numpy as np
import pandas as pd


def split_frame_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    train_df = work[work["Date"] <= pd.Timestamp(train_end_date)].copy()
    val_df = work[(work["Date"] > pd.Timestamp(train_end_date)) & (work["Date"] <= pd.Timestamp(val_end_date))].copy()
    test_df = work[work["Date"] > pd.Timestamp(val_end_date)].copy()
    return train_df, val_df, test_df


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    window_size: int,
    extra_meta_columns: tuple[str, ...] = (),
    sequence_normalization: str = "none",
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
            window = feature_values[start_idx : end_idx + 1].copy()
            if sequence_normalization == "instance_zscore":
                mean = np.nanmean(window, axis=0, keepdims=True)
                std = np.nanstd(window, axis=0, keepdims=True)
                std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
                window = (window - mean) / std
            x_list.append(window)
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
