"""Universe definition and point-in-time filtering utilities."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_historical_universe_mask(
    df: pd.DataFrame,
    csv_path: Path,
    date_col: str = "Date",
    symbol_col: str = "code",
) -> np.ndarray:
    """Computes a boolean mask indicating if each row in df (date_col, symbol_col)
    was a constituent of the universe at that date based on review periods in csv_path.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe definition file {csv_path} not found.")

    df_hist = pd.read_csv(csv_path)
    df_hist["start_date"] = pd.to_datetime(df_hist["start_date"])
    df_hist["end_date"] = pd.to_datetime(df_hist["end_date"])
    df_hist["symbol"] = df_hist["symbol"].str.strip().str.upper()

    df_keys = df[[date_col, symbol_col]].copy()
    df_keys[date_col] = pd.to_datetime(df_keys[date_col])
    df_keys[symbol_col] = df_keys[symbol_col].astype(str).str.strip().str.upper()
    df_keys["idx"] = range(len(df_keys))

    merged = df_keys.merge(df_hist, left_on=symbol_col, right_on="symbol", how="inner")
    active = merged[(merged[date_col] >= merged["start_date"]) & (merged[date_col] <= merged["end_date"])]

    active_indices = active["idx"].unique()
    mask = np.zeros(len(df), dtype=bool)
    mask[active_indices] = True
    return mask


def filter_sequence_by_universe(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    csv_path: Path,
    date_col: str = "Date",
    symbol_col: str = "code",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Filters sequence datasets (x, y, meta) to keep only point-in-time universe constituents."""
    mask = load_historical_universe_mask(meta, csv_path, date_col=date_col, symbol_col=symbol_col)
    return x[mask], y[mask], meta.loc[mask].reset_index(drop=True)
