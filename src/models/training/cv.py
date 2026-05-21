"""Walk-forward / expanding-window cross-validation utilities.

Mục đích:

- Cho phép selection (hyperparam, feature set, threshold) trên **train-only**
  thông qua các fold tách rời theo thời gian, KHÔNG chạm vào validation set
  Apr-2020 → Nov-2022.
- Cung cấp embargo (purge) để loại bỏ rò rỉ giữa train và val fold liền kề
  khi target lấy thông tin từ tương lai (vd: `target_next_return` ở row `t`
  dùng `adjust[t+1]`).

Có 2 mode chính:

1. **Expanding window** (mặc định, paper-style): train mở rộng dần, val luôn
   là cửa sổ tiếp theo. Mỗi fold dùng nhiều dữ liệu hơn fold trước.
2. **Rolling window**: train cố định độ dài, trượt theo `step_days`. Dùng khi
   muốn so sánh stability theo thời gian.

Reuse:

- [src/models/selection/holding_period.py:257](../selection/holding_period.py:257)
  có `build_walk_forward_folds(dates, train_days, test_days, step_days)`
  dùng rolling window (train cố định). Module này KHÔNG thay thế nó; hai hàm
  bổ sung cho nhau:
  - `build_walk_forward_folds` (rolling) — dùng cho analysis turnover/holding.
  - `expanding_walk_forward_folds` (expanding) — dùng cho model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_days: int

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.train_start) & (dates <= self.train_end)

    def val_mask(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.val_start) & (dates <= self.val_end)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "embargo_days": self.embargo_days,
        }


def _normalize_dates(dates: Iterable) -> list[pd.Timestamp]:
    series = pd.to_datetime(pd.Series(list(dates)))
    unique = series.drop_duplicates().sort_values().reset_index(drop=True)
    return list(unique)


def expanding_walk_forward_folds(
    dates: Iterable,
    *,
    min_train_days: int = 750,
    val_days: int = 126,
    step_days: int = 63,
    embargo_days: int = 5,
    max_folds: int | None = None,
) -> list[Fold]:
    """Generate expanding-window folds confined to the given date list.

    Args:
        dates: iterable of dates (must be train-period dates only — caller is
            responsible for filtering out the validation/holdout periods).
        min_train_days: minimum number of trading days in the initial train fold.
        val_days: number of trading days in each val fold.
        step_days: how many days to advance val start between folds.
        embargo_days: number of train days dropped at the tail before each
            val fold begins, to prevent target-leakage via `shift(-1)`.
        max_folds: optional cap on number of folds returned.

    Returns:
        list[Fold] sorted by fold_id (chronological).

    Notes:
        - Embargo is implemented by truncating the train end. Train ends
          `embargo_days` calendar trading days BEFORE the val start.
        - First fold: train = dates[0 .. min_train_days - embargo_days - 1],
          val = dates[min_train_days .. min_train_days + val_days - 1].
    """
    if min_train_days <= 0 or val_days <= 0 or step_days <= 0:
        raise ValueError("min_train_days, val_days, step_days must be > 0")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    dates_list = _normalize_dates(dates)
    n = len(dates_list)
    folds: list[Fold] = []
    val_start_idx = min_train_days
    fold_id = 1
    while val_start_idx + val_days <= n:
        train_end_idx = val_start_idx - embargo_days - 1
        if train_end_idx < 0:
            break
        train_start = dates_list[0]
        train_end = dates_list[train_end_idx]
        val_start = dates_list[val_start_idx]
        val_end = dates_list[val_start_idx + val_days - 1]
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                embargo_days=embargo_days,
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
        fold_id += 1
        val_start_idx += step_days
    return folds


def rolling_walk_forward_folds(
    dates: Iterable,
    *,
    train_days: int = 750,
    val_days: int = 126,
    step_days: int = 63,
    embargo_days: int = 5,
    max_folds: int | None = None,
) -> list[Fold]:
    """Generate rolling-window folds with fixed train length.

    Useful for stability analysis: do early folds beat late folds?
    """
    if train_days <= 0 or val_days <= 0 or step_days <= 0:
        raise ValueError("train_days, val_days, step_days must be > 0")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    dates_list = _normalize_dates(dates)
    n = len(dates_list)
    folds: list[Fold] = []
    val_start_idx = train_days
    fold_id = 1
    while val_start_idx + val_days <= n:
        train_start_idx = val_start_idx - train_days
        train_end_idx = val_start_idx - embargo_days - 1
        if train_end_idx < train_start_idx:
            break
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=dates_list[train_start_idx],
                train_end=dates_list[train_end_idx],
                val_start=dates_list[val_start_idx],
                val_end=dates_list[val_start_idx + val_days - 1],
                embargo_days=embargo_days,
            )
        )
        if max_folds is not None and len(folds) >= max_folds:
            break
        fold_id += 1
        val_start_idx += step_days
    return folds


def iter_fold_frames(
    frame: pd.DataFrame,
    folds: Iterable[Fold],
    date_column: str = "Date",
) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold, train_frame, val_frame) tuples for each fold.

    The fold masks are computed on the `date_column` of the input frame.
    Rows outside both train and val windows are dropped.
    """
    if date_column not in frame.columns:
        raise KeyError(f"date_column '{date_column}' not found in frame.")
    dates = pd.to_datetime(frame[date_column])
    for fold in folds:
        train_mask = fold.train_mask(dates)
        val_mask = fold.val_mask(dates)
        yield (
            fold,
            frame.loc[train_mask].copy(),
            frame.loc[val_mask].copy(),
        )


def summarize_folds(folds: Iterable[Fold]) -> pd.DataFrame:
    """One-row-per-fold summary frame for diagnostic logs."""
    rows = [fold.to_dict() for fold in folds]
    if not rows:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "embargo_days",
                "train_days_approx",
                "val_days_approx",
            ]
        )
    df = pd.DataFrame(rows)
    df["train_days_approx"] = (
        pd.to_datetime(df["train_end"]) - pd.to_datetime(df["train_start"])
    ).dt.days
    df["val_days_approx"] = (
        pd.to_datetime(df["val_end"]) - pd.to_datetime(df["val_start"])
    ).dt.days
    return df


def validate_no_overlap(folds: Iterable[Fold]) -> None:
    """Sanity check: train_end < val_start for every fold (after embargo)."""
    for fold in folds:
        if fold.train_end >= fold.val_start:
            raise AssertionError(
                f"Fold {fold.fold_id}: train_end {fold.train_end} must be < val_start {fold.val_start}"
            )
