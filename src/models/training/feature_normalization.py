from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_MARKET_CONTEXT_FEATURES = frozenset(
    {
        "a_d_ratio",
        "breadth_20",
        "market_ad_ratio_1",
        "market_ad_ratio_20",
        "market_breadth_20",
        "market_index_return",
        "market_leader_excess_return",
        "market_leader_return",
        "market_leader_return_1",
        "market_proxy_return_1",
        "market_return_5",
        "market_return_20",
        "market_return_60",
        "market_volatility_20",
        "vnindex_return",
        "volatility_expanding_median",
        "vingroup_momentum",
    }
)

DEFAULT_CALENDAR_FEATURES = frozenset({"day_of_week"})


@dataclass(frozen=True)
class FeatureNormalizationResult:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]


def _market_group_columns(df: pd.DataFrame) -> list[str]:
    return ["market"] if "market" in df.columns else []


def _market_date_group_columns(df: pd.DataFrame) -> list[str]:
    return [*_market_group_columns(df), "Date"]


def _stock_group_columns(df: pd.DataFrame) -> list[str]:
    return [*_market_group_columns(df), "code"]


def _rolling_zscore(
    series: pd.Series,
    *,
    window: int,
    min_periods: int,
    epsilon: float,
    strict_past: bool,
) -> pd.Series:
    base = series.shift(1) if strict_past else series
    mean = base.rolling(window, min_periods=min_periods).mean()
    std = base.rolling(window, min_periods=min_periods).std()
    std = std.where(std.abs() > epsilon)
    return (series - mean) / (std + epsilon)


def _is_passthrough_feature(feature: str) -> bool:
    return (
        feature.endswith("_rank")
        or feature.endswith("_rank_pct")
        or feature.startswith("is_")
        or feature.startswith("above_")
        or feature.startswith("has_")
        or feature.startswith("below_")
        or feature == "wyckoff_phase_60d"
    )


def _neutral_passthrough_value(feature: str) -> float:
    if feature.endswith("_rank_pct"):
        return 0.5
    if feature.endswith("_rank"):
        return 0.0
    if feature.startswith("is_") or feature.startswith("above_") or feature.startswith("has_") or feature.startswith("below_"):
        return 0.0
    if feature == "wyckoff_phase_60d":
        return 0.0
    return 0.0


def _fill_numeric_neutral(series: pd.Series, neutral_value: float) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).fillna(neutral_value)


def _safe_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def add_multimarket_feature_normalization(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    rolling_window: int = 60,
    min_periods: int = 20,
    include_cross_sectional_z: bool = True,
    include_cross_sectional_rank: bool = True,
    strict_past: bool = True,
    epsilon: float = 1e-6,
) -> FeatureNormalizationResult:
    """Build normalized feature views for stock-panel LSTM training.

    The transform keeps existing feature engineering intact, then derives:
    - rolling per-stock z-scores for technical stock-level features;
    - cross-sectional z-scores/ranks within each market and date;
    - rolling market-level z-scores for market context features;
    - cyclical calendar encodings for day_of_week.
    """
    if rolling_window <= 1:
        raise ValueError("rolling_window must be greater than 1.")
    if min_periods <= 1:
        raise ValueError("min_periods must be greater than 1.")
    if min_periods > rolling_window:
        raise ValueError("min_periods must be less than or equal to rolling_window.")

    missing = [column for column in (*feature_columns, "Date", "code") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature normalization: {missing}")

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])

    stock_group_columns = _stock_group_columns(work)
    market_group_columns = _market_group_columns(work)
    market_date_group_columns = _market_date_group_columns(work)
    output_columns: list[str] = []
    roll_z_columns: list[str] = []
    cross_sectional_z_columns: list[str] = []
    cross_sectional_rank_columns: list[str] = []
    market_roll_z_columns: list[str] = []
    calendar_columns: list[str] = []
    passthrough_columns: list[str] = []

    for feature in feature_columns:
        values = work[feature].astype(float)
        if feature in DEFAULT_CALENDAR_FEATURES:
            radians = 2.0 * np.pi * values / 5.0
            sin_column = f"{feature}_sin"
            cos_column = f"{feature}_cos"
            work[sin_column] = np.sin(radians)
            work[cos_column] = np.cos(radians)
            calendar_columns.extend([sin_column, cos_column])
            output_columns.extend([sin_column, cos_column])
            continue

        if feature in DEFAULT_MARKET_CONTEXT_FEATURES:
            normalized_column = f"{feature}_market_roll_z"
            market_daily = work.loc[:, [*market_date_group_columns, feature]].drop_duplicates(
                subset=market_date_group_columns,
                keep="last",
            )
            if market_group_columns:
                market_daily[normalized_column] = market_daily.groupby(market_group_columns, sort=False)[
                    feature
                ].transform(
                    lambda series: _rolling_zscore(
                        series.astype(float),
                        window=rolling_window,
                        min_periods=min_periods,
                        epsilon=epsilon,
                        strict_past=strict_past,
                    )
                )
            else:
                market_daily[normalized_column] = _rolling_zscore(
                    market_daily[feature].astype(float),
                    window=rolling_window,
                    min_periods=min_periods,
                    epsilon=epsilon,
                    strict_past=strict_past,
                )
            work = work.merge(
                market_daily[[*market_date_group_columns, normalized_column]],
                on=market_date_group_columns,
                how="left",
            )
            work[normalized_column] = _fill_numeric_neutral(work[normalized_column], 0.0)
            market_roll_z_columns.append(normalized_column)
            output_columns.append(normalized_column)
            continue

        if _is_passthrough_feature(feature):
            work[feature] = _fill_numeric_neutral(work[feature].astype(float), _neutral_passthrough_value(feature))
            passthrough_columns.append(feature)
            output_columns.append(feature)
            continue

        roll_column = f"{feature}_roll_z"
        work[roll_column] = work.groupby(stock_group_columns, sort=False)[feature].transform(
            lambda series: _rolling_zscore(
                series.astype(float),
                window=rolling_window,
                min_periods=min_periods,
                epsilon=epsilon,
                strict_past=strict_past,
            )
        )
        work[roll_column] = _fill_numeric_neutral(work[roll_column], 0.0)
        roll_z_columns.append(roll_column)
        output_columns.append(roll_column)

        if include_cross_sectional_z:
            cs_z_column = f"{feature}_cs_z"
            cs_mean = work.groupby(market_date_group_columns, sort=False)[feature].transform("mean")
            cs_std = work.groupby(market_date_group_columns, sort=False)[feature].transform("std")
            valid_std = cs_std.notna() & (cs_std.abs() > epsilon)
            work[cs_z_column] = 0.0
            work.loc[valid_std, cs_z_column] = (
                values.loc[valid_std] - cs_mean.loc[valid_std]
            ) / (cs_std.loc[valid_std] + epsilon)
            work[cs_z_column] = _fill_numeric_neutral(work[cs_z_column], 0.0)
            cross_sectional_z_columns.append(cs_z_column)
            output_columns.append(cs_z_column)

        if include_cross_sectional_rank:
            rank_column = f"{feature}_cs_rank"
            group = work.groupby(market_date_group_columns, sort=False)[feature]
            count = group.transform("count")
            rank = group.rank(method="average", ascending=True)
            work[rank_column] = 0.5
            valid_count = count > 1
            work.loc[valid_count, rank_column] = (
                (rank.loc[valid_count] - 1.0) / (count.loc[valid_count] - 1.0)
            )
            work[rank_column] = _fill_numeric_neutral(work[rank_column], 0.5)
            cross_sectional_rank_columns.append(rank_column)
            output_columns.append(rank_column)

    normalized_columns = _safe_unique(output_columns)
    metadata: dict[str, object] = {
        "mode": "multimarket_v1",
        "base_feature_columns": list(feature_columns),
        "feature_columns": list(normalized_columns),
        "rolling_window": int(rolling_window),
        "min_periods": int(min_periods),
        "strict_past": bool(strict_past),
        "include_cross_sectional_z": bool(include_cross_sectional_z),
        "include_cross_sectional_rank": bool(include_cross_sectional_rank),
        "roll_z_columns": roll_z_columns,
        "cross_sectional_z_columns": cross_sectional_z_columns,
        "cross_sectional_rank_columns": cross_sectional_rank_columns,
        "market_roll_z_columns": market_roll_z_columns,
        "calendar_columns": calendar_columns,
        "passthrough_columns": passthrough_columns,
        "market_group_columns": market_group_columns,
    }
    return FeatureNormalizationResult(frame=work, feature_columns=normalized_columns, metadata=metadata)
