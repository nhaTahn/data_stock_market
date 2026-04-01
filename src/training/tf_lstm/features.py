from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = ["adjust", "volume_match"]
FEATURE_GROUP_ORDER = ["base", "momentum", "liquidity", "volatility", "trend"]

FEATURE_GROUP_MAP = {
    "base": ["adjust", "volume_match"],
    "momentum": ["ret_1", "ret_3", "ret_5", "momentum_5", "momentum_10"],
    "liquidity": ["log_volume", "volume_ratio_5", "volume_ratio_20", "volume_change_1"],
    "volatility": ["volatility_5", "volatility_10", "high_low_range"],
    "trend": ["ma_gap_5", "ma_gap_10", "ma_gap_20"],
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, np.nan)
    return numerator / safe_denominator


def normalize_feature_groups(feature_groups: list[str] | tuple[str, ...] | None) -> list[str]:
    requested = list(feature_groups) if feature_groups else ["base"]
    if "base" not in requested:
        requested = ["base", *requested]

    normalized: list[str] = []
    for group in FEATURE_GROUP_ORDER:
        if group in requested:
            normalized.append(group)

    invalid_groups = sorted(set(requested) - set(FEATURE_GROUP_MAP))
    if invalid_groups:
        raise ValueError(f"Unknown feature groups: {invalid_groups}")
    return normalized


def get_feature_columns(feature_groups: list[str] | tuple[str, ...] | None) -> list[str]:
    groups = normalize_feature_groups(feature_groups)
    columns: list[str] = []
    for group in groups:
        columns.extend(FEATURE_GROUP_MAP[group])
    return columns


def engineer_feature_groups(df: pd.DataFrame, feature_groups: list[str] | tuple[str, ...] | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    groups = normalize_feature_groups(feature_groups)
    out = df.copy().sort_values(["code", "Date"]).reset_index(drop=True)

    grouped = out.groupby("code", group_keys=False)
    adjust = grouped["adjust"]
    volume = grouped["volume_match"]

    out["ret_1"] = adjust.pct_change()
    out["ret_3"] = adjust.pct_change(3)
    out["ret_5"] = adjust.pct_change(5)

    out["momentum_5"] = grouped["adjust"].transform(lambda s: s / s.shift(5) - 1)
    out["momentum_10"] = grouped["adjust"].transform(lambda s: s / s.shift(10) - 1)

    out["log_volume"] = np.log1p(out["volume_match"].clip(lower=0))
    out["volume_ratio_5"] = grouped["volume_match"].transform(
        lambda s: _safe_ratio(s, s.rolling(window=5, min_periods=5).mean())
    )
    out["volume_ratio_20"] = grouped["volume_match"].transform(
        lambda s: _safe_ratio(s, s.rolling(window=20, min_periods=20).mean())
    )
    out["volume_change_1"] = grouped["volume_match"].transform(
        lambda s: _safe_ratio(s, s.shift(1)) - 1
    )

    grouped_after_returns = out.groupby("code", group_keys=False)
    out["volatility_5"] = grouped_after_returns["ret_1"].transform(lambda s: s.rolling(window=5, min_periods=5).std())
    out["volatility_10"] = grouped_after_returns["ret_1"].transform(lambda s: s.rolling(window=10, min_periods=10).std())
    if {"high", "low", "close"}.issubset(out.columns):
        out["high_low_range"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    else:
        out["high_low_range"] = np.nan

    out["ma_gap_5"] = grouped["adjust"].transform(
        lambda s: s / s.rolling(window=5, min_periods=5).mean() - 1
    )
    out["ma_gap_10"] = grouped["adjust"].transform(
        lambda s: s / s.rolling(window=10, min_periods=10).mean() - 1
    )
    out["ma_gap_20"] = grouped["adjust"].transform(
        lambda s: s / s.rolling(window=20, min_periods=20).mean() - 1
    )

    feature_columns = get_feature_columns(groups)
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    return out, feature_columns, groups
