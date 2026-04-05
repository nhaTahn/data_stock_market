from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline.market_config import CleanConfig, ROOT
from src.models.config import FEATURE_COLUMNS
from src.utils.features import ensure_columns


BASE_PRICE_COLS = [
    "Date",
    "code",
    "exchange",
    "open",
    "high",
    "low",
    "close",
    "adjust",
]

BASE_REVIEW_COLS = [
    "ma_5",
    "ma_20",
    "volume_match",
    "volume_ma_5",
    "volume_ma_20",
]

BASE_TARGET_COLS = [
    "target_next_price",
    "target_next_growth_pct",
    "target_next_return",
    "target_next_3d_return",
    "target_next_5d_return",
]

BASE_KEEP_COLS = BASE_PRICE_COLS + BASE_REVIEW_COLS + list(FEATURE_COLUMNS) + BASE_TARGET_COLS


def load_market_data(data_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["code", "Date"]).reset_index(drop=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "adjust", "volume_match", "value_match"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "value_match_imputed" not in df.columns:
        df["value_match_imputed"] = 0
    df["value_match_imputed"] = pd.to_numeric(df["value_match_imputed"], errors="coerce").fillna(0).astype(int)

    required_cols = ["Date", "code", "open", "high", "low", "close", "adjust", "volume_match"]
    df["missing_required"] = df[required_cols].isna().any(axis=1)
    df["duplicate_code_date"] = df.duplicated(subset=["code", "Date"], keep=False)
    df["negative_price"] = (df[["open", "high", "low", "close", "adjust"]] <= 0).any(axis=1)
    df["negative_volume"] = df["volume_match"] < 0
    df["ohlc_invalid"] = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    df["has_hard_issue"] = (
        df["missing_required"]
        | df["duplicate_code_date"]
        | df["negative_price"]
        | df["negative_volume"]
        | df["ohlc_invalid"]
    )

    df = ensure_columns(df)

    by_code = df.groupby("code", group_keys=False)
    next_adjust = by_code["adjust"].shift(-1)
    next_adjust_3d = by_code["adjust"].shift(-3)
    next_adjust_5d = by_code["adjust"].shift(-5)
    df["target_next_price"] = next_adjust
    df["target_next_growth_pct"] = (next_adjust / df["adjust"] - 1) * 100
    df["target_next_return"] = next_adjust / df["adjust"] - 1
    df["target_next_3d_return"] = next_adjust_3d / df["adjust"] - 1
    df["target_next_5d_return"] = next_adjust_5d / df["adjust"] - 1

    return df.replace([np.inf, -np.inf], np.nan)


def _get_event_mask(df: pd.DataFrame, config: CleanConfig) -> pd.Series:
    if config.exchange_limits and "exchange" in df.columns:
        default_c = config.max_close_return_abs or 0.15
        default_a = config.max_adjust_return_abs or 0.30

        def get_limit(ex, key, default):
            if pd.isna(ex): return default
            if not isinstance(config.exchange_limits, dict): return default
            return config.exchange_limits.get(ex, {}).get(key, default)

        limit_close = df["exchange"].apply(lambda ex: get_limit(ex, "max_close_return_abs", default_c))
        limit_adjust = df["exchange"].apply(lambda ex: get_limit(ex, "max_adjust_return_abs", default_a))

        return (
            (df["close_return"].abs() > limit_close)
            | (df["adjust_return"].abs() > limit_adjust)
        ).fillna(False)
    else:
        return (
            (df["close_return"].abs() > config.max_close_return_abs)
            | (df["adjust_return"].abs() > config.max_adjust_return_abs)
        ).fillna(False)


def summarize_tickers(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    recent = df[df["Date"] >= pd.Timestamp(config.train_start_date)].copy()
    total_days = recent["Date"].nunique()
    market_now = recent["Date"].max()
    recent["event_row"] = _get_event_mask(recent, config)

    ticker = (
        recent.groupby("code")
        .agg(
            stock_days=("Date", "nunique"),
            latest_date=("Date", "max"),
            hard_issue_rows=("has_hard_issue", "sum"),
            imputed_rows=("value_match_imputed", "sum"),
            event_rows=("event_row", "sum"),
        )
        .reset_index()
    )
    ticker["coverage_pct"] = ticker["stock_days"] / total_days
    ticker["days_since_latest"] = (market_now - ticker["latest_date"]).dt.days
    ticker["quality_score"] = (
        100
        - (1 - ticker["coverage_pct"]) * 60
        - ticker["hard_issue_rows"] * 10
        - ticker["imputed_rows"] * 0.05
        - ticker["event_rows"] * 2
    ).clip(lower=0)
    return ticker.sort_values(["quality_score", "coverage_pct", "code"], ascending=[False, False, True])


def build_clean_dataset(df: pd.DataFrame, ticker_summary: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    valid_tickers = ticker_summary[
        (ticker_summary["coverage_pct"] >= config.min_coverage)
        & (ticker_summary["days_since_latest"] <= config.recent_active_tolerance_days)
    ]["code"]

    clean = df[df["code"].isin(valid_tickers)].copy()
    clean["event_row"] = _get_event_mask(clean, config)

    if config.drop_neighbors_around_events:
        by_code = clean.groupby("code")["event_row"]
        clean["drop_event_buffer"] = clean["event_row"] | by_code.shift(1).eq(True) | by_code.shift(-1).eq(True)
    else:
        clean["drop_event_buffer"] = clean["event_row"]

    keep_mask = ~clean["has_hard_issue"] & ~clean["drop_event_buffer"]
    if config.drop_imputed_value_match:
        keep_mask &= clean["value_match_imputed"].eq(0)

    clean = clean[keep_mask].copy()
    clean = clean.dropna(subset=["target_next_return"])
    keep_cols = [col for col in BASE_KEEP_COLS if col in clean.columns]
    return clean[keep_cols].sort_values(["code", "Date"]).reset_index(drop=True)


def save_outputs(clean: pd.DataFrame, ticker_summary: pd.DataFrame, config: CleanConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            ("market", config.market),
            ("rows", len(clean)),
            ("tickers", clean["code"].nunique()),
            ("date_start", clean["Date"].min().date().isoformat()),
            ("date_end", clean["Date"].max().date().isoformat()),
            ("target_price_mean", clean["target_next_price"].mean()),
            ("target_growth_pct_mean", clean["target_next_growth_pct"].mean()),
            ("target_return_mean", clean["target_next_return"].mean()),
            ("target_return_std", clean["target_next_return"].std()),
            ("target_return_3d_mean", clean["target_next_3d_return"].mean()),
            ("target_return_3d_std", clean["target_next_3d_return"].std()),
            ("target_return_5d_mean", clean["target_next_5d_return"].mean()),
            ("target_return_5d_std", clean["target_next_5d_return"].std()),
        ],
        columns=["metric", "value"],
    )
    prefix = config.output_prefix
    clean.to_csv(config.output_dir / f"{prefix}_gold_recommended.csv", index=False)
    clean.to_csv(config.output_dir / f"{prefix}_quality_dataset.csv", index=False)
    ticker_summary.to_csv(config.output_dir / f"{prefix}_ticker_quality_summary.csv", index=False)
    summary.to_csv(config.output_dir / f"{prefix}_dataset_summary.csv", index=False)


def build_market_quality_dataset(config: CleanConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prepare_dataset(load_market_data(config.data_dir))
    
    # Load exchange mappings if they exist
    zinfo_dir = ROOT / "data" / "external" / "zInfo" / f"data_info_{config.market.lower()}"
    exchange_csv = zinfo_dir / "symbols_by_exchange.csv"
    
    if exchange_csv.exists():
        exchange_df = pd.read_csv(exchange_csv)
        if "symbol" in exchange_df.columns and "exchange" in exchange_df.columns:
            exchange_df["exchange"] = exchange_df["exchange"].str.upper()
            exchange_map = dict(zip(exchange_df["symbol"], exchange_df["exchange"]))
            
            # Map code to exchange if info exists in the map
            base["exchange"] = base["code"].map(exchange_map)

    ticker_summary = summarize_tickers(base, config)
    clean = build_clean_dataset(base, ticker_summary, config)
    save_outputs(clean, ticker_summary, config)
    return clean, ticker_summary
