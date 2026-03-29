from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VN_DATA_DIR = ROOT / "data" / "VN"
OUTPUT_DIR = ROOT / "data" / "assests" / "data_info_vn" / "history"

TRAIN_START_DATE = "2020-01-01"
RECENT_ACTIVE_TOLERANCE_DAYS = 30


@dataclass(frozen=True)
class CleanProfile:
    name: str
    min_coverage: float
    max_close_return_abs: float
    max_adjust_return_abs: float
    drop_imputed_value_match: bool
    drop_neighbors_around_events: bool
    recent_active_tolerance_days: int = RECENT_ACTIVE_TOLERANCE_DAYS


CLEAN_PROFILES = (
    CleanProfile(
        name="balanced",
        min_coverage=0.70,
        max_close_return_abs=0.15,
        max_adjust_return_abs=0.20,
        drop_imputed_value_match=False,
        drop_neighbors_around_events=False,
    ),
    CleanProfile(
        name="model_strict",
        min_coverage=0.90,
        max_close_return_abs=0.10,
        max_adjust_return_abs=0.16,
        drop_imputed_value_match=True,
        drop_neighbors_around_events=True,
    ),
    CleanProfile(
        name="trust_max",
        min_coverage=0.95,
        max_close_return_abs=0.075,
        max_adjust_return_abs=0.155,
        drop_imputed_value_match=True,
        drop_neighbors_around_events=True,
    ),
)


def load_vn_data(data_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["code", "Date"]).reset_index(drop=True)
    return combined


def compute_row_flags(df: pd.DataFrame) -> pd.DataFrame:
    row_flags = df.copy()
    row_flags["row_id"] = np.arange(len(row_flags))

    required_cols = ["Date", "code", "open", "high", "low", "close", "adjust", "volume_match"]
    row_flags["missing_required"] = row_flags[required_cols].isna().any(axis=1)
    row_flags["duplicate_code_date"] = row_flags.duplicated(subset=["code", "Date"], keep=False)

    numeric_price_cols = ["open", "high", "low", "close", "adjust"]
    for col in numeric_price_cols + ["volume_match"]:
        row_flags[col] = pd.to_numeric(row_flags[col], errors="coerce")

    row_flags["negative_price"] = (row_flags[numeric_price_cols] < 0).any(axis=1)
    row_flags["negative_volume"] = row_flags["volume_match"] < 0
    row_flags["ohlc_invalid"] = (
        (row_flags["high"] < row_flags[["open", "close", "low"]].max(axis=1))
        | (row_flags["low"] > row_flags[["open", "close", "high"]].min(axis=1))
        | (row_flags["high"] < row_flags["low"])
    )

    if "value_match_imputed" in row_flags.columns:
        imputed_series = row_flags["value_match_imputed"]
    else:
        imputed_series = pd.Series(0, index=row_flags.index)
    row_flags["value_match_imputed"] = pd.to_numeric(imputed_series, errors="coerce").fillna(0).astype(int)

    row_flags["close_return"] = row_flags.groupby("code")["close"].pct_change()
    row_flags["adjust_return"] = row_flags.groupby("code")["adjust"].pct_change()

    row_flags["close_bound_violation_7pct"] = row_flags["close_return"].abs() > 0.075
    row_flags["close_bound_violation_10pct"] = row_flags["close_return"].abs() > 0.10
    row_flags["close_bound_violation_15pct"] = row_flags["close_return"].abs() > 0.15
    row_flags["adjust_event_jump_155pct"] = row_flags["adjust_return"].abs() > 0.155
    row_flags["adjust_event_jump_20pct"] = row_flags["adjust_return"].abs() > 0.20

    row_flags["has_hard_issue"] = (
        row_flags["missing_required"]
        | row_flags["duplicate_code_date"]
        | row_flags["negative_price"]
        | row_flags["negative_volume"]
        | row_flags["ohlc_invalid"]
    )
    return row_flags


def compute_neighbor_event_flags(df: pd.DataFrame, event_mask: pd.Series) -> pd.Series:
    group = df.groupby("code")[event_mask.name]
    prev_event = group.shift(1).eq(True)
    next_event = group.shift(-1).eq(True)
    return event_mask | prev_event | next_event


def compute_ticker_quality(row_flags: pd.DataFrame, train_start: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(train_start)
    market_now = row_flags["Date"].max()

    recent = row_flags[row_flags["Date"] >= start_dt].copy()
    total_market_days = recent["Date"].nunique()

    ticker = (
        recent.groupby("code")
        .agg(
            stock_days=("Date", "nunique"),
            first_date=("Date", "min"),
            latest_date=("Date", "max"),
            hard_issue_rows=("has_hard_issue", "sum"),
            imputed_rows=("value_match_imputed", "sum"),
            close_violation_7pct_rows=("close_bound_violation_7pct", "sum"),
            close_violation_10pct_rows=("close_bound_violation_10pct", "sum"),
            close_violation_15pct_rows=("close_bound_violation_15pct", "sum"),
            adjust_jump_155pct_rows=("adjust_event_jump_155pct", "sum"),
            adjust_jump_20pct_rows=("adjust_event_jump_20pct", "sum"),
        )
        .reset_index()
    )

    ticker["coverage_pct"] = ticker["stock_days"] / total_market_days
    ticker["days_since_latest"] = (market_now - ticker["latest_date"]).dt.days
    ticker["is_recently_active"] = ticker["days_since_latest"] <= RECENT_ACTIVE_TOLERANCE_DAYS
    ticker["quality_score"] = (
        100
        - ticker["hard_issue_rows"] * 10
        - ticker["imputed_rows"] * 0.05
        - ticker["close_violation_7pct_rows"] * 1.5
        - ticker["adjust_jump_155pct_rows"] * 2.5
        - (1 - ticker["coverage_pct"]) * 50
    ).clip(lower=0)
    ticker["quality_band"] = pd.cut(
        ticker["quality_score"],
        bins=[-1, 60, 80, 90, 101],
        labels=["review", "usable", "good", "high_trust"],
    )
    return ticker.sort_values(["quality_score", "coverage_pct", "code"], ascending=[False, False, True])


def build_profile_dataset(
    row_flags: pd.DataFrame,
    ticker_quality: pd.DataFrame,
    profile: CleanProfile,
) -> pd.DataFrame:
    valid_tickers = ticker_quality[
        (ticker_quality["coverage_pct"] >= profile.min_coverage)
        & (ticker_quality["days_since_latest"] <= profile.recent_active_tolerance_days)
    ]["code"]

    df = row_flags[row_flags["code"].isin(valid_tickers)].copy()
    event_mask = (
        (df["close_return"].abs() > profile.max_close_return_abs)
        | (df["adjust_return"].abs() > profile.max_adjust_return_abs)
    ).fillna(False)
    event_mask.name = "profile_event"
    df["profile_event"] = event_mask

    if profile.drop_neighbors_around_events:
        df["drop_due_to_event_buffer"] = compute_neighbor_event_flags(df, event_mask)
    else:
        df["drop_due_to_event_buffer"] = df["profile_event"]

    keep_mask = ~df["has_hard_issue"]
    if profile.drop_imputed_value_match:
        keep_mask &= df["value_match_imputed"].eq(0)
    keep_mask &= ~df["drop_due_to_event_buffer"]

    clean = df[keep_mask].copy()
    clean["target_next_adjust_return"] = clean.groupby("code")["adjust"].shift(-1) / clean["adjust"] - 1
    clean["target_next_close_return"] = clean.groupby("code")["close"].shift(-1) / clean["close"] - 1
    clean["target_next_adjust_price"] = clean.groupby("code")["adjust"].shift(-1)
    clean["target_next_close_price"] = clean.groupby("code")["close"].shift(-1)
    clean["clean_profile"] = profile.name
    return clean


def build_overview_table(row_flags: pd.DataFrame, ticker_quality: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(row_flags)
    total_tickers = row_flags["code"].nunique()
    total_days = row_flags["Date"].nunique()
    market_start = row_flags["Date"].min().date().isoformat()
    market_end = row_flags["Date"].max().date().isoformat()

    metrics = [
        ("total_rows", total_rows),
        ("total_tickers", total_tickers),
        ("total_market_days", total_days),
        ("market_start", market_start),
        ("market_end", market_end),
        ("missing_required_rows", int(row_flags["missing_required"].sum())),
        ("duplicate_code_date_rows", int(row_flags["duplicate_code_date"].sum())),
        ("negative_price_rows", int(row_flags["negative_price"].sum())),
        ("negative_volume_rows", int(row_flags["negative_volume"].sum())),
        ("ohlc_invalid_rows", int(row_flags["ohlc_invalid"].sum())),
        ("value_match_imputed_rows", int(row_flags["value_match_imputed"].sum())),
        ("close_violation_7pct_rows", int(row_flags["close_bound_violation_7pct"].sum())),
        ("close_violation_10pct_rows", int(row_flags["close_bound_violation_10pct"].sum())),
        ("close_violation_15pct_rows", int(row_flags["close_bound_violation_15pct"].sum())),
        ("adjust_jump_155pct_rows", int(row_flags["adjust_event_jump_155pct"].sum())),
        ("adjust_jump_20pct_rows", int(row_flags["adjust_event_jump_20pct"].sum())),
        ("high_trust_tickers", int((ticker_quality["quality_band"] == "high_trust").sum())),
        ("good_or_better_tickers", int(ticker_quality["quality_band"].isin(["good", "high_trust"]).sum())),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_profile_summary(
    profile_data: Iterable[tuple[CleanProfile, pd.DataFrame]],
    baseline_rows: int,
    baseline_tickers: int,
) -> pd.DataFrame:
    rows = []
    for profile, df in profile_data:
        rows.append(
            {
                "profile": profile.name,
                "rows_kept": len(df),
                "tickers_kept": df["code"].nunique(),
                "date_start": df["Date"].min().date().isoformat() if not df.empty else None,
                "date_end": df["Date"].max().date().isoformat() if not df.empty else None,
                "row_retention_pct": round(len(df) / baseline_rows * 100, 2),
                "ticker_retention_pct": round(df["code"].nunique() / baseline_tickers * 100, 2),
                "rows_with_target": int(df["target_next_adjust_return"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_drop_reason_summary(
    row_flags: pd.DataFrame,
    ticker_quality: pd.DataFrame,
    profile: CleanProfile,
) -> pd.DataFrame:
    valid_tickers = ticker_quality[
        (ticker_quality["coverage_pct"] >= profile.min_coverage)
        & (ticker_quality["days_since_latest"] <= profile.recent_active_tolerance_days)
    ]["code"]

    df = row_flags[row_flags["code"].isin(valid_tickers)].copy()
    event_mask = (
        (df["close_return"].abs() > profile.max_close_return_abs)
        | (df["adjust_return"].abs() > profile.max_adjust_return_abs)
    ).fillna(False)
    event_mask.name = "profile_event"
    df["profile_event"] = event_mask

    if profile.drop_neighbors_around_events:
        event_buffer_mask = compute_neighbor_event_flags(df, event_mask)
    else:
        event_buffer_mask = event_mask

    reasons = [
        ("hard_issue_rows", int(df["has_hard_issue"].sum())),
        ("imputed_value_match_rows", int(df["value_match_imputed"].eq(1).sum()) if profile.drop_imputed_value_match else 0),
        ("event_buffer_rows", int(event_buffer_mask.sum())),
        ("rows_before_filter", int(len(df))),
        (
            "rows_after_filter",
            int(
                (
                    ~df["has_hard_issue"]
                    & (~df["value_match_imputed"].eq(1) if profile.drop_imputed_value_match else True)
                    & ~event_buffer_mask
                ).sum()
            ),
        ),
    ]
    return pd.DataFrame(reasons, columns=["metric", "value"])


def write_report(
    output_dir: Path,
    overview: pd.DataFrame,
    profile_summary: pd.DataFrame,
    ticker_quality: pd.DataFrame,
    recommended_profile: str,
) -> None:
    recommended_path = output_dir / "vn_gold_recommended.csv"
    recommended_df = pd.read_csv(recommended_path)
    recommended_codes = set(recommended_df["code"].unique())
    removed = ticker_quality[~ticker_quality["code"].isin(recommended_codes)].copy()
    removed["remove_reason"] = ""
    coverage_threshold = next(profile.min_coverage for profile in CLEAN_PROFILES if profile.name == recommended_profile)
    removed.loc[removed["coverage_pct"] < coverage_threshold, "remove_reason"] += f"coverage<{coverage_threshold:.0%};"
    removed.loc[removed["days_since_latest"] > RECENT_ACTIVE_TOLERANCE_DAYS, "remove_reason"] += "not_recent;"
    removed.loc[removed["quality_band"] == "review", "remove_reason"] += "quality_band=review;"

    low_quality = ticker_quality.sort_values(["quality_score", "coverage_pct"]).head(10)
    report_lines = [
        "VN Data Quality Report",
        "",
        f"Recommended training dataset: vn_gold_{recommended_profile}.csv",
        "",
        "Input overview:",
        overview.to_string(index=False),
        "",
        "Clean profile summary:",
        profile_summary.to_string(index=False),
        "",
        "Lowest quality tickers to inspect first:",
        low_quality[
            [
                "code",
                "coverage_pct",
                "days_since_latest",
                "hard_issue_rows",
                "imputed_rows",
                "close_violation_7pct_rows",
                "adjust_jump_155pct_rows",
                "quality_score",
                "quality_band",
            ]
        ].to_string(index=False),
        "",
        "Tickers removed from the recommended dataset:",
        removed[
            [
                "code",
                "remove_reason",
                "coverage_pct",
                "days_since_latest",
                "hard_issue_rows",
                "imputed_rows",
                "close_violation_7pct_rows",
                "adjust_jump_155pct_rows",
                "quality_score",
                "quality_band",
            ]
        ]
        .sort_values(["coverage_pct", "quality_score"])
        .to_string(index=False),
        "",
        "How to use:",
        f"1. Review vn_input_quality_overview.csv and vn_ticker_quality_summary.csv in {output_dir}",
        f"2. Train baseline models on vn_gold_{recommended_profile}.csv",
        "3. Only fall back to model_strict or balanced if you need more rows for experiments",
    ]
    (output_dir / "vn_quality_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def write_outputs(
    row_flags: pd.DataFrame,
    ticker_quality: pd.DataFrame,
    overview: pd.DataFrame,
    profile_summary: pd.DataFrame,
    profile_data: Iterable[tuple[CleanProfile, pd.DataFrame]],
    output_dir: Path,
    recommended_profile: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    overview.to_csv(output_dir / "vn_input_quality_overview.csv", index=False)
    ticker_quality.to_csv(output_dir / "vn_ticker_quality_summary.csv", index=False)
    row_flags.to_csv(output_dir / "vn_row_quality_flags.csv", index=False)
    profile_summary.to_csv(output_dir / "vn_clean_profile_summary.csv", index=False)

    for profile, df in profile_data:
        df.to_csv(output_dir / f"vn_gold_{profile.name}.csv", index=False)
        build_drop_reason_summary(row_flags, ticker_quality, profile).to_csv(
            output_dir / f"vn_drop_reason_summary_{profile.name}.csv",
            index=False,
        )

    recommended_src = output_dir / f"vn_gold_{recommended_profile}.csv"
    recommended_dst = output_dir / "vn_gold_recommended.csv"
    shutil.copyfile(recommended_src, recommended_dst)
    write_report(output_dir, overview, profile_summary, ticker_quality, recommended_profile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trust-first VN quality datasets for local training.")
    parser.add_argument("--data-dir", type=Path, default=VN_DATA_DIR, help="Directory containing VN CSV files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to write quality outputs.")
    parser.add_argument(
        "--recommended-profile",
        choices=[profile.name for profile in CLEAN_PROFILES],
        default="trust_max",
        help="Profile copied to vn_gold_recommended.csv and highlighted in the report.",
    )
    parser.add_argument(
        "--train-start-date",
        default=TRAIN_START_DATE,
        help="Date used to compute coverage and recency checks. Format: YYYY-MM-DD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    row_flags = compute_row_flags(load_vn_data(args.data_dir))
    ticker_quality = compute_ticker_quality(row_flags, train_start=args.train_start_date)
    overview = build_overview_table(row_flags, ticker_quality)

    profile_data = []
    for profile in CLEAN_PROFILES:
        clean = build_profile_dataset(row_flags, ticker_quality, profile)
        profile_data.append((profile, clean))

    profile_summary = build_profile_summary(
        profile_data,
        baseline_rows=len(row_flags),
        baseline_tickers=row_flags["code"].nunique(),
    )
    write_outputs(
        row_flags,
        ticker_quality,
        overview,
        profile_summary,
        profile_data,
        args.output_dir,
        args.recommended_profile,
    )

    print("Saved quality outputs to:", args.output_dir)
    print("Recommended training dataset:", args.output_dir / "vn_gold_recommended.csv")
    print("Quality report:", args.output_dir / "vn_quality_report.txt")
    print("\nInput overview:")
    print(overview.to_string(index=False))
    print("\nClean profiles:")
    print(profile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
