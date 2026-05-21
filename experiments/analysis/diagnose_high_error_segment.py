from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512"
DEFAULT_DAILY_ERROR = (
    DEFAULT_GOLD_DIR
    / "plots"
    / "teacher_style_abs_error_vn100_insample"
    / "teacher_style_abs_error.csv"
)
DEFAULT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "filter_signal"
    / "portable_lstm_filter_signal_20260512_r02_no_leader_seed43"
    / "filter_predictions.csv.gz"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN30_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn30_symbols.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"


FEATURE_COLUMNS = [
    "sector",
    "exchange",
    "close",
    "adjust",
    "volume_match",
    "volume_ratio_20",
    "intraday_return",
    "gap_open",
    "close_position",
    "momentum_20",
    "volatility_20",
    "bb_width",
    "macd_hist",
    "sector_momentum_rank",
    "alpha_sector",
    "target_next_return",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose high q90 absolute prediction-error segment.")
    parser.add_argument("--daily-error", type=Path, default=DEFAULT_DAILY_ERROR)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="segment_2017_d200_250_vn100_high_error")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"])
    parser.add_argument("--universe", default="vn100", choices=["vn30", "vn100", "all"])
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--start-day", type=int, default=200)
    parser.add_argument("--end-day", type=int, default=250)
    parser.add_argument("--spike-threshold", type=float, default=0.035)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args(argv)


def read_symbol_file(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def read_universe(name: str) -> tuple[str, set[str] | None]:
    if name == "vn30":
        return "VN30", read_symbol_file(DEFAULT_VN30_SYMBOLS)
    if name == "vn100":
        return "VN100", read_symbol_file(DEFAULT_VN100_SYMBOLS)
    return "ALL", None


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(
            handle,
            usecols=["code", "split", "Date", "actual_date", "actual_aligned", "base_prediction"],
        )
    frame["code"] = frame["code"].astype(str).str.upper()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    frame["actual_return"] = frame["actual_aligned"].astype(float)
    frame["predicted_return"] = frame["base_prediction"].astype(float)
    frame["error"] = frame["actual_return"] - frame["predicted_return"]
    frame["abs_error"] = frame["error"].abs()
    frame["actual_abs_return"] = frame["actual_return"].abs()
    frame["pred_abs_return"] = frame["predicted_return"].abs()
    frame["sign_mismatch"] = np.sign(frame["actual_return"]) != np.sign(frame["predicted_return"])
    return frame


def read_feature_frame(path: Path) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = ["Date", "code"] + [column for column in FEATURE_COLUMNS if column in available]
    frame = pd.read_csv(path, usecols=usecols)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    return frame.rename(columns={column: f"feature_{column}" for column in usecols if column not in {"Date", "code"}})


def year_segment(daily: pd.DataFrame, year: int, start_day: int, end_day: int) -> pd.DataFrame:
    work = daily.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["year"] = work["Date"].dt.year
    work = work[work["year"].eq(year)].reset_index(drop=True)
    work["trading_day_in_year"] = np.arange(len(work))
    return work[
        work["trading_day_in_year"].between(start_day, end_day, inclusive="both")
    ].copy()


def summarize_prediction_distribution(segment_predictions: pd.DataFrame) -> pd.DataFrame:
    daily = (
        segment_predictions.groupby("actual_date", sort=True)
        .agg(
            n_predictions=("code", "nunique"),
            actual_mean=("actual_return", "mean"),
            actual_std=("actual_return", "std"),
            actual_abs_mean=("actual_abs_return", "mean"),
            actual_abs_q90=("actual_abs_return", lambda values: float(np.quantile(values, 0.90))),
            predicted_mean=("predicted_return", "mean"),
            predicted_std=("predicted_return", "std"),
            predicted_abs_mean=("pred_abs_return", "mean"),
            predicted_abs_q90=("pred_abs_return", lambda values: float(np.quantile(values, 0.90))),
            abs_error_mean=("abs_error", "mean"),
            abs_error_q90=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            sign_mismatch_share=("sign_mismatch", "mean"),
        )
        .reset_index()
        .rename(columns={"actual_date": "Date"})
    )
    daily["tail_shrinkage_ratio_q90"] = daily["predicted_abs_q90"] / daily["actual_abs_q90"].replace(0.0, np.nan)
    daily["tail_shrinkage_gap_q90"] = daily["actual_abs_q90"] - daily["predicted_abs_q90"]
    return daily


def top_errors_by_day(predictions: pd.DataFrame, spike_dates: pd.Series, top_n: int) -> pd.DataFrame:
    rows = predictions[predictions["actual_date"].isin(spike_dates)].copy()
    rows["rank_abs_error_in_day"] = rows.groupby("actual_date")["abs_error"].rank(
        method="first",
        ascending=False,
    )
    return (
        rows[rows["rank_abs_error_in_day"].le(top_n)]
        .sort_values(["actual_date", "rank_abs_error_in_day"], kind="stable")
        .reset_index(drop=True)
    )


def summarize_codes(spike_top_errors: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        spike_top_errors.groupby(["code", "feature_sector"], dropna=False)
        .agg(
            top_error_appearances=("actual_date", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
            mean_actual_return=("actual_return", "mean"),
            mean_predicted_return=("predicted_return", "mean"),
            sign_mismatch_share=("sign_mismatch", "mean"),
            mean_feature_momentum_20=("feature_momentum_20", "mean"),
            mean_feature_volatility_20=("feature_volatility_20", "mean"),
            mean_feature_volume_ratio_20=("feature_volume_ratio_20", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["top_error_appearances", "mean_abs_error"], ascending=[False, False], kind="stable")


def summarize_sectors(spike_top_errors: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        spike_top_errors.groupby("feature_sector", dropna=False)
        .agg(
            top_error_rows=("code", "count"),
            unique_codes=("code", "nunique"),
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            sign_mismatch_share=("sign_mismatch", "mean"),
            mean_actual_return=("actual_return", "mean"),
        )
        .reset_index()
        .rename(columns={"feature_sector": "sector"})
    )
    return grouped.sort_values(["top_error_rows", "mean_abs_error"], ascending=[False, False], kind="stable")


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def write_report(
    output_dir: Path,
    label: str,
    args: argparse.Namespace,
    segment_daily: pd.DataFrame,
    daily_distribution: pd.DataFrame,
    spike_daily: pd.DataFrame,
    code_summary: pd.DataFrame,
    sector_summary: pd.DataFrame,
) -> None:
    segment_start = segment_daily["Date"].min().date()
    segment_end = segment_daily["Date"].max().date()
    spike_dates = ", ".join(str(date.date()) for date in spike_daily["Date"].head(10))
    if len(spike_daily) > 10:
        spike_dates += ", ..."

    merged = segment_daily.merge(daily_distribution, on="Date", how="left")
    high_error_days = int(spike_daily.shape[0])
    total_days = int(segment_daily.shape[0])
    median_q90 = float(segment_daily["q90_abs_error"].median())
    spike_median_q90 = float(spike_daily["q90_abs_error"].median()) if high_error_days else float("nan")
    median_actual_tail = float(merged["actual_abs_q90"].median())
    median_pred_tail = float(merged["predicted_abs_q90"].median())
    median_shrink_ratio = float(merged["tail_shrinkage_ratio_q90"].median())
    sign_mismatch_spikes = float(
        merged.loc[merged["q90_abs_error"].ge(args.spike_threshold), "sign_mismatch_share"].median()
    )

    top_codes = code_summary.head(12)[
        [
            "code",
            "feature_sector",
            "top_error_appearances",
            "mean_abs_error",
            "max_abs_error",
            "sign_mismatch_share",
        ]
    ].copy()
    for column in ["mean_abs_error", "max_abs_error", "sign_mismatch_share"]:
        top_codes[column] = top_codes[column].map(format_pct)

    top_sectors = sector_summary.head(10).copy()
    for column in ["mean_abs_error", "max_abs_error", "sign_mismatch_share", "mean_actual_return"]:
        top_sectors[column] = top_sectors[column].map(format_pct)

    lines = [
        "# High-Error Segment Diagnosis",
        "",
        f"Scope: `{label}`, split `{args.split}`, year `{args.year}`, trading-day segment `{args.start_day}-{args.end_day}`.",
        f"Date range: `{segment_start}` to `{segment_end}`. Holdout/test is not used.",
        "",
        "## What Happened",
        "",
        f"- Segment days: `{total_days}`.",
        f"- Spike rule: `q90(|actual_return - predicted_return|) > {args.spike_threshold:.1%}`.",
        f"- Spike days: `{high_error_days}` / `{total_days}`.",
        f"- Segment median q90(|E|): `{format_pct(median_q90)}`.",
        f"- Spike-day median q90(|E|): `{format_pct(spike_median_q90)}`.",
        f"- Spike dates: {spike_dates}.",
        "",
        "## Main Diagnostic",
        "",
        f"- Median q90 actual absolute return in segment: `{format_pct(median_actual_tail)}`.",
        f"- Median q90 predicted absolute return in segment: `{format_pct(median_pred_tail)}`.",
        f"- Median predicted/actual q90 tail ratio: `{median_shrink_ratio:.3f}`.",
        f"- Median sign-mismatch share on spike days: `{format_pct(sign_mismatch_spikes)}`.",
        "",
        "The high q90 error is therefore not just one bad stock. On spike days, the cross-section has many stocks moving several percent while the base LSTM prediction remains much closer to zero. This is a shrinkage/tail-response problem in an uptrend regime with strong rotation.",
        "",
        "## Repeated High-Error Codes",
        "",
        top_codes.to_markdown(index=False),
        "",
        "## Sector Concentration",
        "",
        top_sectors.to_markdown(index=False),
        "",
        "## Hypotheses To Test Next",
        "",
        "1. **Target/objective shrinkage**: `rel_score` makes the LSTM learn a conservative conditional mean. It helps robust next-day error but under-reacts when the market enters broad uptrend rotation or single-name tail moves.",
        "2. **Feature processing gap**: current features may not encode cross-sectional dispersion, limit-like moves, leader breadth, and market/sector rotation strongly enough. The model sees the uptrend but does not know which stocks are becoming high-move candidates.",
        "3. **Feature selection noise**: portable features are useful, but a few VN-specific microstructure signals may be needed as filter features, not necessarily as base LSTM features.",
        "4. **Model optimization is secondary for now**: simply making the LSTM larger can reduce training error but may not fix tail timing. A small tail-risk/filter head is the cleaner next test.",
        "",
        "## Next Ablation",
        "",
        "Keep the base LSTM frozen, then train/evaluate a sidecar filter on in-sample/validation with new regime features:",
        "",
        "- daily cross-sectional return dispersion",
        "- q90 absolute market/universe return",
        "- market breadth and advance/decline",
        "- sector dispersion and top-sector concentration",
        "- leader/liquidity-weighted return from top traded-value stocks",
        "- limit-like move count or high absolute return count",
        "",
        "Success criterion: reduce `q90(|E|)` and high-error trade exposure in this 2017 uptrend segment and similar high-dispersion buckets, while not materially hurting full validation `rel_score`.",
        "",
        "## Files",
        "",
        "- `segment_daily.csv`",
        "- `segment_daily_distribution.csv`",
        "- `segment_spike_days.csv`",
        "- `segment_spike_top_errors.csv`",
        "- `segment_code_error_summary.csv`",
        "- `segment_sector_error_summary.csv`",
    ]
    (output_dir / "hypothesis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    label, symbols = read_universe(args.universe)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_error = pd.read_csv(args.daily_error)
    segment_daily = year_segment(daily_error, args.year, args.start_day, args.end_day)
    segment_daily["is_spike"] = segment_daily["q90_abs_error"].ge(args.spike_threshold)
    spike_daily = segment_daily[segment_daily["is_spike"]].copy()

    predictions = read_predictions(args.predictions)
    if args.split != "all":
        predictions = predictions[predictions["split"].eq(args.split)].copy()
    if symbols is not None:
        predictions = predictions[predictions["code"].isin(symbols)].copy()
    predictions = predictions[predictions["actual_date"].isin(segment_daily["Date"])].copy()

    features = read_feature_frame(args.data)
    predictions = predictions.merge(features, on=["Date", "code"], how="left")

    daily_distribution = summarize_prediction_distribution(predictions)
    spike_top_errors = top_errors_by_day(predictions, spike_daily["Date"], args.top_n)
    code_summary = summarize_codes(spike_top_errors)
    sector_summary = summarize_sectors(spike_top_errors)

    segment_daily.to_csv(output_dir / "segment_daily.csv", index=False)
    daily_distribution.to_csv(output_dir / "segment_daily_distribution.csv", index=False)
    spike_daily.to_csv(output_dir / "segment_spike_days.csv", index=False)
    spike_top_errors.to_csv(output_dir / "segment_spike_top_errors.csv", index=False)
    code_summary.to_csv(output_dir / "segment_code_error_summary.csv", index=False)
    sector_summary.to_csv(output_dir / "segment_sector_error_summary.csv", index=False)
    write_report(output_dir, label, args, segment_daily, daily_distribution, spike_daily, code_summary, sector_summary)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "daily_error": str(args.daily_error),
                "predictions": str(args.predictions),
                "data": str(args.data),
                "split": args.split,
                "universe": args.universe,
                "year": args.year,
                "start_day": args.start_day,
                "end_day": args.end_day,
                "spike_threshold": args.spike_threshold,
                "top_n": args.top_n,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "segment_days": int(segment_daily.shape[0]),
                "spike_days": int(spike_daily.shape[0]),
                "top_error_rows": int(spike_top_errors.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
