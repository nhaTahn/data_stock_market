from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.features import ensure_paper_features


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
GLOBAL_HISTORY_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_global" / "history"
REPORT_ROOT = GLOBAL_HISTORY_ROOT / "training_runs" / "reports" / "portable_signal_search"
DEFAULT_SOURCE_RUN_DIR = RUN_ROOT / "broad_signmag_portable_no_identity_20260428_allvn_r01"
DEFAULT_DATA_PATH = GLOBAL_HISTORY_ROOT / "multimarket_vn_jp_us_portable_fullcore_20260428_r01.csv"
MARKETS = ("VN", "JP", "US")
RECOMPUTABLE_TECHNICAL_COLUMNS = (
    "close_prev",
    "adjust_prev",
    "close_return",
    "adjust_return",
    "intraday_return",
    "overnight_return",
    "gap_open",
    "range_pct",
    "body_pct",
    "high_close_gap",
    "close_low_gap",
    "close_position",
    "upper_shadow",
    "lower_shadow",
    "volume_ma_5",
    "volume_ma_20",
    "volume_change",
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_zscore_20",
    "return_3",
    "momentum_5",
    "momentum_20",
    "return_10",
    "price_acceleration",
    "rsi_14",
    "day_of_week",
    "true_range",
    "atr_14",
    "atr_gap",
    "volatility_5",
    "volatility_20",
    "volatility_10",
    "volatility_ratio",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_50",
    "ma_200",
    "ma_5_gap",
    "ma_10_gap",
    "ma_20_gap",
    "ma_50_gap",
    "ma_cross_5_20",
    "ma_200_gap",
    "above_ma_200",
    "ma_20_ma_200_gap",
    "rolling_max_20_gap",
    "rolling_min_20_gap",
    "bb_mid_20",
    "bb_std_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width",
    "bb_position",
    "bb_zscore",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume_level_20",
    "volume_delta_1",
)


@dataclass(frozen=True)
class SignalSummary:
    feature: str
    direction: int
    score: float
    train_mean_ic: float
    val_mean_ic: float
    min_market_val_ic: float
    positive_val_markets: int
    val_ic_t_stat: float
    val_positive_days: float
    val_days: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search raw portable feature signals by daily cross-market IC without touching holdout."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--output-name", default=f"portable_signal_search_{datetime.now().strftime('%Y%m%d_r%H%M')}")
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--min-names", type=int, default=15)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument(
        "--extra-features",
        default=(
            "close_return,adjust_return,intraday_return,gap_open,close_position,bb_width,"
            "volume_ratio_20,volatility_20,momentum_5,momentum_20,ma_5_gap,ma_20_gap,"
            "macd_hist,alpha_market,market_return,sector_return,alpha_sector,"
            "sector_momentum_rank,sector_momentum_rank_pct,relative_sector_momentum_20,"
            "sector_positive_ratio,sector_ad_ratio,day_of_week"
        ),
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def resolve_target_column(config: dict[str, object], requested: str | None) -> str:
    if requested:
        return requested
    target_mode = str(config.get("target_mode", "return"))
    if target_mode == "return":
        return "target_next_return"
    if target_mode == "growth_pct":
        return "target_next_growth_pct"
    return str(config.get("target_column", "target_next_return"))


def add_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["market"] if "market" in out.columns else []
    date_group = [*group_cols, "Date"]
    if "adjust_return" in out.columns:
        daily = out.groupby(date_group, as_index=False)["adjust_return"].mean().rename(
            columns={"adjust_return": "market_return_raw"}
        )
        shift_group = group_cols if group_cols else None
        if shift_group:
            grouped = daily.groupby(shift_group, group_keys=False)["market_return_raw"]
        else:
            grouped = daily["market_return_raw"]
        daily["market_return_5"] = grouped.transform(lambda s: s.rolling(5, min_periods=3).mean())
        daily["market_return_20"] = grouped.transform(lambda s: s.rolling(20, min_periods=5).mean())
        daily["market_return_60"] = grouped.transform(lambda s: s.rolling(60, min_periods=20).mean())
        daily["market_volatility_20"] = grouped.transform(lambda s: s.rolling(20, min_periods=5).std())
        out = out.merge(daily.drop(columns=["market_return_raw"]), on=date_group, how="left")

        pos = (out["adjust_return"] > 0.0).astype(float)
        neg = (out["adjust_return"] < 0.0).astype(float)
        breadth = out.loc[:, date_group].copy()
        breadth["positive"] = pos
        breadth["negative"] = neg
        breadth = breadth.groupby(date_group, as_index=False).agg(positive=("positive", "sum"), negative=("negative", "sum"))
        breadth["a_d_ratio"] = breadth["positive"] / (breadth["negative"] + 1.0)
        breadth["breadth"] = breadth["positive"] / (breadth["positive"] + breadth["negative"]).replace(0.0, np.nan)
        if shift_group:
            breadth_grouped = breadth.groupby(shift_group, group_keys=False)
            breadth["market_ad_ratio_20"] = breadth_grouped["a_d_ratio"].transform(
                lambda s: s.rolling(20, min_periods=5).mean()
            )
            breadth["breadth_20"] = breadth_grouped["breadth"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        else:
            breadth["market_ad_ratio_20"] = breadth["a_d_ratio"].rolling(20, min_periods=5).mean()
            breadth["breadth_20"] = breadth["breadth"].rolling(20, min_periods=5).mean()
        out = out.merge(
            breadth[[*date_group, "a_d_ratio", "market_ad_ratio_20", "breadth_20"]],
            on=date_group,
            how="left",
            suffixes=("", "_computed"),
        )
        for column in ("a_d_ratio", "market_ad_ratio_20", "breadth_20"):
            computed = f"{column}_computed"
            if computed in out.columns:
                out[column] = out[column].fillna(out[computed]) if column in out.columns else out[computed]
                out = out.drop(columns=[computed])
    return out


def prepare_frame(data_path: Path, config: dict[str, object], target_column: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    if "market" not in df.columns:
        df["market"] = "VN"
    df = df.sort_values(["market", "code", "Date"], kind="stable").reset_index(drop=True)
    recomputable = [column for column in RECOMPUTABLE_TECHNICAL_COLUMNS if column in df.columns]
    if recomputable and {"open", "high", "low", "close", "adjust", "volume_match"}.issubset(df.columns):
        df = df.drop(columns=recomputable)
    if config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)
    else:
        from src.utils.features import ensure_columns

        df = ensure_columns(df)
    df = add_market_context_features(df)
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    return df.replace([np.inf, -np.inf], np.nan)


def split_frame(df: pd.DataFrame, train_end_date: str, val_end_date: str) -> pd.DataFrame:
    train_end = pd.Timestamp(train_end_date)
    val_end = pd.Timestamp(val_end_date)
    out = df.loc[df["Date"] <= val_end].copy()
    out["split"] = np.where(out["Date"] <= train_end, "train", "val")
    return out


def daily_ic_rows(df: pd.DataFrame, feature: str, target_column: str, min_names: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    needed = ["market", "split", "Date", feature, target_column]
    work = df.loc[:, needed].dropna()
    if work.empty:
        return rows
    for (market, split, date), group in work.groupby(["market", "split", "Date"], sort=True):
        if len(group) < min_names:
            continue
        if group[feature].nunique(dropna=True) < 3 or group[target_column].nunique(dropna=True) < 3:
            continue
        ic = group[feature].corr(group[target_column], method="spearman")
        if pd.isna(ic):
            continue
        rows.append(
            {
                "feature": feature,
                "market": str(market),
                "split": str(split),
                "Date": date,
                "names": int(len(group)),
                "ic": float(ic),
            }
        )
    return rows


def summarize_ic(values: pd.Series) -> tuple[float, float, float, int]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")
    t_stat = float(np.mean(clean) / (std / np.sqrt(len(clean)))) if std and np.isfinite(std) and std > 0 else float("nan")
    return float(np.mean(clean)), t_stat, float(np.mean(clean > 0.0)), int(len(clean))


def build_summary(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_market_rows: list[dict[str, object]] = []
    for (feature, split, market), group in daily.groupby(["feature", "split", "market"], sort=True):
        mean_ic, t_stat, positive_days, days = summarize_ic(group["ic"])
        by_market_rows.append(
            {
                "feature": feature,
                "split": split,
                "market": market,
                "mean_ic": mean_ic,
                "ic_t_stat": t_stat,
                "positive_days": positive_days,
                "days": days,
                "avg_names": float(group["names"].mean()),
            }
        )
    by_market = pd.DataFrame(by_market_rows)

    candidates: list[SignalSummary] = []
    for feature, group in daily.groupby("feature", sort=True):
        train_ic = group.loc[group["split"] == "train", "ic"]
        train_mean, _, _, _ = summarize_ic(train_ic)
        if pd.isna(train_mean):
            continue
        direction = 1 if train_mean >= 0.0 else -1
        signed = group.copy()
        signed["signed_ic"] = signed["ic"] * direction
        train_signed_mean, _, _, _ = summarize_ic(signed.loc[signed["split"] == "train", "signed_ic"])
        val_signed = signed.loc[signed["split"] == "val"].copy()
        val_mean, val_t, val_pos, val_days = summarize_ic(val_signed["signed_ic"])
        if val_signed.empty or pd.isna(val_mean):
            continue
        market_val = (
            val_signed.groupby("market", sort=True)["signed_ic"]
            .mean()
            .reindex(MARKETS)
            .dropna()
        )
        min_market_val = float(market_val.min()) if not market_val.empty else float("nan")
        positive_val_markets = int((market_val > 0.0).sum())
        score = float(val_mean + 0.5 * min_market_val + 0.001 * min(val_t if np.isfinite(val_t) else 0.0, 10.0))
        candidates.append(
            SignalSummary(
                feature=feature,
                direction=direction,
                score=score,
                train_mean_ic=train_signed_mean,
                val_mean_ic=val_mean,
                min_market_val_ic=min_market_val,
                positive_val_markets=positive_val_markets,
                val_ic_t_stat=val_t,
                val_positive_days=val_pos,
                val_days=val_days,
            )
        )
    summary = pd.DataFrame([item.__dict__ for item in candidates])
    if not summary.empty:
        summary = summary.sort_values(
            ["positive_val_markets", "score", "val_mean_ic"],
            ascending=[False, False, False],
            kind="stable",
        )
    return summary, by_market


def write_markdown(output_dir: Path, summary: pd.DataFrame, by_market: pd.DataFrame, top_n: int) -> None:
    lines = [
        "# Portable Signal Search",
        "",
        "Signals are oriented using train-only mean daily Spearman IC. Validation columns are not holdout/test.",
        "",
        "## Top Signals",
        "",
        "| Rank | Feature | Direction | Val IC | Min market IC | Positive markets | t-stat | Positive days |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(summary.head(top_n).itertuples(index=False), start=1):
        lines.append(
            f"| {idx} | `{row.feature}` | `{int(row.direction):+d}` | {float(row.val_mean_ic):+.5f} | "
            f"{float(row.min_market_val_ic):+.5f} | {int(row.positive_val_markets)} | "
            f"{float(row.val_ic_t_stat):+.2f} | {float(row.val_positive_days):.1%} |"
        )
    if not summary.empty:
        best_features = list(summary.head(10)["feature"])
        lines.extend(["", "## By Market For Top 10", ""])
        for feature in best_features:
            lines.append(f"### `{feature}`")
            rows = by_market[(by_market["feature"] == feature) & (by_market["split"] == "val")]
            for market in MARKETS:
                part = rows[rows["market"] == market]
                if part.empty:
                    continue
                row = part.iloc[0]
                direction = int(summary.loc[summary["feature"] == feature, "direction"].iloc[0])
                lines.append(
                    f"- `{market}`: signed mean IC `{float(row['mean_ic']) * direction:+.5f}`, "
                    f"t-stat `{float(row['ic_t_stat']) * direction:+.2f}`, days `{int(row['days'])}`"
                )
            lines.append("")
    output_dir.joinpath("summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_config = load_json(args.source_run_dir / "reports" / "core" / "config.json")
    target_column = resolve_target_column(source_config, args.target_column)
    feature_columns = list(dict.fromkeys([*source_config.get("feature_columns", []), *parse_csv_list(args.extra_features)]))
    output_dir = REPORT_ROOT / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_frame(args.data_path, source_config, target_column)
    split_df = split_frame(
        df,
        train_end_date=str(args.train_end_date or source_config["train_end_date"]),
        val_end_date=str(args.val_end_date or source_config["val_end_date"]),
    )
    available_features = [feature for feature in feature_columns if feature in split_df.columns]
    daily_rows: list[dict[str, object]] = []
    for feature in available_features:
        if not pd.api.types.is_numeric_dtype(split_df[feature]):
            continue
        daily_rows.extend(daily_ic_rows(split_df, feature, target_column, args.min_names))
    daily = pd.DataFrame(daily_rows)
    summary, by_market = build_summary(daily)

    daily.to_csv(output_dir / "daily_ic.csv", index=False)
    by_market.to_csv(output_dir / "by_market_summary.csv", index=False)
    summary.to_csv(output_dir / "signal_summary.csv", index=False)
    manifest = {
        "data_path": str(args.data_path),
        "source_run_dir": str(args.source_run_dir),
        "target_column": target_column,
        "train_end_date": str(args.train_end_date or source_config["train_end_date"]),
        "val_end_date": str(args.val_end_date or source_config["val_end_date"]),
        "min_names": args.min_names,
        "available_features": available_features,
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(output_dir, summary, by_market, args.top_n)
    print(json.dumps({"output_dir": str(output_dir), "features": len(available_features), "daily_rows": len(daily)}, indent=2))


if __name__ == "__main__":
    main()
