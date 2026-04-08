from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.visualization.model_plots import save_equity_curve_plot
from src.models.report_layout import report_backtest_path, resolve_run_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-strategy portfolio backtest from predictions.csv.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--models", default=None, help="Comma-separated model names. Defaults to all models in predictions.csv.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--entry-threshold", type=float, default=0.0)
    parser.add_argument("--holding-period", type=int, default=None)
    parser.add_argument("--rebalance-step", type=int, default=None)
    return parser.parse_args()


def infer_holding_period(run_dir: Path) -> int:
    config_path = resolve_run_artifact(run_dir, "config.json", bucket="core")
    if not config_path.exists():
        return 1
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    target_mode = config.get("target_mode", "return")
    if target_mode == "return_3d":
        return 3
    if target_mode == "return_5d":
        return 5
    return 1


def summarize_curve(strategy_df: pd.DataFrame) -> dict[str, float | int | str]:
    if strategy_df.empty:
        return {
            "trade_count": 0,
            "active_periods": 0,
            "avg_strategy_return": 0.0,
            "cumulative_strategy_return": 0.0,
            "final_equity": 1.0,
            "directional_accuracy": np.nan,
        }
    active = strategy_df[strategy_df["is_trade"] == 1]
    return {
        "trade_count": int(strategy_df["position_count"].sum()),
        "active_periods": int(strategy_df["is_trade"].sum()),
        "avg_strategy_return": float(strategy_df["strategy_return"].mean()),
        "cumulative_strategy_return": float(strategy_df["strategy_return"].sum()),
        "final_equity": float(strategy_df["equity"].iloc[-1]),
        "directional_accuracy": float((active["strategy_return"] >= 0.0).mean()) if not active.empty else np.nan,
    }


def build_equal_weight(date_df: pd.DataFrame) -> tuple[float, int]:
    if date_df.empty:
        return 0.0, 0
    return float(date_df["actual"].mean()), int(len(date_df))


def build_long_only(date_df: pd.DataFrame, top_k: int, entry_threshold: float) -> tuple[float, int]:
    eligible = date_df[date_df["prediction"] >= entry_threshold].nlargest(top_k, "prediction")
    if eligible.empty:
        return 0.0, 0
    return float(eligible["actual"].mean()), int(len(eligible))


def build_long_short(date_df: pd.DataFrame, top_k: int, entry_threshold: float) -> tuple[float, int]:
    longs = date_df[date_df["prediction"] >= entry_threshold].nlargest(top_k, "prediction")
    shorts = date_df[date_df["prediction"] <= -entry_threshold].nsmallest(top_k, "prediction")
    if longs.empty and shorts.empty:
        return 0.0, 0

    parts: list[float] = []
    if not longs.empty:
        parts.append(float(longs["actual"].mean()))
    if not shorts.empty:
        parts.append(float((-shorts["actual"]).mean()))
    return float(np.mean(parts)), int(len(longs) + len(shorts))


def build_best_bet(date_df: pd.DataFrame, entry_threshold: float) -> tuple[float, int]:
    if date_df.empty:
        return 0.0, 0
    best_row = date_df.iloc[date_df["prediction"].abs().argmax()]
    if abs(float(best_row["prediction"])) < entry_threshold:
        return 0.0, 0
    trade_return = float(best_row["actual"]) if float(best_row["prediction"]) >= 0.0 else float(-best_row["actual"])
    return trade_return, 1


def build_strategy_curve(
    model_df: pd.DataFrame,
    strategy_name: str,
    top_k: int,
    entry_threshold: float,
    rebalance_step: int,
) -> pd.DataFrame:
    unique_dates = sorted(model_df["Date"].drop_duplicates())
    selected_dates = unique_dates[:: max(1, rebalance_step)]
    rows: list[dict[str, object]] = []

    strategy_builders = {
        "EqualWeight": lambda df: build_equal_weight(df),
        "LongOnly": lambda df: build_long_only(df, top_k=top_k, entry_threshold=entry_threshold),
        "LongShort": lambda df: build_long_short(df, top_k=top_k, entry_threshold=entry_threshold),
        "BestBet": lambda df: build_best_bet(df, entry_threshold=entry_threshold),
    }
    strategy_fn = strategy_builders[strategy_name]

    for current_date in selected_dates:
        date_df = model_df[model_df["Date"] == current_date].copy()
        strategy_return, position_count = strategy_fn(date_df)
        rows.append(
            {
                "Date": current_date,
                "strategy": strategy_name,
                "strategy_return": strategy_return,
                "position_count": position_count,
                "is_trade": int(position_count > 0),
            }
        )

    curve_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    curve_df["equity"] = (1.0 + curve_df["strategy_return"]).cumprod()
    return curve_df


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    predictions = pd.read_csv(resolve_run_artifact(run_dir, "predictions.csv", bucket="core"))
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    holding_period = args.holding_period or infer_holding_period(run_dir)
    rebalance_step = args.rebalance_step or holding_period

    model_names = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else sorted(predictions["model"].unique())
    )

    summary_rows: list[dict[str, object]] = []
    for model_name in model_names:
        model_df = predictions[(predictions["model"] == model_name) & (predictions["split"] == "test")].copy()
        if model_df.empty:
            continue

        curve_frames: list[pd.DataFrame] = []
        for strategy_name in ("LongOnly", "LongShort", "BestBet", "EqualWeight"):
            curve_df = build_strategy_curve(
                model_df=model_df,
                strategy_name=strategy_name,
                top_k=args.top_k,
                entry_threshold=args.entry_threshold,
                rebalance_step=rebalance_step,
            )
            curve_df["label"] = strategy_name
            curve_frames.append(curve_df[["Date", "label", "equity", "strategy_return", "position_count", "is_trade"]])
            summary = summarize_curve(curve_df)
            summary_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy_name,
                    "holding_period": holding_period,
                    "rebalance_step": rebalance_step,
                    "top_k": args.top_k,
                    "entry_threshold": args.entry_threshold,
                    **summary,
                }
            )

        combined_curve_df = pd.concat(curve_frames, ignore_index=True)
        combined_curve_df.to_csv(report_backtest_path(run_dir, f"strategy_backtest_curves_{model_name}.csv"), index=False)
        save_equity_curve_plot(
            combined_curve_df[["Date", "label", "equity"]],
            report_backtest_path(run_dir, f"strategy_equity_{model_name}.png"),
            f"Strategy Equity Curves - {model_name}",
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["model", "final_equity", "avg_strategy_return"],
        ascending=[True, False, False],
    )
    summary_path = report_backtest_path(run_dir, "strategy_backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(summary_path)


if __name__ == "__main__":
    main()
