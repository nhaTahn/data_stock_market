from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple threshold backtest from predictions.csv.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thresholds", default="0,0.0025,0.005,0.01,0.015,0.02")
    parser.add_argument("--non-overlap", action="store_true")
    parser.add_argument("--holding-period", type=int, default=None)
    return parser.parse_args()


def infer_holding_period(run_dir: Path) -> int:
    config_path = run_dir / "config.json"
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


def select_non_overlap_trades(model_df: pd.DataFrame, threshold: float, holding_period: int) -> pd.DataFrame:
    selected_indices: list[int] = []
    i = 0
    model_df = model_df.reset_index(drop=True)
    while i < len(model_df):
        if float(model_df.loc[i, "prediction"]) >= threshold:
            selected_indices.append(i)
            i += holding_period
            continue
        i += 1
    if not selected_indices:
        return model_df.iloc[0:0].copy()
    return model_df.iloc[selected_indices].copy()


def summarize_active_rows(
    active: pd.DataFrame,
    total_rows: int,
    model_name: str,
    threshold: float,
    holding_period: int,
) -> dict[str, float | str | int]:
    if active.empty:
        return {
            "model": model_name,
            "threshold": threshold,
            "holding_period": holding_period,
            "trade_count": 0,
            "coverage": 0.0,
            "directional_accuracy": np.nan,
            "avg_actual_return": np.nan,
            "avg_strategy_return": np.nan,
            "cumulative_strategy_return": 0.0,
            "final_equity": 1.0,
            "sampled_buy_hold_equity": 1.0,
        }
    strategy_return = active["actual"]
    sampled_buy_hold_equity = float((1.0 + active["actual"]).cumprod().iloc[-1])
    final_equity = float((1.0 + strategy_return).cumprod().iloc[-1])
    return {
        "model": model_name,
        "threshold": threshold,
        "holding_period": holding_period,
        "trade_count": int(len(active)),
        "coverage": float(len(active) / total_rows),
        "directional_accuracy": float(np.mean(active["actual"] >= 0.0)),
        "avg_actual_return": float(active["actual"].mean()),
        "avg_strategy_return": float(strategy_return.mean()),
        "cumulative_strategy_return": float(strategy_return.sum()),
        "final_equity": final_equity,
        "sampled_buy_hold_equity": sampled_buy_hold_equity,
    }


def compute_backtest_rows(
    df: pd.DataFrame,
    thresholds: list[float],
    non_overlap: bool,
    holding_period: int,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for model_name in sorted(df["model"].unique()):
        model_df = df[(df["model"] == model_name) & (df["split"] == "test")].sort_values("Date").copy()
        if model_df.empty:
            continue
        for threshold in thresholds:
            if non_overlap and holding_period > 1:
                active = select_non_overlap_trades(model_df, threshold, holding_period)
            else:
                active = model_df[model_df["prediction"] >= threshold].copy()
            rows.append(summarize_active_rows(active, len(model_df), model_name, threshold, holding_period))
    return rows


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    holding_period = args.holding_period or infer_holding_period(run_dir)
    predictions = pd.read_csv(run_dir / "predictions.csv")
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    result_df = pd.DataFrame(compute_backtest_rows(predictions, thresholds, args.non_overlap, holding_period))
    suffix = "_non_overlap" if args.non_overlap and holding_period > 1 else ""
    result_df.to_csv(run_dir / f"threshold_backtest{suffix}.csv", index=False)

    best_rows = (
        result_df.dropna(subset=["avg_strategy_return"])
        .sort_values(["final_equity", "avg_strategy_return", "directional_accuracy", "coverage"], ascending=[False, False, False, False])
        .groupby("model", as_index=False)
        .head(1)
    )
    summary = {
        row["model"]: {
            "threshold": float(row["threshold"]),
            "holding_period": int(row["holding_period"]),
            "trade_count": int(row["trade_count"]),
            "coverage": float(row["coverage"]),
            "directional_accuracy": float(row["directional_accuracy"]),
            "avg_actual_return": float(row["avg_actual_return"]),
            "avg_strategy_return": float(row["avg_strategy_return"]),
            "cumulative_strategy_return": float(row["cumulative_strategy_return"]),
            "final_equity": float(row["final_equity"]),
            "sampled_buy_hold_equity": float(row["sampled_buy_hold_equity"]),
        }
        for _, row in best_rows.iterrows()
    }
    with (run_dir / f"threshold_backtest_summary{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", run_dir / f"threshold_backtest{suffix}.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
