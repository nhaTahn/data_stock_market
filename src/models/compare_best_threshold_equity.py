from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.visualization.model_plots import save_equity_curve_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare equity curves using each run's best threshold summary.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-prefix", default="equity_compare")
    return parser.parse_args()


def load_best_threshold(run_dir: Path) -> dict[str, float]:
    with (run_dir / "threshold_backtest_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return {model_name: float(values["threshold"]) for model_name, values in summary.items()}


def build_curve(model_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    curve_df = model_df.sort_values("Date").copy()
    curve_df["signal"] = np.where(curve_df["prediction"].abs() >= threshold, np.sign(curve_df["prediction"]), 0.0)
    curve_df["trade_return"] = curve_df["signal"] * curve_df["actual"]
    curve_df["equity"] = (1.0 + curve_df["trade_return"]).cumprod()
    curve_df["buy_hold_equity"] = (1.0 + curve_df["actual"]).cumprod()
    curve_df["is_trade"] = (curve_df["signal"] != 0).astype(int)
    return curve_df


def main() -> None:
    args = parse_args()
    all_summary_rows: list[dict[str, float | str | int]] = []
    all_curve_frames: list[pd.DataFrame] = []

    for run_dir in args.run_dirs:
        predictions = pd.read_csv(run_dir / "predictions.csv")
        predictions["Date"] = pd.to_datetime(predictions["Date"])
        best_thresholds = load_best_threshold(run_dir)

        for model_name, threshold in best_thresholds.items():
            model_df = predictions[(predictions["model"] == model_name) & (predictions["split"] == "test")].copy()
            if model_df.empty:
                continue
            curve_df = build_curve(model_df, threshold)
            label = f"{run_dir.name}:{model_name}"
            curve_df["label"] = label
            all_curve_frames.append(curve_df[["Date", "label", "equity", "buy_hold_equity", "trade_return", "actual", "is_trade"]])
            all_summary_rows.append(
                {
                    "run_name": run_dir.name,
                    "model": model_name,
                    "threshold": threshold,
                    "coverage": float(curve_df["is_trade"].mean()),
                    "trade_count": int(curve_df["is_trade"].sum()),
                    "final_equity": float(curve_df["equity"].iloc[-1]),
                    "buy_hold_equity": float(curve_df["buy_hold_equity"].iloc[-1]),
                    "avg_trade_return": float(curve_df.loc[curve_df["is_trade"] == 1, "trade_return"].mean()) if curve_df["is_trade"].sum() > 0 else np.nan,
                    "directional_accuracy_when_trading": float(
                        np.mean(
                            np.sign(curve_df.loc[curve_df["is_trade"] == 1, "trade_return"]) >= 0
                        )
                    ) if curve_df["is_trade"].sum() > 0 else np.nan,
                }
            )

    summary_df = pd.DataFrame(all_summary_rows).sort_values(["final_equity", "trade_count"], ascending=[False, False])
    curve_df = pd.concat(all_curve_frames, ignore_index=True)

    output_dir = args.run_dirs[0].parent
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    curve_path = output_dir / f"{args.output_prefix}_curves.csv"
    summary_df.to_csv(summary_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    save_equity_curve_plot(curve_df, output_dir / f"{args.output_prefix}.png", "Best-Threshold Equity Curves")

    print(summary_path)
    print(curve_path)
    print(output_dir / f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
