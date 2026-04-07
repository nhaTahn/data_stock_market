from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.metric import directional_accuracy, evaluate
from src.visualization.model_plots import save_actual_vs_prediction_plot, save_rel_score_hist_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update run reports with directional accuracy and prediction plots.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args()


def update_run(run_dir: Path) -> None:
    predictions_path = run_dir / "predictions.csv"
    metrics_path = run_dir / "metrics.json"
    detail_path = run_dir / "metric_details.json"

    prediction_df = pd.read_csv(predictions_path)
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"])

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8") as f:
            metric_details = json.load(f)
    else:
        metric_details = {model_name: {} for model_name in metrics.keys()}

    for model_name in prediction_df["model"].unique():
        model_df = prediction_df[prediction_df["model"] == model_name]
        for split_name in prediction_df["split"].unique():
            split_df = model_df[model_df["split"] == split_name].copy()
            if split_df.empty:
                continue
            if {"code", "Date"}.issubset(split_df.columns):
                split_df = split_df.sort_values(["code", "Date"], kind="stable")
            predict = split_df["prediction"].to_numpy()
            actual = split_df["actual"].to_numpy()
            group_ids = split_df["code"].to_numpy() if "code" in split_df.columns else None
            score = directional_accuracy(predict, actual, group_ids=group_ids)
            result = evaluate(predict, actual, group_ids=group_ids)
            metrics[model_name][split_name]["directional_accuracy"] = score
            metrics[model_name][split_name]["base_loss"] = float(result["base_loss"])
            metrics[model_name][split_name]["abs_loss"] = float(result["abs_loss"])
            metrics[model_name][split_name]["rel_score"] = float(result["rel_score"])
            pd.DataFrame({"error": result["error"], "base": result["base"]}).to_csv(
                run_dir / f"metric_series_{model_name}_{split_name}.csv",
                index=False,
            )
            metric_details[model_name][split_name] = {
                "base_loss": float(result["base_loss"]),
                "abs_loss": float(result["abs_loss"]),
                "rel_score": float(result["rel_score"]),
                "directional_accuracy": score,
                "error_len": int(len(result["error"])),
                "base_len": int(len(result["base"])),
            }
        save_actual_vs_prediction_plot(run_dir, prediction_df, model_name)
        save_rel_score_hist_plot(run_dir, model_name)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(metric_details, f, indent=2)

    print(f"Updated: {run_dir}")


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        update_run(run_dir)


if __name__ == "__main__":
    main()
