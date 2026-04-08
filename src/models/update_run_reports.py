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
from src.models.report_layout import (
    clear_report_files,
    cleanup_legacy_report_artifacts,
    cleanup_report_noise,
    mirror_run_artifacts,
    report_core_path,
    report_metric_series_path,
    resolve_run_artifact,
)
from src.visualization.model_plots import save_actual_vs_prediction_plot, save_rel_score_hist_plot


def select_report_model_names(model_names: list[str]) -> list[str]:
    available = set(model_names)
    selected: list[str] = []
    for baseline_name in ("linear_regression", "arima", "fischer_krauss"):
        if baseline_name in available:
            selected.append(baseline_name)
    family_prefixes = ("lstm", "lstm_quantile", "lstm_signmag", "lstm_attention", "lstm_event")
    for prefix in family_prefixes:
        preferred = [f"{prefix}_best_by_val", f"{prefix}_ensemble", prefix]
        added = False
        for candidate in preferred:
            if candidate in available:
                selected.append(candidate)
                added = True
        if not added:
            fallback = sorted(name for name in available if name.startswith(prefix) and "_seed_" not in name)
            selected.extend(fallback[:1])
    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name in available and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update run reports with directional accuracy and prediction plots.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args()


def update_run(run_dir: Path) -> None:
    predictions_path = resolve_run_artifact(run_dir, "predictions.csv", bucket="core")
    metrics_path = resolve_run_artifact(run_dir, "metrics.json", bucket="core")
    detail_path = resolve_run_artifact(run_dir, "metric_details.json", bucket="core")

    prediction_df = pd.read_csv(predictions_path)
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"])
    report_model_names = set(select_report_model_names(sorted(prediction_df["model"].dropna().unique().tolist())))
    clear_report_files(run_dir, ("plots", "metric_series"))

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8") as f:
            metric_details = json.load(f)
    else:
        metric_details = {model_name: {} for model_name in metrics.keys()}

    for model_name in prediction_df["model"].unique():
        metrics.setdefault(model_name, {})
        metric_details.setdefault(model_name, {})
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
            metrics[model_name].setdefault(split_name, {})
            metrics[model_name][split_name]["directional_accuracy"] = score
            metrics[model_name][split_name]["base_loss"] = float(result["base_loss"])
            metrics[model_name][split_name]["abs_loss"] = float(result["abs_loss"])
            metrics[model_name][split_name]["rel_score"] = float(result["rel_score"])
            if model_name in report_model_names:
                metric_series_df = pd.DataFrame({"error": result["error"], "base": result["base"]})
                metric_series_df.to_csv(run_dir / f"metric_series_{model_name}_{split_name}.csv", index=False)
                metric_series_df.to_csv(
                    report_metric_series_path(run_dir, f"metric_series_{model_name}_{split_name}.csv"),
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
        if model_name in report_model_names:
            save_actual_vs_prediction_plot(run_dir, prediction_df, model_name)
            save_rel_score_hist_plot(run_dir, model_name)

    with report_core_path(run_dir, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with report_core_path(run_dir, "metric_details.json").open("w", encoding="utf-8") as f:
        json.dump(metric_details, f, indent=2)
    mirror_run_artifacts(run_dir)
    cleanup_report_noise(run_dir)
    cleanup_legacy_report_artifacts(run_dir)

    print(f"Updated: {run_dir}")


def main() -> None:
    args = parse_args()
    for run_dir in args.run_dirs:
        update_run(run_dir)


if __name__ == "__main__":
    main()
