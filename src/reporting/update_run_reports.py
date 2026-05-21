from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.reporting import (
    clear_report_files,
    cleanup_legacy_report_artifacts,
    cleanup_report_noise,
    get_default_reporting_standard,
    mirror_run_artifacts,
    report_core_path,
    refresh_run_report_artifacts,
    resolve_run_artifact,
    select_report_model_names,
    resolve_standard_from_config,
)


def add_update_run_reports_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--reveal-out-sample",
        action="store_true",
        help="Include the hidden out-sample holdout split in refreshed report artifacts.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh run reports with standard VN evaluation artifacts.")
    add_update_run_reports_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def update_run(run_dir: Path, *, reveal_out_sample: bool = False) -> None:
    private_predictions_path = run_dir / "holdout_private" / "predictions_full.csv"
    predictions_path = private_predictions_path if private_predictions_path.exists() else resolve_run_artifact(run_dir, "predictions.csv", bucket="core")
    metrics_path = resolve_run_artifact(run_dir, "metrics.json", bucket="core")
    detail_path = resolve_run_artifact(run_dir, "metric_details.json", bucket="core")
    config_path = resolve_run_artifact(run_dir, "config.json", bucket="core")

    prediction_df = pd.read_csv(predictions_path)
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"])
    report_model_names = set(select_report_model_names(sorted(prediction_df["model"].dropna().unique().tolist())))
    clear_report_files(run_dir, ("plots", "metric_series", "backtests", "diagnostics"))

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8") as f:
            metric_details = json.load(f)
    else:
        metric_details = {model_name: {} for model_name in metrics.keys()}

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config_payload = json.load(f)
        standard = resolve_standard_from_config(config_payload)
        feature_columns = tuple(config_payload.get("feature_columns", []))
    else:
        standard = get_default_reporting_standard()
        feature_columns = ()

    metrics, metric_details = refresh_run_report_artifacts(
        run_dir,
        prediction_df,
        metrics,
        metric_details,
        report_model_names,
        standard=standard,
        feature_columns=feature_columns,
        reveal_out_sample=reveal_out_sample,
    )

    with report_core_path(run_dir, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with report_core_path(run_dir, "metric_details.json").open("w", encoding="utf-8") as f:
        json.dump(metric_details, f, indent=2)
    mirror_run_artifacts(run_dir)
    cleanup_report_noise(run_dir)
    cleanup_legacy_report_artifacts(run_dir)

    print(f"Updated: {run_dir}")


def update_run_reports_command(args: argparse.Namespace) -> None:
    for run_dir in args.run_dirs:
        update_run(run_dir, reveal_out_sample=bool(args.reveal_out_sample))


def main(argv: list[str] | None = None) -> None:
    update_run_reports_command(parse_args(argv))


if __name__ == "__main__":
    main()
