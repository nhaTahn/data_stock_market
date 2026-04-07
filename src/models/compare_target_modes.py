from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LSTM results across target modes.")
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--run-names", nargs="*", default=None)
    parser.add_argument("--details-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_run_dirs(run_base: Path, run_names: list[str] | None) -> list[Path]:
    if run_names:
        return [run_base / name for name in run_names if (run_base / name).exists()]
    return sorted(
        [path for path in run_base.iterdir() if path.is_dir() and (path / "metrics.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def collect_model_rel(metrics: dict[str, object], model_name: str, split: str) -> float | None:
    model_metrics = metrics.get(model_name, {})
    split_metrics = model_metrics.get(split, {}) if isinstance(model_metrics, dict) else {}
    value = split_metrics.get("rel_score") if isinstance(split_metrics, dict) else None
    return float(value) if value is not None else None


def collect_row(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not metrics_path.exists() or not config_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    lstm_models = [name for name in metrics if name.startswith("lstm")]
    if not lstm_models:
        return None

    ranked_models = sorted(
        lstm_models,
        key=lambda name: (
            collect_model_rel(metrics, name, "test") if collect_model_rel(metrics, name, "test") is not None else float("-inf"),
            collect_model_rel(metrics, name, "val") if collect_model_rel(metrics, name, "val") is not None else float("-inf"),
        ),
        reverse=True,
    )
    best_model = ranked_models[0]

    row = {
        "run_name": run_dir.name,
        "target_mode": config.get("target_mode"),
        "stocks": config.get("stocks"),
        "feature_columns": ",".join(config.get("feature_columns", [])),
        "window_size": config.get("window_size"),
        "lstm_units": json.dumps(config.get("lstm_units")),
        "best_lstm_model": best_model,
        "best_lstm_val_rel_score": collect_model_rel(metrics, best_model, "val"),
        "best_lstm_test_rel_score": collect_model_rel(metrics, best_model, "test"),
        "lstm_test_rel_score": collect_model_rel(metrics, "lstm", "test"),
        "lstm_ensemble_test_rel_score": collect_model_rel(metrics, "lstm_ensemble", "test"),
        "lstm_signmag_ensemble_test_rel_score": collect_model_rel(metrics, "lstm_signmag_ensemble", "test"),
        "lstm_attention_ensemble_test_rel_score": collect_model_rel(metrics, "lstm_attention_ensemble", "test"),
        "run_dir": str(run_dir),
    }
    return row


def build_target_summary(details_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for target_mode, group in details_df.groupby("target_mode", dropna=False):
        ranked = group.sort_values(
            ["best_lstm_test_rel_score", "best_lstm_val_rel_score"],
            ascending=[False, False],
        )
        best = ranked.iloc[0]
        summary_rows.append(
            {
                "target_mode": target_mode,
                "run_count": int(len(group)),
                "best_run_name": best["run_name"],
                "best_lstm_model": best["best_lstm_model"],
                "max_best_lstm_test_rel_score": best["best_lstm_test_rel_score"],
                "median_best_lstm_test_rel_score": float(group["best_lstm_test_rel_score"].median()),
                "mean_best_lstm_test_rel_score": float(group["best_lstm_test_rel_score"].mean()),
                "median_best_lstm_val_rel_score": float(group["best_lstm_val_rel_score"].median()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["max_best_lstm_test_rel_score", "median_best_lstm_test_rel_score"],
        ascending=[False, False],
    )


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.run_base, args.run_names)
    rows = [row for run_dir in run_dirs if (row := collect_row(run_dir)) is not None]
    if not rows:
        raise ValueError("No run directories with config.json and metrics.json found.")

    details_df = pd.DataFrame(rows).sort_values(
        ["target_mode", "best_lstm_test_rel_score", "best_lstm_val_rel_score"],
        ascending=[True, False, False],
    )
    summary_df = build_target_summary(details_df)

    details_csv = args.details_csv or (args.run_base / "reports" / "target_mode_comparison_details.csv")
    summary_csv = args.summary_csv or (args.run_base / "reports" / "target_mode_comparison_summary.csv")
    details_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    details_df.to_csv(details_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(details_df.to_string(index=False))
    print(f"Saved details: {details_csv}")
    print(summary_df.to_string(index=False))
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()
