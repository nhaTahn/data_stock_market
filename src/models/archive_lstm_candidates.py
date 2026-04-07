from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive and summarize LSTM candidates by rel_score.")
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument("--run-names", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--candidates-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_run_dirs(run_base: Path, run_names: list[str] | None) -> list[Path]:
    if run_names:
        return [run_base / run_name for run_name in run_names if (run_base / run_name).exists()]
    return sorted(
        [path for path in run_base.iterdir() if path.is_dir() and (path / "metrics.json").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def collect_row(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text())
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    backtest_path = run_dir / "threshold_backtest_summary_non_overlap.json"
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}

    lstm_models = [name for name in metrics if name.startswith("lstm")]
    if not lstm_models:
        return None
    ranked_models = sorted(
        lstm_models,
        key=lambda name: (
            metrics.get(name, {}).get("test", {}).get("rel_score", float("-inf")),
            metrics.get(name, {}).get("val", {}).get("rel_score", float("-inf")),
        ),
        reverse=True,
    )
    best_model = ranked_models[0]
    lstm_train = metrics.get("lstm", {}).get("train", {})
    lstm_val = metrics.get("lstm", {}).get("val", {})
    lstm_test = metrics.get("lstm", {}).get("test", {})
    best_val = metrics.get(best_model, {}).get("val", {})
    best_test = metrics.get(best_model, {}).get("test", {})
    linear_test = metrics.get("linear_regression", {}).get("test", {})
    arima_test = metrics.get("arima", {}).get("test", {})
    lstm_backtest = backtest.get("lstm", {})
    best_backtest = backtest.get(best_model, {})

    return {
        "run_name": run_dir.name,
        "target_mode": config.get("target_mode"),
        "stocks": config.get("stocks"),
        "feature_columns": ",".join(config.get("feature_columns", [])),
        "window_size": config.get("window_size"),
        "lstm_units": json.dumps(config.get("lstm_units")),
        "dropout": config.get("dropout"),
        "lr": config.get("lr"),
        "epochs": config.get("epochs"),
        "patience": config.get("patience"),
        "best_lstm_model": best_model,
        "best_lstm_val_rel_score": best_val.get("rel_score"),
        "best_lstm_test_rel_score": best_test.get("rel_score"),
        "best_lstm_test_directional_accuracy": best_test.get("directional_accuracy"),
        "best_lstm_backtest_final_equity": best_backtest.get("final_equity"),
        "best_lstm_backtest_trade_count": best_backtest.get("trade_count"),
        "best_lstm_backtest_threshold": best_backtest.get("threshold"),
        "lstm_train_rel_score": lstm_train.get("rel_score"),
        "lstm_val_rel_score": lstm_val.get("rel_score"),
        "lstm_test_rel_score": lstm_test.get("rel_score"),
        "lstm_test_directional_accuracy": lstm_test.get("directional_accuracy"),
        "linear_test_rel_score": linear_test.get("rel_score"),
        "arima_test_rel_score": arima_test.get("rel_score"),
        "lstm_backtest_final_equity": lstm_backtest.get("final_equity"),
        "lstm_backtest_trade_count": lstm_backtest.get("trade_count"),
        "lstm_backtest_threshold": lstm_backtest.get("threshold"),
        "run_dir": str(run_dir),
    }


def ensure_link(source_dir: Path, save_dir: Path) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)
    target = save_dir / source_dir.name
    if target.exists() or target.is_symlink():
        return str(target)
    try:
        os.symlink(source_dir.resolve(), target, target_is_directory=True)
    except OSError:
        shutil.copytree(source_dir, target)
    return str(target)


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.run_base, args.run_names)
    rows = [row for run_dir in run_dirs if (row := collect_row(run_dir)) is not None]
    if not rows:
        raise ValueError("No run directories with metrics.json found.")

    summary_df = pd.DataFrame(rows).sort_values(
        ["best_lstm_test_rel_score", "best_lstm_val_rel_score", "best_lstm_backtest_final_equity"],
        ascending=[False, False, False],
    )

    save_dir = args.save_dir or (args.run_base / "representative_runs" / f"lstm_relscore_ge_{str(args.threshold).replace('.', '_')}")
    candidates_df = summary_df[summary_df["best_lstm_test_rel_score"] >= args.threshold].copy()
    if not candidates_df.empty:
        candidates_df["saved_path"] = [
            ensure_link(Path(run_dir), save_dir)
            for run_dir in candidates_df["run_dir"].tolist()
        ]

    summary_csv = args.summary_csv or (args.run_base / "reports" / "lstm_run_summary.csv")
    candidates_csv = args.candidates_csv or (args.run_base / "reports" / "lstm_candidates_ge_threshold.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_csv, index=False)
    candidates_df.to_csv(candidates_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"Saved summary: {summary_csv}")
    print(f"Saved candidates: {candidates_csv}")
    if not candidates_df.empty:
        print(f"Archived candidates: {save_dir}")


if __name__ == "__main__":
    main()
