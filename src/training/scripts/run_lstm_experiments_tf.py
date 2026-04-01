from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from tf_lstm.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TRAIN_END_DATE, VAL_END_DATE
from tf_lstm.experiment import run_experiment
from tf_lstm.plotting import plot_experiment_comparison


DEFAULT_EXPERIMENTS = [
    {
        "name": "return_window20_units32",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 20,
        "lstm_units": 32,
        "dropout": 0.0,
        "batch_size": 128,
        "epochs": 20,
        "lr": 1e-3,
        "seed": 42,
    },
    {
        "name": "return_window30_units64",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 30,
        "lstm_units": 64,
        "dropout": 0.1,
        "batch_size": 128,
        "epochs": 20,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "return_window40_units64",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 40,
        "lstm_units": 64,
        "dropout": 0.1,
        "batch_size": 128,
        "epochs": 25,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "return_window25_units64",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 25,
        "lstm_units": 64,
        "dropout": 0.1,
        "batch_size": 128,
        "epochs": 20,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "return_window30_units96",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 30,
        "lstm_units": 96,
        "dropout": 0.1,
        "batch_size": 128,
        "epochs": 20,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "return_window30_units64_dropout20",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 30,
        "lstm_units": 64,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 20,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "return_window35_units64_lr3e4",
        "data_path": str(DEFAULT_DATA_PATH),
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "train_end": TRAIN_END_DATE,
        "val_end": VAL_END_DATE,
        "window_size": 35,
        "lstm_units": 64,
        "dropout": 0.1,
        "batch_size": 128,
        "epochs": 25,
        "lr": 3e-4,
        "seed": 42,
    },
]


def load_experiments(config_path: Path | None) -> list[dict]:
    if config_path is None:
        return DEFAULT_EXPERIMENTS
    return json.loads(config_path.read_text(encoding="utf-8"))


def summarize_results(experiment_name: str, baseline_metrics: dict, lstm_metrics: dict) -> list[dict]:
    rows = []
    for split in ["train", "val", "test"]:
        rows.append(
            {
                "experiment": experiment_name,
                "split": split,
                "baseline_mae": baseline_metrics[split]["mae"],
                "baseline_rmse": baseline_metrics[split]["rmse"],
                "baseline_directional_accuracy": baseline_metrics[split]["directional_accuracy"],
                "baseline_thresholded_directional_accuracy": baseline_metrics[split]["thresholded_directional_accuracy"],
                "lstm_mae": lstm_metrics[split]["mae"],
                "lstm_rmse": lstm_metrics[split]["rmse"],
                "lstm_directional_accuracy": lstm_metrics[split]["directional_accuracy"],
                "lstm_thresholded_directional_accuracy": lstm_metrics[split]["thresholded_directional_accuracy"],
            }
        )
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple TensorFlow LSTM experiments and plot comparisons.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional JSON file describing experiment configs.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tf_experiment_comparison",
        help="Directory for aggregated comparison tables and plots.",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.config_path)
    all_rows: list[dict] = []

    for experiment in experiments:
        name = experiment["name"]
        print(f"\n=== Running experiment: {name} ===")
        run_dir, baseline_metrics, lstm_metrics = run_experiment(SimpleNamespace(**experiment), run_name=name)
        print(f"Saved run to: {run_dir}")
        all_rows.extend(summarize_results(name, baseline_metrics, lstm_metrics))

    summary_df = pd.DataFrame(all_rows)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.comparison_dir / "experiment_summary.csv", index=False)
    plot_experiment_comparison(summary_df, args.comparison_dir)
    print(f"\nSaved comparison outputs to: {args.comparison_dir}")


if __name__ == "__main__":
    main()
