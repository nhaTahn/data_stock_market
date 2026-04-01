from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from tf_lstm.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TRAIN_END_DATE, VAL_END_DATE
from tf_lstm.experiment import run_experiment
from tf_lstm.plotting import plot_experiment_comparison


ABLATION_EXPERIMENTS = [
    {
        "name": "ablation_base",
        "feature_groups": ["base"],
    },
    {
        "name": "ablation_base_momentum",
        "feature_groups": ["base", "momentum"],
    },
    {
        "name": "ablation_base_liquidity",
        "feature_groups": ["base", "liquidity"],
    },
    {
        "name": "ablation_base_volatility",
        "feature_groups": ["base", "volatility"],
    },
    {
        "name": "ablation_base_trend",
        "feature_groups": ["base", "trend"],
    },
    {
        "name": "ablation_base_momentum_liquidity",
        "feature_groups": ["base", "momentum", "liquidity"],
    },
    {
        "name": "ablation_all_groups",
        "feature_groups": ["base", "momentum", "liquidity", "volatility", "trend"],
    },
]

BASE_CONFIG = {
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
}


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

    parser = argparse.ArgumentParser(description="Run feature-group ablation experiments for TensorFlow LSTM.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "tf_feature_ablation",
        help="Directory for aggregated ablation tables and plots.",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for experiment in ABLATION_EXPERIMENTS:
        config = {**BASE_CONFIG, **experiment}
        print(f"\n=== Running ablation: {config['name']} | groups={config['feature_groups']} ===")
        run_dir, baseline_metrics, lstm_metrics = run_experiment(SimpleNamespace(**config), run_name=config["name"])
        print(f"Saved run to: {run_dir}")
        all_rows.extend(summarize_results(config["name"], baseline_metrics, lstm_metrics))

    summary_df = pd.DataFrame(all_rows)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.comparison_dir / "feature_ablation_summary.csv", index=False)
    plot_experiment_comparison(summary_df, args.comparison_dir)
    print(f"\nSaved feature ablation outputs to: {args.comparison_dir}")


if __name__ == "__main__":
    main()
