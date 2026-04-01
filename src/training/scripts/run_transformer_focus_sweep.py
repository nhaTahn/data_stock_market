from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is not installed in the current Python environment.\n"
        "Use a local virtualenv with matplotlib, then rerun this script.\n"
        "Example: pip install matplotlib"
    ) from exc

from model_benchmark.data import prepare_sequence_data
from model_benchmark.reporting import save_model_metrics
from model_benchmark.transformer import run_transformer
from tf_lstm.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TRAIN_END_DATE, VAL_END_DATE


def build_focus_grid() -> list[dict]:
    configs: list[dict] = []
    for dropout in [0.15, 0.2, 0.25]:
        for lr in [3e-4, 5e-4]:
            for num_heads in [4, 6]:
                name = (
                    "transformer_focus_"
                    f"w30_ff64_h16_heads{num_heads}_"
                    f"do{int(dropout * 100):02d}_"
                    f"lr{str(lr).replace('.', '').replace('-', '')}"
                )
                configs.append(
                    {
                        "name": name,
                        "window_size": 30,
                        "feature_groups": ["base", "liquidity"],
                        "dropout": dropout,
                        "batch_size": 128,
                        "epochs": 10,
                        "lr": lr,
                        "seed": 42,
                        "transformer_head_size": 16,
                        "transformer_num_heads": num_heads,
                        "transformer_ff_dim": 64,
                    }
                )
    return configs


def summarize_results(experiment_name: str, baseline_metrics: dict, model_metrics: dict, config: dict) -> list[dict]:
    rows = []
    for split in ["train", "val", "test"]:
        rows.append(
            {
                "experiment": experiment_name,
                "split": split,
                "dropout": config["dropout"],
                "lr": config["lr"],
                "num_heads": config["transformer_num_heads"],
                "baseline_mae": baseline_metrics[split]["mae"],
                "baseline_rmse": baseline_metrics[split]["rmse"],
                "baseline_directional_accuracy": baseline_metrics[split]["directional_accuracy"],
                "baseline_thresholded_directional_accuracy": baseline_metrics[split]["thresholded_directional_accuracy"],
                "model_mae": model_metrics[split]["mae"],
                "model_rmse": model_metrics[split]["rmse"],
                "model_directional_accuracy": model_metrics[split]["directional_accuracy"],
                "model_thresholded_directional_accuracy": model_metrics[split]["thresholded_directional_accuracy"],
            }
        )
    return rows


def plot_focus(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df = summary_df[summary_df["split"] == "test"].copy()
    if test_df.empty:
        return

    for metric in ["mae", "rmse", "directional_accuracy", "thresholded_directional_accuracy"]:
        ordered = test_df.sort_values(f"model_{metric}", ascending=(metric in {"mae", "rmse"}))
        plt.figure(figsize=(14, 5))
        plt.bar(ordered["experiment"], ordered[f"model_{metric}"], color="#2563eb")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Transformer focus sweep test {metric.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"transformer_focus_test_{metric}.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused Transformer hyperparameter sweep on the base+liquidity branch.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "transformer_focus_sweep",
        help="Directory for sweep outputs.",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for experiment in build_focus_grid():
        name = experiment["name"]
        config = {
            "data_path": str(DEFAULT_DATA_PATH),
            "train_end": TRAIN_END_DATE,
            "val_end": VAL_END_DATE,
            **experiment,
        }
        print(f"\n=== Running transformer focus: {name} ===")
        bundle = prepare_sequence_data(
            data_path=Path(config["data_path"]),
            train_end=config["train_end"],
            val_end=config["val_end"],
            window_size=config["window_size"],
            feature_groups=config["feature_groups"],
        )
        model_metrics, history_df, model = run_transformer(bundle, SimpleNamespace(**config))
        run_dir = args.comparison_dir / name
        save_model_metrics(
            run_dir,
            "transformer",
            model_metrics,
            {
                "feature_groups": bundle.feature_groups,
                "feature_count": len(bundle.feature_columns),
                "window_size": config["window_size"],
                "head_size": config["transformer_head_size"],
                "num_heads": config["transformer_num_heads"],
                "ff_dim": config["transformer_ff_dim"],
                "dropout": config["dropout"],
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
                "baseline_1": "predict next-day return = 0",
            },
        )
        save_model_metrics(
            run_dir,
            "baseline",
            bundle.baseline_metrics,
            {
                "feature_groups": bundle.feature_groups,
                "window_size": config["window_size"],
                "baseline_1": "predict next-day return = 0",
            },
        )
        history_df.to_csv(run_dir / "transformer_fit_history.csv", index=False)
        model.save(run_dir / "transformer_model.keras")
        all_rows.extend(summarize_results(name, bundle.baseline_metrics, model_metrics, config))
        print(f"Saved run to: {run_dir}")

    summary_df = pd.DataFrame(all_rows)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.comparison_dir / "transformer_focus_summary.csv", index=False)
    plot_focus(summary_df, args.comparison_dir)
    print(f"\nSaved Transformer focus outputs to: {args.comparison_dir}")


if __name__ == "__main__":
    main()
