from __future__ import annotations

import argparse
import json
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
from tf_lstm.experiment import run_experiment


DEFAULT_SWEEP = [
    {
        "name": "lstm_bliq_w30_u64_do20_lr5e4",
        "model_type": "lstm",
        "feature_groups": ["base", "liquidity"],
        "window_size": 30,
        "lstm_units": 64,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 20,
        "lr": 5e-4,
        "seed": 42,
    },
    {
        "name": "lstm_bliq_w35_u64_do15_lr3e4",
        "model_type": "lstm",
        "feature_groups": ["base", "liquidity"],
        "window_size": 35,
        "lstm_units": 64,
        "dropout": 0.15,
        "batch_size": 128,
        "epochs": 25,
        "lr": 3e-4,
        "seed": 42,
    },
    {
        "name": "lstm_bliq_w30_u96_do20_lr4e4",
        "model_type": "lstm",
        "feature_groups": ["base", "liquidity"],
        "window_size": 30,
        "lstm_units": 96,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 20,
        "lr": 4e-4,
        "seed": 42,
    },
    {
        "name": "transformer_bliq_w30_ff64_h16_do20_lr5e4",
        "model_type": "transformer",
        "feature_groups": ["base", "liquidity"],
        "window_size": 30,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 10,
        "lr": 5e-4,
        "seed": 42,
        "transformer_head_size": 16,
        "transformer_num_heads": 4,
        "transformer_ff_dim": 64,
    },
    {
        "name": "transformer_bliq_w35_ff64_h16_do20_lr5e4",
        "model_type": "transformer",
        "feature_groups": ["base", "liquidity"],
        "window_size": 35,
        "dropout": 0.2,
        "batch_size": 128,
        "epochs": 10,
        "lr": 5e-4,
        "seed": 42,
        "transformer_head_size": 16,
        "transformer_num_heads": 4,
        "transformer_ff_dim": 64,
    },
    {
        "name": "transformer_bliq_w30_ff96_h16_do15_lr3e4",
        "model_type": "transformer",
        "feature_groups": ["base", "liquidity"],
        "window_size": 30,
        "dropout": 0.15,
        "batch_size": 128,
        "epochs": 12,
        "lr": 3e-4,
        "seed": 42,
        "transformer_head_size": 16,
        "transformer_num_heads": 4,
        "transformer_ff_dim": 96,
    },
]


def load_sweep_configs(config_path: Path | None) -> list[dict]:
    if config_path is None:
        return DEFAULT_SWEEP
    return json.loads(config_path.read_text(encoding="utf-8"))


def summarize_results(experiment_name: str, model_type: str, baseline_metrics: dict, model_metrics: dict) -> list[dict]:
    rows = []
    for split in ["train", "val", "test"]:
        rows.append(
            {
                "experiment": experiment_name,
                "model_type": model_type,
                "split": split,
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


def plot_sweep(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df = summary_df[summary_df["split"] == "test"].copy()
    if test_df.empty:
        return

    for metric in ["mae", "rmse", "directional_accuracy", "thresholded_directional_accuracy"]:
        plt.figure(figsize=(12, 5))
        plt.bar(test_df["experiment"], test_df[f"model_{metric}"], color="#2563eb", label="Model")
        plt.plot(
            test_df["experiment"],
            test_df[f"baseline_{metric}"],
            color="#dc2626",
            marker="o",
            label="Baseline 1",
        )
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Sequence model sweep test {metric.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"sweep_test_{metric}.png", dpi=150)
        plt.close()


def run_transformer_experiment(args: SimpleNamespace, run_name: str, output_dir: Path) -> tuple[Path, dict, dict]:
    bundle = prepare_sequence_data(
        data_path=Path(args.data_path),
        train_end=args.train_end,
        val_end=args.val_end,
        window_size=args.window_size,
        feature_groups=args.feature_groups,
    )
    metrics, history_df, model = run_transformer(bundle, args)
    run_dir = output_dir / run_name
    save_model_metrics(
        run_dir,
        "transformer",
        metrics,
        {
            "feature_groups": bundle.feature_groups,
            "feature_count": len(bundle.feature_columns),
            "window_size": args.window_size,
            "head_size": args.transformer_head_size,
            "num_heads": args.transformer_num_heads,
            "ff_dim": args.transformer_ff_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "baseline_1": "predict next-day return = 0",
        },
    )
    save_model_metrics(
        run_dir,
        "baseline",
        bundle.baseline_metrics,
        {
            "feature_groups": bundle.feature_groups,
            "window_size": args.window_size,
            "baseline_1": "predict next-day return = 0",
        },
    )
    history_df.to_csv(run_dir / "transformer_fit_history.csv", index=False)
    model.save(run_dir / "transformer_model.keras")
    return run_dir, bundle.baseline_metrics, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a focused tuning sweep for LSTM and Transformer sequence models.")
    parser.add_argument("--config-path", type=Path, default=None, help="Optional JSON file describing sweep configs.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sequence_model_sweep",
        help="Directory for sweep outputs.",
    )
    args = parser.parse_args()

    sweep_configs = load_sweep_configs(args.config_path)
    all_rows: list[dict] = []

    for experiment in sweep_configs:
        name = experiment["name"]
        model_type = experiment["model_type"]
        config = {
            "data_path": str(DEFAULT_DATA_PATH),
            "output_dir": str(args.comparison_dir),
            "train_end": TRAIN_END_DATE,
            "val_end": VAL_END_DATE,
            **experiment,
        }
        print(f"\n=== Running {model_type}: {name} ===")
        if model_type == "lstm":
            run_dir, baseline_metrics, model_metrics = run_experiment(SimpleNamespace(**config), run_name=name)
        elif model_type == "transformer":
            run_dir, baseline_metrics, model_metrics = run_transformer_experiment(
                SimpleNamespace(**config),
                run_name=name,
                output_dir=args.comparison_dir,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        print(f"Saved run to: {run_dir}")
        all_rows.extend(summarize_results(name, model_type, baseline_metrics, model_metrics))

    summary_df = pd.DataFrame(all_rows)
    args.comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.comparison_dir / "sequence_sweep_summary.csv", index=False)
    plot_sweep(summary_df, args.comparison_dir)
    print(f"\nSaved sweep outputs to: {args.comparison_dir}")


if __name__ == "__main__":
    main()
