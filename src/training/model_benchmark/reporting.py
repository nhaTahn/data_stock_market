from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "matplotlib is not installed in the current Python environment.\n"
        "Use a local virtualenv with matplotlib, then rerun this script.\n"
        "Example: pip install matplotlib"
    ) from exc


def save_model_metrics(output_dir: Path, model_name: str, metrics: dict[str, dict[str, float]], config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([{"split": split, **split_metrics} for split, split_metrics in metrics.items()])
    metrics_df.to_csv(output_dir / f"{model_name}_metrics.csv", index=False)

    lines = [f"{model_name} benchmark", "", "Configuration:"]
    lines.extend(f"{key}: {value}" for key, value in config.items())
    lines.extend(["", "Metrics:", metrics_df.to_string(index=False)])
    (output_dir / f"{model_name}_report.txt").write_text("\n".join(lines), encoding="utf-8")


def save_summary(output_dir: Path, summary_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)


def plot_summary(output_dir: Path, summary_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_df = summary_df[summary_df["split"] == "test"].copy()
    if test_df.empty:
        return

    for metric in ["mae", "rmse", "directional_accuracy", "thresholded_directional_accuracy"]:
        plt.figure(figsize=(10, 5))
        plt.bar(test_df["model"], test_df[metric], color="#3b82f6")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Benchmark test {metric.upper()} by model")
        plt.tight_layout()
        plt.savefig(output_dir / f"benchmark_test_{metric}.png", dpi=150)
        plt.close()

