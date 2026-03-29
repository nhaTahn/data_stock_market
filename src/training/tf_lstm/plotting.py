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


def plot_experiment_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = summary_df[summary_df["split"] == "test"].copy()
    if test_df.empty:
        return

    for metric in ["mae", "rmse", "directional_accuracy", "thresholded_directional_accuracy"]:
        plt.figure(figsize=(10, 5))
        plt.bar(test_df["experiment"], test_df[f"lstm_{metric}"], label="LSTM")
        plt.plot(test_df["experiment"], test_df[f"baseline_{metric}"], color="red", marker="o", label="Baseline 1")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Next-day return test {metric.upper()} by experiment")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_test_{metric}.png", dpi=150)
        plt.close()
