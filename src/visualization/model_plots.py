from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_actual_vs_prediction_plot(run_dir: Path, prediction_df: pd.DataFrame, model_name: str) -> None:
    model_df = prediction_df[prediction_df["model"] == model_name].copy()
    if model_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    for ax, split_name in zip(axes, ["train", "val", "test"]):
        split_df = model_df[model_df["split"] == split_name].sort_values("Date")
        if split_df.empty:
            ax.set_visible(False)
            continue
        ax.plot(split_df["Date"], split_df["actual"], label="actual", color="#1f77b4", linewidth=1.2)
        ax.plot(split_df["Date"], split_df["prediction"], label="prediction", color="#ff7f0e", linewidth=1.2, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(f"{model_name} | {split_name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Actual vs Prediction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(run_dir / f"actual_vs_prediction_{model_name}.png", dpi=200)
    plt.close(fig)


def save_equity_curve_plot(curve_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, label_df in curve_df.groupby("label"):
        ax.plot(label_df["Date"], label_df["equity"], linewidth=1.6, label=label)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
