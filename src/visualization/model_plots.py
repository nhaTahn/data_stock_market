from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_actual_vs_prediction_plot(run_dir: Path, prediction_df: pd.DataFrame, model_name: str) -> None:
    model_df = prediction_df[prediction_df["model"] == model_name].copy()
    if model_df.empty:
        return

    split_names = ["train", "val", "test"]
    if "code" in model_df.columns:
        codes = sorted(model_df["code"].dropna().unique().tolist())
    else:
        codes = ["all"]
        model_df = model_df.assign(code="all")

    n_rows = len(codes)
    n_cols = len(split_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 2.8 * max(n_rows, 1)),
        sharex=False,
        squeeze=False,
    )

    for row_idx, code in enumerate(codes):
        code_df = model_df[model_df["code"] == code].copy()
        for col_idx, split_name in enumerate(split_names):
            ax = axes[row_idx, col_idx]
            split_df = code_df[code_df["split"] == split_name].copy()
            split_df = split_df.sort_values("Date", kind="stable")
            if split_df.empty:
                ax.set_visible(False)
                continue
            ax.plot(split_df["Date"], split_df["actual"], label="actual", color="#1f77b4", linewidth=1.2)
            ax.plot(split_df["Date"], split_df["prediction"], label="prediction", color="#ff7f0e", linewidth=1.2, alpha=0.9)
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(f"{code} | {split_name}")
            ax.grid(True, alpha=0.2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right")

    fig.suptitle(f"Actual vs Prediction | {model_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(run_dir / f"actual_vs_prediction_{model_name}.png", dpi=200)
    plt.close(fig)


def save_rel_score_hist_plot(run_dir: Path, model_name: str) -> None:
    split_frames: list[tuple[str, pd.DataFrame]] = []
    for split_name in ("train", "val", "test"):
        path = run_dir / f"metric_series_{model_name}_{split_name}.csv"
        if not path.exists():
            continue
        split_df = pd.read_csv(path)
        if split_df.empty:
            continue
        split_frames.append((split_name, split_df))

    if not split_frames:
        return

    fig, axes = plt.subplots(len(split_frames), 1, figsize=(14, 3.5 * len(split_frames)), squeeze=False)

    for row_idx, (split_name, split_df) in enumerate(split_frames):
        base_abs = np.abs(split_df["base"].to_numpy(dtype=float))
        error_abs = np.abs(split_df["error"].to_numpy(dtype=float))
        proxy_rel_score = 1.0 - (error_abs / np.maximum(base_abs, 1e-6))
        proxy_rel_score = np.clip(proxy_rel_score, -3.0, 1.0)

        q50_base, q90_base = np.quantile(base_abs, [0.5, 0.9])
        q50_error, q90_error = np.quantile(error_abs, [0.5, 0.9])
        base_loss = float(q50_base + 0.5 * q90_base)
        abs_loss = float(q50_error + 0.5 * q90_error)
        rel_score = 1.0 - (abs_loss / base_loss) if base_loss > 0 else np.nan
        hist_ax = axes[row_idx, 0]
        hist_ax.hist(proxy_rel_score, bins=50, color="#1f77b4", alpha=0.75)
        hist_ax.axvline(float(np.mean(proxy_rel_score)), color="#d62728", linewidth=1.5, label="mean proxy")
        hist_ax.axvline(rel_score, color="#2ca02c", linewidth=1.5, linestyle="--", label="aggregate rel_score")
        hist_ax.set_title(f"{model_name} | {split_name} | Rel Score Histogram")
        hist_ax.set_xlabel("local rel score proxy")
        hist_ax.set_ylabel("count")
        hist_ax.grid(True, alpha=0.2)
        hist_ax.legend(loc="upper left")

        summary = (
            f"base_loss={base_loss:.4f}\n"
            f"abs_loss={abs_loss:.4f}\n"
            f"rel_score={rel_score:.4f}\n"
            f"mean_proxy={np.mean(proxy_rel_score):.4f}\n"
            f"median_proxy={np.median(proxy_rel_score):.4f}\n"
            f"q50(|base|)={q50_base:.4f}\n"
            f"q90(|error|)={q90_error:.4f}"
        )
        hist_ax.text(
            0.98,
            0.98,
            summary,
            transform=hist_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.suptitle("Rel Score Histogram", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(run_dir / f"rel_score_hist_{model_name}.png", dpi=200)
    plt.close(fig)


def save_metric_evaluation_plot(run_dir: Path, model_name: str) -> None:
    save_rel_score_hist_plot(run_dir, model_name)


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
