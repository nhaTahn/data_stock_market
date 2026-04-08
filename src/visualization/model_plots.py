from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.report_layout import report_plot_path


def _prepare_rel_score_histogram_stats(split_df: pd.DataFrame) -> dict[str, float | np.ndarray]:
    base_abs = np.abs(split_df["base"].to_numpy(dtype=float))
    error_abs = np.abs(split_df["error"].to_numpy(dtype=float))

    q50_base, q90_base = np.quantile(base_abs, [0.5, 0.9])
    q50_error, q90_error = np.quantile(error_abs, [0.5, 0.9])

    base_loss = float(q50_base + 0.5 * q90_base)
    abs_loss = float(q50_error + 0.5 * q90_error)
    rel_score = 1.0 - (abs_loss / base_loss) if base_loss > 0 else np.nan

    raw_proxy_rel_score = 1.0 - (error_abs / np.maximum(base_abs, 1e-6))
    raw_proxy_rel_score = np.clip(raw_proxy_rel_score, -3.0, 1.0)

    # Stabilize the per-row proxy with the same robust scale used by the
    # aggregate rel_score, so rows with tiny |base| do not collapse at -3.
    proxy_floor = max(base_loss, 1e-4)
    proxy_denom = np.maximum(base_abs, proxy_floor)
    stabilized_proxy_rel_score = 1.0 - (error_abs / proxy_denom)
    stabilized_proxy_rel_score = np.clip(stabilized_proxy_rel_score, -1.5, 1.0)

    near_zero_threshold = max(1e-6, 0.05 * q50_base)
    near_zero_share = float(np.mean(base_abs <= near_zero_threshold))
    raw_positive_share = float(np.mean(raw_proxy_rel_score > 0.0))
    raw_left_edge_share = float(np.mean(raw_proxy_rel_score <= -2.9))
    stabilized_positive_share = float(np.mean(stabilized_proxy_rel_score > 0.0))
    stabilized_hard_left_share = float(np.mean(stabilized_proxy_rel_score < -0.5))

    return {
        "raw_proxy_rel_score": raw_proxy_rel_score,
        "stabilized_proxy_rel_score": stabilized_proxy_rel_score,
        "q50_base": float(q50_base),
        "q90_base": float(q90_base),
        "q50_error": float(q50_error),
        "q90_error": float(q90_error),
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": float(rel_score),
        "proxy_floor": float(proxy_floor),
        "near_zero_share": near_zero_share,
        "raw_positive_share": raw_positive_share,
        "raw_left_edge_share": raw_left_edge_share,
        "stabilized_positive_share": stabilized_positive_share,
        "stabilized_hard_left_share": stabilized_hard_left_share,
    }


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
            actual = split_df["actual"].to_numpy(dtype=float)
            prediction = split_df["prediction"].to_numpy(dtype=float)
            actual_std = float(np.nanstd(actual))
            pred_std = float(np.nanstd(prediction))
            amplitude_ratio = pred_std / max(actual_std, 1e-8)
            scale_factor = min(12.0, max(1.0, actual_std / max(pred_std, 1e-8)))
            rolling_window = max(3, min(15, len(split_df) // 20 if len(split_df) >= 20 else 5))
            actual_roll = pd.Series(actual).rolling(rolling_window, min_periods=1).mean().to_numpy()
            prediction_roll = pd.Series(prediction).rolling(rolling_window, min_periods=1).mean().to_numpy()

            display_prediction = prediction_roll.copy()
            prediction_label = f"prediction roll({rolling_window})"
            if np.isfinite(scale_factor) and scale_factor > 1.5:
                display_prediction = (prediction_roll - float(np.nanmean(prediction_roll))) * scale_factor
                prediction_label = f"prediction x{scale_factor:.1f} roll({rolling_window})"

            ax.plot(
                split_df["Date"],
                actual_roll,
                label=f"actual roll({rolling_window})",
                color="#1f4e79",
                linewidth=1.8,
                alpha=0.95,
            )
            ax.plot(
                split_df["Date"],
                display_prediction,
                label=prediction_label,
                color="#c55a11",
                linewidth=1.7,
                alpha=0.95,
            )
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
            ax.set_title(f"{code} | {split_name}")
            ax.grid(True, alpha=0.2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right")
            stats_text = f"pred_std={pred_std:.4f}\nactual_std={actual_std:.4f}\namp={amplitude_ratio:.2f}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "#cccccc"},
            )

    fig.suptitle(f"Actual vs Prediction | {model_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    filename = f"actual_vs_prediction_{model_name}.png"
    fig.savefig(report_plot_path(run_dir, filename), dpi=200)
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

    fig, axes = plt.subplots(
        len(split_frames),
        2,
        figsize=(16, 3.8 * len(split_frames)),
        squeeze=False,
    )

    for row_idx, (split_name, split_df) in enumerate(split_frames):
        stats = _prepare_rel_score_histogram_stats(split_df)
        raw_proxy_rel_score = stats["raw_proxy_rel_score"]
        stabilized_proxy_rel_score = stats["stabilized_proxy_rel_score"]

        raw_ax = axes[row_idx, 0]
        raw_ax.hist(raw_proxy_rel_score, bins=np.linspace(-3.0, 1.0, 41), color="#1f77b4", alpha=0.75)
        raw_ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.35)
        raw_ax.axvline(float(np.mean(raw_proxy_rel_score)), color="#d62728", linewidth=1.5, label="mean proxy")
        raw_ax.axvline(stats["rel_score"], color="#2ca02c", linewidth=1.5, linestyle="--", label="aggregate rel_score")
        raw_ax.set_title(f"{model_name} | {split_name} | Raw Proxy")
        raw_ax.set_xlabel("raw local rel score proxy")
        raw_ax.set_ylabel("count")
        raw_ax.grid(True, alpha=0.2)
        raw_ax.legend(loc="upper left")

        raw_summary = (
            f"base_loss={stats['base_loss']:.4f}\n"
            f"abs_loss={stats['abs_loss']:.4f}\n"
            f"rel_score={stats['rel_score']:.4f}\n"
            f"mean_proxy={np.mean(raw_proxy_rel_score):.4f}\n"
            f"median_proxy={np.median(raw_proxy_rel_score):.4f}\n"
            f"share(proxy>0)={stats['raw_positive_share']:.1%}\n"
            f"share(proxy<=-2.9)={stats['raw_left_edge_share']:.1%}\n"
            f"near_zero_base={stats['near_zero_share']:.1%}"
        )
        raw_ax.text(
            0.98,
            0.98,
            raw_summary,
            transform=raw_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        stable_ax = axes[row_idx, 1]
        stable_ax.hist(
            stabilized_proxy_rel_score,
            bins=np.linspace(-1.5, 1.0, 41),
            color="#1f77b4",
            alpha=0.75,
        )
        stable_ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.35)
        stable_ax.axvline(
            float(np.mean(stabilized_proxy_rel_score)),
            color="#d62728",
            linewidth=1.5,
            label="mean proxy",
        )
        stable_ax.axvline(
            stats["rel_score"],
            color="#2ca02c",
            linewidth=1.5,
            linestyle="--",
            label="aggregate rel_score",
        )
        stable_ax.set_title(f"{model_name} | {split_name} | Stabilized Proxy")
        stable_ax.set_xlabel("stabilized local rel score proxy")
        stable_ax.set_ylabel("count")
        stable_ax.grid(True, alpha=0.2)
        stable_ax.legend(loc="upper left")

        stable_summary = (
            f"base_loss={stats['base_loss']:.4f}\n"
            f"abs_loss={stats['abs_loss']:.4f}\n"
            f"rel_score={stats['rel_score']:.4f}\n"
            f"mean_proxy={np.mean(stabilized_proxy_rel_score):.4f}\n"
            f"median_proxy={np.median(stabilized_proxy_rel_score):.4f}\n"
            f"share(proxy>0)={stats['stabilized_positive_share']:.1%}\n"
            f"share(proxy<-0.5)={stats['stabilized_hard_left_share']:.1%}\n"
            f"proxy_floor={stats['proxy_floor']:.4f}"
        )
        stable_ax.text(
            0.98,
            0.98,
            stable_summary,
            transform=stable_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.suptitle("Rel Score Histogram: Raw vs Stabilized Proxy", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    filename = f"rel_score_hist_{model_name}.png"
    fig.savefig(report_plot_path(run_dir, filename), dpi=200)
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
    if output_path.parent.name != "plots":
        fig.savefig(report_plot_path(output_path.parent, output_path.name), dpi=200)
    plt.close(fig)
