from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

from src.reporting import resolve_run_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot comparison charts for multiple training runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="trial_compare")
    return parser.parse_args()


def load_backtest_summary(run_dir: Path, target_mode: str) -> tuple[str, dict[str, dict[str, float]]]:
    if target_mode in {"return_3d", "return_5d"}:
        non_overlap = resolve_run_artifact(run_dir, "threshold_backtest_summary_non_overlap.json", "backtests")
        if non_overlap.exists():
            with non_overlap.open("r", encoding="utf-8") as f:
                return "non_overlap", json.load(f)
    with resolve_run_artifact(run_dir, "threshold_backtest_summary.json", "backtests").open("r", encoding="utf-8") as f:
        return "overlap", json.load(f)


def select_non_overlap_mask(model_df: pd.DataFrame, threshold: float, holding_period: int) -> pd.Series:
    chosen = np.zeros(len(model_df), dtype=bool)
    i = 0
    while i < len(model_df):
        if abs(float(model_df.iloc[i]["prediction"])) >= threshold:
            chosen[i] = True
            i += holding_period
        else:
            i += 1
    return pd.Series(chosen, index=model_df.index)


def build_equity_frame(run_dir: Path, model_name: str, threshold: float, holding_period: int, non_overlap: bool) -> pd.DataFrame:
    predictions = pd.read_csv(resolve_run_artifact(run_dir, "predictions.csv", "core"))
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    model_df = predictions[(predictions["model"] == model_name) & (predictions["split"] == "test")].sort_values("Date").copy()
    if non_overlap and holding_period > 1:
        trade_mask = select_non_overlap_mask(model_df.reset_index(drop=True), threshold, holding_period).to_numpy()
        model_df = model_df.reset_index(drop=True)
        model_df["signal"] = 0.0
        model_df.loc[trade_mask, "signal"] = np.sign(model_df.loc[trade_mask, "prediction"])
    else:
        model_df["signal"] = np.where(model_df["prediction"].abs() >= threshold, np.sign(model_df["prediction"]), 0.0)
    model_df["trade_return"] = model_df["signal"] * model_df["actual"]
    model_df["equity"] = (1.0 + model_df["trade_return"]).cumprod()
    model_df["label"] = f"{run_dir.name}:{model_name}"
    return model_df


def build_summary_rows(run_dirs: list[Path]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        with resolve_run_artifact(run_dir, "config.json", "core").open("r", encoding="utf-8") as f:
            config = json.load(f)
        with resolve_run_artifact(run_dir, "metrics.json", "core").open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        backtest_mode, summary = load_backtest_summary(run_dir, config["target_mode"])
        for model_name, split_metrics in metrics.items():
            test = split_metrics["test"]
            bt = summary.get(model_name, {})
            rows.append(
                {
                    "run_name": run_dir.name,
                    "target_mode": config["target_mode"],
                    "feature_columns": ",".join(config["feature_columns"]),
                    "model": model_name,
                    "label": f"{run_dir.name}\n{model_name}",
                    "test_rel_score": float(test.get("rel_score", np.nan)),
                    "test_directional_accuracy": float(test.get("directional_accuracy", np.nan)),
                    "best_threshold": float(bt.get("threshold", np.nan)),
                    "holding_period": int(bt.get("holding_period", 1)),
                    "trade_count": int(bt.get("trade_count", 0)),
                    "coverage": float(bt.get("coverage", np.nan)),
                    "final_equity": float(bt.get("final_equity", np.nan)),
                    "backtest_mode": backtest_mode,
                }
            )
            if bt:
                curves.append(
                    build_equity_frame(
                        run_dir,
                        model_name,
                        float(bt["threshold"]),
                        int(bt.get("holding_period", 1)),
                        backtest_mode == "non_overlap",
                    )
                )
    return pd.DataFrame(rows), curves


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    x = np.arange(len(summary_df))
    axes[0].bar(x, summary_df["test_rel_score"], color="#1f77b4")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Test rel_score")
    axes[0].grid(True, axis="y", alpha=0.2)

    axes[1].bar(x, summary_df["test_directional_accuracy"], color="#ff7f0e")
    axes[1].axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Test direction")
    axes[1].grid(True, axis="y", alpha=0.2)

    axes[2].bar(x, summary_df["final_equity"], color="#2ca02c")
    axes[2].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Best-threshold equity")
    axes[2].grid(True, axis="y", alpha=0.2)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(summary_df["label"], rotation=45, ha="right")

    fig.suptitle("Run Comparison Summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_equity_curves(curves: list[pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    for curve_df in curves:
        ax.plot(curve_df["Date"], curve_df["equity"], linewidth=1.4, label=curve_df["label"].iloc[0])
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Best-Threshold Equity Curves")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_test_predictions(run_dirs: list[Path], output_path: Path) -> None:
    items: list[tuple[str, pd.DataFrame]] = []
    for run_dir in run_dirs:
        predictions = pd.read_csv(resolve_run_artifact(run_dir, "predictions.csv", "core"))
        predictions["Date"] = pd.to_datetime(predictions["Date"])
        for model_name in sorted(predictions["model"].unique()):
            split_df = predictions[(predictions["split"] == "test") & (predictions["model"] == model_name)].sort_values("Date")
            items.append((f"{run_dir.name}\n{model_name}", split_df))

    n = len(items)
    ncols = 2
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), sharex=False)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, (title, split_df) in zip(axes.flatten(), items):
        ax.plot(split_df["Date"], split_df["actual"], color="#1f77b4", linewidth=1.1, label="actual")
        ax.plot(split_df["Date"], split_df["prediction"], color="#ff7f0e", linewidth=1.1, alpha=0.9, label="prediction")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes.flatten()[len(items):]:
        ax.set_visible(False)

    fig.suptitle("Test Actual vs Prediction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df, curves = build_summary_rows(args.run_dirs)
    summary_df.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)
    plot_summary(summary_df, args.output_dir / f"{args.prefix}_summary.png")
    plot_equity_curves(curves, args.output_dir / f"{args.prefix}_equity.png")
    plot_test_predictions(args.run_dirs, args.output_dir / f"{args.prefix}_test_predictions.png")
    print(args.output_dir / f"{args.prefix}_summary.csv")
    print(args.output_dir / f"{args.prefix}_summary.png")
    print(args.output_dir / f"{args.prefix}_equity.png")
    print(args.output_dir / f"{args.prefix}_test_predictions.png")


if __name__ == "__main__":
    main()
