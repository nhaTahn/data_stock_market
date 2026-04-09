from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


DEFAULT_BUNDLE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "gold"
    / "best_committee_fnb_20260408_235445"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create price-vs-predicted and return-error plots for a gold bundle.",
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--bins", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_data_path(bundle_dir: Path) -> Path:
    source_config_path = bundle_dir / "core" / "source_run_config.json"
    if source_config_path.exists():
        config = read_json(source_config_path)
        data_path = config.get("data_path")
        if data_path:
            return Path(data_path)
    raise FileNotFoundError(f"Could not resolve source data path from {source_config_path}")


def resolve_codes(bundle_dir: Path, merged_df: pd.DataFrame) -> list[str]:
    config = read_json(bundle_dir / "core" / "config.json")
    overlap_codes = config.get("committee_overlap_codes")
    if isinstance(overlap_codes, list) and overlap_codes:
        return [str(code) for code in overlap_codes]
    return sorted(merged_df["code"].dropna().astype(str).unique().tolist())


def load_merged_predictions(bundle_dir: Path, split: str) -> pd.DataFrame:
    predictions_path = bundle_dir / "core" / "predictions.csv"
    predictions_df = pd.read_csv(predictions_path, parse_dates=["Date"])
    filtered_predictions = predictions_df.loc[predictions_df["split"] == split].copy()
    if filtered_predictions.empty:
        raise ValueError(f"No rows found for split={split!r} in {predictions_path}")

    market_df = pd.read_csv(
        resolve_data_path(bundle_dir),
        usecols=["Date", "code", "close", "target_next_price"],
        parse_dates=["Date"],
    )

    merged_df = filtered_predictions.merge(
        market_df,
        on=["Date", "code"],
        how="left",
        validate="many_to_one",
    )
    merged_df["predicted_next_price"] = merged_df["close"] * (1.0 + merged_df["prediction"])
    merged_df["actual_next_price_from_return"] = merged_df["close"] * (1.0 + merged_df["actual"])
    merged_df["return_error"] = merged_df["prediction"] - merged_df["actual"]
    return merged_df.sort_values(["code", "Date"]).reset_index(drop=True)


def build_price_plot(bundle_dir: Path, merged_df: pd.DataFrame, codes: list[str], split: str) -> Path:
    n_codes = len(codes)
    ncols = 2 if n_codes > 1 else 1
    nrows = math.ceil(n_codes / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.5 * nrows), sharex=False)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    flat_axes = axes_array.ravel()

    for ax, code in zip(flat_axes, codes):
        code_df = merged_df.loc[merged_df["code"] == code].copy()
        if code_df.empty:
            ax.set_visible(False)
            continue
        actual_price = code_df["target_next_price"].fillna(code_df["actual_next_price_from_return"])
        ax.plot(code_df["Date"], actual_price, label="Actual next price", color="#1f77b4", linewidth=1.8)
        ax.plot(
            code_df["Date"],
            code_df["predicted_next_price"],
            label="Predicted next price",
            color="#d62728",
            linewidth=1.4,
            alpha=0.9,
        )
        mae = float(np.mean(np.abs(code_df["predicted_next_price"] - actual_price)))
        ax.set_title(f"{code} | MAE={mae:,.1f}")
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

    for ax in flat_axes[n_codes:]:
        ax.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Actual vs Predicted Next Price (from predicted return)", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = bundle_dir / "plots" / f"{split}_actual_vs_predicted_price_from_return.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_error_hist_plot(bundle_dir: Path, merged_df: pd.DataFrame, split: str, bins: int) -> Path:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    errors = merged_df["return_error"].dropna().to_numpy()
    axes[0].hist(errors, bins=bins, color="#4e79a7", alpha=0.85, edgecolor="white")
    axes[0].axvline(0.0, color="#d62728", linestyle="--", linewidth=1.4)
    axes[0].set_title("Overall Return Error Histogram")
    axes[0].set_xlabel("prediction - actual")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.2)

    code_groups = []
    code_labels = []
    for code, group in merged_df.groupby("code", sort=True):
        code_groups.append(group["return_error"].dropna().to_numpy())
        code_labels.append(code)
    axes[1].boxplot(code_groups, tick_labels=code_labels, showfliers=False)
    axes[1].axhline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
    axes[1].set_title("Return Error by Code")
    axes[1].set_xlabel("Code")
    axes[1].set_ylabel("prediction - actual")
    axes[1].grid(alpha=0.2)

    mean_error = float(np.mean(errors))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    fig.suptitle(
        f"Return Error Distribution | mean={mean_error:.5f} | mae={mae:.5f} | rmse={rmse:.5f}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    output_path = bundle_dir / "plots" / f"{split}_return_error_hist.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    merged_df = load_merged_predictions(args.bundle_dir, args.split)
    codes = resolve_codes(args.bundle_dir, merged_df)
    build_price_plot(args.bundle_dir, merged_df, codes, args.split)
    build_error_hist_plot(args.bundle_dir, merged_df, args.split, args.bins)
    print(f"Saved plots under {args.bundle_dir / 'plots'}")


if __name__ == "__main__":
    main()
