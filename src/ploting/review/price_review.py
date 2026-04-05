from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.model.config import DEFAULT_DATA_PATH, DEFAULT_PLOT_DIR, FEATURE_COLUMNS
from src.utils.features import ensure_columns


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["code", "Date"]).reset_index(drop=True)


def ensure_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "target_next_return" not in df.columns and "adjust" in df.columns:
        by_code = df.groupby("code", group_keys=False)
        next_adjust = by_code["adjust"].shift(-1)
        df["target_next_return"] = next_adjust / df["adjust"] - 1
    return df.replace([np.inf, -np.inf], np.nan)


def rank_features(df: pd.DataFrame) -> list[str]:
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature not in df.columns:
            continue
        sample = df[[feature, "target_next_return"]].dropna()
        if len(sample) < 50:
            continue
        corr_next_return = sample[feature].corr(sample["target_next_return"])
        rows.append((feature, abs(corr_next_return), corr_next_return))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def split_chunks(items, chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_price_panel(ax, sample: pd.DataFrame, stock: str, page_no: int, page_total: int) -> None:
    x = np.arange(len(sample))
    up_mask = sample["close"] >= sample["open"]
    colors = np.where(up_mask, "#00c2a8", "#ff4d6d")
    body_bottom = np.minimum(sample["open"], sample["close"])
    body_height = (sample["close"] - sample["open"]).abs()

    ax.vlines(x, sample["low"], sample["high"], color=colors, linewidth=0.8, alpha=0.9)
    ax.bar(x, body_height, bottom=body_bottom, color=colors, width=0.6, alpha=0.9)
    if "ma_5" in sample.columns:
        ax.plot(x, sample["ma_5"], color="#3da5ff", linewidth=1.2, label="ma_5")
    if "ma_20" in sample.columns:
        ax.plot(x, sample["ma_20"], color="#ff9800", linewidth=1.2, label="ma_20")
    ax.set_title(f"{stock} Price | page {page_no}/{page_total}")
    ax.legend(loc="upper left")
    style_axis(ax)


def plot_volume_panel(ax, sample: pd.DataFrame) -> None:
    x = np.arange(len(sample))
    up_mask = sample["close"] >= sample["open"]
    colors = np.where(up_mask, "#00c2a8", "#ff4d6d")
    if "volume_match" in sample.columns:
        ax.bar(x, sample["volume_match"], color=colors, width=0.8, alpha=0.5, label="volume")
    if "volume_ma_5" in sample.columns:
        ax.plot(x, sample["volume_ma_5"], color="#3da5ff", linewidth=1.2, label="volume_ma_5")
    if "volume_ma_20" in sample.columns:
        ax.plot(x, sample["volume_ma_20"], color="#ff9800", linewidth=1.2, label="volume_ma_20")
    ax.set_title("Volume With MA")
    ax.legend(loc="upper left")
    style_axis(ax)


def format_date_axis(ax, sample: pd.DataFrame) -> None:
    x = np.arange(len(sample))
    step = max(len(sample) // 8, 1)
    ticks = x[::step]
    labels = sample["Date"].dt.strftime("%Y-%m-%d").iloc[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))


def plot_feature_series(ax, sample: pd.DataFrame, feature: str, corr_next_return: float) -> None:
    series = sample[feature]
    std = series.std()
    scaled = (series - series.mean()) / (std if std and not np.isnan(std) else 1)
    ax.plot(np.arange(len(sample)), scaled, label=feature, color="#6c63ff")
    ax.set_title(f"{feature} | corr(next_return)={corr_next_return:.3f}")
    ax.legend(loc="upper left")
    style_axis(ax)


def plot_feature_decile(ax, sample: pd.DataFrame, feature: str) -> None:
    decile_sample = sample[[feature, "target_next_return"]].dropna().copy()
    if len(decile_sample) < 20:
        ax.set_title(f"{feature} decile next_return")
        style_axis(ax)
        return
    decile_sample["bucket"] = pd.qcut(decile_sample[feature], 10, duplicates="drop")
    grouped = decile_sample.groupby("bucket", observed=False)["target_next_return"].mean().reset_index(drop=True)
    x = np.arange(1, len(grouped) + 1)
    colors = np.where(grouped.values >= 0, "#00c2a8", "#ff4d6d")
    ax.bar(x, grouped.values, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{feature} decile -> next_return")
    ax.set_xlabel("decile")
    style_axis(ax)


def save_price_review_page(
    sample: pd.DataFrame,
    feature_chunk,
    output_path: Path,
    stock: str,
    page_no: int,
    page_total: int,
) -> None:
    nrows = 2 + max(len(feature_chunk), 1)
    fig = plt.figure(figsize=(14, 3.2 * nrows), constrained_layout=True)
    grid = GridSpec(nrows, 2, figure=fig, hspace=0.45, wspace=0.2)

    price_ax = fig.add_subplot(grid[0, :])
    volume_ax = fig.add_subplot(grid[1, :], sharex=price_ax)
    plot_price_panel(price_ax, sample, stock, page_no, page_total)
    plot_volume_panel(volume_ax, sample)

    if feature_chunk:
        bottom_axes = []
        for idx, (feature, _, corr_next_return) in enumerate(feature_chunk, start=2):
            left_ax = fig.add_subplot(grid[idx, 0], sharex=price_ax)
            right_ax = fig.add_subplot(grid[idx, 1])
            plot_feature_series(left_ax, sample, feature, corr_next_return)
            plot_feature_decile(right_ax, sample, feature)
            bottom_axes.append(left_ax)
    else:
        left_ax = fig.add_subplot(grid[2, 0], sharex=price_ax)
        right_ax = fig.add_subplot(grid[2, 1])
        left_ax.set_title("No feature available")
        right_ax.set_title("No feature available")
        style_axis(left_ax)
        style_axis(right_ax)
        bottom_axes = [left_ax]

    format_date_axis(bottom_axes[-1], sample)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_price_review(
    df: pd.DataFrame,
    output_dir: Path,
    stock: str,
    lookback: int,
    max_features: int | None,
    features_per_page: int,
) -> list[Path]:
    sample = df[df["code"] == stock].dropna(subset=["close", "adjust"]).tail(lookback).copy()
    if sample.empty:
        return []
    sample = ensure_forecast_columns(sample)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("review*.png"):
        old_file.unlink()

    ranked_features = rank_features(sample)
    if max_features is not None:
        ranked_features = ranked_features[:max_features]
    feature_chunks = list(split_chunks(ranked_features, features_per_page)) or [[]]
    page_total = len(feature_chunks)
    output_paths = []

    for page_idx, feature_chunk in enumerate(feature_chunks, start=1):
        page_output = output_dir / f"review_p{page_idx:02d}.png"
        save_price_review_page(sample, feature_chunk, page_output, stock, page_idx, page_total)
        output_paths.append(page_output)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create price review plots.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR / "price_review")
    parser.add_argument("--stock", default=None)
    parser.add_argument("--all-stocks", action="store_true")
    parser.add_argument("--all-tickers", action="store_true")
    parser.add_argument("--lookback", type=int, default=250)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--features-per-page", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = ensure_columns(load_data(args.data_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in args.output_dir.glob("*.png"):
        old_file.unlink()

    if args.all_stocks or args.all_tickers:
        for stock in sorted(df["code"].dropna().unique()):
            plot_price_review(
                df,
                args.output_dir / stock,
                stock,
                args.lookback,
                None if args.max_features <= 0 else args.max_features,
                args.features_per_page,
            )
        print("Saved plots to:", args.output_dir)
        return

    stock = args.stock or df["code"].value_counts().idxmax()
    saved_paths = plot_price_review(
        df,
        args.output_dir / stock,
        stock,
        args.lookback,
        None if args.max_features <= 0 else args.max_features,
        args.features_per_page,
    )
    if len(saved_paths) == 1:
        print("Saved plot to:", saved_paths[0])
    else:
        print("Saved plots to:", args.output_dir)


if __name__ == "__main__":
    main()
