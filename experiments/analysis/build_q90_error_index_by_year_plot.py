"""Plot VN index proxy vs daily q90 absolute prediction error by year."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTIONS = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_key_report_plots_20260601/validation_predictions.csv"
DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
VN100_SYMBOLS = ROOT / "data/external/zInfo/data_info_vn/vn100_symbols.csv"
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/teacher_style_abs_error_vn100_insample"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/teacher_style_abs_error_vn100_insample"


def read_symbols(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    col = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[col].dropna().astype(str).str.upper())


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].sort_values(["code", "Date"], kind="stable").copy()
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = raw.groupby("Date", sort=True)["stock_return"].mean().rename("index_proxy_return").reset_index()
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def rebase(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return clean / clean.dropna().iloc[0] * 100.0


def build_daily_error(predictions_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path, parse_dates=["Date"])
    pred["abs_error"] = (pred["actual"].astype(float) - pred["prediction"].astype(float)).abs()
    daily = (
        pred.groupby("Date", sort=True)["abs_error"]
        .agg(
            n_stocks="count",
            q90_abs_error=lambda values: float(np.quantile(values, 0.90)),
            median_abs_error="median",
        )
        .reset_index()
    )
    return daily


def plot_by_year(frame: pd.DataFrame, output_path: Path) -> None:
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    years = sorted(work["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.5, 3.6 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    legend_handles = None
    for ax, year in zip(axes, years):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        x = np.arange(len(part))
        ax2 = ax.twinx()
        l1 = ax.plot(x, part["index_proxy_rebased"], color="#1f8bb6", linewidth=1.25, label="VN100")
        l2 = ax2.plot(x, part["q90_abs_error"] * 100, color="#e63946", linestyle="--", linewidth=1.05, alpha=0.9, label="q90(|E|)")
        ax.set_title(str(year), loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Trading day")
        ax.grid(True, alpha=0.22)
        ax.tick_params(axis="y", labelcolor="#1f8bb6")
        ax2.tick_params(axis="y", labelcolor="#e63946")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
        y2max = max(float((part["q90_abs_error"] * 100).max()) * 1.12, 3.5)
        y2min = max(0.0, float((part["q90_abs_error"] * 100).min()) * 0.85)
        ax2.set_ylim(y2min, y2max)
        if legend_handles is None:
            legend_handles = l1 + l2

    for ax in axes[len(years):]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper right", ncol=2, frameon=True)
    fig.suptitle("VN100 vs q90 absolute prediction error by year, validation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = read_symbols(VN100_SYMBOLS)
    daily_error = build_daily_error(PREDICTIONS)
    index = build_index_proxy(DATA, symbols)
    frame = daily_error.merge(index, on="Date", how="inner").sort_values("Date", kind="stable").reset_index(drop=True)
    frame["index_proxy_rebased"] = rebase(frame["index_proxy"])
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        frame.to_csv(output_dir / "vn100_vs_q90_abs_error_by_year_validation.csv", index=False)
        plot_by_year(frame, output_dir / "vn100_vs_q90_abs_error_by_year_validation.png")
    yearly = (
        frame.assign(year=frame["Date"].dt.year)
        .groupby("year")
        .agg(
            n_days=("Date", "nunique"),
            median_q90_abs_error=("q90_abs_error", "median"),
            p90_q90_abs_error=("q90_abs_error", lambda values: float(np.quantile(values, 0.90))),
            max_q90_abs_error=("q90_abs_error", "max"),
            median_n_stocks=("n_stocks", "median"),
        )
        .reset_index()
    )
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        yearly.to_csv(output_dir / "vn100_vs_q90_abs_error_by_year_validation_summary.csv", index=False)
    print(yearly.round(6).to_markdown(index=False))
    print(OUTPUT_DIR / "vn100_vs_q90_abs_error_by_year_validation.png")


if __name__ == "__main__":
    main()
