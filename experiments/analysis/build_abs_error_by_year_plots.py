"""Build absolute-error by-year plots for the current best VN validation model."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTIONS = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_key_report_plots_20260601/validation_predictions.csv"
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/teacher_style_abs_error_vn100_insample"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/teacher_style_abs_error_vn100_insample"


def daily_abs_error(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["year"] = work["Date"].dt.year
    work["abs_error"] = (work["prediction"].astype(float) - work["actual"].astype(float)).abs()
    daily = (
        work.groupby("Date", sort=True)["abs_error"]
        .agg(
            n_obs="count",
            median_abs_error="median",
            mean_abs_error="mean",
            q75_abs_error=lambda values: float(np.quantile(values, 0.75)),
            q90_abs_error=lambda values: float(np.quantile(values, 0.90)),
        )
        .reset_index()
    )
    daily["year"] = daily["Date"].dt.year
    return daily


def plot_daily_by_year(daily: pd.DataFrame, output_path: Path) -> None:
    years = sorted(daily["year"].unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.7 * len(years)), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, year in zip(axes, years):
        part = daily[daily["year"].eq(year)].copy()
        ax.plot(part["Date"], part["median_abs_error"] * 100, color="#f59e0b", linewidth=1.2, label="Median |E|")
        ax.plot(part["Date"], part["q75_abs_error"] * 100, color="#7c3aed", linewidth=1.0, alpha=0.8, label="Q75 |E|")
        ax.plot(part["Date"], part["q90_abs_error"] * 100, color="#dc2626", linewidth=1.0, alpha=0.75, label="Q90 |E|")
        ax.axhline(3.0, color="#059669", linestyle="--", linewidth=1.0, label="3.0% target")
        ax.axhline(3.5, color="#991b1b", linestyle=":", linewidth=1.0, label="3.5% line")
        median_q50 = float(part["median_abs_error"].median() * 100)
        median_q90 = float(part["q90_abs_error"].median() * 100)
        days_gt_35 = int((part["q90_abs_error"] > 0.035).sum())
        ax.set_title(f"{year}: median daily |E|={median_q50:.2f}%, median daily Q90={median_q90:.2f}%, Q90 days>3.5%={days_gt_35}", fontsize=10)
        ax.set_ylabel("Abs error (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left", ncol=5, fontsize=9)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily Absolute Return Error by Year — Best VN Validation Meta-Ensemble", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_hist_by_year(frame: pd.DataFrame, output_path: Path) -> None:
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["year"] = work["Date"].dt.year
    work["abs_error"] = (work["prediction"].astype(float) - work["actual"].astype(float)).abs()
    years = sorted(work["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    x_high = float(np.quantile(work["abs_error"], 0.995))
    bins = np.linspace(0, x_high, 44)
    for ax, year in zip(axes, years):
        part = work[work["year"].eq(year)]
        values = np.clip(part["abs_error"].to_numpy(float), 0, x_high)
        ax.hist(values * 100, bins=bins * 100, color="#4189c7", alpha=0.86, edgecolor="white", linewidth=0.75)
        q50 = float(np.quantile(part["abs_error"], 0.50) * 100)
        q75 = float(np.quantile(part["abs_error"], 0.75) * 100)
        q90 = float(np.quantile(part["abs_error"], 0.90) * 100)
        ax.axvline(q50, color="#f59e0b", linewidth=1.4, label=f"Q50={q50:.2f}%")
        ax.axvline(q75, color="#7c3aed", linestyle=":", linewidth=1.4, label=f"Q75={q75:.2f}%")
        ax.axvline(q90, color="#dc2626", linestyle=":", linewidth=1.4, label=f"Q90={q90:.2f}%")
        ax.axvline(3.0, color="#059669", linestyle="--", linewidth=1.0)
        ax.set_title(f"{year} | n={len(part):,}", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[len(years):]:
        ax.axis("off")
    fig.suptitle("Histogram of Absolute Error |prediction - actual| by Year", fontsize=14, fontweight="bold")
    fig.supxlabel("Absolute error (%)")
    fig.supylabel("Frequency")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    daily = daily_abs_error(frame)
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        daily.to_csv(output_dir / "daily_abs_error_by_year_validation.csv", index=False)
        plot_daily_by_year(daily, output_dir / "daily_abs_error_by_year_validation.png")
        plot_hist_by_year(frame, output_dir / "abs_error_histogram_by_year_validation.png")
    summary = daily.groupby("year").agg(
        n_days=("Date", "nunique"),
        median_daily_abs_error=("median_abs_error", "median"),
        p90_daily_abs_error=("median_abs_error", lambda values: float(np.quantile(values, 0.90))),
        median_daily_q90_abs_error=("q90_abs_error", "median"),
        days_q90_gt_3p5=("q90_abs_error", lambda values: int((values > 0.035).sum())),
    ).reset_index()
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        summary.to_csv(output_dir / "abs_error_by_year_summary_validation.csv", index=False)
        (output_dir / "abs_error_by_year_summary_validation.md").write_text(
            "# Absolute Error by Year — Validation\n\n" + summary.round(6).to_markdown(index=False) + "\n",
            encoding="utf-8",
        )
    print(summary.round(6).to_markdown(index=False))
    print(OUTPUT_DIR / "daily_abs_error_by_year_validation.png")
    print(OUTPUT_DIR / "abs_error_histogram_by_year_validation.png")


if __name__ == "__main__":
    main()
