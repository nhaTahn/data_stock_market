"""Build extended VN100 vs daily q90(|E|) by-year plot across train+validation artifacts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/teacher_style_abs_error_vn100_insample"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/teacher_style_abs_error_vn100_insample"
TRAIN_CSV = OUTPUT_DIR / "teacher_style_abs_error.csv"
VAL_CSV = OUTPUT_DIR / "vn100_vs_q90_abs_error_by_year_validation.csv"


def rebase(values: pd.Series, base: float = 100.0) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return clean / clean.dropna().iloc[0] * base


def load_combined() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"])
    train = train.rename(columns={"q90_abs_error": "q90_abs_error"})
    train["source"] = "train_artifact"
    train = train[["Date", "n_stocks", "q90_abs_error", "index_proxy_return", "index_proxy", "index_proxy_rebased", "source"]]

    val = pd.read_csv(VAL_CSV, parse_dates=["Date"])
    val = val.rename(columns={"n_obs": "n_stocks"}) if "n_obs" in val.columns else val
    if "index_proxy_rebased" not in val.columns:
        val["index_proxy_rebased"] = rebase(val["index_proxy"])
    val["source"] = "validation_best_meta"
    val = val[["Date", "n_stocks", "q90_abs_error", "index_proxy_return", "index_proxy", "index_proxy_rebased", "source"]]

    combined = pd.concat([train, val], ignore_index=True).sort_values("Date", kind="stable")
    combined = combined.drop_duplicates(["Date", "source"], keep="last").reset_index(drop=True)
    combined["year"] = combined["Date"].dt.year
    return combined


def plot_extended(frame: pd.DataFrame, output_path: Path) -> None:
    years = sorted(frame["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.45 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    legend_handles = None

    for ax, year in zip(axes, years):
        part = frame[frame["year"].eq(year)].sort_values("Date").reset_index(drop=True)
        if part.empty:
            ax.axis("off")
            continue
        x = np.arange(len(part))
        ax2 = ax.twinx()
        index_line = ax.plot(x, part["index_proxy_rebased"], color="#1f8bb6", linewidth=1.25, label="VN100")
        error_line = ax2.plot(x, part["q90_abs_error"] * 100, color="#e63946", linestyle="--", linewidth=1.0, alpha=0.9, label="q90(|E|)")
        above35 = part["q90_abs_error"].gt(0.035)
        above50 = part["q90_abs_error"].gt(0.050)
        if above35.any():
            ax2.scatter(x[above35.to_numpy()], part.loc[above35, "q90_abs_error"] * 100, s=13, color="#e63946", alpha=0.75, zorder=5)
        if above50.any():
            ax2.scatter(x[above50.to_numpy()], part.loc[above50, "q90_abs_error"] * 100, s=22, color="#7f1d1d", edgecolor="white", linewidth=0.4, zorder=6, label=">5% spike")
        ax2.axhline(3.5, color="#991b1b", linestyle=":", linewidth=0.8, alpha=0.75)
        ax2.axhline(5.0, color="#7f1d1d", linestyle="-.", linewidth=0.8, alpha=0.65)
        median_q90 = float(part["q90_abs_error"].median() * 100)
        p90_q90 = float(part["q90_abs_error"].quantile(0.90) * 100)
        n35 = int(above35.sum())
        n50 = int(above50.sum())
        idx_delta = float(part["index_proxy_rebased"].iloc[-1] / part["index_proxy_rebased"].iloc[0] - 1.0) * 100
        src = "+".join(sorted(part["source"].unique()))
        ax.set_title(
            f"{year} | med q90={median_q90:.2f}%, p90={p90_q90:.2f}%, >3.5%={n35}, >5%={n50}, VN100 Δ={idx_delta:+.1f}%",
            loc="left",
            fontsize=8.8,
            fontweight="bold",
        )
        ax.text(0.01, 0.04, src, transform=ax.transAxes, fontsize=7.5, color="#4b5563")
        ax.set_xlabel("Trading day")
        ax.grid(True, alpha=0.20)
        ax.tick_params(axis="y", labelcolor="#1f8bb6", labelsize=8)
        ax2.tick_params(axis="y", labelcolor="#e63946", labelsize=8)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
        y2max = max(float((part["q90_abs_error"] * 100).max()) * 1.12, 5.6)
        y2min = max(0.0, float((part["q90_abs_error"] * 100).min()) * 0.82)
        ax2.set_ylim(y2min, y2max)
        if legend_handles is None:
            legend_handles = index_line + error_line

    for ax in axes[len(years):]:
        ax.axis("off")
    if legend_handles is not None:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper right", ncol=2, frameon=True)
    fig.suptitle("VN100 vs daily q90(|actual - prediction|) by year — extended train + validation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_summary(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for year, part in frame.groupby("year"):
        above35 = part["q90_abs_error"].gt(0.035)
        above50 = part["q90_abs_error"].gt(0.050)
        idx_delta = float(part["index_proxy_rebased"].iloc[-1] / part["index_proxy_rebased"].iloc[0] - 1.0)
        rows.append({
            "year": int(year),
            "n_days": int(part["Date"].nunique()),
            "source": "+".join(sorted(part["source"].unique())),
            "median_q90_abs_error": float(part["q90_abs_error"].median()),
            "p90_q90_abs_error": float(part["q90_abs_error"].quantile(0.90)),
            "max_q90_abs_error": float(part["q90_abs_error"].max()),
            "days_gt_3p5": int(above35.sum()),
            "share_gt_3p5": float(above35.mean()),
            "days_gt_5": int(above50.sum()),
            "share_gt_5": float(above50.mean()),
            "vn100_delta": idx_delta,
        })
    summary = pd.DataFrame(rows).sort_values("year")
    summary.to_csv(output_dir / "extended_vn100_q90_abs_error_by_year_summary.csv", index=False)
    display = summary.copy()
    for col in ["median_q90_abs_error", "p90_q90_abs_error", "max_q90_abs_error", "share_gt_3p5", "share_gt_5", "vn100_delta"]:
        display[col] = (display[col] * 100).map(lambda value: f"{value:.2f}%")
    (output_dir / "extended_vn100_q90_abs_error_by_year_summary.md").write_text(
        "# Extended VN100 q90(|E|) by Year\n\n" + display.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_combined()
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        frame.to_csv(output_dir / "extended_vn100_q90_abs_error_by_year_series.csv", index=False)
        plot_extended(frame, output_dir / "extended_vn100_q90_abs_error_by_year.png")
        summary = write_summary(frame, output_dir)
    print(summary.round(6).to_markdown(index=False))
    print(OUTPUT_DIR / "extended_vn100_q90_abs_error_by_year.png")


if __name__ == "__main__":
    main()
