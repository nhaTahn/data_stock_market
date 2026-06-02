"""Build train-ranked risk-gate coverage/error frontier for validation years.

This diagnostic keeps holdout/test closed. It evaluates stricter/lighter
lagged-market-risk gates on validation only and plots the trade-off between
accepted-day coverage and daily q90(|E|) tail error.
"""
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

from experiments.analysis.build_risk_gated_q90_error_plot import (  # noqa: E402
    OUTPUT_DIR,
    REPORT_DIR,
    add_train_rank_risk,
    build_daily_predictions,
    build_index_proxy,
)

THRESHOLD_GRID = tuple(np.round(np.arange(0.10, 0.71, 0.05), 2))
TARGET_MEDIAN = 0.030
TARGET_P90 = 0.040


def summarize_thresholds(val: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLD_GRID:
        work = val.copy()
        work["accepted"] = work["risk_score"] <= threshold
        work["year"] = work["Date"].dt.year
        for year, part in work.groupby("year", sort=True):
            accepted = part[part["accepted"]]
            rows.append(
                {
                    "threshold": float(threshold),
                    "year": int(year),
                    "n_days": int(part["Date"].nunique()),
                    "accepted_days": int(len(accepted)),
                    "coverage": float(len(accepted) / len(part)) if len(part) else np.nan,
                    "full_median_q90": float(part["q90_abs_error"].median()),
                    "accepted_median_q90": float(accepted["q90_abs_error"].median()) if len(accepted) else np.nan,
                    "accepted_p90_q90": float(accepted["q90_abs_error"].quantile(0.90)) if len(accepted) else np.nan,
                    "accepted_days_gt_3p5": int((accepted["q90_abs_error"] > 0.035).sum()),
                    "accepted_days_gt_5": int((accepted["q90_abs_error"] > 0.050).sum()),
                }
            )
        accepted_all = work[work["accepted"]]
        rows.append(
            {
                "threshold": float(threshold),
                "year": 0,
                "n_days": int(work["Date"].nunique()),
                "accepted_days": int(len(accepted_all)),
                "coverage": float(len(accepted_all) / len(work)) if len(work) else np.nan,
                "full_median_q90": float(work["q90_abs_error"].median()),
                "accepted_median_q90": float(accepted_all["q90_abs_error"].median()) if len(accepted_all) else np.nan,
                "accepted_p90_q90": float(accepted_all["q90_abs_error"].quantile(0.90)) if len(accepted_all) else np.nan,
                "accepted_days_gt_3p5": int((accepted_all["q90_abs_error"] > 0.035).sum()),
                "accepted_days_gt_5": int((accepted_all["q90_abs_error"] > 0.050).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(summary: pd.DataFrame) -> float:
    by_year = summary[summary["year"].ne(0)].copy()
    feasible: list[float] = []
    for threshold, part in by_year.groupby("threshold", sort=True):
        ok_median = bool((part["accepted_median_q90"] <= TARGET_MEDIAN).all())
        ok_p90 = bool((part["accepted_p90_q90"] <= TARGET_P90).all())
        has_coverage = bool((part["accepted_days"] >= 10).all())
        if ok_median and ok_p90 and has_coverage:
            feasible.append(float(threshold))
    if feasible:
        return max(feasible)
    safer = summary[summary["year"].eq(0)].dropna(subset=["accepted_p90_q90"])
    safer = safer.sort_values(["accepted_p90_q90", "accepted_median_q90", "threshold"])
    return float(safer.iloc[0]["threshold"])


def plot_frontier(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    years = [0] + sorted(int(year) for year in summary["year"].unique() if int(year) != 0)
    labels = {0: "All validation"}
    colors = {0: "#111827"}
    palette = ["#2563eb", "#16a34a", "#f97316", "#9333ea", "#dc2626"]
    for idx, year in enumerate(years):
        part = summary[summary["year"].eq(year)].sort_values("threshold")
        label = labels.get(year, str(year))
        color = colors.get(year, palette[(idx - 1) % len(palette)])
        axes[0].plot(part["coverage"] * 100, part["accepted_median_q90"] * 100, marker="o", linewidth=1.1, label=label, color=color)
        axes[1].plot(part["coverage"] * 100, part["accepted_p90_q90"] * 100, marker="o", linewidth=1.1, label=label, color=color)
    for ax, title, target in [(axes[0], "Median daily q90(|E|)", 3.0), (axes[1], "P90 of daily q90(|E|)", 4.0)]:
        ax.axhline(target, color="#991b1b", linestyle=":", linewidth=1.0)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Accepted-day coverage (%)")
        ax.set_ylabel("Error (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
        ax.grid(alpha=0.22)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Risk-gate coverage/error frontier — validation only", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_selected_gate(val: pd.DataFrame, index: pd.DataFrame, threshold: float, output_path: Path) -> pd.DataFrame:
    frame = val.merge(index[["Date", "index_proxy_rebased"]], on="Date", how="left").sort_values("Date")
    frame["accepted"] = frame["risk_score"] <= threshold
    frame["year"] = frame["Date"].dt.year
    years = sorted(frame["year"].unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.7 * len(years)), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, year in zip(axes, years):
        part = frame[frame["year"].eq(year)].reset_index(drop=True)
        x = np.arange(len(part))
        ax2 = ax.twinx()
        accepted = part["accepted"]
        ax.plot(x, part["index_proxy_rebased"], color="#1f8bb6", linewidth=1.25, label="VN100")
        ax2.plot(x, part["q90_abs_error"] * 100, color="#e63946", linestyle="--", linewidth=0.9, alpha=0.25, label="full q90(|E|)")
        ax2.scatter(x[accepted], part.loc[accepted, "q90_abs_error"] * 100, color="#e63946", s=15, alpha=0.95, label="accepted q90(|E|)")
        ax2.axhline(3.0, color="#991b1b", linestyle=":", linewidth=0.9)
        ax2.axhline(4.0, color="#7f1d1d", linestyle="-.", linewidth=0.8)
        accepted_part = part[accepted]
        med = accepted_part["q90_abs_error"].median() * 100 if len(accepted_part) else np.nan
        p90 = accepted_part["q90_abs_error"].quantile(0.90) * 100 if len(accepted_part) else np.nan
        ax.set_title(
            f"{year} accepted={int(accepted.sum())}/{len(part)} | med={med:.2f}%, p90={p90:.2f}%",
            loc="left",
            fontsize=9,
            fontweight="bold",
        )
        ax.grid(alpha=0.20)
        ax.set_xlabel("Trading day")
        ax.tick_params(axis="y", labelcolor="#1f8bb6")
        ax2.tick_params(axis="y", labelcolor="#e63946")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    fig.suptitle(f"Stricter risk-gated VN100 vs q90(|E|), threshold={threshold:.2f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train, val = build_daily_predictions()
    train, val = add_train_rank_risk(train, val)
    summary = summarize_thresholds(val)
    selected_threshold = choose_threshold(summary)
    index = build_index_proxy()
    for output_dir in [OUTPUT_DIR, REPORT_DIR]:
        summary.to_csv(output_dir / "risk_gate_coverage_error_frontier_validation.csv", index=False)
        plot_frontier(summary, output_dir / "risk_gate_coverage_error_frontier_validation.png")
        selected = plot_selected_gate(
            val,
            index,
            selected_threshold,
            output_dir / "risk_gate_selected_strict_vn100_q90_by_year_validation.png",
        )
        selected.to_csv(output_dir / "risk_gate_selected_strict_vn100_q90_by_year_validation.csv", index=False)
    print(f"selected_threshold={selected_threshold:.2f}")
    print(summary[summary["threshold"].eq(selected_threshold)].round(6).to_markdown(index=False))
    print(OUTPUT_DIR / "risk_gate_coverage_error_frontier_validation.png")
    print(OUTPUT_DIR / "risk_gate_selected_strict_vn100_q90_by_year_validation.png")


if __name__ == "__main__":
    main()
