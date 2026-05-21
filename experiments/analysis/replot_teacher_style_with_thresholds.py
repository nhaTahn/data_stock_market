"""Re-plot teacher-style q90(|E|) time-series with explicit thresholds and
violation annotations. Companion plots for the 2026-05-21 advisor report.

Reads:  gold/.../teacher_style_abs_error_vn100_insample/teacher_style_abs_error.csv

Outputs (under same plots/teacher_style_threshold_replot_20260521/):
  - q90_error_by_year_threshold.png   (single multi-panel figure, all years)
  - by_year/q90_error_<year>_threshold.png  (per-year zoomed plots)
  - violation_segments.csv            (consecutive violation segments)
  - per_year_summary.csv              (yearly statistics)
  - summary.md                        (markdown explanation)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "gold" / "vn_transition_pressure_20260512" / "plots"
    / "teacher_style_abs_error_vn100_insample" / "teacher_style_abs_error.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "gold" / "vn_transition_pressure_20260512" / "plots"
    / "teacher_style_threshold_replot_20260521"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-low", type=float, default=0.03)
    parser.add_argument("--threshold-high", type=float, default=0.035)
    parser.add_argument("--violation-threshold", type=float, default=0.035)
    return parser.parse_args(argv)


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["year"] = df["Date"].dt.year
    df["q90_abs_error_pct"] = df["q90_abs_error"] * 100.0
    return df


def find_violation_segments(df: pd.DataFrame, threshold: float, min_len: int = 2) -> pd.DataFrame:
    """Find consecutive runs of days with q90_abs_error > threshold."""
    above = df["q90_abs_error"] > threshold
    segments: list[dict[str, object]] = []
    in_seg = False
    seg_start = 0
    for idx, flag in enumerate(above.values):
        if flag and not in_seg:
            seg_start = idx
            in_seg = True
        elif not flag and in_seg:
            seg_end = idx - 1
            if seg_end - seg_start + 1 >= min_len:
                segments.append({
                    "year": int(df.iloc[seg_start]["year"]),
                    "start_date": df.iloc[seg_start]["Date"],
                    "end_date": df.iloc[seg_end]["Date"],
                    "n_days": int(seg_end - seg_start + 1),
                    "max_error": float(df.iloc[seg_start:seg_end+1]["q90_abs_error"].max()),
                    "median_error": float(df.iloc[seg_start:seg_end+1]["q90_abs_error"].median()),
                    "index_change": float(
                        df.iloc[seg_end]["index_proxy_rebased"] / df.iloc[seg_start]["index_proxy_rebased"] - 1.0
                    ),
                })
            in_seg = False
    if in_seg:
        seg_end = len(df) - 1
        if seg_end - seg_start + 1 >= min_len:
            segments.append({
                "year": int(df.iloc[seg_start]["year"]),
                "start_date": df.iloc[seg_start]["Date"],
                "end_date": df.iloc[seg_end]["Date"],
                "n_days": int(seg_end - seg_start + 1),
                "max_error": float(df.iloc[seg_start:seg_end+1]["q90_abs_error"].max()),
                "median_error": float(df.iloc[seg_start:seg_end+1]["q90_abs_error"].median()),
                "index_change": float(
                    df.iloc[seg_end]["index_proxy_rebased"] / df.iloc[seg_start]["index_proxy_rebased"] - 1.0
                ),
            })
    return pd.DataFrame(segments).sort_values(["n_days", "max_error"], ascending=[False, False]).reset_index(drop=True)


def yearly_summary(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in df.groupby("year"):
        row: dict[str, object] = {
            "year": int(year),
            "n_days": int(len(group)),
            "median_q90": float(group["q90_abs_error"].median()),
            "p90_q90": float(group["q90_abs_error"].quantile(0.90)),
            "max_q90": float(group["q90_abs_error"].max()),
            "index_year_change": float(
                group["index_proxy_rebased"].iloc[-1] / group["index_proxy_rebased"].iloc[0] - 1.0
            ),
        }
        for thr in thresholds:
            key = f"{int(round(thr * 1000))}"
            row[f"days_above_{key}bp"] = int((group["q90_abs_error"] > thr).sum())
            row[f"share_above_{key}bp"] = float((group["q90_abs_error"] > thr).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def plot_per_year(
    df_year: pd.DataFrame,
    threshold_low: float,
    threshold_high: float,
    output_path: Path,
) -> None:
    fig, ax_left = plt.subplots(figsize=(13, 5.5))
    ax_right = ax_left.twinx()
    # Left axis: VN100 index proxy (rebased)
    ax_left.plot(df_year["Date"], df_year["index_proxy_rebased"],
                 color="#1f77b4", linewidth=1.2, alpha=0.85, label="VN100 index proxy (rebased)")
    ax_left.set_ylabel("VN100 index proxy (rebased=100 at 2012-05-28)", color="#1f77b4")
    ax_left.tick_params(axis="y", labelcolor="#1f77b4")
    ax_left.spines["left"].set_color("#1f77b4")
    # Right axis: q90 abs error %
    error_pct = df_year["q90_abs_error"] * 100.0
    ax_right.plot(df_year["Date"], error_pct,
                  color="#d62728", linewidth=1.0, alpha=0.85, label=r"$Q_{0.90}(|E|)$ daily")
    above_high = df_year["q90_abs_error"] > threshold_high
    if above_high.any():
        ax_right.scatter(
            df_year.loc[above_high, "Date"],
            error_pct[above_high],
            color="#d62728", s=26, edgecolor="black", linewidth=0.4, zorder=5,
            label=f"Violations > {threshold_high*100:.1f}%",
        )
    # Threshold bands
    ax_right.axhspan(0, threshold_low * 100, color="#2ca02c", alpha=0.07)
    ax_right.axhspan(threshold_low * 100, threshold_high * 100, color="#ff7f0e", alpha=0.08)
    ax_right.axhspan(threshold_high * 100, max(error_pct.max() * 1.05, threshold_high * 100 * 1.5),
                     color="#d62728", alpha=0.08)
    ax_right.axhline(threshold_low * 100, color="#2ca02c", linewidth=1.0, linestyle="--",
                     label=f"target {threshold_low*100:.1f}%")
    ax_right.axhline(threshold_high * 100, color="#d62728", linewidth=1.0, linestyle="--",
                     label=f"violation {threshold_high*100:.1f}%")
    ax_right.set_ylabel(r"$Q_{0.90}(|\mathrm{actual}-\mathrm{prediction}|)$ [%]", color="#d62728")
    ax_right.tick_params(axis="y", labelcolor="#d62728")
    ax_right.spines["right"].set_color("#d62728")
    ax_right.set_ylim(0, max(error_pct.max() * 1.10, threshold_high * 100 * 1.5))
    # Title
    year = int(df_year["year"].iloc[0])
    n_violations = int(above_high.sum())
    median_error = float(df_year["q90_abs_error"].median()) * 100.0
    index_change = (df_year["index_proxy_rebased"].iloc[-1] / df_year["index_proxy_rebased"].iloc[0] - 1.0) * 100.0
    title = (
        f"VN100 index vs daily $Q_{{0.90}}(|E|)$, year {year}\n"
        f"days={len(df_year)}, median $Q_{{0.90}}(|E|)$={median_error:.2f}%, "
        f"violations >{threshold_high*100:.1f}%={n_violations}, index Δ={index_change:+.1f}%"
    )
    ax_left.set_title(title, fontsize=11)
    # Legend combined
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right,
                   loc="upper left", fontsize=8, framealpha=0.9)
    # X-axis formatting
    ax_left.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax_left.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_all_years_grid(
    df: pd.DataFrame,
    threshold_low: float,
    threshold_high: float,
    output_path: Path,
) -> None:
    years = sorted(df["year"].unique())
    n_years = len(years)
    n_cols = 2
    n_rows = (n_years + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, year in enumerate(years):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax_right = ax.twinx()
        df_year = df[df["year"] == year].copy()
        ax.plot(df_year["Date"], df_year["index_proxy_rebased"],
                color="#1f77b4", linewidth=1.0, alpha=0.85)
        ax.set_ylabel("VN100", color="#1f77b4", fontsize=9)
        ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
        error_pct = df_year["q90_abs_error"] * 100.0
        ax_right.plot(df_year["Date"], error_pct,
                      color="#d62728", linewidth=0.9, alpha=0.85)
        above_high = df_year["q90_abs_error"] > threshold_high
        if above_high.any():
            ax_right.scatter(df_year.loc[above_high, "Date"], error_pct[above_high],
                             color="#d62728", s=15, edgecolor="black", linewidth=0.3, zorder=5)
        ax_right.axhline(threshold_low * 100, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_right.axhline(threshold_high * 100, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_right.set_ylim(0, max(error_pct.max() * 1.10, threshold_high * 100 * 1.5))
        ax_right.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
        n_v = int(above_high.sum())
        median_e = float(df_year["q90_abs_error"].median()) * 100.0
        ax.set_title(f"{year}: median Q90={median_e:.2f}%, violations>{threshold_high*100:.1f}%={n_v}",
                     fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.4)
    # Hide unused axes
    for idx in range(n_years, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")
    fig.suptitle(
        f"Daily $Q_{{0.90}}(|\\mathrm{{actual}}-\\mathrm{{prediction}}|)$ vs VN100 — by year (train split)\n"
        f"green dashed = target {threshold_low*100:.1f}%, red dashed = violation {threshold_high*100:.1f}%",
        fontsize=12, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_readout(
    output_dir: Path,
    df: pd.DataFrame,
    yearly: pd.DataFrame,
    segments: pd.DataFrame,
    threshold_low: float,
    threshold_high: float,
) -> None:
    median_overall = float(df["q90_abs_error"].median())
    p90_overall = float(df["q90_abs_error"].quantile(0.90))
    n_above = int((df["q90_abs_error"] > threshold_high).sum())
    n_total = int(len(df))
    share_above = n_above / max(n_total, 1)

    lines: list[str] = []
    lines.append(f"# Teacher-Style Q90(|E|) Re-Plot — Threshold {threshold_high*100:.1f}%")
    lines.append("")
    lines.append("Scope: VN100 train split only. Holdout/test is not used.")
    lines.append("")
    lines.append("## Mục Tiêu Đường Threshold")
    lines.append("")
    lines.append(f"- Mục tiêu: đường đỏ $Q_{{0.90}}(|E|)$ nằm dưới **{threshold_low*100:.1f}% (target)**.")
    lines.append(f"- Vi phạm: vượt **{threshold_high*100:.1f}% (violation line)** → bị flag để chẩn đoán.")
    lines.append(f"- Tổng quan: median = **{median_overall*100:.2f}%**, p90 = **{p90_overall*100:.2f}%**, violations = **{n_above}/{n_total}** ({share_above*100:.1f}%).")
    lines.append("")
    lines.append("## Yearly Summary")
    lines.append("")
    display = yearly.copy()
    for col in ["median_q90", "p90_q90", "max_q90", "share_above_30bp", "share_above_35bp", "index_year_change"]:
        if col in display.columns:
            display[col] = (display[col] * 100).map(lambda v: f"{v:.2f}%")
    show_cols = ["year", "n_days", "median_q90", "p90_q90", "max_q90",
                 "days_above_30bp", "days_above_35bp", "share_above_35bp", "index_year_change"]
    show_cols = [c for c in show_cols if c in display.columns]
    lines.append(display[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Top Violation Segments (≥ 2 ngày liên tiếp)")
    lines.append("")
    if len(segments) > 0:
        seg_disp = segments.head(20).copy()
        seg_disp["max_error"] = (seg_disp["max_error"] * 100).map(lambda v: f"{v:.2f}%")
        seg_disp["median_error"] = (seg_disp["median_error"] * 100).map(lambda v: f"{v:.2f}%")
        seg_disp["index_change"] = (seg_disp["index_change"] * 100).map(lambda v: f"{v:+.1f}%")
        seg_disp["start_date"] = pd.to_datetime(seg_disp["start_date"]).dt.strftime("%Y-%m-%d")
        seg_disp["end_date"] = pd.to_datetime(seg_disp["end_date"]).dt.strftime("%Y-%m-%d")
        lines.append(seg_disp.to_markdown(index=False))
    else:
        lines.append("_Không có segment vi phạm dài hơn 2 ngày._")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `q90_error_by_year_threshold.png` — multi-panel grid all years")
    lines.append("- `by_year/q90_error_<year>_threshold.png` — zoomed plot từng năm")
    lines.append("- `violation_segments.csv` — bảng segments")
    lines.append("- `per_year_summary.csv` — bảng tổng hợp theo năm")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_year_dir = args.output_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)
    df = load_panel(args.input)
    print(f"Loaded {len(df)} days from {df['Date'].min()} to {df['Date'].max()}")
    # Multi-panel grid
    plot_all_years_grid(df, args.threshold_low, args.threshold_high,
                        args.output_dir / "q90_error_by_year_threshold.png")
    # Per-year detailed
    for year, group in df.groupby("year"):
        out = by_year_dir / f"q90_error_{int(year)}_threshold.png"
        plot_per_year(group, args.threshold_low, args.threshold_high, out)
        print(f"  wrote {out.name}")
    # Tables
    yearly = yearly_summary(df, [args.threshold_low, args.threshold_high])
    yearly.to_csv(args.output_dir / "per_year_summary.csv", index=False)
    segments = find_violation_segments(df, args.violation_threshold, min_len=2)
    segments.to_csv(args.output_dir / "violation_segments.csv", index=False)
    # Readout
    write_readout(args.output_dir, df, yearly, segments, args.threshold_low, args.threshold_high)
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
