from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "day_guard_error_control_20260520"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"
DEFAULT_POLICY = "risk_input_noise_q30__day_guard_selected_score_q90_q40"
DEFAULT_OUTPUT = (
    ROOT
    / "gold"
    / "vn_transition_pressure_20260512"
    / "plots"
    / "teacher_style_day_guard_lowspike_vn100_index_all_error_val"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-style plot for day-guard accepted rows.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-stocks-per-seed-day", type=int, default=5)
    return parser.parse_args(argv)


def read_symbols(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].copy().sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = raw.groupby("Date", sort=True)["stock_return"].mean().rename("index_proxy_return").reset_index()
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def rebase_to_100(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return clean / clean.dropna().iloc[0] * 100.0


def build_daily_error(source_dir: Path, policy: str, min_stocks: int) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_csv(source_dir / "day_guard_accepted_rows.csv", parse_dates=["Date"])
    frame = frame[frame["policy"].eq(policy)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for policy: {policy}")
    frame["error_abs"] = (frame["actual"].astype(float) - frame["prediction"].astype(float)).abs()
    seed_daily = (
        frame.groupby(["seed", "Date"], sort=True)
        .agg(
            n_stocks=("code", "nunique"),
            q90_abs_error=("error_abs", lambda values: float(np.quantile(values, 0.90))),
        )
        .reset_index()
    )
    seed_daily = seed_daily[seed_daily["n_stocks"].ge(min_stocks)].copy()
    daily = (
        seed_daily.groupby("Date", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            n_stocks=("n_stocks", "mean"),
            n_stocks_min=("n_stocks", "min"),
            q90_abs_error=("q90_abs_error", "mean"),
            q90_abs_error_min=("q90_abs_error", "min"),
            q90_abs_error_max=("q90_abs_error", "max"),
        )
        .reset_index()
    )
    return daily, sorted(int(seed) for seed in seed_daily["seed"].dropna().unique())


def plot_teacher_style(frame: pd.DataFrame, output_path: Path, policy: str) -> None:
    x = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(12.5, 5.2))
    ax1.plot(x, frame["index_proxy_rebased"], color="#1f77b4", linewidth=1.6, label="VN100 (left axis)")
    ax1.set_ylabel("VN100, rebased to 100")
    ax1.grid(True, alpha=0.22)
    years = frame["Date"].dt.year.to_numpy()
    ticks: list[int] = []
    labels: list[str] = []
    last_year: int | None = None
    for idx, year in enumerate(years):
        if last_year != int(year):
            ticks.append(idx)
            labels.append(str(year))
            last_year = int(year)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(x, frame["q90_abs_error"], color="#d62728", linestyle="--", linewidth=1.15, label="q90(|E|) accepted error (right axis)")
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"VN100 vs q90(|actual return - predicted return|), day-guard accepted\n{policy}")
    ax1.set_xlabel("Trading days in validation period")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_by_year(frame: pd.DataFrame, output_path: Path, policy: str) -> None:
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    years = sorted(work["year"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 6.4))
    axes = axes.reshape(-1)
    for ax1, year in zip(axes, years):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        x = np.arange(len(part))
        ax1.plot(x, part["index_proxy_rebased"], color="#1f77b4", linewidth=1.25)
        ax1.set_title(str(year), loc="left", fontsize=10, fontweight="bold")
        ax1.grid(True, alpha=0.18)
        ax2 = ax1.twinx()
        ax2.plot(x, part["q90_abs_error"], color="#d62728", linestyle="--", linewidth=1.0)
        ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax1.set_xlabel("Trading day")
    for ax in axes[len(years) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label="VN100"),
        plt.Line2D([0], [0], color="#d62728", lw=1.3, linestyle="--", label="q90(|E|)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(f"VN100 vs q90 accepted prediction error by year, day guard\n{policy}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    symbols = read_symbols(DEFAULT_VN100_SYMBOLS)
    daily, seeds = build_daily_error(args.source_dir, args.policy, args.min_stocks_per_seed_day)
    index_frame = build_index_proxy(args.data, symbols)
    frame = daily.merge(index_frame, on="Date", how="inner").sort_values("Date", kind="stable").reset_index(drop=True)
    frame["index_proxy_rebased"] = rebase_to_100(frame["index_proxy"])
    frame.to_csv(args.output_dir / "teacher_style_abs_error.csv", index=False)
    plot_teacher_style(frame, args.output_dir / "teacher_style_index_vs_q90_abs_error.png", args.policy)
    plot_by_year(frame, args.output_dir / "teacher_style_index_vs_q90_abs_error_by_year.png", args.policy)
    summary = [
        "# Teacher-Style Day-Guard Low-Spike Plot",
        "",
        f"Policy: `{args.policy}`.",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        f"- Days: `{len(frame)}`",
        f"- Date range: `{frame['Date'].min().date()}` to `{frame['Date'].max().date()}`",
        f"- Median accepted stocks/seed/day: `{frame['n_stocks'].median():.1f}`",
        f"- Median q90(|E|): `{frame['q90_abs_error'].median():.5f}`",
        f"- P90 q90(|E|): `{frame['q90_abs_error'].quantile(0.90):.5f}`",
        f"- Max q90(|E|): `{frame['q90_abs_error'].max():.5f}`",
        "",
        "Files:",
        "",
        "- `teacher_style_index_vs_q90_abs_error.png`",
        "- `teacher_style_index_vs_q90_abs_error_by_year.png`",
        "- `teacher_style_abs_error.csv`",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "policy": args.policy,
                "min_stocks_per_seed_day": args.min_stocks_per_seed_day,
                "seeds": seeds,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "days": int(len(frame))}, indent=2))


if __name__ == "__main__":
    main()
