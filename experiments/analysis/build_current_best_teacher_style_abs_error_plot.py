from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    / "selective_error_control_target3p0_20260520"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN30_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn30_symbols.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "gold"
    / "vn_transition_pressure_20260512"
    / "plots"
    / "teacher_style_best_error_control_vn100_val"
)


@dataclass(frozen=True)
class PolicySpec:
    score: str
    policy: str

    @property
    def label(self) -> str:
        return f"{self.score}_{self.policy}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-style index proxy vs q90 absolute prediction error for current best error-control policy.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--score", default="risk_hgb")
    parser.add_argument("--policy", default="coverage_q40")
    parser.add_argument("--index-universe", default="vn100", choices=["vn30", "vn100"])
    parser.add_argument(
        "--error-universe",
        default="same_as_index",
        choices=["same_as_index", "all"],
        help="Use same symbols as the index, or all accepted model rows for q90(|E|).",
    )
    parser.add_argument("--min-stocks-per-seed-day", type=int, default=5)
    return parser.parse_args(argv)


def read_symbol_file(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def index_symbols(name: str) -> tuple[str, set[str]]:
    if name == "vn30":
        return "VN30", read_symbol_file(DEFAULT_VN30_SYMBOLS)
    if name == "vn100":
        return "VN100", read_symbol_file(DEFAULT_VN100_SYMBOLS)
    raise ValueError(f"Unsupported index universe: {name}")


def threshold_for_seed(by_seed: pd.DataFrame, seed: int, spec: PolicySpec) -> float:
    row = by_seed[
        by_seed["seed"].eq(seed)
        & by_seed["score"].eq(spec.score)
        & by_seed["policy"].eq(spec.policy)
    ]
    if row.empty:
        raise ValueError(f"Missing threshold for seed={seed}, score={spec.score}, policy={spec.policy}")
    return float(row.iloc[0]["threshold"])


def build_seed_daily_error(
    source_dir: Path,
    spec: PolicySpec,
    symbols: set[str] | None,
    *,
    min_stocks: int,
) -> tuple[pd.DataFrame, list[int]]:
    by_seed = pd.read_csv(source_dir / "selective_error_by_seed.csv")
    seeds = sorted(int(seed) for seed in by_seed["seed"].dropna().unique())
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        threshold = threshold_for_seed(by_seed, seed, spec)
        frame = pd.read_csv(source_dir / f"val_selective_scores_seed_{seed}.csv", parse_dates=["Date"])
        frame["code"] = frame["code"].astype(str).str.upper()
        if symbols is not None:
            frame = frame[frame["code"].isin(symbols)].copy()
        frame = frame[frame[spec.score].astype(float).le(threshold)].copy()
        frame["error_abs"] = (frame["actual"].astype(float) - frame["prediction"].astype(float)).abs()
        daily = (
            frame.groupby("Date", sort=True)
            .agg(
                n_stocks=("code", "nunique"),
                q90_abs_error=("error_abs", lambda values: float(np.quantile(values, 0.90))),
            )
            .reset_index()
        )
        daily = daily[daily["n_stocks"].ge(min_stocks)].copy()
        daily["seed"] = seed
        daily["threshold"] = threshold
        parts.append(daily)
    if not parts:
        raise ValueError("No daily error rows were built.")
    return pd.concat(parts, ignore_index=True), seeds


def aggregate_seed_daily(seed_daily: pd.DataFrame) -> pd.DataFrame:
    return (
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


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].copy()
    if raw.empty:
        raise ValueError("No raw rows match selected index symbols.")
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = (
        raw.groupby("Date", sort=True)["stock_return"]
        .mean()
        .rename("index_proxy_return")
        .reset_index()
        .sort_values("Date", kind="stable")
    )
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def rebase_to_100(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    first = clean.dropna().iloc[0]
    return clean / first * 100.0


def add_year_ticks(ax: plt.Axes, dates: pd.Series) -> None:
    years = pd.to_datetime(dates).dt.year.to_numpy()
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    last_year: int | None = None
    for idx, year in enumerate(years):
        if last_year != int(year):
            tick_positions.append(idx)
            tick_labels.append(str(year))
            last_year = int(year)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def plot_teacher_style(frame: pd.DataFrame, output_path: Path, index_label: str, spec: PolicySpec) -> None:
    x = np.arange(len(frame))
    fig, ax1 = plt.subplots(figsize=(12.5, 5.2))
    index_color = "#1f77b4"
    error_color = "#d62728"
    ax1.plot(
        x,
        frame["index_proxy_rebased"],
        color=index_color,
        linewidth=1.6,
        label=f"{index_label} (left axis)",
    )
    ax1.set_ylabel(f"{index_label}, rebased to 100")
    ax1.grid(True, alpha=0.22)
    add_year_ticks(ax1, frame["Date"])

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        frame["q90_abs_error"],
        color=error_color,
        linestyle="--",
        linewidth=1.15,
        label="q90(|E|) accepted error (right axis)",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{index_label} vs q90(|actual return - predicted return|), validation accepted\n{spec.score}/{spec.policy}")
    ax1.set_xlabel("Trading days in validation period")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_teacher_style_by_year(frame: pd.DataFrame, output_path: Path, index_label: str, spec: PolicySpec) -> None:
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    years = sorted(work["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 3.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax1, year in zip(axes, years):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        x = np.arange(len(part))
        ax1.plot(x, part["index_proxy_rebased"], color="#1f77b4", linewidth=1.25, label=index_label)
        ax1.set_title(str(year), loc="left", fontsize=10, fontweight="bold")
        ax1.grid(True, alpha=0.18)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            part["q90_abs_error"],
            color="#d62728",
            linestyle="--",
            linewidth=1.0,
            label="q90(|E|)",
        )
        ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_xlabel("Trading day")
    for ax in axes[len(years) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label=index_label),
        plt.Line2D([0], [0], color="#d62728", lw=1.3, linestyle="--", label="q90(|E|)"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(f"{index_label} vs q90 accepted prediction error by year, validation\n{spec.score}/{spec.policy}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_single_year(part: pd.DataFrame, output_path: Path, index_label: str, spec: PolicySpec, year: int) -> None:
    x = np.arange(len(part))
    fig, ax1 = plt.subplots(figsize=(10.5, 4.2))
    ax1.plot(x, part["index_proxy_rebased"], color="#1f77b4", linewidth=1.6, label=f"{index_label} (left axis)")
    ax1.set_ylabel(f"{index_label}, rebased to 100")
    ax1.grid(True, alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        part["q90_abs_error"],
        color="#d62728",
        linestyle="--",
        linewidth=1.25,
        label="q90(|E|) accepted error (right axis)",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{index_label} vs q90(|actual return - predicted return|), {year}\n{spec.score}/{spec.policy}")
    ax1.set_xlabel("Trading day in year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_individual_years(frame: pd.DataFrame, output_dir: Path, index_label: str, spec: PolicySpec) -> list[str]:
    by_year_dir = output_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    written: list[str] = []
    for year in sorted(work["year"].unique()):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        file_name = f"{index_label.lower()}_accepted_q90_abs_error_{int(year)}.png"
        plot_single_year(part, by_year_dir / file_name, index_label, spec, int(year))
        written.append(f"by_year/{file_name}")
    return written


def write_summary(
    output_dir: Path,
    frame: pd.DataFrame,
    index_label: str,
    spec: PolicySpec,
    year_files: list[str],
    seeds: list[int],
    error_universe: str,
) -> None:
    lines = [
        "# Teacher-Style Current Best Error-Control Plot",
        "",
        "Scope: validation accepted samples only. Holdout/test is not used.",
        f"Policy: `{spec.score}/{spec.policy}`.",
        f"Index line: `{index_label}`.",
        f"Error universe: `{error_universe}`.",
        f"Seeds aggregated by daily mean q90: `{', '.join(str(seed) for seed in seeds)}`.",
        "",
        "Formula:",
        "",
        "```text",
        "E_d = { actual_return_{i,d} - predicted_return_{i,d} } for accepted stocks",
        "seed_error_s(d) = Q_0.90(|E_d|) within each seed s",
        "ts_error(d) = mean_s seed_error_s(d)",
        "```",
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
    for file_name in year_files:
        lines.append(f"- `{file_name}`")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = PolicySpec(args.score, args.policy)
    index_label, symbols = index_symbols(args.index_universe)
    error_symbols = symbols if args.error_universe == "same_as_index" else None
    seed_daily, seeds = build_seed_daily_error(args.source_dir, spec, error_symbols, min_stocks=args.min_stocks_per_seed_day)
    daily_error = aggregate_seed_daily(seed_daily)
    index_frame = build_index_proxy(args.data, symbols)
    frame = daily_error.merge(index_frame, on="Date", how="inner").sort_values("Date", kind="stable").reset_index(drop=True)
    frame["index_proxy_rebased"] = rebase_to_100(frame["index_proxy"])
    frame.to_csv(args.output_dir / "teacher_style_abs_error.csv", index=False)
    seed_daily.to_csv(args.output_dir / "teacher_style_abs_error_by_seed.csv", index=False)

    plot_teacher_style(frame, args.output_dir / "teacher_style_index_vs_q90_abs_error.png", index_label, spec)
    plot_teacher_style_by_year(frame, args.output_dir / "teacher_style_index_vs_q90_abs_error_by_year.png", index_label, spec)
    year_files = plot_individual_years(frame, args.output_dir, index_label, spec)
    write_summary(args.output_dir, frame, index_label, spec, year_files, seeds, args.error_universe)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_dir": str(args.source_dir),
                "data": str(args.data),
                "score": args.score,
                "policy": args.policy,
                "index_universe": args.index_universe,
                "error_universe": args.error_universe,
                "index_label": index_label,
                "min_stocks_per_seed_day": args.min_stocks_per_seed_day,
                "seeds": seeds,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "days": int(len(frame)), "index": index_label}, indent=2))


if __name__ == "__main__":
    main()
