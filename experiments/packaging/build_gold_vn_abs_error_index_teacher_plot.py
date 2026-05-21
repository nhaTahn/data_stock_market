from __future__ import annotations

import argparse
import gzip
import json
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

DEFAULT_GOLD_DIR = ROOT / "gold" / "vn_transition_pressure_20260512"
DEFAULT_PREDICTIONS = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "filter_signal"
    / "portable_lstm_filter_signal_20260512_r02_no_leader_seed43"
    / "filter_predictions.csv.gz"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN30_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn30_symbols.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-style VN index proxy vs q90 absolute prediction error plot.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="teacher_style_abs_error_vn30_insample")
    parser.add_argument("--split", default="train", choices=["train", "val", "all"])
    parser.add_argument("--index-universe", default="vn30", choices=["vn30", "vn100"])
    parser.add_argument(
        "--error-universe",
        default="same_as_index",
        choices=["same_as_index", "all"],
        help="Use same index symbols for error quantile, or all model-predicted stocks.",
    )
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


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(
            handle,
            usecols=["split", "Date", "actual_date", "code", "actual_aligned", "base_prediction"],
        )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    return frame.sort_values(["split", "actual_date", "code"], kind="stable").reset_index(drop=True)


def build_daily_abs_error(
    predictions: pd.DataFrame,
    split: str,
    symbols: set[str] | None,
) -> pd.DataFrame:
    work = predictions.copy()
    if split != "all":
        work = work[work["split"].eq(split)].copy()
    if symbols is not None:
        work = work[work["code"].isin(symbols)].copy()
    work = work.dropna(subset=["actual_date", "actual_aligned", "base_prediction"])
    work["error_abs"] = (work["actual_aligned"].astype(float) - work["base_prediction"].astype(float)).abs()
    daily = (
        work.groupby("actual_date", sort=True)
        .agg(
            n_stocks=("code", "nunique"),
            q90_abs_error=("error_abs", lambda values: float(np.quantile(values, 0.90))),
        )
        .reset_index()
        .rename(columns={"actual_date": "Date"})
    )
    return daily


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


def plot_teacher_style(frame: pd.DataFrame, output_path: Path, index_label: str) -> None:
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
        label="q90(|E|) prediction error (right axis)",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{index_label} vs q90(|actual return - predicted return|), in-sample")
    ax1.set_xlabel("Trading days in in-sample period")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_teacher_style_by_year(frame: pd.DataFrame, output_path: Path, index_label: str) -> None:
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
    fig.suptitle(f"{index_label} vs q90 absolute prediction error by year, in-sample")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_single_year(part: pd.DataFrame, output_path: Path, index_label: str, year: int) -> None:
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
        label="q90(|E|) prediction error (right axis)",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("q90 absolute return error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"{index_label} vs q90(|actual return - predicted return|), {year}")
    ax1.set_xlabel("Trading day in year")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_individual_years(frame: pd.DataFrame, output_dir: Path, index_label: str) -> list[str]:
    by_year_dir = output_dir / "by_year"
    by_year_dir.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    written: list[str] = []
    for year in sorted(work["year"].unique()):
        part = work[work["year"].eq(year)].reset_index(drop=True)
        file_name = f"{index_label.lower()}_q90_abs_error_{int(year)}.png"
        plot_single_year(part, by_year_dir / file_name, index_label, int(year))
        written.append(f"by_year/{file_name}")
    return written


def write_summary(output_dir: Path, frame: pd.DataFrame, index_label: str, split: str, error_universe: str) -> None:
    lines = [
        "# Teacher-Style Absolute Error Plot",
        "",
        f"Scope: `{split}` split only. Holdout/test is not used.",
        f"Index line: `{index_label}`.",
        f"Error universe: `{error_universe}`.",
        "",
        "Formula:",
        "",
        "```text",
        "E_d = { actual_return_{i,d} - predicted_return_{i,d} }",
        "ts_error(d) = Q_0.90(|E_d|)",
        "```",
        "",
        "The plot intentionally keeps only two lines: index proxy and q90 absolute return error.",
        "",
        f"- Days: `{len(frame)}`",
        f"- Date range: `{frame['Date'].min().date()}` to `{frame['Date'].max().date()}`",
        f"- Median stocks/day used in error: `{frame['n_stocks'].median():.0f}`",
        f"- Median q90(|E|): `{frame['q90_abs_error'].median():.5f}`",
        "",
        "Files:",
        "",
        "- `teacher_style_index_vs_q90_abs_error.png`",
        "- `teacher_style_index_vs_q90_abs_error_by_year.png`",
        "- `by_year/*.png`",
        "- `teacher_style_abs_error.csv`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    index_label, symbols = index_symbols(args.index_universe)
    error_symbols = symbols if args.error_universe == "same_as_index" else None
    predictions = read_predictions(args.predictions)
    daily_error = build_daily_abs_error(predictions, args.split, error_symbols)
    index_frame = build_index_proxy(args.data, symbols)
    frame = daily_error.merge(index_frame, on="Date", how="inner").sort_values("Date", kind="stable").reset_index(drop=True)
    frame["index_proxy_rebased"] = rebase_to_100(frame["index_proxy"])
    frame.to_csv(output_dir / "teacher_style_abs_error.csv", index=False)

    plot_teacher_style(frame, output_dir / "teacher_style_index_vs_q90_abs_error.png", index_label)
    plot_teacher_style_by_year(frame, output_dir / "teacher_style_index_vs_q90_abs_error_by_year.png", index_label)
    year_files = plot_individual_years(frame, output_dir, index_label)
    write_summary(output_dir, frame, index_label, args.split, args.error_universe)

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "data": str(args.data),
                "split": args.split,
                "index_universe": args.index_universe,
                "error_universe": args.error_universe,
                "index_label": index_label,
                "error_formula": "q90(abs(actual_aligned - base_prediction)) by actual_date",
                "by_year_files": year_files,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "days": int(len(frame))}, indent=2))


if __name__ == "__main__":
    main()
