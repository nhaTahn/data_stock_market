from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
    parser = argparse.ArgumentParser(
        description="Plot daily cross-sectional prediction error quantiles with a VN market proxy index."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="daily_error_vnindex_proxy")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--prediction-column", default="base_prediction")
    parser.add_argument("--actual-column", default="actual_aligned")
    parser.add_argument("--index-universe", default="all", choices=["all", "vn30", "vn100"])
    parser.add_argument("--symbol-file", type=Path, default=None)
    return parser.parse_args(argv)


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(
            handle,
            usecols=["split", "Date", "actual_date", "code", "actual_aligned", "base_prediction"],
        )
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    return frame.sort_values(["split", "actual_date", "code"], kind="stable").reset_index(drop=True)


def build_daily_error_frame(
    predictions: pd.DataFrame,
    split: str,
    actual_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    work = predictions.copy()
    if split != "all":
        work = work[work["split"].eq(split)].copy()
    work = work.dropna(subset=["actual_date", "code", actual_column, prediction_column])
    work["error"] = work[actual_column].astype(float) - work[prediction_column].astype(float)
    work["abs_error"] = work["error"].abs()
    daily = (
        work.groupby("actual_date", sort=True)
        .agg(
            n_stocks=("code", "nunique"),
            error_q10=("error", lambda values: float(np.quantile(values, 0.10))),
            error_q50=("error", lambda values: float(np.quantile(values, 0.50))),
            error_q90=("error", lambda values: float(np.quantile(values, 0.90))),
            abs_error_q90=("abs_error", lambda values: float(np.quantile(values, 0.90))),
            error_mean=("error", "mean"),
        )
        .reset_index()
        .rename(columns={"actual_date": "Date"})
    )
    return daily


def read_symbol_file(path: Path) -> set[str]:
    frame = pd.read_csv(path)
    column = "symbol" if "symbol" in frame.columns else "code"
    return set(frame[column].dropna().astype(str).str.upper())


def resolve_index_symbols(index_universe: str, symbol_file: Path | None) -> tuple[str, set[str] | None]:
    if symbol_file is not None:
        return symbol_file.stem.upper(), read_symbol_file(symbol_file)
    if index_universe == "vn30":
        return "VN30 proxy", read_symbol_file(DEFAULT_VN30_SYMBOLS)
    if index_universe == "vn100":
        return "VN100 proxy", read_symbol_file(DEFAULT_VN100_SYMBOLS)
    return "VN all-stock proxy", None


def build_market_proxy_index(data_path: Path, symbols: set[str] | None = None) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    if symbols is not None:
        raw = raw[raw["code"].isin(symbols)].copy()
        if raw.empty:
            raise ValueError("No matching symbols found for selected index universe.")
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    market = (
        raw.groupby("Date", sort=True)["stock_return"]
        .mean()
        .rename("market_proxy_return_1")
        .reset_index()
        .sort_values("Date", kind="stable")
    )
    market["market_proxy_index"] = (1.0 + market["market_proxy_return_1"].fillna(0.0)).cumprod()
    return market


def rebase_index(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    first = clean.dropna().iloc[0]
    if not np.isfinite(first) or first == 0.0:
        return clean
    return clean / first * 100.0


def plot_2scale(
    frame: pd.DataFrame,
    left_series: dict[str, str],
    right_series: dict[str, str],
    output_path: Path,
    *,
    title: str,
    left_label: str,
    right_label: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(13.5, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_iter = iter(colors)
    for label, column in left_series.items():
        ax1.plot(frame["Date"], frame[column], color=next(color_iter), linewidth=1.6, label=label)
    ax1.grid(True, alpha=0.22)
    ax1.set_ylabel(left_label)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    ax2 = ax1.twinx()
    for label, column in right_series.items():
        ax2.plot(
            frame["Date"],
            frame[column],
            color=next(color_iter),
            linestyle="--",
            linewidth=1.45,
            label=label,
        )
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel(right_label)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_index_with_abs_error(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    index_label: str,
    title: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(13.5, 5.8))
    ax1.plot(
        frame["Date"],
        frame["market_proxy_index_rebased"],
        color="#1f77b4",
        linewidth=1.7,
        label=index_label,
    )
    ax1.set_ylabel(f"{index_label}, rebased to 100")
    ax1.grid(True, alpha=0.22)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    ax2 = ax1.twinx()
    ax2.plot(
        frame["Date"],
        frame["abs_error_q90"],
        color="#d62728",
        linestyle="--",
        linewidth=1.35,
        alpha=0.90,
        label="q90(|actual - prediction|)",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel("Daily q90 absolute return error")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_index_with_abs_error_ma(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    index_label: str,
    title: str,
    window: int = 20,
) -> None:
    work = frame.copy()
    work["abs_error_q90_ma"] = work["abs_error_q90"].rolling(window, min_periods=max(5, window // 4)).mean()

    fig, ax1 = plt.subplots(figsize=(13.5, 5.8))
    ax1.plot(
        work["Date"],
        work["market_proxy_index_rebased"],
        color="#1f77b4",
        linewidth=1.8,
        label=index_label,
    )
    ax1.set_ylabel(f"{index_label}, rebased to 100")
    ax1.grid(True, alpha=0.22)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    ax2 = ax1.twinx()
    ax2.plot(
        work["Date"],
        work["abs_error_q90_ma"],
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        label=f"q90(|actual - prediction|) MA{window}",
    )
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax2.set_ylabel(f"Daily q90 absolute return error MA{window}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_same_scale_returns(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(
        frame["Date"],
        frame["market_proxy_return_1"],
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.55,
        label="Market proxy daily return",
    )
    ax.plot(
        frame["Date"],
        frame["error_q90"],
        color="#ff7f0e",
        linewidth=1.05,
        alpha=0.78,
        label="q90(actual - prediction)",
    )
    ax.plot(
        frame["Date"],
        frame["abs_error_q90"],
        color="#d62728",
        linewidth=1.05,
        alpha=0.72,
        label="q90(abs(actual - prediction))",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylabel("Return / error quantile")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_same_scale_rolling_returns(frame: pd.DataFrame, output_path: Path, *, title: str, window: int = 20) -> None:
    work = frame.copy()
    work["market_return_roll"] = work["market_proxy_return_1"].rolling(window, min_periods=max(5, window // 4)).mean()
    work["error_q90_roll"] = work["error_q90"].rolling(window, min_periods=max(5, window // 4)).mean()
    work["abs_error_q90_roll"] = work["abs_error_q90"].rolling(window, min_periods=max(5, window // 4)).mean()

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.plot(
        work["Date"],
        work["market_return_roll"],
        color="#1f77b4",
        linewidth=1.8,
        label=f"Market proxy return MA{window}",
    )
    ax.plot(
        work["Date"],
        work["error_q90_roll"],
        color="#ff7f0e",
        linewidth=1.8,
        label=f"q90(actual - prediction) MA{window}",
    )
    ax.plot(
        work["Date"],
        work["abs_error_q90_roll"],
        color="#d62728",
        linewidth=1.8,
        label=f"q90(abs error) MA{window}",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylabel("Rolling return / rolling error quantile")
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(output_dir: Path, frame: pd.DataFrame, split: str, index_label: str) -> None:
    corr_q90 = frame["market_proxy_return_1"].corr(frame["error_q90"])
    corr_abs_q90 = frame["market_proxy_return_1"].corr(frame["abs_error_q90"])
    lines = [
        "# Daily Error Quantile vs VN Market Proxy",
        "",
        f"Scope: `{split}` split from the current VN gold prediction artifact. Holdout/test is not used.",
        "",
        "Formula:",
        "",
        "```text",
        "E_d = { actual_{i,d} - prediction_{i,d} | all stocks i on day d }",
        "ts_error_q90(d) = Q_0.90(E_d)",
        "ts_abs_error_q90(d) = Q_0.90(|E_d|)",
        "```",
        "",
        "Index note: the current cleaned dataset does not contain an original official VNINDEX/VN30/VN100 "
        "level column. The plotted index is therefore an equal-weight proxy built from the selected symbol "
        "universe and rebased to 100 over the plotted period.",
        "",
        f"- Index line: `{index_label}`",
        f"- Days: `{len(frame)}`",
        f"- Date range: `{frame['Date'].min().date()}` to `{frame['Date'].max().date()}`",
        f"- Median stocks per day: `{frame['n_stocks'].median():.0f}`",
        f"- Median `q90(error)`: `{frame['error_q90'].median():+.5f}`",
        f"- Median `q90(abs_error)`: `{frame['abs_error_q90'].median():+.5f}`",
        f"- Corr market proxy return vs `q90(error)`: `{corr_q90:+.4f}`",
        f"- Corr market proxy return vs `q90(abs_error)`: `{corr_abs_q90:+.4f}`",
        "",
        "Files:",
        "",
        "- `daily_q90_error_vs_market_proxy.png`",
        "- `daily_abs_q90_error_vs_market_proxy.png`",
        "- `daily_error_quantile_band_vs_market_proxy.png`",
        "- `daily_error_vs_market_return_same_scale.png`",
        "- `daily_error_vs_market_return_same_scale_ma20.png`",
        "- `index_proxy_vs_q90_abs_error.png`",
        "- `index_proxy_vs_q90_abs_error_ma20.png`",
        "- `daily_error_vnindex_proxy.csv`",
        "",
        "Read: the same-scale return plots are the main diagnostic. The index-level plots are only a visual overlay "
        "for market context and should not be interpreted as same-unit series.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_predictions(args.predictions)
    index_label, index_symbols = resolve_index_symbols(args.index_universe, args.symbol_file)
    daily_error = build_daily_error_frame(
        predictions,
        split=args.split,
        actual_column=args.actual_column,
        prediction_column=args.prediction_column,
    )
    market = build_market_proxy_index(args.data, index_symbols)
    frame = daily_error.merge(market, on="Date", how="left").sort_values("Date", kind="stable")
    frame["market_proxy_index_rebased"] = rebase_index(frame["market_proxy_index"])
    frame.to_csv(output_dir / "daily_error_vnindex_proxy.csv", index=False)

    plot_2scale(
        frame,
        {index_label: "market_proxy_index_rebased"},
        {"q90(actual - prediction)": "error_q90"},
        output_dir / "daily_q90_error_vs_market_proxy.png",
        title="Daily cross-sectional q90 error vs VN market proxy index",
        left_label="Market proxy index, rebased to 100",
        right_label="Daily error quantile",
    )
    plot_2scale(
        frame,
        {index_label: "market_proxy_index_rebased"},
        {"q90(abs(actual - prediction))": "abs_error_q90"},
        output_dir / "daily_abs_q90_error_vs_market_proxy.png",
        title="Daily cross-sectional q90 absolute error vs VN market proxy index",
        left_label="Market proxy index, rebased to 100",
        right_label="Daily absolute error quantile",
    )
    plot_index_with_abs_error(
        frame,
        output_dir / "index_proxy_vs_q90_abs_error.png",
        index_label=index_label,
        title=f"{index_label} vs daily q90 absolute prediction error",
    )
    plot_index_with_abs_error_ma(
        frame,
        output_dir / "index_proxy_vs_q90_abs_error_ma20.png",
        index_label=index_label,
        title=f"{index_label} vs MA20 q90 absolute prediction error",
        window=20,
    )
    plot_2scale(
        frame,
        {"VN market proxy index": "market_proxy_index_rebased"},
        {
            "q10(error)": "error_q10",
            "q50(error)": "error_q50",
            "q90(error)": "error_q90",
        },
        output_dir / "daily_error_quantile_band_vs_market_proxy.png",
        title="Daily cross-sectional error quantiles vs VN market proxy index",
        left_label="Market proxy index, rebased to 100",
        right_label="Daily error quantile",
    )
    plot_same_scale_returns(
        frame,
        output_dir / "daily_error_vs_market_return_same_scale.png",
        title="Daily prediction error quantiles vs VN market proxy return (same return scale)",
    )
    plot_same_scale_rolling_returns(
        frame,
        output_dir / "daily_error_vs_market_return_same_scale_ma20.png",
        title="MA20 prediction error quantiles vs VN market proxy return (same return scale)",
        window=20,
    )
    write_summary(output_dir, frame, args.split, index_label)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "data": str(args.data),
                "output_dir": str(output_dir),
                "split": args.split,
                "prediction_column": args.prediction_column,
                "actual_column": args.actual_column,
                "error_formula": "actual - prediction",
                "market_proxy": "equal_weight_adjust_return_cumprod_rebased_100",
                "index_universe": args.index_universe,
                "index_label": index_label,
                "symbol_file": str(args.symbol_file) if args.symbol_file is not None else None,
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "days": int(len(frame))}, indent=2))


if __name__ == "__main__":
    main()
