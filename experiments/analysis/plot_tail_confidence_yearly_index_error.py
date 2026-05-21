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
DEFAULT_GOLD_ROOT = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_confidence_lstm_ablation"
DEFAULT_OUTPUT = DEFAULT_GOLD_ROOT / "yearly_index_error"
DEFAULT_RUN_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "tail_confidence_lstm_ablation"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_VN30_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn30_symbols.csv"
DEFAULT_VN100_SYMBOLS = ROOT / "data" / "external" / "zInfo" / "data_info_vn" / "vn100_symbols.csv"


@dataclass(frozen=True)
class PredictionCase:
    label: str
    run_name: str
    calibration: str
    color: str


PREDICTION_CASES: tuple[PredictionCase, ...] = (
    PredictionCase("Base LSTM", "base52_loaded_no_finetune_e0", "identity", "#6b7280"),
    PredictionCase("Weighted + sign scale", "rel_weighted_finetune_base52_lr1e4_e8", "sign_split_grid", "#dc2626"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot yearly VN30/VN100 index proxy vs q90 absolute prediction error.")
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
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


def rebase_to_100(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    first_valid = clean.dropna()
    if first_valid.empty:
        return clean * np.nan
    return clean / first_valid.iloc[0] * 100.0


def build_index_proxy(data_path: Path, symbols: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(data_path, usecols=["Date", "code", "adjust"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = raw[raw["code"].isin(symbols)].copy()
    raw = raw.sort_values(["code", "Date"], kind="stable")
    raw["stock_return"] = raw.groupby("code", sort=False)["adjust"].pct_change()
    daily = (
        raw.groupby("Date", sort=True)
        .agg(index_proxy_return=("stock_return", "mean"), n_index_stocks=("code", "nunique"))
        .reset_index()
        .sort_values("Date", kind="stable")
    )
    daily["index_proxy"] = (1.0 + daily["index_proxy_return"].fillna(0.0)).cumprod()
    return daily


def load_scales(gold_root: Path, run_name: str, calibration: str) -> dict[str, float]:
    if calibration == "identity":
        return {"scale": 1.0}
    path = gold_root / "amplitude_calibration" / "amplitude_calibration_by_run.csv"
    frame = pd.read_csv(path)
    selected = frame[
        frame["run_name"].eq(run_name) & frame["calibration"].eq(calibration) & frame["scope"].eq("val_full")
    ]
    if selected.empty:
        raise ValueError(f"No calibration found for {run_name} / {calibration}")
    return {key: float(value) for key, value in json.loads(str(selected.iloc[0]["scales_json"])).items()}


def apply_calibration(prediction: np.ndarray, calibration: str, scales: dict[str, float]) -> np.ndarray:
    if calibration == "identity":
        return prediction
    if calibration == "sign_split_grid":
        return np.where(
            prediction >= 0.0,
            prediction * float(scales["positive_scale"]),
            prediction * float(scales["negative_scale"]),
        )
    raise ValueError(f"Unsupported calibration: {calibration}")


def read_case_predictions(
    run_root: Path,
    gold_root: Path,
    case: PredictionCase,
    split: str,
    symbols: set[str],
) -> pd.DataFrame:
    path = run_root / case.run_name / "reports" / "core" / "predictions.csv"
    frame = pd.read_csv(path, usecols=["Date", "code", "split", "model", "prediction", "actual"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    if split != "all":
        frame = frame[frame["split"].astype(str).eq(split)].copy()
    frame = frame[frame["code"].isin(symbols)].copy()
    models = set(frame["model"].astype(str).unique())
    model = "lstm" if "lstm" in models else sorted(model for model in models if model.startswith("lstm"))[0]
    frame = frame[frame["model"].astype(str).eq(model)].copy()
    scales = load_scales(gold_root, case.run_name, case.calibration)
    frame["calibrated_prediction"] = apply_calibration(
        frame["prediction"].to_numpy(dtype=float),
        case.calibration,
        scales,
    )
    frame["abs_error"] = (frame["actual"].astype(float) - frame["calibrated_prediction"].astype(float)).abs()
    daily = (
        frame.groupby("Date", sort=True)
        .agg(
            n_error_stocks=("code", "nunique"),
            q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))),
        )
        .reset_index()
    )
    daily["case"] = case.label
    daily["run_name"] = case.run_name
    daily["calibration"] = case.calibration
    return daily


def build_plot_frame(
    data_path: Path,
    run_root: Path,
    gold_root: Path,
    split: str,
    index_name: str,
) -> tuple[str, pd.DataFrame]:
    index_label, symbols = index_symbols(index_name)
    index_frame = build_index_proxy(data_path, symbols)
    case_frames = [
        read_case_predictions(run_root, gold_root, case, split, symbols)
        for case in PREDICTION_CASES
    ]
    merged = index_frame.copy()
    for daily in case_frames:
        case_label = str(daily["case"].iloc[0])
        slim = daily[["Date", "q90_abs_error", "n_error_stocks"]].rename(
            columns={
                "q90_abs_error": f"{case_label} q90_abs_error",
                "n_error_stocks": f"{case_label} n_error_stocks",
            }
        )
        merged = merged.merge(slim, on="Date", how="inner")
    merged = merged.sort_values("Date", kind="stable").reset_index(drop=True)
    merged["index_proxy_rebased"] = rebase_to_100(merged["index_proxy"])
    return index_label, merged


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
        }
    )


def plot_year_panel(frame: pd.DataFrame, output_path: Path, index_label: str, split: str) -> None:
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    years = sorted(work["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, max(3.6, 3.35 * n_rows)))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax1, year in zip(axes, years):
        part = work[work["year"].eq(year)].reset_index(drop=True).copy()
        part["index_proxy_year_rebased"] = rebase_to_100(part["index_proxy"])
        x = np.arange(len(part))
        ax1.plot(x, part["index_proxy_year_rebased"], color="#1f77b4", linewidth=1.45, label=index_label)
        ax1.set_title(str(year), loc="left", fontweight="bold", fontsize=10.5)
        ax1.set_ylabel(f"{index_label}=100")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        for case in PREDICTION_CASES:
            column = f"{case.label} q90_abs_error"
            ax2.plot(
                x,
                part[column],
                color=case.color,
                linestyle="--",
                linewidth=1.15,
                label=case.label,
                alpha=0.92,
            )
        ax2.axhline(0.035, color="#111827", linestyle=":", linewidth=0.9, alpha=0.65)
        ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax2.set_ylabel("q90 absolute error")
        ax1.set_xlabel("Trading day in year")
    for ax in axes[len(years) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=1.5, label=index_label),
        *[
            plt.Line2D([0], [0], color=case.color, lw=1.2, linestyle="--", label=case.label)
            for case in PREDICTION_CASES
        ],
        plt.Line2D([0], [0], color="#111827", lw=0.9, linestyle=":", label="3.5% threshold"),
    ]
    fig.suptitle(f"{index_label} vs q90(|actual return - predicted return|), {split} split by year", y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_individual_years(frame: pd.DataFrame, output_dir: Path, index_label: str, split: str) -> list[str]:
    by_year = output_dir / "by_year"
    by_year.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    written: list[str] = []
    for year in sorted(work["year"].unique()):
        part = work[work["year"].eq(year)].reset_index(drop=True).copy()
        part["index_proxy_year_rebased"] = rebase_to_100(part["index_proxy"])
        x = np.arange(len(part))
        fig, ax1 = plt.subplots(figsize=(10.8, 4.4))
        ax1.plot(x, part["index_proxy_year_rebased"], color="#1f77b4", linewidth=1.6, label=index_label)
        ax1.set_ylabel(f"{index_label}, rebased to 100")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        for case in PREDICTION_CASES:
            ax2.plot(
                x,
                part[f"{case.label} q90_abs_error"],
                color=case.color,
                linestyle="--",
                linewidth=1.25,
                label=case.label,
                alpha=0.95,
            )
        ax2.axhline(0.035, color="#111827", linestyle=":", linewidth=1.0, alpha=0.7, label="3.5% threshold")
        ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax2.set_ylabel("q90 absolute return error")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True, fontsize=9)
        ax1.set_title(f"{index_label} vs yearly q90 prediction error, {int(year)} ({split})")
        ax1.set_xlabel("Trading day in year")
        fig.tight_layout()
        file_name = f"{index_label.lower()}_{split}_q90_abs_error_{int(year)}.png"
        fig.savefig(by_year / file_name, bbox_inches="tight", dpi=180)
        plt.close(fig)
        written.append(f"by_year/{file_name}")
    return written


def summarize(frame: pd.DataFrame, index_label: str) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = work["Date"].dt.year
    rows: list[dict[str, object]] = []
    for year, group in work.groupby("year", sort=True):
        row: dict[str, object] = {
            "index": index_label,
            "year": int(year),
            "days": int(len(group)),
            "index_return": float(group["index_proxy"].iloc[-1] / group["index_proxy"].iloc[0] - 1.0),
        }
        for case in PREDICTION_CASES:
            values = group[f"{case.label} q90_abs_error"].astype(float)
            row[f"{case.label} median_q90_error"] = float(values.median())
            row[f"{case.label} p90_q90_error"] = float(values.quantile(0.90))
            row[f"{case.label} spike_rate_3p5"] = float((values >= 0.035).mean())
        row["delta_median_current_minus_base"] = (
            row["Weighted + sign scale median_q90_error"] - row["Base LSTM median_q90_error"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.to_numpy()]
        widths = [
            max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
            for idx, column in enumerate(columns)
        ]
        header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
        separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
        body = [
            "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
            for row in rows
        ]
        return "\n".join([header, separator, *body])


def write_summary(output_dir: Path, split: str, yearly: pd.DataFrame, files: list[str]) -> None:
    display = yearly.copy()
    for column in display.columns:
        if column not in {"index", "year", "days"}:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if isinstance(value, float) else value)
    lines = [
        "# Yearly Index vs q90 Error",
        "",
        f"Scope: `{split}` split. Index proxy is equal-weight from VN30/VN100 symbols and rebased inside each yearly plot.",
        "",
        "Formula:",
        "",
        "```text",
        "E_d = { actual_return_{i,d} - predicted_return_{i,d} }",
        "ts_error(d) = Q_0.90(|E_d|)",
        "```",
        "",
        "Cases:",
        "",
        "- `Base LSTM`: seed 52 base checkpoint.",
        "- `Weighted + sign scale`: seed 52 weighted fine-tune with train-fitted sign calibration.",
        "",
        "## Year Summary",
        "",
        markdown_table(display),
        "",
        "## Files",
        "",
        *[f"- `{file}`" for file in files],
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    all_yearly: list[pd.DataFrame] = []
    files: list[str] = []
    for index_name in ("vn100", "vn30"):
        index_label, frame = build_plot_frame(args.data, args.run_root, args.gold_root, args.split, index_name)
        index_dir = args.output_dir / index_label.lower()
        index_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(index_dir / "yearly_index_error.csv", index=False)
        panel_file = f"{index_label.lower()}_{args.split}_index_vs_q90_error_by_year.png"
        plot_year_panel(frame, index_dir / panel_file, index_label, args.split)
        files.append(f"{index_label.lower()}/{panel_file}")
        for file_name in plot_individual_years(frame, index_dir, index_label, args.split):
            files.append(f"{index_label.lower()}/{file_name}")
        all_yearly.append(summarize(frame, index_label))
    yearly = pd.concat(all_yearly, ignore_index=True)
    yearly.to_csv(args.output_dir / "year_summary.csv", index=False)
    write_summary(args.output_dir, args.split, yearly, files)
    print(json.dumps({"output_dir": str(args.output_dir), "files": files}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
