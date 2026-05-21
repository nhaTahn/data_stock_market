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

PREFERRED_CODES = ("VCB", "FPT", "HPG", "SSI", "VIC", "VHM", "MWG", "MSN")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VN next-day return prediction plots for the gold report.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="next_day_prediction_plots")
    parser.add_argument("--min-codes-per-day", type=int, default=50)
    parser.add_argument("--timeseries-days", type=int, default=160)
    parser.add_argument("--per-code-days", type=int, default=252)
    return parser.parse_args(argv)


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    return frame.sort_values(["split", "Date", "code"], kind="stable").reset_index(drop=True)


def choose_latest_cross_section(frame: pd.DataFrame, min_codes: int) -> pd.Timestamp:
    val = frame[frame["split"].eq("val")]
    counts = val.groupby("Date")["code"].nunique()
    eligible = counts[counts >= min_codes]
    if eligible.empty:
        return pd.Timestamp(counts.idxmax())
    return pd.Timestamp(eligible.index.max())


def percent_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")


def robust_loss(values: pd.Series) -> float:
    clean = values.dropna().abs()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.5) + 0.5 * clean.quantile(0.9))


def directional_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    clean = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if clean.empty:
        return float("nan")
    return float((np.sign(clean["actual"]) == np.sign(clean["predicted"])).mean())


def save_all_code_bar(day: pd.DataFrame, output_path: Path) -> None:
    work = day.sort_values("base_prediction", ascending=True).copy()
    colors = np.where(work["base_prediction"].to_numpy(dtype=float) >= 0.0, "#2ca02c", "#d62728")
    fig_height = max(10.0, 0.18 * len(work))
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    ax.barh(work["code"].astype(str), work["base_prediction"], color=colors, alpha=0.86)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(
        f"Base LSTM next-day return prediction by VN code\n"
        f"signal date={work['Date'].iloc[0].date()} | actual date={work['actual_date'].iloc[0].date()} | codes={len(work)}"
    )
    ax.set_xlabel("Predicted next-day return")
    ax.set_ylabel("Code")
    percent_axis(ax)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_bottom_prediction_actual(day: pd.DataFrame, output_path: Path, n_each: int = 20) -> pd.DataFrame:
    low = day.nsmallest(n_each, "base_prediction")
    high = day.nlargest(n_each, "base_prediction")
    work = pd.concat([low, high], ignore_index=True).drop_duplicates("code").sort_values("base_prediction")
    y = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(13.5, max(8.0, 0.28 * len(work))))
    ax.barh(y, work["base_prediction"], color="#1f77b4", alpha=0.78, label="Base LSTM prediction")
    ax.scatter(work["actual_aligned"], y, color="#ff7f0e", s=30, label="Actual next-day return", zorder=3)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(work["code"].astype(str))
    ax.set_title(
        f"Base LSTM prediction vs actual next-day return (top/bottom signals)\n"
        f"signal date={work['Date'].iloc[0].date()} | actual date={work['actual_date'].iloc[0].date()}"
    )
    ax.set_xlabel("Return")
    ax.set_ylabel("Code")
    percent_axis(ax)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return work


def choose_timeseries_codes(frame: pd.DataFrame) -> list[str]:
    available = set(frame["code"].astype(str).unique())
    chosen = [code for code in PREFERRED_CODES if code in available]
    if len(chosen) >= 6:
        return chosen[:6]
    counts = frame.groupby("code")["Date"].count().sort_values(ascending=False)
    for code in counts.index.astype(str):
        if code not in chosen:
            chosen.append(code)
        if len(chosen) >= 6:
            break
    return chosen


def save_selected_timeseries(frame: pd.DataFrame, output_path: Path, days: int) -> list[str]:
    val = frame[frame["split"].eq("val")].copy()
    codes = choose_timeseries_codes(val)
    fig, axes = plt.subplots(len(codes), 1, figsize=(13.8, 2.45 * len(codes)), sharex=False)
    if len(codes) == 1:
        axes = [axes]
    for ax, code in zip(axes, codes):
        work = val[val["code"].astype(str).eq(code)].sort_values("Date").tail(days)
        ax.plot(work["actual_date"], work["actual_aligned"], color="#ff7f0e", linewidth=1.0, alpha=0.78, label="Actual")
        ax.plot(work["actual_date"], work["base_prediction"], color="#1f77b4", linewidth=1.15, alpha=0.92, label="Base LSTM pred")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(code, loc="left", fontsize=10.5, fontweight="bold")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle(f"Base LSTM predicted vs actual next-day return on selected VN codes (last {days} in-sample days)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return codes


def save_cross_section_scatter(frame: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    val = frame[frame["split"].eq("val")].dropna(subset=["base_prediction", "actual_aligned"]).copy()
    summary = (
        val.groupby("code", sort=True)
        .agg(
            mean_prediction=("base_prediction", "mean"),
            mean_actual=("actual_aligned", "mean"),
            prediction_std=("base_prediction", "std"),
            actual_std=("actual_aligned", "std"),
            rows=("Date", "count"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(summary["mean_prediction"], summary["mean_actual"], s=34, alpha=0.72, color="#1f77b4")
    for _, row in summary.iterrows():
        if abs(float(row["mean_prediction"])) >= summary["mean_prediction"].abs().quantile(0.88):
            ax.text(float(row["mean_prediction"]), float(row["mean_actual"]), str(row["code"]), fontsize=8)
    low = min(summary["mean_prediction"].min(), summary["mean_actual"].min())
    high = max(summary["mean_prediction"].max(), summary["mean_actual"].max())
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Per-code mean next-day return: Base LSTM prediction vs actual (in-sample)")
    ax.set_xlabel("Mean predicted next-day return")
    ax.set_ylabel("Mean actual next-day return")
    percent_axis(ax)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.2f}%")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summary


def build_per_code_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    val = frame[frame["split"].eq("val")].dropna(subset=["base_prediction", "actual_aligned"]).copy()
    rows: list[dict[str, float | int | str]] = []
    for code, work in val.groupby("code", sort=True):
        actual = work["actual_aligned"].astype(float)
        predicted = work["base_prediction"].astype(float)
        residual = predicted - actual
        base_score = robust_loss(actual)
        abs_score = robust_loss(residual)
        corr = actual.corr(predicted) if len(work) > 2 else np.nan
        rows.append(
            {
                "code": str(code),
                "rows": int(len(work)),
                "date_start": str(pd.Timestamp(work["Date"].min()).date()),
                "date_end": str(pd.Timestamp(work["Date"].max()).date()),
                "actual_mean": float(actual.mean()),
                "prediction_mean": float(predicted.mean()),
                "actual_std": float(actual.std()),
                "prediction_std": float(predicted.std()),
                "mae": float(residual.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "corr": float(corr) if pd.notna(corr) else np.nan,
                "dir_acc": directional_accuracy(actual, predicted),
                "rel_score": float(1.0 - abs_score / base_score) if base_score and pd.notna(base_score) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["corr", "rel_score"], ascending=[False, False]).reset_index(drop=True)


def save_timeseries_for_code(work: pd.DataFrame, output_path: Path, days: int) -> None:
    plot = work.sort_values("actual_date").tail(days).copy()
    residual = plot["base_prediction"] - plot["actual_aligned"]
    corr = plot["actual_aligned"].corr(plot["base_prediction"]) if len(plot) > 2 else np.nan
    dir_acc = directional_accuracy(plot["actual_aligned"], plot["base_prediction"])

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.8, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0]},
    )
    ax.plot(
        plot["actual_date"],
        plot["actual_aligned"],
        color="#d95f02",
        linewidth=1.0,
        alpha=0.72,
        label="Actual next-day return",
    )
    ax.plot(
        plot["actual_date"],
        plot["base_prediction"],
        color="#1b75bb",
        linewidth=1.25,
        alpha=0.94,
        label="Base LSTM prediction",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper right", ncol=2)

    ax2.fill_between(plot["actual_date"], residual, 0.0, color="#6b7280", alpha=0.32)
    ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax2.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax2.set_ylabel("Pred - actual")
    ax2.grid(axis="y", alpha=0.18)

    code = str(plot["code"].iloc[0])
    title = (
        f"{code}: Base LSTM prediction vs actual next-day return"
        f" | days={len(plot)} | corr={corr:+.3f}" if pd.notna(corr) else
        f"{code}: Base LSTM prediction vs actual next-day return | days={len(plot)}"
    )
    if pd.notna(dir_acc):
        title += f" | dir_acc={dir_acc * 100:.1f}%"
    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_per_code_timeseries(frame: pd.DataFrame, output_dir: Path, days: int) -> pd.DataFrame:
    val = frame[frame["split"].eq("val")].dropna(subset=["base_prediction", "actual_aligned"]).copy()
    per_code_dir = output_dir / "per_code_timeseries"
    per_code_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_per_code_metrics(frame)
    metrics.to_csv(output_dir / "per_code_timeseries_metrics.csv", index=False)
    for code, work in val.groupby("code", sort=True):
        save_timeseries_for_code(work, per_code_dir / f"{code}.png", days)
    return metrics


def save_timeseries_grid(
    frame: pd.DataFrame,
    codes: list[str],
    output_path: Path,
    days: int,
    title: str,
) -> None:
    val = frame[frame["split"].eq("val")].dropna(subset=["base_prediction", "actual_aligned"]).copy()
    n_cols = 3
    n_rows = int(np.ceil(len(codes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.5, 3.1 * n_rows), sharex=False)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, code in zip(axes, codes):
        work = val[val["code"].astype(str).eq(code)].sort_values("actual_date").tail(days)
        ax.plot(work["actual_date"], work["actual_aligned"], color="#d95f02", linewidth=0.9, alpha=0.68)
        ax.plot(work["actual_date"], work["base_prediction"], color="#1b75bb", linewidth=1.05, alpha=0.92)
        corr = work["actual_aligned"].corr(work["base_prediction"]) if len(work) > 2 else np.nan
        suffix = f" corr={corr:+.2f}" if pd.notna(corr) else ""
        ax.set_title(f"{code}{suffix}", loc="left", fontsize=10, fontweight="bold")
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.42)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
        ax.grid(axis="y", alpha=0.15)
    for ax in axes[len(codes) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="#d95f02", lw=1.2, label="Actual"),
        plt.Line2D([0], [0], color="#1b75bb", lw=1.2, label="Base LSTM pred"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=True)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown(
    output_dir: Path,
    signal_date: pd.Timestamp,
    actual_date: pd.Timestamp,
    codes: list[str],
    metrics: pd.DataFrame,
) -> None:
    top_corr = metrics.sort_values("corr", ascending=False).head(10)
    bottom_corr = metrics.sort_values("corr", ascending=True).head(10)
    lines = [
        "# VN Next-Day Return Prediction Plots",
        "",
        "Scope: VN train/in-sample artifact only. Holdout/test is not used.",
        "",
        f"- Latest cross-section signal date: `{signal_date.date()}`",
        f"- Actual next-day date: `{actual_date.date()}`",
        f"- Time-series sample codes: `{', '.join(codes)}`",
        "",
        "## Plots",
        "",
        "- `base_lstm_latest_all_codes.png`: Base LSTM predicted next-day return sorted across all VN codes on the latest eligible in-sample signal date.",
        "- `base_lstm_latest_top_bottom_prediction_actual.png`: Top/bottom Base LSTM signals with actual next-day return overlay.",
        "- `base_lstm_selected_codes_timeseries.png`: Base LSTM prediction vs actual return over time for representative VN codes.",
        "- `base_lstm_per_code_mean_prediction_vs_actual.png`: Per-code mean prediction vs mean actual return in-sample.",
        "- `base_lstm_top_corr_timeseries_grid.png`: Time-series overlay for the highest-correlation VN codes.",
        "- `base_lstm_core_codes_timeseries_grid.png`: Time-series overlay for core report codes.",
        "- `per_code_timeseries/`: one prediction-vs-actual time-series PNG per VN code.",
        "- `per_code_timeseries_metrics.csv`: per-code correlation, directional accuracy, MAE, RMSE, and rel_score.",
        "",
        "Read: this plot checks whether Base LSTM has usable forecasting signal. It is not a trading backtest; trading suitability is evaluated separately with net equity, Sharpe, drawdown, turnover, and cost.",
        "",
        "## Per-Code Signal Quality",
        "",
        f"- Codes with per-code plots: `{len(metrics)}`",
        f"- Median per-code correlation: `{metrics['corr'].median():+.4f}`",
        f"- Median per-code directional accuracy: `{metrics['dir_acc'].median() * 100:.2f}%`",
        "",
        "Top per-code correlations:",
        "",
        top_corr[["code", "corr", "dir_acc", "rel_score", "mae"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Lowest per-code correlations:",
        "",
        bottom_corr[["code", "corr", "dir_acc", "rel_score", "mae"]].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_predictions(args.predictions)
    signal_date = choose_latest_cross_section(frame, args.min_codes_per_day)
    day = frame[(frame["split"].eq("val")) & (frame["Date"].eq(signal_date))].dropna(
        subset=["base_prediction", "actual_aligned"]
    )
    if day.empty:
        raise ValueError(f"No val rows for signal date {signal_date}")
    actual_date = pd.Timestamp(day["actual_date"].iloc[0])

    save_all_code_bar(day, output_dir / "base_lstm_latest_all_codes.png")
    top_bottom = save_top_bottom_prediction_actual(day, output_dir / "base_lstm_latest_top_bottom_prediction_actual.png")
    top_bottom.to_csv(output_dir / "latest_top_bottom_prediction_actual.csv", index=False)
    codes = save_selected_timeseries(frame, output_dir / "base_lstm_selected_codes_timeseries.png", args.timeseries_days)
    code_summary = save_cross_section_scatter(frame, output_dir / "base_lstm_per_code_mean_prediction_vs_actual.png")
    code_summary.to_csv(output_dir / "per_code_mean_prediction_vs_actual.csv", index=False)
    per_code_metrics = save_per_code_timeseries(frame, output_dir, args.per_code_days)
    recent_val = (
        frame[frame["split"].eq("val")]
        .dropna(subset=["base_prediction", "actual_aligned"])
        .sort_values(["code", "actual_date"], kind="stable")
        .groupby("code", group_keys=False)
        .tail(args.per_code_days)
    )
    recent_per_code_metrics = build_per_code_metrics(recent_val)
    recent_per_code_metrics.to_csv(output_dir / "per_code_timeseries_recent_metrics.csv", index=False)
    top_corr_codes = recent_per_code_metrics.dropna(subset=["corr"]).head(12)["code"].astype(str).tolist()
    core_codes = choose_timeseries_codes(frame[frame["split"].eq("val")])
    save_timeseries_grid(
        frame,
        top_corr_codes,
        output_dir / "base_lstm_top_corr_timeseries_grid.png",
        args.per_code_days,
        f"Base LSTM prediction vs actual next-day return: highest-correlation VN codes (last {args.per_code_days} val days)",
    )
    save_timeseries_grid(
        frame,
        core_codes,
        output_dir / "base_lstm_core_codes_timeseries_grid.png",
        args.per_code_days,
        f"Base LSTM prediction vs actual next-day return: core VN codes (last {args.per_code_days} val days)",
    )

    write_markdown(output_dir, signal_date, actual_date, codes, per_code_metrics)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "output_dir": str(output_dir),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "actual_date": actual_date.strftime("%Y-%m-%d"),
                "codes": codes,
                "per_code_timeseries_days": args.per_code_days,
                "per_code_plot_count": int(len(per_code_metrics)),
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "signal_date": str(signal_date.date()), "codes": len(day)}, indent=2))


if __name__ == "__main__":
    main()
