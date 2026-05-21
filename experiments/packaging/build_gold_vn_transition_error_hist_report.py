from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metric import evaluate  # noqa: E402


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


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    column: str
    title: str


CANDIDATES = (
    CandidateSpec("base_lstm", "prediction_base", "Base LSTM"),
    CandidateSpec(
        "lstm_filter_move_top_train_ic",
        "prediction_move_top_train_ic_selected",
        "Base LSTM + LSTM filter expected-move selector",
    ),
)

SPLIT_LABELS = {
    "train": "train",
    "val": "in_sample",
    "visible": "train_in_sample",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold VN transition pressure error histogram report.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-name", default="error_hist_report")
    return parser.parse_args(argv)


def read_predictions(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        frame = pd.read_csv(handle)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["actual_date"] = pd.to_datetime(frame["actual_date"])
    return frame.sort_values(["split", "code", "Date"], kind="stable").reset_index(drop=True)


def split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "visible":
        return frame[frame["split"].isin(["train", "val"])].copy()
    return frame[frame["split"].eq(split)].copy()


def grouped_metric_arrays(frame: pd.DataFrame, prediction_column: str) -> dict[str, np.ndarray | float | int]:
    parts: list[dict[str, np.ndarray | float | int]] = []
    for _, split_part in frame.groupby("split", sort=True):
        clean = split_part.dropna(subset=[prediction_column, "actual_aligned", "code"]).sort_values(
            ["code", "Date"], kind="stable"
        )
        if len(clean) < 10:
            continue
        metric = evaluate(
            clean[prediction_column].to_numpy(dtype=float),
            clean["actual_aligned"].to_numpy(dtype=float),
            group_ids=clean["code"].astype(str).to_numpy(),
        )
        parts.append(
            {
                "base": np.asarray(metric["base"], dtype=float),
                "error_actual_minus_prediction": np.asarray(metric["error"], dtype=float),
                "rel_score": float(metric["rel_score"]),
                "base_loss": float(metric["base_loss"]),
                "abs_loss": float(metric["abs_loss"]),
                "directional_accuracy": float(metric["directional_accuracy"]),
                "row_count": int(len(clean)),
                "code_count": int(clean["code"].astype(str).nunique()),
            }
        )
    if not parts:
        raise ValueError(f"No usable rows for {prediction_column}")

    base = np.concatenate([np.asarray(item["base"], dtype=float) for item in parts])
    error_actual_minus_prediction = np.concatenate(
        [np.asarray(item["error_actual_minus_prediction"], dtype=float) for item in parts]
    )
    base_loss = float(np.quantile(np.abs(base), 0.5) + 0.5 * np.quantile(np.abs(base), 0.9))
    abs_loss = float(
        np.quantile(np.abs(error_actual_minus_prediction), 0.5)
        + 0.5 * np.quantile(np.abs(error_actual_minus_prediction), 0.9)
    )
    rel_score = float(1.0 - abs_loss / base_loss) if base_loss > 0 else float("nan")
    return {
        "base": base,
        "error_prediction_minus_actual": -error_actual_minus_prediction,
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": rel_score,
        "directional_accuracy": float(np.average([item["directional_accuracy"] for item in parts])),
        "row_count": int(sum(int(item["row_count"]) for item in parts)),
        "code_count": int(max(int(item["code_count"]) for item in parts)),
    }


def summarize_errors(arrays: dict[str, np.ndarray | float | int], split_label: str, candidate: CandidateSpec) -> dict[str, object]:
    errors = np.asarray(arrays["error_prediction_minus_actual"], dtype=float)
    return {
        "candidate": candidate.label,
        "candidate_title": candidate.title,
        "split": split_label,
        "row_count": int(arrays["row_count"]),
        "aligned_rows": int(len(errors)),
        "code_count": int(arrays["code_count"]),
        "rel_score": float(arrays["rel_score"]),
        "base_score": float(arrays["base_loss"]),
        "abs_score": float(arrays["abs_loss"]),
        "directional_accuracy": float(arrays["directional_accuracy"]),
        "error_min": float(np.min(errors)),
        "error_q20": float(np.quantile(errors, 0.20)),
        "error_q25": float(np.quantile(errors, 0.25)),
        "error_median": float(np.quantile(errors, 0.50)),
        "error_q75": float(np.quantile(errors, 0.75)),
        "error_q80": float(np.quantile(errors, 0.80)),
        "error_max": float(np.max(errors)),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
    }


def save_error_histogram(errors: np.ndarray, summary: dict[str, object], output_path: Path) -> None:
    x_low = float(np.quantile(errors, 0.005))
    x_high = float(np.quantile(errors, 0.995))
    visible = errors[(errors >= x_low) & (errors <= x_high)]
    clipped_share = 1.0 - (len(visible) / len(errors)) if len(errors) else 0.0

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    bins = np.linspace(x_low, x_high, 52)
    ax.hist(visible, bins=bins, color="#1f77b4", alpha=0.86, edgecolor="white", linewidth=0.7)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="E = 0")
    ax.axvline(float(summary["error_mean"]), color="#d62728", linewidth=1.5, label=f"mean={summary['error_mean']:.5f}")
    ax.axvline(float(summary["error_q20"]), color="#17becf", linestyle=":", linewidth=1.35, label=f"q20={summary['error_q20']:.5f}")
    ax.axvline(float(summary["error_q25"]), color="#2ca02c", linestyle=":", linewidth=1.35, label=f"q25={summary['error_q25']:.5f}")
    ax.axvline(float(summary["error_q75"]), color="#9467bd", linestyle=":", linewidth=1.35, label=f"q75={summary['error_q75']:.5f}")
    ax.axvline(float(summary["error_q80"]), color="#bcbd22", linestyle=":", linewidth=1.35, label=f"q80={summary['error_q80']:.5f}")
    ax.set_title(
        f"Histogram of E = prediction - actual ({summary['split']}, all days, all VN stocks)\n"
        f"{summary['candidate_title']}"
    )
    ax.set_xlabel("E = prediction - actual")
    ax.set_ylabel("Frequency")
    ax.set_xlim(x_low, x_high)
    ax.grid(axis="y", alpha=0.18)
    stats_text = (
        f"rel_score={summary['rel_score']:.5f}\n"
        f"base_score={summary['base_score']:.5f}\n"
        f"abs_score={summary['abs_score']:.5f}\n"
        f"dir_acc={summary['directional_accuracy']:.2%}\n"
        f"stocks={summary['code_count']} rows={summary['aligned_rows']}\n"
        f"xlim=q0.5..q99.5 clipped={clipped_share:.1%}\n"
        f"min={summary['error_min']:.5f}  max={summary['error_max']:.5f}\n"
        f"q20={summary['error_q20']:.5f}  q25={summary['error_q25']:.5f}\n"
        f"q75={summary['error_q75']:.5f}  q80={summary['error_q80']:.5f}\n"
        f"mean={summary['error_mean']:.5f}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        fontsize=10.5,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown(output_dir: Path, summary: pd.DataFrame, plots: dict[tuple[str, str], str]) -> None:
    lines = [
        "# VN Gold Error Histogram Report",
        "",
        "Scope: train and in-sample validation only. Holdout/test is not used.",
        "",
        "E is defined as:",
        "",
        "```text",
        "E = prediction - actual",
        "```",
        "",
        "## Summary",
        "",
        "| Candidate | Split | Stocks | Rows | rel_score | base_score | abs_score | dir_acc | q20 | q80 | mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['candidate']}` | `{row['split']}` | {int(row['code_count'])} | {int(row['aligned_rows'])} | "
            f"{float(row['rel_score']):+.5f} | {float(row['base_score']):.5f} | {float(row['abs_score']):.5f} | "
            f"{float(row['directional_accuracy']):.2%} | {float(row['error_q20']):+.5f} | "
            f"{float(row['error_q80']):+.5f} | {float(row['error_mean']):+.5f} |"
        )
    lines.extend(["", "## Plots", ""])
    for _, row in summary.iterrows():
        key = (str(row["candidate"]), str(row["split"]))
        if key not in plots:
            continue
        lines.append(f"- `{row['candidate']}` `{row['split']}`: `{plots[key]}`")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.gold_dir / "plots" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_predictions(args.predictions)

    rows: list[dict[str, object]] = []
    plots: dict[tuple[str, str], str] = {}
    for candidate in CANDIDATES:
        if candidate.column not in frame.columns:
            raise ValueError(f"Missing prediction column: {candidate.column}")
        for split in ("train", "val", "visible"):
            work = split_frame(frame, split)
            split_label = SPLIT_LABELS[split]
            arrays = grouped_metric_arrays(work, candidate.column)
            summary = summarize_errors(arrays, split_label, candidate)
            errors = np.asarray(arrays["error_prediction_minus_actual"], dtype=float)
            filename = f"error_hist_{candidate.label}_{split_label}.png"
            save_error_histogram(errors, summary, output_dir / filename)
            rows.append(summary)
            plots[(candidate.label, split_label)] = filename

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "error_hist_summary.csv", index=False)
    write_markdown(output_dir, summary_df, plots)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "predictions": str(args.predictions),
                "output_dir": str(output_dir),
                "candidates": [candidate.__dict__ for candidate in CANDIDATES],
                "holdout_test_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "plots": len(plots)}, indent=2))


if __name__ == "__main__":
    main()
