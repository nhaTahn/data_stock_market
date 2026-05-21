from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_ROOT = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_confidence_lstm_ablation"
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


@dataclass(frozen=True)
class PlotRow:
    label: str
    candidate: str
    calibration: str


PLOT_ROWS: tuple[PlotRow, ...] = (
    PlotRow("Base LSTM", "base_loaded_no_finetune", "identity"),
    PlotRow("Base + sign scale", "base_loaded_no_finetune", "sign_split_grid"),
    PlotRow("Sharp FT", "rel_sharp_finetune", "identity"),
    PlotRow("Sharp + predmag scale", "rel_sharp_finetune", "predmag_bucket_grid"),
    PlotRow("Weighted FT", "rel_weighted_finetune", "identity"),
    PlotRow("Weighted + sign scale", "rel_weighted_finetune", "sign_split_grid"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot current tail-confidence LSTM evaluation artifacts.")
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_GOLD_ROOT / "current_evaluation_plots")
    parser.add_argument("--hist-seed", type=int, default=52)
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base_loss = robust_loss(actual)
    abs_loss = robust_loss(actual - prediction)
    return float(1.0 - abs_loss / base_loss) if np.isfinite(base_loss) and base_loss > 0.0 else float("nan")


def q90_abs_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.quantile(np.abs(actual - prediction), 0.90))


def pred_actual_abs_q90_ratio(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual_q90 = float(np.quantile(np.abs(actual), 0.90))
    return float(np.quantile(np.abs(prediction), 0.90) / actual_q90) if actual_q90 > 0.0 else float("nan")


def load_calibration_aggregate(gold_root: Path) -> pd.DataFrame:
    path = gold_root / "amplitude_calibration" / "amplitude_calibration_aggregate.csv"
    frame = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for plot_row in PLOT_ROWS:
        selected = frame[
            frame["candidate"].eq(plot_row.candidate) & frame["calibration"].eq(plot_row.calibration)
        ].iloc[0]
        rows.append(
            {
                "label": plot_row.label,
                "candidate": plot_row.candidate,
                "calibration": plot_row.calibration,
                "val_rel_score_mean": float(selected["val_rel_score_mean"]),
                "val_q90_abs_error_mean": float(selected["val_q90_abs_error_mean"]),
                "val_pred_actual_abs_q90_ratio_mean": float(selected["val_pred_actual_abs_q90_ratio_mean"]),
                "val_directional_accuracy_mean": float(selected["val_directional_accuracy_mean"]),
            }
        )
    return pd.DataFrame(rows)


def set_common_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


def plot_metric_panels(frame: pd.DataFrame, output_dir: Path) -> Path:
    colors = ["#6b7280", "#9ca3af", "#2563eb", "#60a5fa", "#dc2626", "#f87171"]
    metrics = [
        ("val_rel_score_mean", "Validation rel_score", "higher is better", "{:.4f}"),
        ("val_q90_abs_error_mean", "q90(|actual - predicted|)", "lower is better", "{:.4f}"),
        ("val_pred_actual_abs_q90_ratio_mean", "Prediction / actual abs q90", "higher = less conservative", "{:.1%}"),
        ("val_directional_accuracy_mean", "Directional accuracy", "higher is better", "{:.1%}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    labels = frame["label"].to_list()
    x = np.arange(len(labels))
    for ax, (column, title, subtitle, fmt) in zip(axes.ravel(), metrics):
        values = frame[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=colors, width=0.68)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.suptitle("Current LSTM Tail-Confidence Evaluation, Validation Mean Across 3 Seeds", fontsize=14, y=0.995)
    fig.tight_layout()
    path = output_dir / "current_metric_panels.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed_stability(gold_root: Path, output_dir: Path) -> Path:
    frame = pd.read_csv(gold_root / "tail_confidence_seed_sweep_comparison.csv")
    labels = {
        "base_loaded_no_finetune": "Base",
        "rel_sharp_finetune": "Sharp FT",
        "rel_weighted_finetune": "Weighted FT",
    }
    colors = {
        "base_loaded_no_finetune": "#6b7280",
        "rel_sharp_finetune": "#2563eb",
        "rel_weighted_finetune": "#dc2626",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for candidate, group in frame.groupby("candidate", sort=False):
        group = group.sort_values("seed")
        axes[0].plot(
            group["seed"],
            group["official_val_rel_score"],
            marker="o",
            linewidth=2,
            color=colors.get(candidate, None),
            label=labels.get(candidate, candidate),
        )
        axes[1].plot(
            group["seed"],
            group["val_q90_abs_error"],
            marker="o",
            linewidth=2,
            color=colors.get(candidate, None),
            label=labels.get(candidate, candidate),
        )
    axes[0].axhline(0.0, color="#111827", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_title("Validation rel_score by seed")
    axes[0].set_ylabel("rel_score")
    axes[1].set_title("Validation q90 absolute error by seed")
    axes[1].set_ylabel("q90(|E|)")
    for ax in axes:
        ax.set_xlabel("seed")
        ax.set_xticks(sorted(frame["seed"].unique()))
        ax.legend(frameon=False)
    fig.suptitle("Seed Stability Check", fontsize=14)
    fig.tight_layout()
    path = output_dir / "seed_stability.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def read_lstm_predictions(run_root: Path, run_name: str) -> pd.DataFrame:
    frame = pd.read_csv(run_root / run_name / "reports" / "core" / "predictions.csv")
    models = set(frame["model"].astype(str).unique())
    if "lstm" in models:
        chosen = "lstm"
    else:
        chosen = sorted(model for model in models if str(model).startswith("lstm"))[0]
    out = frame[frame["model"].astype(str).eq(chosen)].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def load_calibrated_prediction(gold_root: Path, run_root: Path, run_name: str, calibration: str) -> pd.DataFrame:
    frame = read_lstm_predictions(run_root, run_name)
    row = pd.read_csv(gold_root / "amplitude_calibration" / "amplitude_calibration_by_run.csv")
    selected = row[row["run_name"].eq(run_name) & row["calibration"].eq(calibration) & row["scope"].eq("val_full")].iloc[0]
    scales = json.loads(str(selected["scales_json"]))
    val = frame[frame["split"].astype(str).eq("val")].copy()
    prediction = val["prediction"].to_numpy(dtype=float).copy()
    if calibration == "sign_split_grid":
        positive_scale = float(scales["positive_scale"])
        negative_scale = float(scales["negative_scale"])
        prediction = np.where(prediction >= 0.0, prediction * positive_scale, prediction * negative_scale)
    elif calibration == "predmag_bucket_grid":
        thresholds = np.array([float(scales["threshold_1"]), float(scales["threshold_2"])])
        bucket = np.digitize(np.abs(prediction), thresholds, right=False)
        scale_values = np.array([float(scales["scale_low"]), float(scales["scale_mid"]), float(scales["scale_high"])])
        prediction = prediction * scale_values[bucket]
    elif calibration == "identity":
        pass
    else:
        raise ValueError(f"Unsupported calibration for plotting: {calibration}")
    val["plot_prediction"] = prediction
    return val


def plot_error_histograms(gold_root: Path, run_root: Path, output_dir: Path, seed: int) -> Path:
    cases = [
        ("Base LSTM", f"base{seed}_loaded_no_finetune_e0", "identity", "#6b7280"),
        ("Sharp FT", f"rel_sharp_finetune_base{seed}_lr1e4_e8", "identity", "#2563eb"),
        ("Weighted FT", f"rel_weighted_finetune_base{seed}_lr1e4_e8", "identity", "#dc2626"),
        ("Weighted + sign scale", f"rel_weighted_finetune_base{seed}_lr1e4_e8", "sign_split_grid", "#f97316"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.linspace(-0.09, 0.09, 73)
    for ax, (title, run_name, calibration, color) in zip(axes.ravel(), cases):
        frame = load_calibrated_prediction(gold_root, run_root, run_name, calibration)
        actual = frame["actual"].to_numpy(dtype=float)
        prediction = frame["plot_prediction"].to_numpy(dtype=float)
        error = actual - prediction
        ax.hist(error, bins=bins, color=color, alpha=0.72)
        ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
        ax.axvline(float(np.mean(error)), color="#facc15", linewidth=1.4)
        q10, q90 = np.quantile(error, [0.10, 0.90])
        ax.axvline(q10, color="#0f766e", linestyle=":", linewidth=1.4)
        ax.axvline(q90, color="#0f766e", linestyle=":", linewidth=1.4)
        ax.set_title(
            f"{title}\nrel={rel_score(actual, prediction):.4f}, q90|E|={q90_abs_error(actual, prediction):.4f}, "
            f"amp={pred_actual_abs_q90_ratio(actual, prediction):.1%}",
            fontsize=10.5,
        )
        ax.set_xlabel("E = actual return - predicted return")
        ax.set_ylabel("count")
    fig.suptitle(f"Validation Error Histograms, Seed {seed}", fontsize=14)
    fig.tight_layout()
    path = output_dir / f"error_histograms_seed{seed}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_daily_tail_error(gold_root: Path, output_dir: Path, seed: int) -> Path:
    cases = [
        ("Base LSTM", f"base{seed}_loaded_no_finetune_e0", "#6b7280"),
        ("Sharp FT", f"rel_sharp_finetune_base{seed}_lr1e4_e8", "#2563eb"),
        ("Weighted FT", f"rel_weighted_finetune_base{seed}_lr1e4_e8", "#dc2626"),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for label, artifact, color in cases:
        frame = pd.read_csv(gold_root / artifact / "daily_tail_error.csv")
        model = "lstm" if frame["model"].astype(str).eq("lstm").any() else sorted(frame["model"].unique())[0]
        val = frame[frame["split"].astype(str).eq("val") & frame["model"].astype(str).eq(model)].copy()
        val["eval_date"] = pd.to_datetime(val["eval_date"])
        ax.plot(val["eval_date"], val["daily_q90_abs_error"], color=color, linewidth=1.15, alpha=0.9, label=label)
    ax.axhline(0.035, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75, label="3.5% threshold")
    ax.set_title(f"Daily q90 absolute prediction error on validation, seed {seed}")
    ax.set_ylabel("q90(|actual - predicted|)")
    ax.set_xlabel("date")
    ax.legend(frameon=False, ncols=4, loc="upper left")
    fig.tight_layout()
    path = output_dir / f"daily_tail_error_seed{seed}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_index(paths: list[Path], output_dir: Path) -> None:
    lines = [
        "# Current Evaluation Plots",
        "",
        "Generated from tail-confidence seed sweep and amplitude-calibration artifacts.",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_common_style()
    paths = [
        plot_metric_panels(load_calibration_aggregate(args.gold_root), args.output_dir),
        plot_seed_stability(args.gold_root, args.output_dir),
        plot_error_histograms(args.gold_root, args.run_root, args.output_dir, args.hist_seed),
        plot_daily_tail_error(args.gold_root, args.output_dir, args.hist_seed),
    ]
    write_index(paths, args.output_dir)
    print(json.dumps({"plots": [str(path) for path in paths], "index": str(args.output_dir / "README.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
