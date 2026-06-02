"""Build teacher-style signed error histogram for the BEST VN validation predictions.

Style matches the provided report figure:
E = prediction - actual, with mean/quantile markers and rel_score box.
Uses clipping to replicate the boundary spikes.
"""
from __future__ import annotations

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

PREDICTIONS = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_key_report_plots_20260601/validation_predictions.csv"
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/teacher_style_abs_error_vn100_insample"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/teacher_style_abs_error_vn100_insample"


def robust_loss(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def build_histogram(frame: pd.DataFrame, output_path: Path) -> dict[str, float]:
    actual = frame["actual"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    error = prediction - actual
    
    base_score = robust_loss(actual)
    abs_score = robust_loss(error)
    rel = rel_score(actual, prediction)
    
    stats = {
        "rel_score": rel,
        "base_score": base_score,
        "abs_score": abs_score,
        "min": float(np.min(error)),
        "max": float(np.max(error)),
        "q20": float(np.quantile(error, 0.20)),
        "q25": float(np.quantile(error, 0.25)),
        "q75": float(np.quantile(error, 0.75)),
        "q80": float(np.quantile(error, 0.80)),
        "mean": float(np.mean(error)),
    }

    # Replicating the exact style of the sample image
    fig, ax = plt.subplots(figsize=(15, 8.5))
    
    # 0.5th and 99.5th quantiles for clipping limits
    x_low = float(np.quantile(error, 0.005))
    x_high = float(np.quantile(error, 0.995))
    
    # Clip errors to replicate the boundary spikes
    clipped_error = np.clip(error, x_low, x_high)
    
    bins = np.linspace(x_low, x_high, 52)
    ax.hist(clipped_error, bins=bins, color="#4189c7", alpha=0.86, edgecolor="white", linewidth=0.8)

    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.3, label="E = 0")
    ax.axvline(stats["mean"], color="#d62728", linestyle="-", linewidth=1.5, label=f"mean={stats['mean']:.5f}")
    ax.axvline(stats["q20"], color="#17becf", linestyle=":", linewidth=1.3, label=f"q20={stats['q20']:.5f}")
    ax.axvline(stats["q25"], color="#2ca02c", linestyle=":", linewidth=1.3, label=f"q25={stats['q25']:.5f}")
    ax.axvline(stats["q75"], color="#7f7f99", linestyle=":", linewidth=1.3, label=f"q75={stats['q75']:.5f}")
    ax.axvline(stats["q80"], color="#bcbd22", linestyle=":", linewidth=1.3, label=f"q80={stats['q80']:.5f}")

    text = "\n".join([
        f"rel_score={stats['rel_score']:.5f}",
        f"base_score={stats['base_score']:.5f}",
        f"abs_score={stats['abs_score']:.5f}",
        f"min={stats['min']:.5f}  max={stats['max']:.5f}",
        f"q20={stats['q20']:.5f}  q25={stats['q25']:.5f}",
        f"q75={stats['q75']:.5f}  q80={stats['q80']:.5f}",
        f"mean={stats['mean']:.5f}",
    ])
    
    ax.text(
        0.985,
        0.985,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )

    ax.set_title("Histogram of E = prediction - actual (validation, all days, all stocks)", fontsize=14, pad=12)
    ax.set_xlabel("E = prediction - actual", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=11.5)
    
    # Leave a little padding on the sides
    ax.set_xlim(x_low - 0.008, x_high + 0.008)
    
    # Hide top and right spines to look clean
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    
    # Overwrite the validation histogram with the correct style
    stats = build_histogram(frame, OUTPUT_DIR / "histogram_prediction_minus_actual_validation.png")
    build_histogram(frame, REPORT_DIR / "histogram_prediction_minus_actual_validation.png")
    
    pd.DataFrame([stats]).to_csv(OUTPUT_DIR / "histogram_prediction_minus_actual_validation_stats.csv", index=False)
    pd.DataFrame([stats]).to_csv(REPORT_DIR / "histogram_prediction_minus_actual_validation_stats.csv", index=False)
    
    summary = "\n".join([
        "# Teacher-Style Signed Error Histogram",
        "",
        "Scope: VN validation only. Holdout/test is not used.",
        "",
        "Formula: `E = prediction - actual`.",
        "",
        f"- rel_score: `{stats['rel_score']:.5f}`",
        f"- base_score: `{stats['base_score']:.5f}`",
        f"- abs_score: `{stats['abs_score']:.5f}`",
        f"- mean error: `{stats['mean']:.5f}`",
        "",
        "Files:",
        "",
        "- `histogram_prediction_minus_actual_validation.png`",
        "- `histogram_prediction_minus_actual_validation_stats.csv`",
    ])
    (OUTPUT_DIR / "histogram_prediction_minus_actual_validation_summary.md").write_text(summary, encoding="utf-8")
    (REPORT_DIR / "histogram_prediction_minus_actual_validation_summary.md").write_text(summary, encoding="utf-8")
    print("Generated best model validation histogram with correct style successfully.")


if __name__ == "__main__":
    main()
