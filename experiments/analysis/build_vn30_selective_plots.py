"""Plot the selective prediction coverage-error frontier for dynamic VN30 panel LSTM."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREDICTIONS = ROOT / "data/processed/assets/data_info_vn/history/training_runs/vn30_dynamic_panel_tuned_5seed/reports/core/predictions.csv"
OUTPUT_DIR = ROOT / "gold/vn_transition_pressure_20260512/plots/vn30_dynamic_panel_tuned_5seed"
REPORT_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/vn30_dynamic_panel_tuned_5seed/reports/plots"


def robust_loss(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    base = robust_loss(actual[mask])
    return float(1.0 - robust_loss((actual - pred)[mask]) / base) if base > 0 else float("nan")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREDICTIONS)
    val = df[df["split"] == "val"].copy()
    wide = val.pivot_table(index=["code", "Date"], columns="model", values="prediction", aggfunc="first").reset_index()
    actuals = val.drop_duplicates(["code", "Date"])[["code", "Date", "actual"]]
    wide = actuals.merge(wide, on=["code", "Date"], how="inner")

    seed_cols = [c for c in wide.columns if str(c).startswith("panel_lstm_seed_")]
    wide["seed_std"] = wide[seed_cols].std(axis=1)

    coverages = np.linspace(0.50, 1.0, 51)
    results = []

    for cov in coverages:
        thr = wide["seed_std"].quantile(cov)
        sel = wide[wide["seed_std"] <= thr]
        err = sel["actual"] - sel["panel_lstm_ensemble"]
        rs = rel_score(sel["actual"].to_numpy(), sel["panel_lstm_ensemble"].to_numpy())
        q90 = np.quantile(np.abs(err), 0.90)
        sh35 = (np.abs(err) > 0.035).mean()
        results.append({
            "coverage": cov * 100,
            "rel_score": rs,
            "q90_error": q90 * 100,
            "share_gt_3p5": sh35 * 100
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "selective_frontier_data.csv", index=False)
    res_df.to_csv(REPORT_DIR / "selective_frontier_data.csv", index=False)

    # Plot frontier
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax2 = ax.twinx()

    l1 = ax.plot(res_df["coverage"], res_df["rel_score"], color="#059669", marker="o", markersize=4, linewidth=1.5, label="rel_score (Left)")
    l2 = ax2.plot(res_df["coverage"], res_df["q90_error"], color="#dc2626", marker="s", markersize=4, linestyle="--", linewidth=1.2, label="Q90 |E| (Right)")

    ax.set_title("Selective Prediction Coverage-Error Frontier (Validation)\nDynamic VN30 Panel LSTM Ensemble Gated by Seed Disagreement", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Coverage Level (%)", fontsize=10)
    ax.set_ylabel("rel_score", fontsize=10, color="#059669")
    ax2.set_ylabel("Q90 Absolute Error (%)", fontsize=10, color="#dc2626")

    ax.tick_params(axis="y", labelcolor="#059669")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}%"))

    # Align legends
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="lower left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "vn30_selective_frontier.png", dpi=200, bbox_inches="tight")
    fig.savefig(REPORT_DIR / "vn30_selective_frontier.png", dpi=200, bbox_inches="tight")

    print("SUCCESS: Selective frontier plot created successfully!")
    print(OUTPUT_DIR / "vn30_selective_frontier.png")


if __name__ == "__main__":
    main()
