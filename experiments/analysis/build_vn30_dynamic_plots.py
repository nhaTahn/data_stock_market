"""Build publication-quality plots for the dynamic VN30 panel run.

Outputs:
1. signed_error_density_vn30.png:
   - Signed error (prediction - actual) distribution of the ensemble.
2. vn30_error_envelope_by_year.png:
   - Multi-panel by-year plot showing rebased VN30 index proxy vs error envelope.
"""
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
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def plot_density(df: pd.DataFrame, output_path: Path) -> None:
    actual = df["actual"].to_numpy(dtype=float)
    prediction = df["prediction"].to_numpy(dtype=float)
    error = prediction - actual

    rel = rel_score(actual, prediction)
    base_s = robust_loss(actual)
    abs_s = robust_loss(error)

    mean_err = float(np.mean(error))
    q25 = float(np.quantile(error, 0.25))
    q75 = float(np.quantile(error, 0.75))

    x_low = float(np.quantile(error, 0.005))
    x_high = float(np.quantile(error, 0.995))
    clipped_error = np.clip(error, x_low, x_high)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)

    bins = np.linspace(x_low, x_high, 50)
    ax.hist(clipped_error, bins=bins, density=True, color="#10b981", alpha=0.8, edgecolor="white", linewidth=0.6, label="Error density")

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(error)
    x_grid = np.linspace(x_low, x_high, 300)
    ax.plot(x_grid, kde(x_grid), color="#047857", linewidth=1.8, label="KDE Fit")

    ax.axvline(0.0, color="#1e293b", linestyle="-", linewidth=1.1)
    ax.axvline(mean_err, color="#ef4444", linestyle="--", linewidth=1.2, label=f"Mean ({mean_err:+.4f})")
    ax.axvline(q25, color="#3b82f6", linestyle=":", linewidth=1.3, label=f"Q25 ({q25:+.4f})")
    ax.axvline(q75, color="#8b5cf6", linestyle=":", linewidth=1.3, label=f"Q75 ({q75:+.4f})")

    ax.set_title("VN30 Signed Error Distribution (E = prediction - actual)\nDynamic VN30 Panel LSTM Ensemble (Validation Split)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Return Error (E)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9)

    metrics_text = (
        f"Ensemble Metrics (Val)\n"
        f"rel_score  = {rel:.5f}\n"
        f"base_score = {base_s:.5f}\n"
        f"abs_score  = {abs_s:.5f}\n\n"
        f"Mean Err   = {mean_err:+.5f}\n"
        f"Q25 / Q75  = {q25:+.5f} / {q75:+.5f}"
    )

    ax.text(
        0.98,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.95,
        }
    )

    ax.set_xlim(x_low - 0.005, x_high + 0.005)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_envelope(df: pd.DataFrame, output_path: Path) -> None:
    # 1. Compute daily stats
    df["abs_error"] = (df["actual"] - df["prediction"]).abs()
    daily = (
        df.groupby("Date")
        .agg(
            median_abs_error=("abs_error", "median"),
            q90_abs_error=("abs_error", lambda s: float(np.quantile(s, 0.90))),
            index_return=("actual", "mean")
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    
    # Rebase index proxy
    daily["index_proxy"] = (1.0 + daily["index_return"].fillna(0.0)).cumprod()
    daily["index_proxy_rebased"] = daily["index_proxy"] / daily["index_proxy"].iloc[0] * 100.0
    daily["year"] = daily["Date"].dt.year

    years = sorted(daily["year"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(years) / n_cols))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.8 * n_rows), dpi=200)
    axes = np.atleast_1d(axes).reshape(-1)

    legend_handles = None

    for ax, year in zip(axes, years):
        part = daily[daily["year"].eq(year)].reset_index(drop=True)
        if part.empty:
            ax.axis("off")
            continue

        x = np.arange(len(part))
        ax2 = ax.twinx()

        l1 = ax.plot(x, part["index_proxy_rebased"], color="#2563eb", linewidth=1.4, label="VN30 Index Proxy")
        ax.fill_between(x, part["index_proxy_rebased"], part["index_proxy_rebased"].min() * 0.98, color="#2563eb", alpha=0.06)

        l2 = ax2.plot(x, part["median_abs_error"] * 100, color="#d97706", linewidth=1.2, label="Median |E_d| (Central)")
        l3 = ax2.plot(x, part["q90_abs_error"] * 100, color="#dc2626", linestyle="--", linewidth=1.1, label="Q90 |E_d| (Tail Risk)")
        ax2.fill_between(x, part["median_abs_error"] * 100, part["q90_abs_error"] * 100, color="#dc2626", alpha=0.10)

        ax2.axhline(3.0, color="#10b981", linestyle=":", linewidth=1.0, alpha=0.85)
        ax2.axhline(3.5, color="#ef4444", linestyle="-.", linewidth=0.9, alpha=0.75)

        med_q90 = float(part["q90_abs_error"].median() * 100)
        p90_q90 = float(part["q90_abs_error"].quantile(0.90) * 100)

        ax.set_title(
            f"{year} (Validation) | n_days={len(part)} | med Q90={med_q90:.2f}%, p90 Q90={p90_q90:.2f}%",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
            color="#1e293b"
        )
        ax.set_xlabel("Trading Days", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax2.tick_params(axis="y", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax2.grid(False)

        ax.tick_params(axis="y", labelcolor="#2563eb")
        ax2.tick_params(axis="y", labelcolor="#dc2626")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))

        ax2.set_ylim(0.0, max(float(part["q90_abs_error"].max() * 100) * 1.15, 6.0))

        if legend_handles is None:
            legend_handles = l1 + l2 + l3

    for ax in axes[len(years):]:
        ax.axis("off")

    if legend_handles is not None:
        fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper right", ncol=3, frameon=True, facecolor="white", edgecolor="#cbd5e1")

    fig.suptitle("VN30 Index vs Dynamic LSTM Ensemble Prediction Error Envelope\nValidation Split (Dynamic Constituents)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter to ensemble model + validation split
    df = pd.read_csv(PREDICTIONS)
    val_ens = df[(df["model"] == "panel_lstm_ensemble") & (df["split"] == "val")].copy()

    if val_ens.empty:
        print("ERROR: No ensemble validation predictions found!")
        return

    plot_density(val_ens, OUTPUT_DIR / "signed_error_density_vn30.png")
    plot_density(val_ens, REPORT_DIR / "signed_error_density_vn30.png")

    plot_envelope(val_ens, OUTPUT_DIR / "vn30_error_envelope_by_year.png")
    plot_envelope(val_ens, REPORT_DIR / "vn30_error_envelope_by_year.png")

    print("SUCCESS: Dynamic VN30 plots created successfully!")
    print(OUTPUT_DIR / "signed_error_density_vn30.png")
    print(OUTPUT_DIR / "vn30_error_envelope_by_year.png")


if __name__ == "__main__":
    main()
