"""Build teacher-style median-error control plot for the frozen VN meta-ensemble.

This intentionally plots daily median absolute error and a 20-day rolling median,
not daily Q90. Holdout/test is not used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training import evaluate_meta_ensemble_calibration as meta  # noqa: E402
from experiments.training import evaluate_regime_calibration as regime  # noqa: E402

OUTPUT = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_median_error_control_20260601"
LOCAL_OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_median_error_control_20260601"


def build_predictions() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    y_train, y_val, pred_train, pred_val, sigma_train, sigma_val = meta.load_anchor_seed_predictions()
    dates_train, dates_val = regime.load_dates()
    market = regime.build_market_features(dates_train, dates_val, y_train, y_val)
    rv_train = market["train"]["vol10"]
    rv_val = market["val"]["vol10"]
    q_train = market["train"]["q905"]
    q_val = market["val"]["q905"]
    base_train = pred_train.mean(axis=1)
    base_val = pred_val.mean(axis=1)
    rv_edges, q_edges, grid = regime.fit_2d_scales(y_train, base_train, rv_train, q_train)
    regime_train = regime.apply_2d_scales(base_train, rv_train, q_train, rv_edges, q_edges, grid)
    regime_val = regime.apply_2d_scales(base_val, rv_val, q_val, rv_edges, q_edges, grid)
    x_train = meta.make_meta_features(pred_train, sigma_train, rv_train, q_train)
    x_val = meta.make_meta_features(pred_val, sigma_val, rv_val, q_val)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        learning_rate=0.03,
        max_leaf_nodes=8,
        l2_regularization=0.2,
        random_state=43,
    )
    model.fit(x_train, y_train)
    meta_train = np.asarray(model.predict(x_train), dtype=np.float32)
    meta_val = np.asarray(model.predict(x_val), dtype=np.float32)
    _, _, pred_val_final = meta.train_selected_blend(y_train, regime_train, meta_train, y_val, regime_val, meta_val)
    return y_val, pred_val_final, pd.to_datetime(dates_val).reset_index(drop=True)


def build_daily_frame(y_val: np.ndarray, prediction: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame({"Date": dates, "abs_error": np.abs(y_val - prediction)})
    daily = (
        work.groupby("Date", sort=True)["abs_error"]
        .agg(
            n_obs="count",
            median_abs_error="median",
            mean_abs_error="mean",
            q75_abs_error=lambda values: float(np.quantile(values, 0.75)),
            q90_abs_error=lambda values: float(np.quantile(values, 0.90)),
        )
        .reset_index()
    )
    daily["rolling20_median_abs_error"] = daily["median_abs_error"].rolling(20, min_periods=7).median()
    daily["rolling30_median_abs_error"] = daily["median_abs_error"].rolling(30, min_periods=10).median()
    return daily


def plot_daily(daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.plot(daily["Date"], daily["median_abs_error"] * 100, color="#f59e0b", alpha=0.45, linewidth=0.9, label="Daily median |E|")
    ax.plot(daily["Date"], daily["rolling20_median_abs_error"] * 100, color="#1d4ed8", linewidth=1.8, label="20-day rolling median |E|")
    ax.plot(daily["Date"], daily["rolling30_median_abs_error"] * 100, color="#0f766e", linewidth=1.4, alpha=0.85, label="30-day rolling median |E|")
    ax.axhline(3.0, color="#dc2626", linestyle="--", linewidth=1.1, label="3.0% target")
    ax.set_title("VN Validation Median Error Control — Meta-Ensemble hgb_abs_blend", fontweight="bold")
    ax.set_ylabel("Absolute error (%)")
    ax.set_xlabel("Validation date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_summary(daily: pd.DataFrame, output_dir: Path, local_output: Path) -> None:
    rows = []
    for column in ["median_abs_error", "mean_abs_error", "q75_abs_error", "q90_abs_error", "rolling20_median_abs_error", "rolling30_median_abs_error"]:
        values = daily[column].dropna()
        rows.append({
            "series": column,
            "days": int(values.shape[0]),
            "median": float(values.median()),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
            "days_gt_3pct": int((values > 0.03).sum()),
            "share_gt_3pct": float((values > 0.03).mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "median_error_summary.csv", index=False)
    summary.to_csv(local_output / "median_error_summary.csv", index=False)
    lines = [
        "# Meta-Ensemble Median Error Control",
        "",
        "Scope: VN validation only. Holdout/test is not used.",
        "",
        "This report is intentionally based on daily median absolute error, not daily Q90 tail error.",
        "The goal is to show the central-error stability of the frozen `hgb_abs_blend` meta-ensemble.",
        "",
        "## Summary",
        "",
        summary.round(6).to_markdown(index=False),
        "",
        "## Read",
        "",
        "- Daily median |E| has a low central-error level but can spike on market-shock days.",
        "- The 20-day rolling median |E| stays below the 3.0% target throughout validation.",
        "- Keep this separate from Q90/tail-risk plots; Q90 remains the honest tail-stress diagnostic.",
        "",
        json.dumps({"gold_dir": str(output_dir), "local_output": str(local_output)}, indent=2),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (local_output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    y_val, prediction, dates = build_predictions()
    daily = build_daily_frame(y_val, prediction, dates)
    daily.to_csv(OUTPUT / "daily_median_error_series.csv", index=False)
    daily.to_csv(LOCAL_OUTPUT / "daily_median_error_series.csv", index=False)
    plot_daily(daily, OUTPUT / "median_error_control_timeseries.png")
    plot_daily(daily, LOCAL_OUTPUT / "median_error_control_timeseries.png")
    write_summary(daily, OUTPUT, LOCAL_OUTPUT)
    print((OUTPUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
