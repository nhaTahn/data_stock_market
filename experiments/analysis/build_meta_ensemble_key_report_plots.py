"""Build two key report plots for the frozen VN meta-ensemble.

Outputs:
- rel_score histogram by stock code
- daily return-error time series

Scope: VN validation only. Holdout/test is not used.
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
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402

DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/meta_ensemble_key_report_plots_20260601"
REPORT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/meta_ensemble_key_report_plots_20260601"


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.50) + 0.5 * np.quantile(np.abs(clean), 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def load_meta_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_training_frame(str(DATA), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    scaler = fit_feature_scaler(frame.loc[frame["Date"] <= "2020-03-31"].dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    _, _, sequence_meta = build_sequence_dataset(scaled, features, "target_next_return", 15, sequence_normalization="none")
    dummy = np.zeros((len(sequence_meta), 1), dtype=np.float32)
    splits = split_sequence_dataset(dummy, sequence_meta["target"].to_numpy(), sequence_meta, "2020-03-31", "2022-11-15")
    return splits["train"][2].reset_index(drop=True), splits["val"][2].reset_index(drop=True)


def build_prediction_frame() -> pd.DataFrame:
    y_train, y_val, pred_train, pred_val, sigma_train, sigma_val = meta.load_anchor_seed_predictions()
    dates_train, dates_val = regime.load_dates()
    _, meta_val = load_meta_frames()
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
    meta_val_pred = np.asarray(model.predict(x_val), dtype=np.float32)
    alpha, _, prediction = meta.train_selected_blend(y_train, regime_train, meta_train, y_val, regime_val, meta_val_pred)
    out = meta_val.loc[:, ["code", "Date"]].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["actual"] = y_val.astype(float)
    out["prediction"] = prediction.astype(float)
    out["error"] = out["actual"] - out["prediction"]
    out["abs_error"] = out["error"].abs()
    out.attrs["alpha"] = float(alpha)
    return out


def build_relscore_by_code(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, group in predictions.groupby("code", sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        rows.append({
            "code": code,
            "n_obs": int(len(group)),
            "rel_score": rel_score(actual, pred),
            "median_abs_error": float(np.quantile(np.abs(actual - pred), 0.50)),
            "q90_abs_error": float(np.quantile(np.abs(actual - pred), 0.90)),
        })
    return pd.DataFrame(rows).sort_values("rel_score", ascending=False).reset_index(drop=True)


def build_daily_error(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("Date", sort=True)["abs_error"]
        .agg(
            n_obs="count",
            median_abs_error="median",
            mean_abs_error="mean",
            q75_abs_error=lambda values: float(np.quantile(values, 0.75)),
            q90_abs_error=lambda values: float(np.quantile(values, 0.90)),
        )
        .reset_index()
    )


def plot_relscore_hist(rel_by_code: pd.DataFrame, output_path: Path) -> None:
    values = rel_by_code["rel_score"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(values, bins=28, color="#2563eb", alpha=0.72, edgecolor="white")
    mean_value = float(values.mean())
    median_value = float(values.median())
    ax.axvline(mean_value, color="#f97316", linestyle="--", linewidth=1.4, label=f"Mean={mean_value:.3f}")
    ax.axvline(median_value, color="#16a34a", linestyle=":", linewidth=1.8, label=f"Median={median_value:.3f}")
    ax.axvline(0.0, color="#111827", linestyle="-", linewidth=1.0, alpha=0.75, label="Zero baseline")
    ax.set_title("Validation rel_score Distribution by Stock — hgb_abs_blend", fontweight="bold")
    ax.set_xlabel("rel_score by stock code")
    ax.set_ylabel("Number of stocks")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_daily_error(daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.plot(daily["Date"], daily["median_abs_error"] * 100, color="#f59e0b", linewidth=1.1, alpha=0.85, label="Daily median |E|")
    ax.plot(daily["Date"], daily["q75_abs_error"] * 100, color="#7c3aed", linewidth=1.0, alpha=0.72, label="Daily Q75 |E|")
    ax.plot(daily["Date"], daily["q90_abs_error"] * 100, color="#dc2626", linewidth=1.0, alpha=0.72, label="Daily Q90 |E|")
    ax.axhline(3.0, color="#059669", linestyle="--", linewidth=1.1, label="3.0% target")
    ax.axhline(3.5, color="#991b1b", linestyle=":", linewidth=1.1, label="3.5% violation line")
    ax.set_title("Validation Daily Return Error Time Series — hgb_abs_blend", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute return error (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(predictions: pd.DataFrame, rel_by_code: pd.DataFrame, daily: pd.DataFrame) -> None:
    overall_rel = rel_score(predictions["actual"].to_numpy(float), predictions["prediction"].to_numpy(float))
    daily_summary = pd.DataFrame([
        {
            "series": col,
            "median": float(daily[col].median()),
            "p90": float(daily[col].quantile(0.90)),
            "max": float(daily[col].max()),
            "days_gt_3pct": int((daily[col] > 0.03).sum()),
            "days_gt_3p5pct": int((daily[col] > 0.035).sum()),
        }
        for col in ["median_abs_error", "q75_abs_error", "q90_abs_error"]
    ])
    lines = [
        "# Meta-Ensemble Key Report Plots",
        "",
        "Scope: VN validation only. Holdout/test is not used.",
        "Model: `hgb_abs_blend` meta-ensemble over frozen 5-seed hetero anchor.",
        "",
        f"Overall validation rel_score: `{overall_rel:.6f}`.",
        f"Blend alpha: `{predictions.attrs.get('alpha', float('nan')):.3f}`.",
        "",
        "## Daily Error Summary",
        "",
        daily_summary.round(6).to_markdown(index=False),
        "",
        "## rel_score by Code Summary",
        "",
        rel_by_code["rel_score"].describe().to_frame("rel_score").round(6).to_markdown(),
        "",
        "## Files",
        "",
        "- `rel_score_histogram_by_code.png`",
        "- `daily_return_error_timeseries.png`",
        "- `rel_score_by_code.csv`",
        "- `daily_return_error_series.csv`",
        "- `validation_predictions.csv`",
        "",
        json.dumps({"gold_dir": str(GOLD), "report_dir": str(REPORT)}, indent=2),
    ]
    for output_dir in [GOLD, REPORT]:
        (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    predictions = build_prediction_frame()
    rel_by_code = build_relscore_by_code(predictions)
    daily = build_daily_error(predictions)
    for output_dir in [GOLD, REPORT]:
        predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
        rel_by_code.to_csv(output_dir / "rel_score_by_code.csv", index=False)
        daily.to_csv(output_dir / "daily_return_error_series.csv", index=False)
        plot_relscore_hist(rel_by_code, output_dir / "rel_score_histogram_by_code.png")
        plot_daily_error(daily, output_dir / "daily_return_error_timeseries.png")
    write_summary(predictions, rel_by_code, daily)
    print((GOLD / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
