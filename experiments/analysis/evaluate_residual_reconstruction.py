"""Step 3B: Evaluate residual model with various market reconstruction strategies.

Instead of training a market head (which failed in Step 3), we use the already-
trained residual models from Step 2A and test different reconstruction rules:

1. oracle: market_actual (upper bound, already tested).
2. ar1: intercept + slope * market_return_lag1 (already tested, weak).
3. momentum_5: scale * rolling_5day_market_return (simple momentum).
4. momentum_20: scale * rolling_20day_market_return.
5. zero: no market component (= residual model as-is, lower bound).
6. shrunk_ar1: ar1 prediction * shrinkage_factor (reduce noise).

All reconstruction parameters are fitted on train only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PREDICTIONS = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "residual_target_probe_20260519"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "residual_reconstruction_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "residual_reconstruction_20260520"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--spike-thresholds", default="0.05,0.07,0.08")
    return parser.parse_args(argv)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.50) + 0.5 * np.quantile(np.abs(clean), 0.90))


def rel_score_fn(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def load_residual_predictions(predictions_dir: Path, seed: int) -> pd.DataFrame:
    """Load residual_oracle predictions (which contain alpha predictions in target space)."""
    path = predictions_dir / f"predictions_residual_oracle_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_market_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build daily market panel with lagged features for reconstruction."""
    daily = df.groupby("Date", sort=True).agg(
        market_actual=("market_component", "first"),  # oracle = market_actual
        market_return_today=("actual", "mean"),  # cross-sectional mean of raw actual
    ).reset_index()
    daily["market_lag1"] = daily["market_return_today"].shift(1)
    daily["market_lag1_5"] = daily["market_return_today"].shift(1).rolling(5, min_periods=3).mean()
    daily["market_lag1_20"] = daily["market_return_today"].shift(1).rolling(20, min_periods=10).mean()
    return daily


def fit_reconstruction_params(train_daily: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fit all reconstruction strategies on train data only."""
    params: dict[str, dict[str, float]] = {}
    # AR(1)
    valid = train_daily.dropna(subset=["market_actual", "market_lag1"])
    if len(valid) >= 30:
        x = valid["market_lag1"].to_numpy(dtype=float)
        y = valid["market_actual"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        params["ar1"] = {"intercept": float(coef[0]), "slope": float(coef[1])}
    else:
        params["ar1"] = {"intercept": 0.0, "slope": 0.0}
    # Momentum scales (grid search for best train rel_score)
    for horizon_name, col in [("momentum_5", "market_lag1_5"), ("momentum_20", "market_lag1_20")]:
        valid_h = train_daily.dropna(subset=["market_actual", col])
        if len(valid_h) < 30:
            params[horizon_name] = {"scale": 0.0}
            continue
        best_scale = 0.0
        best_corr = -np.inf
        for scale in np.arange(0.0, 3.1, 0.1):
            pred = scale * valid_h[col].to_numpy(dtype=float)
            actual = valid_h["market_actual"].to_numpy(dtype=float)
            corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 1e-8 else 0.0
            if corr > best_corr:
                best_corr = corr
                best_scale = float(scale)
        params[horizon_name] = {"scale": best_scale}
    # Shrunk AR(1)
    ar1_pred = params["ar1"]["intercept"] + params["ar1"]["slope"] * train_daily["market_lag1"].fillna(0.0).to_numpy()
    best_shrink = 0.0
    best_score = -np.inf
    actual_train = train_daily["market_actual"].fillna(0.0).to_numpy(dtype=float)
    for shrink in np.arange(0.0, 1.05, 0.05):
        pred = shrink * ar1_pred
        score = rel_score_fn(actual_train, pred)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_shrink = float(shrink)
    params["shrunk_ar1"] = {"shrinkage": best_shrink, **params["ar1"]}
    return params


def apply_reconstruction(daily: pd.DataFrame, strategy: str, params: dict[str, dict[str, float]]) -> np.ndarray:
    """Return per-day market component prediction."""
    n = len(daily)
    if strategy == "oracle":
        return daily["market_actual"].fillna(0.0).to_numpy(dtype=float)
    if strategy == "zero":
        return np.zeros(n, dtype=float)
    if strategy == "ar1":
        p = params["ar1"]
        return (p["intercept"] + p["slope"] * daily["market_lag1"].fillna(0.0).to_numpy(dtype=float))
    if strategy == "shrunk_ar1":
        p = params["shrunk_ar1"]
        ar1 = p["intercept"] + p["slope"] * daily["market_lag1"].fillna(0.0).to_numpy(dtype=float)
        return p["shrinkage"] * ar1
    if strategy == "momentum_5":
        p = params["momentum_5"]
        return p["scale"] * daily["market_lag1_5"].fillna(0.0).to_numpy(dtype=float)
    if strategy == "momentum_20":
        p = params["momentum_20"]
        return p["scale"] * daily["market_lag1_20"].fillna(0.0).to_numpy(dtype=float)
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_reconstruction(
    pred_df: pd.DataFrame,
    daily: pd.DataFrame,
    strategy: str,
    params: dict[str, dict[str, float]],
    spike_thresholds: tuple[float, ...],
    split: str,
) -> dict[str, object]:
    """Evaluate a reconstruction strategy on a given split."""
    split_df = pred_df.loc[pred_df["split"] == split].copy()
    if split_df.empty:
        return {}
    market_pred_daily = apply_reconstruction(daily, strategy, params)
    date_to_market = dict(zip(daily["Date"], market_pred_daily))
    split_df["market_pred"] = split_df["Date"].map(date_to_market).fillna(0.0)
    # alpha prediction is prediction_target_space (residual model output)
    split_df["final_pred"] = split_df["prediction_target_space"] + split_df["market_pred"]
    actual = split_df["actual"].to_numpy(dtype=float)
    final_pred = split_df["final_pred"].to_numpy(dtype=float)
    abs_error = np.abs(actual - final_pred)
    daily_agg = split_df.groupby("Date")["final_pred"].apply(
        lambda g: float(np.quantile(np.abs(split_df.loc[g.index, "actual"] - g), 0.90))
    ).reset_index(name="daily_q90")
    row: dict[str, object] = {
        "strategy": strategy,
        "split": split,
        "rel_score": rel_score_fn(actual, final_pred),
        "median_abs_error": float(np.quantile(abs_error, 0.50)),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)),
        "daily_q90_p90": float(daily_agg["daily_q90"].quantile(0.90)),
        "daily_max": float(daily_agg["daily_q90"].max()),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(final_pred))),
        "pred_actual_q90_ratio": float(np.quantile(np.abs(final_pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8)),
    }
    for threshold in spike_thresholds:
        key = int(round(threshold * 100))
        row[f"spike_days_ge_{key}pct"] = int(daily_agg["daily_q90"].ge(threshold).sum())
    return row


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    spike_thresholds = tuple(float(t.strip()) for t in args.spike_thresholds.split(","))
    strategies = ["oracle", "zero", "ar1", "shrunk_ar1", "momentum_5", "momentum_20"]
    all_results: list[dict[str, object]] = []
    all_params: dict[int, dict] = {}
    for seed in seeds:
        print(f"Processing seed={seed}")
        pred_df = load_residual_predictions(args.predictions_dir, seed)
        daily = build_market_panel(pred_df)
        # Split daily into train/val
        train_dates = pred_df.loc[pred_df["split"] == "train", "Date"].unique()
        train_daily = daily.loc[daily["Date"].isin(train_dates)].copy()
        params = fit_reconstruction_params(train_daily)
        all_params[seed] = params
        for strategy in strategies:
            row = evaluate_reconstruction(pred_df, daily, strategy, params, spike_thresholds, split="val")
            if row:
                row["seed"] = seed
                all_results.append(row)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    # Aggregate
    agg_rows = []
    for strategy, group in results_df.groupby("strategy"):
        row = {"strategy": strategy, "n_seeds": len(group)}
        for col in ["rel_score", "daily_max", "spike_days_ge_8pct", "directional_accuracy", "pred_actual_q90_ratio"]:
            if col in group.columns:
                vals = group[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(args.output_dir / "results_aggregate.csv", index=False)
    agg_df.to_csv(args.gold_dir / "results_aggregate.csv", index=False)
    # Params
    (args.output_dir / "fitted_params.json").write_text(
        json.dumps({str(k): v for k, v in all_params.items()}, indent=2), encoding="utf-8"
    )
    # Readout
    lines = [
        "# Residual Reconstruction Readout",
        "",
        "Step 3B: evaluate different market reconstruction strategies on residual model predictions.",
        "Scope: VN train/validation only. Holdout/test is not used.",
        "",
        "## Aggregate Validation (3 seeds)",
        "",
        agg_df.to_markdown(index=False),
        "",
        "## Per-Seed Validation",
        "",
        results_df.to_markdown(index=False),
        "",
        "## Fitted Parameters",
        "",
        json.dumps({str(k): v for k, v in all_params.items()}, indent=2),
        "",
        "## Decision",
        "",
        "Best non-oracle strategy that beats zero (residual-only) on BOTH rel_score AND spike reduction",
        "is the candidate for production reconstruction.",
    ]
    readout = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(readout, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(readout, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
