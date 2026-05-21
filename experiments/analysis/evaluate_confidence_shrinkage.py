"""Step 4: Confidence shrinkage + volatility abstention (post-processing).

No retraining needed. Uses existing raw_baseline predictions from Step 2A.

Strategies:
1. confidence_shrinkage: p_new = p * (2*|sign_prob - 0.5|)^gamma
   - Requires sign_prob from signmag model. Since raw_baseline is plain LSTM
     (no sign_prob), we approximate confidence via prediction magnitude:
     confidence = min(1, |p| / q75(|p|))  (high |p| = more confident).
   - Alternative: use volatility-based confidence.

2. volatility_clipping: p_new = clip(p, -k*vol_20, +k*vol_20)
   - Reduces tail error by capping prediction at k sigma.

3. volatility_abstention: p_new = p * (1 - abstain_mask)
   - abstain_mask = 1 when predicted_vol > threshold (high vol day → predict 0).

4. combined: clipping + abstention on high-vol days.

All parameters fitted on train split only.
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
    / "training_runs" / "reports" / "confidence_shrinkage_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "confidence_shrinkage_20260520"


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


def load_baseline_predictions(predictions_dir: Path, seed: int) -> pd.DataFrame:
    path = predictions_dir / f"predictions_raw_baseline_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    # Normalize column name
    if "prediction_raw" in df.columns and "prediction" not in df.columns:
        df["prediction"] = df["prediction_raw"]
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-stock and market-level volatility for abstention logic."""
    work = df.copy()
    # Per-day market volatility proxy: cross-sectional std of actual returns
    daily_vol = work.groupby("Date")["actual"].std().rename("daily_cs_vol").reset_index()
    # Lagged daily vol (available at prediction time)
    daily_vol["daily_cs_vol_lag1"] = daily_vol["daily_cs_vol"].shift(1)
    daily_vol["daily_cs_vol_lag1_5"] = daily_vol["daily_cs_vol"].shift(1).rolling(5, min_periods=3).mean()
    work = work.merge(daily_vol[["Date", "daily_cs_vol", "daily_cs_vol_lag1", "daily_cs_vol_lag1_5"]], on="Date", how="left")
    return work


# --- Strategy implementations ---

def apply_clipping(prediction: np.ndarray, k: float, vol: np.ndarray) -> np.ndarray:
    """Clip prediction at ±k * local_volatility."""
    cap = k * np.maximum(vol, 1e-6)
    return np.clip(prediction, -cap, cap)


def apply_magnitude_shrinkage(prediction: np.ndarray, gamma: float, q75_pred: float) -> np.ndarray:
    """Shrink prediction based on relative magnitude confidence."""
    abs_pred = np.abs(prediction)
    confidence = np.minimum(abs_pred / max(q75_pred, 1e-8), 1.0)
    shrinkage = confidence ** gamma
    return prediction * shrinkage


def apply_vol_abstention(prediction: np.ndarray, vol_lag: np.ndarray, threshold: float) -> np.ndarray:
    """Set prediction to 0 on high-vol days."""
    mask = vol_lag > threshold
    result = prediction.copy()
    result[mask] = 0.0
    return result


def apply_vol_shrinkage(prediction: np.ndarray, vol_lag: np.ndarray, vol_median: float, power: float = 1.0) -> np.ndarray:
    """Shrink prediction proportional to how much vol exceeds median."""
    ratio = vol_lag / max(vol_median, 1e-8)
    shrink_factor = np.minimum(1.0, (1.0 / np.maximum(ratio, 0.5)) ** power)
    return prediction * shrink_factor


# --- Parameter fitting (train only) ---

def fit_clipping_k(train_df: pd.DataFrame, grid: np.ndarray) -> float:
    actual = train_df["actual"].to_numpy(dtype=float)
    pred = train_df["prediction"].to_numpy(dtype=float)
    vol = train_df["daily_cs_vol"].to_numpy(dtype=float)
    best_k = float(grid[0])
    best_score = -np.inf
    for k in grid:
        clipped = apply_clipping(pred, float(k), vol)
        score = rel_score_fn(actual, clipped)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_k = float(k)
    return best_k


def fit_vol_abstention_threshold(train_df: pd.DataFrame, grid: np.ndarray) -> float:
    actual = train_df["actual"].to_numpy(dtype=float)
    pred = train_df["prediction"].to_numpy(dtype=float)
    vol_lag = train_df["daily_cs_vol_lag1"].to_numpy(dtype=float)
    best_t = float(grid[-1])  # default: never abstain
    best_score = -np.inf
    for t in grid:
        modified = apply_vol_abstention(pred, vol_lag, float(t))
        score = rel_score_fn(actual, modified)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def fit_vol_shrinkage_power(train_df: pd.DataFrame, vol_median: float, grid: np.ndarray) -> float:
    actual = train_df["actual"].to_numpy(dtype=float)
    pred = train_df["prediction"].to_numpy(dtype=float)
    vol_lag = train_df["daily_cs_vol_lag1"].to_numpy(dtype=float)
    best_p = 0.0
    best_score = -np.inf
    for p in grid:
        modified = apply_vol_shrinkage(pred, vol_lag, vol_median, float(p))
        score = rel_score_fn(actual, modified)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_p = float(p)
    return best_p


def fit_all_params(train_df: pd.DataFrame) -> dict[str, object]:
    params: dict[str, object] = {}
    # Clipping
    k_grid = np.arange(1.5, 4.1, 0.25)
    params["clip_k"] = fit_clipping_k(train_df, k_grid)
    # Vol abstention
    vol_lag_values = train_df["daily_cs_vol_lag1"].dropna()
    if len(vol_lag_values) > 30:
        vol_quantiles = np.quantile(vol_lag_values, [0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        params["vol_abstain_threshold"] = fit_vol_abstention_threshold(train_df, vol_quantiles)
    else:
        params["vol_abstain_threshold"] = 1.0  # never abstain
    # Vol shrinkage
    vol_median = float(vol_lag_values.median()) if len(vol_lag_values) > 10 else 0.01
    params["vol_median"] = vol_median
    power_grid = np.arange(0.0, 3.1, 0.25)
    params["vol_shrink_power"] = fit_vol_shrinkage_power(train_df, vol_median, power_grid)
    # Magnitude shrinkage
    pred_abs = np.abs(train_df["prediction"].to_numpy(dtype=float))
    params["pred_q75"] = float(np.quantile(pred_abs[pred_abs > 0], 0.75)) if np.any(pred_abs > 0) else 0.01
    return params


# --- Evaluation ---

def evaluate_strategy(
    val_df: pd.DataFrame,
    prediction_col: str,
    strategy_name: str,
    spike_thresholds: tuple[float, ...],
) -> dict[str, object]:
    actual = val_df["actual"].to_numpy(dtype=float)
    pred = val_df[prediction_col].to_numpy(dtype=float)
    abs_error = np.abs(actual - pred)
    daily = pd.DataFrame({"Date": val_df["Date"].values, "abs_error": abs_error})
    daily_agg = daily.groupby("Date")["abs_error"].quantile(0.90).reset_index(name="daily_q90")
    row: dict[str, object] = {
        "strategy": strategy_name,
        "rel_score": rel_score_fn(actual, pred),
        "median_abs_error": float(np.quantile(abs_error, 0.50)),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)),
        "daily_q90_p90": float(daily_agg["daily_q90"].quantile(0.90)),
        "daily_max": float(daily_agg["daily_q90"].max()),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        "pred_actual_q90_ratio": float(
            np.quantile(np.abs(pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8)
        ),
        "coverage": float(np.mean(pred != 0.0)),
    }
    for threshold in spike_thresholds:
        key = int(round(threshold * 100))
        row[f"spike_days_ge_{key}pct"] = int(daily_agg["daily_q90"].ge(threshold).sum())
    return row


def apply_all_strategies(val_df: pd.DataFrame, params: dict[str, object]) -> pd.DataFrame:
    """Apply all post-processing strategies and add columns."""
    work = val_df.copy()
    pred = work["prediction"].to_numpy(dtype=float)
    vol = work["daily_cs_vol"].to_numpy(dtype=float)
    vol_lag = work["daily_cs_vol_lag1"].to_numpy(dtype=float)
    # 1. Baseline (no change)
    work["pred_baseline"] = pred
    # 2. Clipping
    work["pred_clip"] = apply_clipping(pred, params["clip_k"], vol)
    # 3. Vol abstention
    work["pred_vol_abstain"] = apply_vol_abstention(pred, vol_lag, params["vol_abstain_threshold"])
    # 4. Vol shrinkage
    work["pred_vol_shrink"] = apply_vol_shrinkage(pred, vol_lag, params["vol_median"], params["vol_shrink_power"])
    # 5. Magnitude shrinkage (gamma=1)
    work["pred_mag_shrink_g1"] = apply_magnitude_shrinkage(pred, 1.0, params["pred_q75"])
    # 6. Magnitude shrinkage (gamma=2)
    work["pred_mag_shrink_g2"] = apply_magnitude_shrinkage(pred, 2.0, params["pred_q75"])
    # 7. Clip + vol_shrink combined
    clipped = apply_clipping(pred, params["clip_k"], vol)
    work["pred_clip_vol_shrink"] = apply_vol_shrinkage(clipped, vol_lag, params["vol_median"], params["vol_shrink_power"])
    # 8. Clip + vol_abstain combined
    work["pred_clip_vol_abstain"] = apply_vol_abstention(clipped, vol_lag, params["vol_abstain_threshold"])
    # 9. Vol shrink + magnitude shrink
    vol_shrunk = apply_vol_shrinkage(pred, vol_lag, params["vol_median"], params["vol_shrink_power"])
    work["pred_vol_mag_shrink"] = apply_magnitude_shrinkage(vol_shrunk, 1.0, params["pred_q75"])
    return work


STRATEGY_COLUMNS = [
    ("baseline", "pred_baseline"),
    ("clip", "pred_clip"),
    ("vol_abstain", "pred_vol_abstain"),
    ("vol_shrink", "pred_vol_shrink"),
    ("mag_shrink_g1", "pred_mag_shrink_g1"),
    ("mag_shrink_g2", "pred_mag_shrink_g2"),
    ("clip_vol_shrink", "pred_clip_vol_shrink"),
    ("clip_vol_abstain", "pred_clip_vol_abstain"),
    ("vol_mag_shrink", "pred_vol_mag_shrink"),
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    spike_thresholds = tuple(float(t.strip()) for t in args.spike_thresholds.split(","))
    all_results: list[dict[str, object]] = []
    all_params: dict[int, dict] = {}
    for seed in seeds:
        print(f"Processing seed={seed}")
        df = load_baseline_predictions(args.predictions_dir, seed)
        df = add_volatility_features(df)
        train_df = df.loc[df["split"] == "train"].copy()
        val_df = df.loc[df["split"] == "val"].copy()
        params = fit_all_params(train_df)
        all_params[seed] = params
        val_processed = apply_all_strategies(val_df, params)
        for strategy_name, col in STRATEGY_COLUMNS:
            row = evaluate_strategy(val_processed, col, strategy_name, spike_thresholds)
            row["seed"] = seed
            all_results.append(row)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    # Aggregate
    agg_rows = []
    for strategy, group in results_df.groupby("strategy"):
        row = {"strategy": strategy, "n_seeds": len(group)}
        for col in ["rel_score", "daily_max", "spike_days_ge_5pct", "spike_days_ge_7pct", "spike_days_ge_8pct", "directional_accuracy", "pred_actual_q90_ratio", "coverage"]:
            if col in group.columns:
                vals = group[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values("rel_score_mean", ascending=False).reset_index(drop=True)
    agg_df.to_csv(args.output_dir / "results_aggregate.csv", index=False)
    agg_df.to_csv(args.gold_dir / "results_aggregate.csv", index=False)
    (args.output_dir / "fitted_params.json").write_text(
        json.dumps({str(k): v for k, v in all_params.items()}, indent=2), encoding="utf-8"
    )
    # Readout
    display_cols = ["strategy", "rel_score_mean", "rel_score_std", "daily_max_mean", "spike_days_ge_8pct_mean", "directional_accuracy_mean", "coverage_mean"]
    display = agg_df[[c for c in display_cols if c in agg_df.columns]].copy()
    lines = [
        "# Confidence Shrinkage & Volatility Abstention Readout",
        "",
        "Step 4: post-processing on raw_baseline predictions (no retraining).",
        "Scope: VN train/validation only. Holdout/test is not used.",
        "",
        "## Aggregate Validation (3 seeds)",
        "",
        display.to_markdown(index=False),
        "",
        "## Fitted Parameters",
        "",
        json.dumps({str(k): v for k, v in all_params.items()}, indent=2),
        "",
        "## Decision",
        "",
        "Best strategy = highest rel_score_mean that ALSO reduces spike_days_ge_8pct vs baseline.",
        "If no strategy improves both, the raw baseline remains gold.",
    ]
    readout = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(readout, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(readout, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
