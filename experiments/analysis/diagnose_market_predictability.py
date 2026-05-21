"""Diagnose market-mean return predictability for VN universe.

Step 1 of the input/target processing improvement plan
(`docs/improvement_plan_v2_input_target_processing_20260519.md`).

Goal: decide whether a two-stream (market_pred + alpha_pred) architecture is
worth implementing. The hypothesis is that if cross-sectional mean return at
day t can be predicted (R^2 >= 0.05) from lagged market features, then a
dedicated market head can absorb spike-day moves and reduce |error| at q90.

Decision rule:
- if best lagged R^2 >= 0.05 OR AR(1) >= 0.05 (abs): proceed to Step 2.
- if 0.02 <= best R^2 < 0.05: marginal, run Step 2 anyway but with low priors.
- if best R^2 < 0.02 AND AR(1) ~ 0: skip two-stream, switch to selective
  abstention path.

Output:
- CSV table of regression results per feature set / horizon.
- Markdown readout summarising decision.
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

DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs" / "reports" / "market_predictability_diagnostic_20260519"
DEFAULT_DOCS_OUTPUT = ROOT / "docs" / "market_predictability_readout.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--docs-output", type=Path, default=DEFAULT_DOCS_OUTPUT)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    return parser.parse_args(argv)


def build_market_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily cross-sectional statistics."""
    work = df.sort_values(["code", "Date"], kind="stable").copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["return_1"] = work.groupby("code", sort=False)["adjust"].pct_change()
    daily = work.groupby("Date", sort=True).agg(
        market_return_actual=("target_next_return", "mean"),
        market_return_today=("return_1", "mean"),
        market_q10_today=("return_1", lambda values: float(np.nanquantile(values, 0.10))),
        market_q90_today=("return_1", lambda values: float(np.nanquantile(values, 0.90))),
        market_negative_ratio_today=("return_1", lambda values: float(np.nanmean(values < 0.0))),
        market_abs_q90_today=("return_1", lambda values: float(np.nanquantile(np.abs(values), 0.90))),
        market_breadth_today=("return_1", lambda values: float(np.nanmean(values > 0.0))),
        n_stocks=("return_1", "count"),
    ).reset_index()
    daily = daily.dropna(subset=["market_return_actual"]).reset_index(drop=True)
    return daily


def add_lagged_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add strictly past features (lag>=1) to avoid leak when predicting target at t."""
    panel = daily.copy()
    base_columns = [
        "market_return_today",
        "market_q10_today",
        "market_q90_today",
        "market_negative_ratio_today",
        "market_abs_q90_today",
        "market_breadth_today",
    ]
    for column in base_columns:
        panel[f"{column}_lag1"] = panel[column].shift(1)
        panel[f"{column}_lag2"] = panel[column].shift(2)
    panel["market_return_lag1_5"] = panel["market_return_today"].shift(1).rolling(5, min_periods=3).mean()
    panel["market_return_lag1_20"] = panel["market_return_today"].shift(1).rolling(20, min_periods=10).mean()
    panel["market_volatility_lag1_20"] = panel["market_return_today"].shift(1).rolling(20, min_periods=10).std()
    panel["market_negative_ratio_ewm5_lag1"] = (
        panel["market_negative_ratio_today"].shift(1).ewm(span=5, adjust=False).mean()
    )
    panel["market_abs_q90_ewm5_lag1"] = (
        panel["market_abs_q90_today"].shift(1).ewm(span=5, adjust=False).mean()
    )
    return panel


def autocorrelation(series: pd.Series, lag: int) -> float:
    s = series.dropna()
    if len(s) < lag + 5:
        return float("nan")
    return float(s.autocorr(lag=lag))


def fit_ols_r2(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Plain OLS using numpy. Returns (R2, coefficients, intercept)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    if len(y_clean) < X_clean.shape[1] + 5:
        return float("nan"), np.zeros(X_clean.shape[1]), float("nan")
    X_design = np.column_stack([np.ones(len(X_clean)), X_clean])
    coef, *_ = np.linalg.lstsq(X_design, y_clean, rcond=None)
    intercept = float(coef[0])
    beta = coef[1:]
    y_hat = X_design @ coef
    ss_res = float(np.sum((y_clean - y_hat) ** 2))
    ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return r2, beta, intercept


def out_of_sample_r2(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> dict[str, float]:
    """Fit on train, evaluate R^2 on validation."""
    features = panel.loc[:, feature_columns].to_numpy(dtype=float)
    target = panel[target_column].to_numpy(dtype=float)
    dates = panel["Date"].to_numpy()
    train_mask = (dates <= np.datetime64(train_end)) & np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    val_mask = (
        (dates > np.datetime64(train_end))
        & (dates <= np.datetime64(val_end))
        & np.isfinite(target)
        & np.all(np.isfinite(features), axis=1)
    )
    if train_mask.sum() < len(feature_columns) + 10 or val_mask.sum() < 30:
        return {
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "r2_train": float("nan"),
            "r2_val": float("nan"),
            "rmse_val": float("nan"),
        }
    X_train = features[train_mask]
    y_train = target[train_mask]
    X_val = features[val_mask]
    y_val = target[val_mask]
    X_design_train = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(X_design_train, y_train, rcond=None)
    y_hat_train = X_design_train @ coef
    X_design_val = np.column_stack([np.ones(len(X_val)), X_val])
    y_hat_val = X_design_val @ coef
    train_mean = float(np.mean(y_train))
    val_mean = float(np.mean(y_val))
    ss_res_train = float(np.sum((y_train - y_hat_train) ** 2))
    ss_tot_train = float(np.sum((y_train - train_mean) ** 2))
    ss_res_val = float(np.sum((y_val - y_hat_val) ** 2))
    ss_tot_val = float(np.sum((y_val - val_mean) ** 2))
    r2_train = float("nan") if ss_tot_train <= 0 else 1.0 - ss_res_train / ss_tot_train
    r2_val = float("nan") if ss_tot_val <= 0 else 1.0 - ss_res_val / ss_tot_val
    rmse_val = float(np.sqrt(np.mean((y_val - y_hat_val) ** 2)))
    return {
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "r2_train": r2_train,
        "r2_val": r2_val,
        "rmse_val": rmse_val,
    }


def tail_day_split_r2(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    tail_threshold: float = 0.04,
) -> dict[str, float]:
    """Fit on full train, then evaluate R^2 on val split into tail vs normal days."""
    features = panel.loc[:, feature_columns].to_numpy(dtype=float)
    target = panel[target_column].to_numpy(dtype=float)
    dates = panel["Date"].to_numpy()
    train_mask = (dates <= np.datetime64(train_end)) & np.isfinite(target) & np.all(np.isfinite(features), axis=1)
    val_mask = (
        (dates > np.datetime64(train_end))
        & (dates <= np.datetime64(val_end))
        & np.isfinite(target)
        & np.all(np.isfinite(features), axis=1)
    )
    if train_mask.sum() < len(feature_columns) + 10 or val_mask.sum() < 30:
        return {"n_tail_val": 0, "n_normal_val": 0, "r2_tail_val": float("nan"), "r2_normal_val": float("nan")}
    X_train = features[train_mask]
    y_train = target[train_mask]
    X_design_train = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(X_design_train, y_train, rcond=None)
    X_val = features[val_mask]
    y_val = target[val_mask]
    X_design_val = np.column_stack([np.ones(len(X_val)), X_val])
    y_hat_val = X_design_val @ coef
    is_tail = np.abs(y_val) >= tail_threshold
    n_tail = int(is_tail.sum())
    n_normal = int((~is_tail).sum())
    if n_tail >= 5:
        ss_res_tail = float(np.sum((y_val[is_tail] - y_hat_val[is_tail]) ** 2))
        ss_tot_tail = float(np.sum((y_val[is_tail] - np.mean(y_val[is_tail])) ** 2))
        r2_tail = float("nan") if ss_tot_tail <= 0 else 1.0 - ss_res_tail / ss_tot_tail
    else:
        r2_tail = float("nan")
    if n_normal >= 5:
        ss_res_normal = float(np.sum((y_val[~is_tail] - y_hat_val[~is_tail]) ** 2))
        ss_tot_normal = float(np.sum((y_val[~is_tail] - np.mean(y_val[~is_tail])) ** 2))
        r2_normal = float("nan") if ss_tot_normal <= 0 else 1.0 - ss_res_normal / ss_tot_normal
    else:
        r2_normal = float("nan")
    return {
        "n_tail_val": n_tail,
        "n_normal_val": n_normal,
        "r2_tail_val": r2_tail,
        "r2_normal_val": r2_normal,
    }


def feature_correlation_with_target(
    panel: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    train_mask = pd.to_datetime(panel["Date"]) <= train_end
    train = panel.loc[train_mask].copy()
    target = train[target_column].astype(float)
    for column in feature_columns:
        feature = train[column].astype(float)
        valid = feature.notna() & target.notna()
        if valid.sum() < 30:
            rows.append({"feature": column, "pearson": float("nan"), "spearman": float("nan"), "n": int(valid.sum())})
            continue
        pearson = float(feature[valid].corr(target[valid], method="pearson"))
        spearman = float(feature[valid].corr(target[valid], method="spearman"))
        rows.append({"feature": column, "pearson": pearson, "spearman": spearman, "n": int(valid.sum())})
    return pd.DataFrame(rows).sort_values("pearson", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def write_readout(
    docs_path: Path,
    autocorr_rows: list[dict[str, float]],
    feature_results: list[dict[str, object]],
    feature_corr: pd.DataFrame,
    decision: dict[str, object],
) -> None:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Market Predictability Diagnostic Readout")
    lines.append("")
    lines.append("Step 1 of input/target processing improvement plan.")
    lines.append("")
    lines.append("Scope: VN train (<= 2020-03-31) and validation (2020-04-01 .. 2022-11-15).")
    lines.append("Holdout/test is not used.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- best train R^2: `{decision['best_train_r2']:.4f}` ({decision['best_train_set']})")
    lines.append(f"- best val R^2: `{decision['best_val_r2']:.4f}` ({decision['best_val_set']})")
    lines.append(f"- best |AR(k)|: `{decision['best_abs_autocorr']:.4f}` (lag={decision['best_abs_autocorr_lag']})")
    lines.append(f"- recommendation: **{decision['recommendation']}**")
    lines.append("")
    lines.append("## Autocorrelation of Target (Cross-Sectional Mean Return)")
    lines.append("")
    lines.append("| Lag | Train AR(k) | Val AR(k) |")
    lines.append("| ---: | ---: | ---: |")
    for row in autocorr_rows:
        lines.append(f"| {row['lag']} | `{row['train_ar']:.4f}` | `{row['val_ar']:.4f}` |")
    lines.append("")
    lines.append("## Linear Regression R^2 (Lagged Market Features -> market_return_actual)")
    lines.append("")
    lines.append("| Feature Set | n_train | n_val | R^2 train | R^2 val | RMSE val | R^2 tail val | R^2 normal val |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in feature_results:
        lines.append(
            f"| {row['name']} | {row['n_train']} | {row['n_val']} | "
            f"`{row['r2_train']:.4f}` | `{row['r2_val']:.4f}` | "
            f"`{row['rmse_val']:.5f}` | `{row['r2_tail_val']:.4f}` | `{row['r2_normal_val']:.4f}` |"
        )
    lines.append("")
    lines.append("## Top Lagged Features by Train Pearson Correlation")
    lines.append("")
    lines.append("| Feature | Pearson | Spearman | n |")
    lines.append("| --- | ---: | ---: | ---: |")
    for _, row in feature_corr.head(15).iterrows():
        lines.append(f"| `{row['feature']}` | `{row['pearson']:.4f}` | `{row['spearman']:.4f}` | {row['n']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if decision["recommendation"] == "proceed_two_stream":
        lines.append("- Market return is sufficiently predictable from lagged features.")
        lines.append("- Two-stream architecture (market_pred + alpha_pred) is worth implementing.")
        lines.append("- Proceed to Step 2 (residual target probe).")
    elif decision["recommendation"] == "marginal_two_stream":
        lines.append("- Market return is marginally predictable.")
        lines.append("- Two-stream may give small gains; run Step 2 with low priors.")
        lines.append("- If Step 2 (residual probe) does not show >= +0.005 rel_score gain, fall back to selective abstention.")
    else:
        lines.append("- Market return is essentially unpredictable from lagged features.")
        lines.append("- Two-stream architecture will not help.")
        lines.append("- Switch to selective abstention path: filter signal + holding period selector.")
    lines.append("")
    docs_path.write_text("\n".join(lines), encoding="utf-8")


def make_decision(
    feature_results: list[dict[str, object]],
    autocorr_rows: list[dict[str, float]],
) -> dict[str, object]:
    finite_train = [(row["name"], row["r2_train"]) for row in feature_results if np.isfinite(row["r2_train"])]
    finite_val = [(row["name"], row["r2_val"]) for row in feature_results if np.isfinite(row["r2_val"])]
    best_train_set, best_train_r2 = max(finite_train, key=lambda kv: kv[1]) if finite_train else ("none", float("nan"))
    best_val_set, best_val_r2 = max(finite_val, key=lambda kv: kv[1]) if finite_val else ("none", float("nan"))
    best_abs_autocorr = float("nan")
    best_abs_autocorr_lag = -1
    for row in autocorr_rows:
        for kind in ("train_ar", "val_ar"):
            value = row.get(kind, float("nan"))
            if np.isfinite(value) and (np.isnan(best_abs_autocorr) or abs(value) > best_abs_autocorr):
                best_abs_autocorr = float(abs(value))
                best_abs_autocorr_lag = int(row["lag"])
    score_for_decision = best_val_r2 if np.isfinite(best_val_r2) else best_train_r2
    if np.isnan(score_for_decision):
        recommendation = "abandon_two_stream"
    elif score_for_decision >= 0.05 or (np.isfinite(best_abs_autocorr) and best_abs_autocorr >= 0.05):
        recommendation = "proceed_two_stream"
    elif score_for_decision >= 0.02 or (np.isfinite(best_abs_autocorr) and best_abs_autocorr >= 0.02):
        recommendation = "marginal_two_stream"
    else:
        recommendation = "abandon_two_stream"
    return {
        "best_train_set": best_train_set,
        "best_train_r2": float(best_train_r2),
        "best_val_set": best_val_set,
        "best_val_r2": float(best_val_r2),
        "best_abs_autocorr": float(best_abs_autocorr),
        "best_abs_autocorr_lag": int(best_abs_autocorr_lag),
        "recommendation": recommendation,
    }


FEATURE_SETS: dict[str, list[str]] = {
    "ar1_only": ["market_return_today_lag1"],
    "ar1_ar2": ["market_return_today_lag1", "market_return_today_lag2"],
    "rolling_returns": ["market_return_lag1_5", "market_return_lag1_20"],
    "vol_only": ["market_volatility_lag1_20"],
    "tail_only": [
        "market_negative_ratio_today_lag1",
        "market_abs_q90_today_lag1",
        "market_q10_today_lag1",
    ],
    "tail_ewm": [
        "market_negative_ratio_ewm5_lag1",
        "market_abs_q90_ewm5_lag1",
    ],
    "combined_minimal": [
        "market_return_today_lag1",
        "market_volatility_lag1_20",
        "market_negative_ratio_today_lag1",
    ],
    "combined_full": [
        "market_return_today_lag1",
        "market_return_today_lag2",
        "market_return_lag1_5",
        "market_return_lag1_20",
        "market_volatility_lag1_20",
        "market_q10_today_lag1",
        "market_q90_today_lag1",
        "market_negative_ratio_today_lag1",
        "market_abs_q90_today_lag1",
        "market_breadth_today_lag1",
        "market_negative_ratio_ewm5_lag1",
        "market_abs_q90_ewm5_lag1",
    ],
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = pd.read_csv(args.data)
    daily = build_market_panel(df)
    panel = add_lagged_features(daily)
    train_end = pd.to_datetime(args.train_end_date)
    val_end = pd.to_datetime(args.val_end_date)

    train_target = panel.loc[panel["Date"] <= train_end, "market_return_actual"]
    val_target = panel.loc[(panel["Date"] > train_end) & (panel["Date"] <= val_end), "market_return_actual"]
    autocorr_rows = []
    for lag in [1, 2, 3, 5, 10]:
        autocorr_rows.append({
            "lag": lag,
            "train_ar": autocorrelation(train_target, lag),
            "val_ar": autocorrelation(val_target, lag),
        })

    feature_results: list[dict[str, object]] = []
    for name, columns in FEATURE_SETS.items():
        missing = [c for c in columns if c not in panel.columns]
        if missing:
            print(f"Skipping {name}: missing columns {missing}")
            continue
        oos = out_of_sample_r2(panel, columns, "market_return_actual", train_end, val_end)
        tail_split = tail_day_split_r2(panel, columns, "market_return_actual", train_end, val_end)
        feature_results.append({
            "name": name,
            "n_features": len(columns),
            **oos,
            **tail_split,
        })

    feature_columns_for_corr = sorted({col for cols in FEATURE_SETS.values() for col in cols})
    feature_corr = feature_correlation_with_target(
        panel, feature_columns_for_corr, "market_return_actual", train_end
    )

    decision = make_decision(feature_results, autocorr_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_dir / "market_panel.csv", index=False)
    pd.DataFrame(autocorr_rows).to_csv(args.output_dir / "autocorrelation.csv", index=False)
    pd.DataFrame(feature_results).to_csv(args.output_dir / "feature_set_r2.csv", index=False)
    feature_corr.to_csv(args.output_dir / "feature_correlation.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_readout(args.docs_output, autocorr_rows, feature_results, feature_corr, decision)
    print(json.dumps(decision, indent=2))
    print(f"Readout: {args.docs_output}")
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
