"""Portable multi-market baseline/significance smoke test.

This is not the final model training. It checks whether a common feature
protocol is viable across VN/US/JP before launching expensive multi-market
heteroscedastic ensemble runs.

Holdout/test is not used. Each market uses:
- train: all data <= 2020-03-31
- validation/in-sample: 2020-04-01..2022-11-15
- portable features common to VN/US/JP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402

OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_baseline_significance_20260525"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/multimarket_portable_baseline_significance_20260525"
TRAIN_END = "2020-03-31"
VAL_START = "2020-04-01"
VAL_END = "2022-11-15"
FOLD_DAYS = 21
MARKETS = {
    "VN": ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv",
    "US": ROOT / "data/processed/assets/data_info_us/history/us_gold_recommended.csv",
    "JP": ROOT / "data/processed/assets/data_info_jp/history/jp_gold_recommended.csv",
}
PORTABLE_FEATURES = (
    "close_return",
    "adjust_return",
    "range_pct",
    "body_pct",
    "volume_change",
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "ma_5_gap",
    "ma_20_gap",
)


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def metric(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    base = robust_loss(actual)
    abs_err = np.abs(err)
    return {
        "n": int(len(actual)),
        "rel_score": float(1.0 - robust_loss(err) / base) if base > 0 else float("nan"),
        "absE_robust": robust_loss(err),
        "base_robust": base,
        "absE_q90": float(np.quantile(abs_err, 0.9)) if len(abs_err) else float("nan"),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8))
            if len(actual) else float("nan")
        ),
    }


def fold_ids(dates: pd.Series) -> np.ndarray:
    unique_dates = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    date_to_pos = {date: idx for idx, date in enumerate(unique_dates)}
    return pd.to_datetime(dates).map(date_to_pos).to_numpy(dtype=int) // FOLD_DAYS


def load_market(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = load_training_frame(path, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    keep = ["Date", "code", "target_next_return", *PORTABLE_FEATURES]
    missing = [col for col in keep if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=["target_next_return", *PORTABLE_FEATURES])
    train = frame[frame["Date"] <= TRAIN_END].copy()
    val = frame[(frame["Date"] >= VAL_START) & (frame["Date"] <= VAL_END)].copy()
    return train, val


def stock_train_mean(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    means = train.groupby("code")["target_next_return"].mean()
    global_mean = float(train["target_next_return"].mean())
    return val["code"].map(means).fillna(global_mean).to_numpy(dtype=np.float32)


def lagged_stock_mean_val_only(val: pd.DataFrame) -> np.ndarray:
    ordered = val[["code", "Date", "target_next_return"]].sort_values(["code", "Date"]).copy()
    ordered["pred"] = (
        ordered.groupby("code")["target_next_return"]
        .transform(lambda series: series.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    return ordered.sort_index()["pred"].to_numpy(dtype=np.float32)


def ridge_pred(train: pd.DataFrame, val: pd.DataFrame, alpha: float = 10.0) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True, random_state=0))
    model.fit(train[list(PORTABLE_FEATURES)].to_numpy(dtype=np.float32), train["target_next_return"].to_numpy(dtype=np.float32))
    return model.predict(val[list(PORTABLE_FEATURES)].to_numpy(dtype=np.float32)).astype(np.float32)


def evaluate_market(market: str, path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train, val = load_market(path)
    actual = val["target_next_return"].to_numpy(dtype=np.float32)
    predictions = {
        "zero": np.zeros_like(actual),
        "global_train_mean": np.full_like(actual, float(train["target_next_return"].mean())),
        "stock_train_mean": stock_train_mean(train, val),
        "lagged_stock_mean5_val_only": lagged_stock_mean_val_only(val),
        "ridge_portable": ridge_pred(train, val),
    }
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    ids = fold_ids(val["Date"])
    for model, pred in predictions.items():
        row = metric(actual, pred)
        row.update({"market": market, "model": model, "train_rows": len(train), "val_rows": len(val), "n_codes": val["code"].nunique()})
        overall_rows.append(row)
        for fold_id in sorted(set(ids)):
            mask = ids == fold_id
            dates = val.loc[mask, "Date"]
            fold = metric(actual[mask], pred[mask])
            fold.update(
                {
                    "market": market,
                    "model": model,
                    "fold_id": int(fold_id),
                    "test_start": dates.min().date().isoformat(),
                    "test_end": dates.max().date().isoformat(),
                }
            )
            fold_rows.append(fold)
    return overall_rows, fold_rows


def bootstrap_best_vs_zero(folds: pd.DataFrame, n_boot: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for market, group in folds.groupby("market"):
        pivot = group.pivot(index="fold_id", columns="model", values="rel_score")
        if "zero" not in pivot:
            continue
        for model in [col for col in pivot.columns if col != "zero"]:
            joined = pivot[[model, "zero"]].dropna()
            diff = (joined[model] - joined["zero"]).to_numpy(float)
            idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
            boot = diff[idx].mean(axis=1)
            rows.append(
                {
                    "market": market,
                    "model": model,
                    "n_folds": int(len(diff)),
                    "mean_delta_vs_zero": float(diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "p_boot_delta_le_0": float(np.mean(boot <= 0)),
                    "positive_delta_folds": int(np.sum(diff > 0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "mean_delta_vs_zero"], ascending=[True, False])


def build_report(overall: pd.DataFrame, folds: pd.DataFrame, sig: pd.DataFrame) -> str:
    summary = folds.groupby(["market", "model"]).agg(
        mean_fold_rel=("rel_score", "mean"),
        median_fold_rel=("rel_score", "median"),
        min_fold_rel=("rel_score", "min"),
        positive_folds=("rel_score", lambda series: int((series > 0).sum())),
        folds=("rel_score", "size"),
    ).reset_index()
    best = overall.sort_values(["market", "rel_score"], ascending=[True, False]).groupby("market").head(2)
    return "\n".join(
        [
            "# Multi-Market Portable Baseline Significance Smoke Test",
            "",
            "Protocol: common VN/US/JP portable features, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
            "",
            "## Overall Metrics",
            "",
            overall.round(6).to_markdown(index=False),
            "",
            "## Best Two Baselines Per Market",
            "",
            best.round(6).to_markdown(index=False),
            "",
            "## Fold Summary",
            "",
            summary.round(6).to_markdown(index=False),
            "",
            "## Bootstrap vs Zero",
            "",
            sig.round(6).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This is a pre-model smoke test for the multi-market paper protocol.",
            "- The goal is to confirm schema/date/feature compatibility and identify how hard each market is before expensive LSTM/heteroscedastic ensemble training.",
            "- If ridge/simple baselines are weak but stable, the next step is a portable heteroscedastic ensemble run on US and JP with the same academic report template.",
        ]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for market, path in MARKETS.items():
        rows, folds = evaluate_market(market, path)
        overall_rows.extend(rows)
        fold_rows.extend(folds)
    overall = pd.DataFrame(overall_rows).sort_values(["market", "rel_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    sig = bootstrap_best_vs_zero(folds)
    overall.to_csv(OUTPUT / "overall_portable_baseline_metrics.csv", index=False)
    folds.to_csv(OUTPUT / "fold_portable_baseline_metrics.csv", index=False)
    sig.to_csv(OUTPUT / "bootstrap_vs_zero.csv", index=False)
    overall.to_csv(GOLD / "overall_portable_baseline_metrics.csv", index=False)
    folds.to_csv(GOLD / "fold_portable_baseline_metrics.csv", index=False)
    sig.to_csv(GOLD / "bootstrap_vs_zero.csv", index=False)
    report = build_report(overall, folds, sig)
    (OUTPUT / "multimarket_baseline_report.md").write_text(report, encoding="utf-8")
    (GOLD / "multimarket_baseline_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
