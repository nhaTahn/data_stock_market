"""Fixed long-train seed ensemble calibration probe.

Protocol:
- train period: all data <= 2020-03-31
- validation/in-sample: 2020-04-01 .. 2022-11-15
- holdout/test: not used

Uses cached hetero_combined full-train predictions and evaluates whether
ensembling seeds improves rel_score and abs(E) stability before holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import (  # noqa: E402
    PRED_DIR,
    SEEDS,
    choose_train_calibration,
    load_meta,
    metric,
)

OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_ensemble_calibration_20260524"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/fixed_train_ensemble_calibration_20260524"


def date_fold_ids(dates: pd.Series, fold_days: int = 21) -> np.ndarray:
    unique_dates = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    date_to_pos = {date: idx for idx, date in enumerate(unique_dates)}
    positions = pd.to_datetime(dates).map(date_to_pos).to_numpy(dtype=int)
    return positions // fold_days


def load_seed_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    y_train_ref: np.ndarray | None = None
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
        y_train = data["y_train"].astype(np.float32)
        scale, _ = choose_train_calibration(y_train, data["mu_train"], data["sigma_train"])
        train_preds.append(data["mu_train"].astype(np.float32) * scale)
        val_preds.append(data["mu_val"].astype(np.float32) * scale)
        y_train_ref = y_train if y_train_ref is None else y_train_ref
    if y_train_ref is None:
        raise RuntimeError("No cached seed predictions found.")
    return np.vstack(train_preds), np.vstack(val_preds), y_train_ref


def build_variants(train_preds: np.ndarray, val_preds: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    raw_train: list[np.ndarray] = []
    raw_val: list[np.ndarray] = []
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
        raw_train.append(data["mu_train"].astype(np.float32))
        raw_val.append(data["mu_val"].astype(np.float32))

    variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ensemble_mean_cal_each": (train_preds.mean(axis=0), val_preds.mean(axis=0)),
        "ensemble_median_cal_each": (np.median(train_preds, axis=0), np.median(val_preds, axis=0)),
        "ensemble_mean_raw": (np.mean(raw_train, axis=0), np.mean(raw_val, axis=0)),
    }

    output: dict[str, np.ndarray] = {}
    for name, (train_pred, val_pred) in variants.items():
        scale, clip = choose_train_calibration(y_train, train_pred, np.full_like(train_pred, 0.02))
        output[name] = val_pred
        output[f"{name}_traincal"] = val_pred * scale
        if clip is not None:
            output[f"{name}_traincal_clip"] = np.clip(val_pred * scale, -clip * 0.02, clip * 0.02)
    return output


def summarize_folds(y_val: np.ndarray, pred: np.ndarray, dates: pd.Series, variant: str) -> list[dict[str, object]]:
    fold_ids = date_fold_ids(dates)
    rows: list[dict[str, object]] = []
    for fold_id in sorted(set(fold_ids)):
        mask = fold_ids == fold_id
        fold_dates = pd.to_datetime(dates[mask])
        row = metric(y_val[mask], pred[mask])
        row.update(
            {
                "variant": variant,
                "fold_id": int(fold_id),
                "test_start": fold_dates.min().date().isoformat(),
                "test_end": fold_dates.max().date().isoformat(),
            }
        )
        rows.append(row)
    return rows


def summarize_days(y_val: np.ndarray, pred: np.ndarray, dates: pd.Series, variant: str) -> list[dict[str, object]]:
    daily = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": y_val, "pred": pred})
    rows: list[dict[str, object]] = []
    for date, group in daily.groupby("Date", sort=True):
        row = metric(group["actual"].to_numpy(), group["pred"].to_numpy())
        row.update({"variant": variant, "Date": date.date().isoformat()})
        rows.append(row)
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    _, y_val, meta_val = load_meta()
    dates = pd.to_datetime(meta_val["Date"])

    train_preds, val_preds, y_train = load_seed_predictions()
    variants = build_variants(train_preds, val_preds, y_train)

    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for variant, pred in variants.items():
        row = metric(y_val, pred)
        row.update({"variant": variant})
        overall_rows.append(row)
        fold_rows.extend(summarize_folds(y_val, pred, dates, variant))
        daily_rows.extend(summarize_days(y_val, pred, dates, variant))

    overall = pd.DataFrame(overall_rows).sort_values("rel_score", ascending=False)
    folds = pd.DataFrame(fold_rows)
    daily = pd.DataFrame(daily_rows)
    fold_summary = folds.groupby("variant").agg(
        mean_fold_rel=("rel_score", "mean"),
        median_fold_rel=("rel_score", "median"),
        min_fold_rel=("rel_score", "min"),
        positive_folds=("rel_score", lambda series: int((series > 0).sum())),
        folds=("rel_score", "size"),
        mean_absE_robust=("absE_robust", "mean"),
        p90_absE_robust=("absE_robust", lambda series: float(series.quantile(0.9))),
        mean_absE_q90=("absE_q90", "mean"),
    ).reset_index().sort_values("mean_fold_rel", ascending=False)

    overall.to_csv(OUTPUT / "overall_metrics.csv", index=False)
    folds.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    daily.to_csv(OUTPUT / "daily_metrics.csv", index=False)
    fold_summary.to_csv(OUTPUT / "fold_summary.csv", index=False)
    overall.to_csv(GOLD / "overall_metrics.csv", index=False)
    folds.to_csv(GOLD / "fold_metrics.csv", index=False)
    daily.to_csv(GOLD / "daily_metrics.csv", index=False)
    fold_summary.to_csv(GOLD / "fold_summary.csv", index=False)

    text = "\n".join(
        [
            "# Fixed Long-Train Ensemble Calibration",
            "",
            "Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
            "",
            "## Overall Metrics",
            "",
            overall.round(5).to_markdown(index=False),
            "",
            "## 21-Day Fold Metrics",
            "",
            fold_summary.round(5).to_markdown(index=False),
        ]
    )
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
