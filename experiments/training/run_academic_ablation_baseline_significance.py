"""Academic ablation, baseline, and significance report.

Validation-only protocol:
- train: all data <= 2020-03-31
- validation/in-sample: 2020-04-01..2022-11-15
- holdout/test: not used

The goal is to turn the current frozen candidate into a paper-grade evidence
package: ablations, simple baselines, fold-level paired improvements, and
bootstrap confidence intervals.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import (  # noqa: E402
    PRED_DIR,
    SEEDS,
    choose_train_calibration,
    metric,
    rel_score,
)
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402

DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/academic_ablation_baseline_significance_20260525"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/academic_ablation_baseline_significance_20260525"
TRAIN_END = "2020-03-31"
VAL_END = "2022-11-15"
FOLD_DAYS = 21


def load_sequence_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    frame = load_training_frame(DATA, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df, _, _ = split_frame_by_date(frame, TRAIN_END, VAL_END)
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        features,
        "target_next_return",
        15,
        extra_meta_columns=("volatility_20",),
        sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, TRAIN_END, VAL_END)
    x_train, y_train, meta_train = splits["train"]
    x_val, y_val, meta_val = splits["val"]
    return x_train, y_train.astype(np.float32), meta_train.reset_index(drop=True), x_val, y_val.astype(np.float32), meta_val.reset_index(drop=True)


def fold_ids(dates: pd.Series) -> np.ndarray:
    unique_dates = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    date_to_pos = {date: idx for idx, date in enumerate(unique_dates)}
    return pd.to_datetime(dates).map(date_to_pos).to_numpy(dtype=int) // FOLD_DAYS


def stock_mean_baseline(y_train: np.ndarray, meta_train: pd.DataFrame, meta_val: pd.DataFrame) -> np.ndarray:
    train = pd.DataFrame({"code": meta_train["code"].to_numpy(), "y": y_train})
    code_means = train.groupby("code")["y"].mean()
    global_mean = float(np.mean(y_train))
    return meta_val["code"].map(code_means).fillna(global_mean).to_numpy(dtype=np.float32)


def recent_stock_mean_baseline(meta_val: pd.DataFrame) -> np.ndarray:
    """Point-in-time lagged 5-observation stock mean within validation only.

    For each validation row this uses only prior validation targets for the
    same stock. The first rows default to zero; it is intentionally simple.
    """
    val = meta_val[["code", "Date", "target"]].copy()
    val["Date"] = pd.to_datetime(val["Date"])
    val = val.sort_values(["code", "Date"])
    pred = (
        val.groupby("code")["target"]
        .transform(lambda series: series.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    val["pred"] = pred
    return val.sort_index()["pred"].to_numpy(dtype=np.float32)


def ridge_baseline(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray) -> np.ndarray:
    model = Ridge(alpha=10.0, fit_intercept=True, random_state=0)
    train_last = x_train[:, -1, :]
    val_last = x_val[:, -1, :]
    model.fit(train_last, y_train)
    return model.predict(val_last).astype(np.float32)


def load_cached_model_variants(y_train: np.ndarray) -> dict[str, np.ndarray]:
    raw_train: list[np.ndarray] = []
    raw_val: list[np.ndarray] = []
    cal_train: list[np.ndarray] = []
    cal_val: list[np.ndarray] = []
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
        scale, _ = choose_train_calibration(data["y_train"], data["mu_train"], data["sigma_train"])
        raw_train.append(data["mu_train"].astype(np.float32))
        raw_val.append(data["mu_val"].astype(np.float32))
        cal_train.append(data["mu_train"].astype(np.float32) * scale)
        cal_val.append(data["mu_val"].astype(np.float32) * scale)

    raw_train_arr = np.vstack(raw_train)
    raw_val_arr = np.vstack(raw_val)
    cal_train_arr = np.vstack(cal_train)
    cal_val_arr = np.vstack(cal_val)

    variants: dict[str, np.ndarray] = {}
    variants["single_seed43_raw"] = raw_val_arr[0]
    variants["single_seed43_train_cal"] = cal_val_arr[0]
    variants["ensemble_mean_raw"] = raw_val_arr.mean(axis=0)
    variants["ensemble_median_train_cal"] = np.median(cal_val_arr, axis=0)
    variants["ensemble_mean_train_cal_each"] = cal_val_arr.mean(axis=0)

    mean_raw_train = raw_train_arr.mean(axis=0)
    mean_raw_val = raw_val_arr.mean(axis=0)
    mean_raw_scale, mean_raw_clip = choose_train_calibration(y_train, mean_raw_train, np.full_like(mean_raw_train, 0.02))
    variants["ensemble_mean_raw_train_cal"] = mean_raw_val * mean_raw_scale
    if mean_raw_clip is not None:
        variants["ensemble_mean_raw_train_cal_clip"] = np.clip(mean_raw_val * mean_raw_scale, -mean_raw_clip * 0.02, mean_raw_clip * 0.02)

    mean_cal_train = cal_train_arr.mean(axis=0)
    mean_cal_val = cal_val_arr.mean(axis=0)
    mean_cal_scale, mean_cal_clip = choose_train_calibration(y_train, mean_cal_train, np.full_like(mean_cal_train, 0.02))
    variants["ensemble_mean_cal_each_train_cal"] = mean_cal_val * mean_cal_scale
    if mean_cal_clip is not None:
        variants["ensemble_mean_cal_each_train_cal_clip"] = np.clip(mean_cal_val * mean_cal_scale, -mean_cal_clip * 0.02, mean_cal_clip * 0.02)

    return variants


def summarize_predictions(
    y_val: np.ndarray,
    meta_val: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = fold_ids(meta_val["Date"])
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for name, pred in predictions.items():
        row = metric(y_val, pred)
        row["model"] = name
        overall_rows.append(row)
        for fold_id in sorted(set(ids)):
            mask = ids == fold_id
            dates = pd.to_datetime(meta_val.loc[mask, "Date"])
            fold_row = metric(y_val[mask], pred[mask])
            fold_row.update(
                {
                    "model": name,
                    "fold_id": int(fold_id),
                    "test_start": dates.min().date().isoformat(),
                    "test_end": dates.max().date().isoformat(),
                }
            )
            fold_rows.append(fold_row)

        daily = pd.DataFrame({"Date": pd.to_datetime(meta_val["Date"]), "actual": y_val, "pred": pred})
        for date, group in daily.groupby("Date", sort=True):
            daily_row = metric(group["actual"].to_numpy(), group["pred"].to_numpy())
            daily_row.update({"model": name, "Date": date.date().isoformat()})
            daily_rows.append(daily_row)
    return (
        pd.DataFrame(overall_rows).sort_values("rel_score", ascending=False),
        pd.DataFrame(fold_rows),
        pd.DataFrame(daily_rows),
    )


def bootstrap_significance(folds: pd.DataFrame, candidate: str, baselines: list[str], n_boot: int = 20000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    pivot = folds.pivot(index="fold_id", columns="model", values="rel_score")
    rows: list[dict[str, object]] = []
    candidate_values = pivot[candidate].dropna()
    for baseline in baselines:
        joined = pd.concat([candidate_values, pivot[baseline]], axis=1, join="inner").dropna()
        joined.columns = ["candidate", "baseline"]
        diff = (joined["candidate"] - joined["baseline"]).to_numpy(dtype=float)
        if len(diff) == 0:
            continue
        indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[indices].mean(axis=1)
        rows.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "n_folds": int(len(diff)),
                "mean_delta_rel_score": float(diff.mean()),
                "median_delta_rel_score": float(np.median(diff)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "p_boot_delta_le_0": float(np.mean(boot <= 0)),
                "positive_delta_folds": int(np.sum(diff > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_delta_rel_score", ascending=False)


def build_report(overall: pd.DataFrame, folds: pd.DataFrame, daily: pd.DataFrame, sig: pd.DataFrame) -> str:
    candidate = "ensemble_mean_cal_each_train_cal_clip"
    top_cols = ["model", "rel_score", "absE_robust", "absE_q90", "DA", "pred_actual_q90_ratio"]
    fold_summary = folds.groupby("model").agg(
        mean_fold_rel=("rel_score", "mean"),
        median_fold_rel=("rel_score", "median"),
        min_fold_rel=("rel_score", "min"),
        positive_folds=("rel_score", lambda series: int((series > 0).sum())),
        folds=("rel_score", "size"),
        mean_absE_robust=("absE_robust", "mean"),
        mean_absE_q90=("absE_q90", "mean"),
    ).reset_index().sort_values("mean_fold_rel", ascending=False)
    year = daily.copy()
    year["year"] = pd.to_datetime(year["Date"]).dt.year
    year_summary = year.groupby(["model", "year"]).agg(
        mean_rel_score=("rel_score", "mean"),
        positive_days=("rel_score", lambda series: int((series > 0).sum())),
        days=("rel_score", "size"),
        mean_absE_robust=("absE_robust", "mean"),
    ).reset_index()
    cand_year = year_summary[year_summary["model"] == candidate]

    return "\n".join(
        [
            "# Academic Ablation + Baseline + Significance Report",
            "",
            "Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
            "",
            "## Overall Ablation And Baseline Ranking",
            "",
            overall[top_cols].round(6).to_markdown(index=False),
            "",
            "## 21-Day Fold Summary",
            "",
            fold_summary.round(6).to_markdown(index=False),
            "",
            "## Bootstrap Significance vs Candidate",
            "",
            sig.round(6).to_markdown(index=False),
            "",
            "## Candidate Yearly Robustness",
            "",
            cand_year.round(6).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The frozen candidate is evaluated against zero, train-only mean baselines, a lagged validation-only baseline, ridge regression, and seed/ensemble ablations.",
            "- Significance is paired by 21-day validation fold and uses bootstrap resampling of fold-level rel_score deltas.",
            "- This package is validation-only evidence for paper development; holdout remains closed.",
        ]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    x_train, y_train, meta_train, x_val, y_val, meta_val = load_sequence_data()

    predictions: dict[str, np.ndarray] = {
        "zero": np.zeros_like(y_val),
        "global_train_mean": np.full_like(y_val, float(np.mean(y_train))),
        "stock_train_mean": stock_mean_baseline(y_train, meta_train, meta_val),
        "lagged_stock_mean5_val_only": recent_stock_mean_baseline(meta_val),
        "ridge_last_step": ridge_baseline(x_train, y_train, x_val),
    }
    predictions.update(load_cached_model_variants(y_train))

    overall, folds, daily = summarize_predictions(y_val, meta_val, predictions)
    candidate = "ensemble_mean_cal_each_train_cal_clip"
    baselines = [name for name in predictions if name != candidate]
    sig = bootstrap_significance(folds, candidate, baselines)

    overall.to_csv(OUTPUT / "overall_ablation_baseline_metrics.csv", index=False)
    folds.to_csv(OUTPUT / "fold_ablation_baseline_metrics.csv", index=False)
    daily.to_csv(OUTPUT / "daily_ablation_baseline_metrics.csv", index=False)
    sig.to_csv(OUTPUT / "paired_bootstrap_significance.csv", index=False)
    overall.to_csv(GOLD / "overall_ablation_baseline_metrics.csv", index=False)
    folds.to_csv(GOLD / "fold_ablation_baseline_metrics.csv", index=False)
    daily.to_csv(GOLD / "daily_ablation_baseline_metrics.csv", index=False)
    sig.to_csv(GOLD / "paired_bootstrap_significance.csv", index=False)

    report = build_report(overall, folds, daily, sig)
    (OUTPUT / "academic_report.md").write_text(report, encoding="utf-8")
    (GOLD / "academic_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
