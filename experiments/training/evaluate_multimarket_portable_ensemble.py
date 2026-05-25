"""Evaluate portable heteroscedastic ensembles across VN/US/JP.

Inputs are saved predictions from run_hetero_nll_probe.py with
--save-predictions. This script builds train-only calibrated ensembles,
cross-sectional demeaned alpha metrics, and paired bootstrap significance.

Holdout/test is not used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_ensemble_academic_20260525"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/multimarket_portable_ensemble_academic_20260525"
BASELINE_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/multimarket_portable_baseline_significance_20260525"
SEEDS = [43, 52, 62, 71, 82]
MARKET_RUNS = {
    "VN": ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/portable_hetero_5seed_preds_20260525_vn",
    "US": ROOT / "data/processed/assets/data_info_us/history/training_runs/reports/portable_hetero_5seed_preds_20260525_us",
    "JP": ROOT / "data/processed/assets/data_info_jp/history/training_runs/reports/portable_hetero_5seed_preds_20260525_jp",
}
VARIANT = "hetero_combined"
FOLD_DAYS = 21


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


def choose_scale(y_train: np.ndarray, pred_train: np.ndarray) -> float:
    best_score = float("-inf")
    best_scale = 1.0
    for scale in np.round(np.arange(0.0, 1.51, 0.05), 2):
        score = metric(y_train, pred_train * scale)["rel_score"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_scale = float(scale)
    return best_scale


def fold_ids(dates: np.ndarray) -> np.ndarray:
    series = pd.Series(pd.to_datetime(dates))
    unique_dates = series.drop_duplicates().sort_values().reset_index(drop=True)
    date_to_pos = {date: idx for idx, date in enumerate(unique_dates)}
    return series.map(date_to_pos).to_numpy(dtype=int) // FOLD_DAYS


def demean_by_date(actual: np.ndarray, pred: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates), "actual": actual, "pred": pred})
    frame["actual_alpha"] = frame["actual"] - frame.groupby("Date")["actual"].transform("mean")
    frame["pred_alpha"] = frame["pred"] - frame.groupby("Date")["pred"].transform("mean")
    return frame["actual_alpha"].to_numpy(dtype=np.float32), frame["pred_alpha"].to_numpy(dtype=np.float32)


def load_market_predictions(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train: dict[str, list[np.ndarray] | np.ndarray] = {"mu": [], "sigma": []}
    val: dict[str, list[np.ndarray] | np.ndarray] = {"mu": [], "sigma": []}
    for seed in SEEDS:
        data = np.load(run_dir / "predictions" / f"{VARIANT}_seed_{seed}.npz", allow_pickle=True)
        train["mu"].append(data["mu_train"])
        train["sigma"].append(data["sigma_train"])
        val["mu"].append(data["mu_val"])
        val["sigma"].append(data["sigma_val"])
        if seed == SEEDS[0]:
            train["y"] = data["y_train"]
            train["dates"] = data["train_dates"]
            train["codes"] = data["train_codes"]
            val["y"] = data["y_val"]
            val["dates"] = data["val_dates"]
            val["codes"] = data["val_codes"]
    return {k: np.asarray(v) for k, v in train.items()}, {k: np.asarray(v) for k, v in val.items()}


def evaluate_market(market: str, run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train, val = load_market_predictions(run_dir)
    y_train = train["y"].astype(np.float32)
    y_val = val["y"].astype(np.float32)
    val_dates = val["dates"]

    train_mean = train["mu"].mean(axis=0)
    val_mean = val["mu"].mean(axis=0)
    train_median = np.median(train["mu"], axis=0)
    val_median = np.median(val["mu"], axis=0)
    scale_mean = choose_scale(y_train, train_mean)
    scale_median = choose_scale(y_train, train_median)

    train_variants = {
        "ensemble_mean_raw": train_mean,
        "ensemble_mean_train_cal": train_mean * scale_mean,
        "ensemble_median_raw": train_median,
        "ensemble_median_train_cal": train_median * scale_median,
    }
    predictions = {
        "ensemble_mean_raw": val_mean,
        "ensemble_mean_train_cal": val_mean * scale_mean,
        "ensemble_median_raw": val_median,
        "ensemble_median_train_cal": val_median * scale_median,
    }
    for idx, seed in enumerate(SEEDS):
        seed_scale = choose_scale(y_train, train["mu"][idx])
        train_variants[f"seed{seed}_raw"] = train["mu"][idx]
        train_variants[f"seed{seed}_train_cal"] = train["mu"][idx] * seed_scale
        predictions[f"seed{seed}_raw"] = val["mu"][idx]
        predictions[f"seed{seed}_train_cal"] = val["mu"][idx] * seed_scale

    train_scores = {
        name: metric(y_train, pred)["rel_score"]
        for name, pred in train_variants.items()
        if name.startswith("ensemble_")
    }
    selected_name = max(train_scores, key=train_scores.get)
    predictions["selected_train_only_portable_ensemble"] = predictions[selected_name]

    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    ids = fold_ids(val_dates)
    for name, pred in predictions.items():
        row = metric(y_val, pred)
        alpha_actual, alpha_pred = demean_by_date(y_val, pred, val_dates)
        alpha_row = metric(alpha_actual, alpha_pred)
        row.update(
            {
                "market": market,
                "model": name,
                "alpha_rel_score": alpha_row["rel_score"],
                "alpha_absE_robust": alpha_row["absE_robust"],
                "selected_from_train": selected_name if name == "selected_train_only_portable_ensemble" else "",
                "selected_train_rel_score": train_scores[selected_name] if name == "selected_train_only_portable_ensemble" else np.nan,
            }
        )
        overall_rows.append(row)
        alpha_row.update({"market": market, "model": name})
        alpha_rows.append(alpha_row)

        for fold_id in sorted(set(ids)):
            mask = ids == fold_id
            dates = pd.to_datetime(val_dates[mask])
            fold = metric(y_val[mask], pred[mask])
            fa, fp = demean_by_date(y_val[mask], pred[mask], val_dates[mask])
            alpha_fold = metric(fa, fp)
            fold.update(
                {
                    "market": market,
                    "model": name,
                    "fold_id": int(fold_id),
                    "test_start": dates.min().date().isoformat(),
                    "test_end": dates.max().date().isoformat(),
                    "alpha_rel_score": alpha_fold["rel_score"],
                    "alpha_absE_robust": alpha_fold["absE_robust"],
                    "selected_from_train": selected_name if name == "selected_train_only_portable_ensemble" else "",
                }
            )
            fold_rows.append(fold)
    return overall_rows, fold_rows, alpha_rows


def bootstrap_pair(folds: pd.DataFrame, candidate: str, baselines: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for market, group in folds.groupby("market"):
        pivot = group.pivot(index="fold_id", columns="model", values=metric_col)
        if candidate not in pivot:
            continue
        for baseline in [col for col in pivot.columns if col != candidate]:
            joined = pivot[[candidate, baseline]].dropna()
            diff = (joined[candidate] - joined[baseline]).to_numpy(float)
            idx = rng.integers(0, len(diff), size=(20000, len(diff)))
            boot = diff[idx].mean(axis=1)
            rows.append(
                {
                    "market": market,
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric_col,
                    "n_folds": int(len(diff)),
                    "mean_delta": float(diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "p_boot_delta_le_0": float(np.mean(boot <= 0)),
                    "positive_delta_folds": int(np.sum(diff > 0)),
                }
            )

        # Compare with simple baselines loaded from baseline smoke.
        base_market = baselines[baselines["market"] == market]
        for baseline in base_market["model"].unique():
            bfold = base_market[base_market["model"] == baseline].sort_values("fold_id")
            joined = pd.DataFrame({"cand": pivot[candidate]}).reset_index().merge(
                bfold[["fold_id", metric_col]].rename(columns={metric_col: "base"}),
                on="fold_id",
                how="inner",
            ).dropna()
            if joined.empty:
                continue
            diff = (joined["cand"] - joined["base"]).to_numpy(float)
            idx = rng.integers(0, len(diff), size=(20000, len(diff)))
            boot = diff[idx].mean(axis=1)
            rows.append(
                {
                    "market": market,
                    "candidate": candidate,
                    "baseline": f"simple:{baseline}",
                    "metric": metric_col,
                    "n_folds": int(len(diff)),
                    "mean_delta": float(diff.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "p_boot_delta_le_0": float(np.mean(boot <= 0)),
                    "positive_delta_folds": int(np.sum(diff > 0)),
                }
            )
    return pd.DataFrame(rows).sort_values(["market", "metric", "mean_delta"], ascending=[True, True, False])


def build_report(overall: pd.DataFrame, folds: pd.DataFrame, sig: pd.DataFrame) -> str:
    candidate = "selected_train_only_portable_ensemble"
    best = overall.sort_values(["market", "rel_score"], ascending=[True, False]).groupby("market").head(4)
    fold_summary = folds[folds["model"] == candidate].groupby("market").agg(
        mean_fold_rel=("rel_score", "mean"),
        positive_folds=("rel_score", lambda series: int((series > 0).sum())),
        folds=("rel_score", "size"),
        mean_alpha_fold_rel=("alpha_rel_score", "mean"),
        positive_alpha_folds=("alpha_rel_score", lambda series: int((series > 0).sum())),
    ).reset_index()
    sig_focus = sig[
        (sig["candidate"] == candidate)
        & (sig["baseline"].isin(["simple:zero", "simple:ridge_portable", "simple:global_train_mean", "ensemble_mean_raw", "ensemble_median_raw", "ensemble_mean_train_cal"]))
    ]
    return "\n".join(
        [
            "# Multi-Market Portable Ensemble Academic Report",
            "",
            "Protocol: VN/US/JP, common portable features, train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
            "",
            "## Top Ensemble/Ablation Metrics",
            "",
            best[["market", "model", "rel_score", "alpha_rel_score", "absE_robust", "alpha_absE_robust", "DA", "pred_actual_q90_ratio"]].round(6).to_markdown(index=False),
            "",
            "## Candidate Fold Summary",
            "",
            fold_summary.round(6).to_markdown(index=False),
            "",
            "## Bootstrap Significance Focus",
            "",
            sig_focus.round(6).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- `selected_train_only_portable_ensemble` is selected separately per market using train rel_score only among raw/calibrated mean/median ensembles.",
            "- The regular rel_score includes market drift; `alpha_rel_score` demeans each date cross-section and is a stricter stock-selection metric.",
            "- This remains validation-only evidence. Holdout/test remains closed.",
        ]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    for market, run_dir in MARKET_RUNS.items():
        rows, folds, alpha = evaluate_market(market, run_dir)
        overall_rows.extend(rows)
        fold_rows.extend(folds)
        alpha_rows.extend(alpha)
    overall = pd.DataFrame(overall_rows).sort_values(["market", "rel_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    alpha = pd.DataFrame(alpha_rows)
    baseline_folds = pd.read_csv(BASELINE_DIR / "fold_portable_baseline_metrics.csv")
    baseline_folds["alpha_rel_score"] = np.nan
    sig = pd.concat(
        [
            bootstrap_pair(folds, "selected_train_only_portable_ensemble", baseline_folds, "rel_score"),
            bootstrap_pair(folds, "selected_train_only_portable_ensemble", baseline_folds, "alpha_rel_score"),
        ],
        ignore_index=True,
    )
    for df, name in [
        (overall, "overall_metrics.csv"),
        (folds, "fold_metrics.csv"),
        (alpha, "alpha_overall_metrics.csv"),
        (sig, "bootstrap_significance.csv"),
    ]:
        df.to_csv(OUTPUT / name, index=False)
        df.to_csv(GOLD / name, index=False)
    report = build_report(overall, folds, sig)
    (OUTPUT / "academic_report.md").write_text(report, encoding="utf-8")
    (GOLD / "academic_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
