"""Train-selected VN anchor + rich-feature blend router.

Uses cached validation-closed predictions only. Holdout/test is not used.

Goal:
- keep the frozen old anchor as fallback,
- use train-period fold stability to decide whether a rich-feature sidecar blend is promotable,
- evaluate the selected rule on VN validation (2020-04-01..2022-11-15) once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.evaluate_fixed_train_relscore_calibration import choose_train_calibration  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402

DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
OLD_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
RICH_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/vn_rich_feat_full5_preds_20260526/predictions/rich_feat"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/vn_train_selected_blend_router_20260526"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/vn_train_selected_blend_router_20260526"
SEEDS = (43, 52, 62, 71, 82)
TRAIN_END = "2020-03-31"
VAL_END = "2022-11-15"


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


def metric(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[mask]
    pred = pred[mask]
    error = actual - pred
    return {
        "n": int(len(actual)),
        "rel_score": rel_score(actual, pred),
        "absE_robust": robust_loss(error),
        "base_robust": robust_loss(actual),
        "absE_q90": float(np.quantile(np.abs(error), 0.9)) if len(error) else float("nan"),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8))
            if len(actual) else float("nan")
        ),
    }


def fold_ids_from_dates(dates: pd.Series, fold_days: int = 21) -> np.ndarray:
    unique = pd.Series(pd.to_datetime(dates).unique()).sort_values().reset_index(drop=True)
    pos = pd.to_datetime(dates).map({date: idx for idx, date in enumerate(unique)}).to_numpy(dtype=int)
    return pos // fold_days


def fold_metrics(actual: np.ndarray, pred: np.ndarray, dates: pd.Series, label: str) -> pd.DataFrame:
    folds = fold_ids_from_dates(dates)
    rows: list[dict[str, object]] = []
    for fold_id in sorted(set(folds)):
        mask = folds == fold_id
        row = metric(actual[mask], pred[mask])
        fold_dates = pd.to_datetime(dates[mask])
        row.update({"variant": label, "fold_id": int(fold_id), "start": fold_dates.min(), "end": fold_dates.max()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_meta_dates() -> tuple[pd.Series, pd.Series]:
    frame = load_training_frame(DATA, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df = frame.loc[frame["Date"] <= TRAIN_END].copy()
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, features, "target_next_return", 15,
        extra_meta_columns=("__tn__",), sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, TRAIN_END, VAL_END)
    return (
        pd.to_datetime(splits["train"][2].reset_index(drop=True)["Date"]),
        pd.to_datetime(splits["val"][2].reset_index(drop=True)["Date"]),
    )


def load_old_anchor() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_preds: list[np.ndarray] = []
    val_preds: list[np.ndarray] = []
    y_train_ref: np.ndarray | None = None
    y_val_ref: np.ndarray | None = None
    for seed in SEEDS:
        data = np.load(OLD_DIR / f"predictions_seed_{seed}.npz", allow_pickle=True)
        y_train = data["y_train"].astype(np.float32)
        a_best, k_best = choose_train_calibration(y_train, data["mu_train"], data["sigma_train"])
        pred_train = data["mu_train"].astype(np.float32) * a_best
        pred_val = data["mu_val"].astype(np.float32) * a_best
        if k_best is not None:
            pred_train = np.clip(pred_train, -k_best * data["sigma_train"], k_best * data["sigma_train"])
            pred_val = np.clip(pred_val, -k_best * data["sigma_val"], k_best * data["sigma_val"])
        train_preds.append(pred_train)
        val_preds.append(pred_val)
        y_train_ref = y_train
        y_val_ref = data["y_val"].astype(np.float32)
    assert y_train_ref is not None and y_val_ref is not None
    train_mean = np.mean(train_preds, axis=0)
    val_mean = np.mean(val_preds, axis=0)
    a_ens, k_ens = choose_train_calibration(y_train_ref, train_mean, np.full_like(train_mean, 0.02))
    train_final = train_mean * a_ens
    val_final = val_mean * a_ens
    if k_ens is not None:
        train_final = np.clip(train_final, -k_ens * 0.02, k_ens * 0.02)
        val_final = np.clip(val_final, -k_ens * 0.02, k_ens * 0.02)
    return y_train_ref, y_val_ref, train_final.astype(np.float32), val_final.astype(np.float32)


def load_rich_traincal() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mu: list[np.ndarray] = []
    val_mu: list[np.ndarray] = []
    train_sigma: list[np.ndarray] = []
    val_sigma: list[np.ndarray] = []
    y_train_ref: np.ndarray | None = None
    y_val_ref: np.ndarray | None = None
    for seed in SEEDS:
        data = np.load(RICH_DIR / f"predictions_seed_{seed}.npz", allow_pickle=True)
        train_mu.append(data["mu_train"].astype(np.float32))
        val_mu.append(data["mu_val"].astype(np.float32))
        train_sigma.append(data["sigma_train"].astype(np.float32))
        val_sigma.append(data["sigma_val"].astype(np.float32))
        y_train_ref = data["y_train"].astype(np.float32)
        y_val_ref = data["y_val"].astype(np.float32)
    assert y_train_ref is not None and y_val_ref is not None
    train_mean = np.mean(train_mu, axis=0)
    val_mean = np.mean(val_mu, axis=0)
    sigma_train = np.mean(train_sigma, axis=0)
    sigma_val = np.mean(val_sigma, axis=0)
    a_best, k_best = choose_rich_calibration(y_train_ref, train_mean, sigma_train)
    train_final = train_mean * a_best
    val_final = val_mean * a_best
    if k_best is not None:
        train_final = np.clip(train_final, -k_best * sigma_train, k_best * sigma_train)
        val_final = np.clip(val_final, -k_best * sigma_val, k_best * sigma_val)
    return y_train_ref, y_val_ref, train_final.astype(np.float32), val_final.astype(np.float32)


def choose_rich_calibration(y_train: np.ndarray, pred_train: np.ndarray, sigma_train: np.ndarray) -> tuple[float, float | None]:
    best = (float("-inf"), 1.0, None)
    scales = np.round(np.arange(0.0, 1.61, 0.05), 2)
    clips: list[float | None] = [None, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for scale in scales:
        base_pred = pred_train * scale
        for clip in clips:
            pred = base_pred if clip is None else np.clip(base_pred, -clip * sigma_train, clip * sigma_train)
            score = rel_score(y_train, pred)
            if np.isfinite(score) and score > best[0]:
                best = (score, float(scale), clip)
    return best[1], best[2]


def candidate_score(train_folds: pd.DataFrame, old_folds: pd.DataFrame) -> float:
    merged = train_folds[["fold_id", "rel_score"]].merge(
        old_folds[["fold_id", "rel_score"]], on="fold_id", suffixes=("", "_old")
    )
    delta = merged["rel_score"] - merged["rel_score_old"]
    # conservative objective: mean fold delta minus instability and downside penalties
    return float(delta.mean() - 0.25 * delta.std(ddof=1) + 0.02 * (delta > 0).mean() + 0.10 * min(delta.min(), 0.0))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    train_dates, val_dates = load_meta_dates()
    y_train_old, y_val, old_train, old_val = load_old_anchor()
    y_train_rich, y_val_rich, rich_train, rich_val = load_rich_traincal()
    if not np.array_equal(y_train_old, y_train_rich) or not np.array_equal(y_val, y_val_rich):
        raise ValueError("Old and rich prediction targets are not aligned.")

    old_train_folds = fold_metrics(y_train_old, old_train, train_dates, "old_anchor")
    old_val_row = metric(y_val, old_val)
    candidates: list[dict[str, object]] = []
    fold_rows: list[pd.DataFrame] = [old_train_folds]
    val_rows: list[dict[str, object]] = [{**old_val_row, "variant": "old_anchor", "blend_weight_rich": 0.0, "selected_by_train": False}]

    for weight in np.round(np.arange(0.0, 0.81, 0.05), 2):
        train_pred = (1.0 - weight) * old_train + weight * rich_train
        val_pred = (1.0 - weight) * old_val + weight * rich_val
        label = f"blend_rich_{weight:.2f}"
        train_folds = fold_metrics(y_train_old, train_pred, train_dates, label)
        fold_rows.append(train_folds)
        score = candidate_score(train_folds, old_train_folds)
        train_summary = train_folds["rel_score"].agg(["mean", "std", "min"])
        val_metric = metric(y_val, val_pred)
        val_rows.append({**val_metric, "variant": label, "blend_weight_rich": float(weight), "selected_by_train": False})
        candidates.append({
            "variant": label,
            "blend_weight_rich": float(weight),
            "train_objective": score,
            "train_mean_rel": float(train_summary["mean"]),
            "train_std_rel": float(train_summary["std"]),
            "train_min_rel": float(train_summary["min"]),
            "train_positive_folds": int((train_folds["rel_score"] > 0).sum()),
            "val_rel_score": val_metric["rel_score"],
            "val_DA": val_metric["DA"],
        })

    candidate_df = pd.DataFrame(candidates).sort_values("train_objective", ascending=False)
    selected = candidate_df.iloc[0].to_dict()
    old_train_mean = float(old_train_folds["rel_score"].mean())
    old_train_min = float(old_train_folds["rel_score"].min())
    # Promote only if the train-fold objective improves and does not reduce min fold too much.
    promote = bool(
        selected["blend_weight_rich"] > 0
        and selected["train_objective"] > 0.001
        and selected["train_mean_rel"] > old_train_mean + 0.001
        and selected["train_min_rel"] >= old_train_min - 0.005
    )
    selected_variant = selected["variant"] if promote else "old_anchor"

    val_df = pd.DataFrame(val_rows)
    val_df.loc[val_df["variant"] == selected_variant, "selected_by_train"] = True
    fold_df = pd.concat(fold_rows, ignore_index=True)

    candidate_df.to_csv(OUTPUT / "train_candidate_grid.csv", index=False)
    val_df.sort_values("rel_score", ascending=False).to_csv(OUTPUT / "validation_blend_grid.csv", index=False)
    fold_df.to_csv(OUTPUT / "train_fold_metrics.csv", index=False)
    candidate_df.to_csv(GOLD / "train_candidate_grid.csv", index=False)
    val_df.sort_values("rel_score", ascending=False).to_csv(GOLD / "validation_blend_grid.csv", index=False)

    selected_val = val_df.loc[val_df["variant"] == selected_variant].iloc[0].to_dict()
    report_payload = {
        "selected_variant": selected_variant,
        "promote_blend": promote,
        "selected_train_candidate": selected,
        "selected_validation_metric": selected_val,
        "old_anchor_validation_rel_score": old_val_row["rel_score"],
        "output_dir": str(OUTPUT),
        "gold_dir": str(GOLD),
    }
    text = "\n".join([
        "# VN Train-Selected Blend Router",
        "",
        "Protocol: train <= 2020-03-31 for selection; validation 2020-04-01..2022-11-15 for readout. Holdout/test not used.",
        "",
        "## Decision",
        "",
        f"- Selected variant: `{selected_variant}`",
        f"- Promote blend: `{promote}`",
        f"- Selected validation rel_score: `{selected_val['rel_score']:.5f}`",
        f"- Old anchor validation rel_score: `{old_val_row['rel_score']:.5f}`",
        "",
        "## Top Train-Selected Candidates",
        "",
        candidate_df.head(12).round(6).to_markdown(index=False),
        "",
        "## Top Validation Readout (diagnostic, not selection)",
        "",
        val_df.sort_values("rel_score", ascending=False).head(12).round(6).to_markdown(index=False),
        "",
        "## Payload",
        "",
        "```json",
        json.dumps(report_payload, indent=2, default=str),
        "```",
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
