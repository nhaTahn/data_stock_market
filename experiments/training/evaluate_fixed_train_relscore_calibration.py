"""Fixed long-train validation rel_score calibration probe.

Protocol matches the intended target setup:
- Train period: all data <= 2020-03-31 (already trained in hetero_combined_full5)
- Validation/in-sample: 2020-04-01 .. 2022-11-15
- Holdout/test: NOT used

Uses saved 5-seed full-train predictions from hetero_combined_full5_20260521.
Tests post-hoc prediction calibration and gates:
- scale: pred * a, a in [0, 1]
- volatility clipping: clip(pred, ±k * volatility_20)
- gates: none, pressure_only, wyck035, wyck040

Outputs daily/fold metric series for rel_score and abs(E).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import apply_feature_scaler, fit_feature_scaler  # noqa: E402

PRED_DIR = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/hetero_combined_full5_20260521"
OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/fixed_train_relscore_calibration_20260524"
GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/fixed_train_relscore_calibration_20260524"
DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
SEEDS = [43, 52, 62, 71, 82]


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    return float(np.quantile(np.abs(clean), 0.5) + 0.5 * np.quantile(np.abs(clean), 0.9))


def rel_score(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


def metric(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = actual - pred
    abs_error = np.abs(error)
    return {
        "n": int(len(actual)),
        "rel_score": rel_score(actual, pred),
        "absE_robust": robust_loss(error),
        "base_robust": robust_loss(actual),
        "absE_median": float(np.quantile(abs_error, 0.5)) if len(abs_error) else float("nan"),
        "absE_q90": float(np.quantile(abs_error, 0.9)) if len(abs_error) else float("nan"),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8))
            if len(actual) else float("nan")
        ),
    }


def load_meta() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = load_training_frame(DATA, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame["volatility_20"].astype(float)
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_df, _, _ = split_frame_by_date(frame, "2020-03-31", "2022-11-15")
    scaler = fit_feature_scaler(train_df.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, features, "target_next_return", 15,
        extra_meta_columns=("__tn__",), sequence_normalization="none"
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, "2020-03-31", "2022-11-15")
    _, y_val, meta_val = splits["val"]
    return frame, y_val.astype(np.float32), meta_val.reset_index(drop=True)


def build_gates(frame: pd.DataFrame, dates: pd.Series) -> dict[str, np.ndarray]:
    daily = frame.groupby("Date").agg(
        buying_pressure=("buying_pressure", "mean"),
        selling_pressure=("selling_pressure", "mean"),
        wyckoff_phase_60d=("wyckoff_phase_60d", "mean"),
    ).sort_index()
    pressure_delta_20 = (daily["buying_pressure"] - daily["selling_pressure"]).rolling(20, min_periods=5).mean()
    gates_daily = {
        "none": pd.Series(True, index=daily.index),
        "pressure_only": pressure_delta_20 >= 0,
        "wyck035": (pressure_delta_20 >= 0) & (daily["wyckoff_phase_60d"] >= 0.35),
        "wyck040": (pressure_delta_20 >= 0) & (daily["wyckoff_phase_60d"] >= 0.40),
    }
    return {name: dates.map(series).fillna(False).to_numpy(dtype=bool) for name, series in gates_daily.items()}


def choose_train_calibration(y_train: np.ndarray, mu_train: np.ndarray, scale_train: np.ndarray) -> tuple[float, float | None]:
    # Train-only grid. Optimizes rel_score on the fixed train predictions.
    best = (float("-inf"), 1.0, None)
    scales = np.round(np.arange(0.0, 1.26, 0.05), 2)
    clips: list[float | None] = [None, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    for a in scales:
        pred0 = mu_train * a
        for k in clips:
            pred = pred0 if k is None else np.clip(pred0, -k * scale_train, k * scale_train)
            score = rel_score(y_train, pred)
            if np.isfinite(score) and score > best[0]:
                best = (score, float(a), k)
    return best[1], best[2]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    frame, y_val, meta_val = load_meta()
    dates = pd.to_datetime(meta_val["Date"])
    val_scale = meta_val["__tn__"].to_numpy(dtype=np.float32)
    gates = build_gates(frame, dates)

    rows = []
    daily_rows = []
    for seed in SEEDS:
        data = np.load(PRED_DIR / f"predictions_seed_{seed}.npz")
        mu_train = data["mu_train"]
        sigma_train = data["sigma_train"]
        y_train = data["y_train"]
        # train scale not saved; use sigma-based calibration-free for train clip? skip clip train scale by sigma_train proxy
        scale_train_proxy = np.maximum(sigma_train, 1e-4)
        a_best, k_best = choose_train_calibration(y_train, mu_train, scale_train_proxy)
        mu_val = data["mu_val"]
        sigma_val = data["sigma_val"]
        assert len(mu_val) == len(y_val)
        variants = {
            "raw": mu_val,
            "train_cal": mu_val * a_best,
            "shrink_0_50": mu_val * 0.50,
            "shrink_0_75": mu_val * 0.75,
            "clip_vol_1_0": np.clip(mu_val, -1.0 * val_scale, 1.0 * val_scale),
            "cal_clip_sigma": np.clip(mu_val * a_best, -1.0 * sigma_val, 1.0 * sigma_val),
            "zero": np.zeros_like(mu_val),
        }
        for variant, pred_base in variants.items():
            for gate_name, mask in gates.items():
                pred = np.where(mask, pred_base, 0.0)
                m = metric(y_val, pred)
                m.update({"seed": seed, "variant": variant, "gate": gate_name, "coverage_days": float(pd.Series(mask).groupby(dates.values).any().mean()), "train_cal_scale": a_best, "train_cal_clip": -1.0 if k_best is None else k_best})
                rows.append(m)
                tmp = pd.DataFrame({"Date": dates, "actual": y_val, "pred": pred})
                for date, group in tmp.groupby("Date", sort=True):
                    dm = metric(group["actual"].to_numpy(), group["pred"].to_numpy())
                    dm.update({"seed": seed, "variant": variant, "gate": gate_name, "Date": date})
                    daily_rows.append(dm)

    result = pd.DataFrame(rows)
    daily = pd.DataFrame(daily_rows)
    result.to_csv(OUTPUT / "per_seed_metrics.csv", index=False)
    daily.to_csv(OUTPUT / "daily_metric_series.csv", index=False)

    summary = result.groupby(["variant", "gate"]).agg(
        mean_rel_score=("rel_score", "mean"),
        std_rel_score=("rel_score", "std"),
        mean_absE_robust=("absE_robust", "mean"),
        mean_absE_q90=("absE_q90", "mean"),
        mean_DA=("DA", "mean"),
        mean_ratio=("pred_actual_q90_ratio", "mean"),
        mean_train_scale=("train_cal_scale", "mean"),
        coverage_days=("coverage_days", "mean"),
    ).reset_index().sort_values("mean_rel_score", ascending=False)
    summary.to_csv(OUTPUT / "summary_metrics.csv", index=False)
    summary.to_csv(GOLD / "summary_metrics.csv", index=False)
    daily.to_csv(GOLD / "daily_metric_series.csv", index=False)

    cols = ["variant", "gate", "mean_rel_score", "std_rel_score", "mean_absE_robust", "mean_absE_q90", "mean_DA", "mean_ratio", "mean_train_scale", "coverage_days"]
    text = "\n".join([
        "# Fixed Long-Train RelScore Calibration Summary",
        "",
        "Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
        "",
        summary[cols].head(40).round(5).to_markdown(index=False),
    ])
    (OUTPUT / "summary.md").write_text(text, encoding="utf-8")
    (GOLD / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
