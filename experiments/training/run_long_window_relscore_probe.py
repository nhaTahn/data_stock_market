"""Long-window rolling rel_score probe.

Validation-only robustness test. Holdout/test is NOT used.

This fixes the previous issue where w126 was too short for LSTM calibration.
It tests larger train windows and expanding-train folds:
- w504/t21/s21
- w756/t21/s21
- expanding/t21/s21

For each fold/seed:
1. Train hetero_combined on the selected training window.
2. Predict train and validation fold.
3. Choose calibration scale on train fold: pred = a * pred, a in [0, 1.25].
4. Evaluate raw and calibrated rel_score / abs(E) on test fold.

Outputs:
- per_fold_metrics.csv
- summary.csv
- daily_metric_series.csv
- summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from experiments.training.run_hetero_nll_probe import CombinedRelScoreNLLLoss  # noqa: E402

DEFAULT_DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/long_window_relscore_probe_20260524"
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/long_window_relscore_probe_20260524"

VAL_START = pd.Timestamp("2020-04-01")
VAL_END = pd.Timestamp("2022-11-15")
TEST_DAYS = 21
STEP_DAYS = 21


def parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
    error = np.asarray(actual) - np.asarray(pred)
    abs_error = np.abs(error)
    return {
        "n": int(len(actual)),
        "rel_score": rel_score(actual, pred),
        "absE_robust": robust_loss(error),
        "base_robust": robust_loss(actual),
        "absE_q90": float(np.quantile(abs_error, 0.9)) if len(abs_error) else float("nan"),
        "absE_median": float(np.quantile(abs_error, 0.5)) if len(abs_error) else float("nan"),
        "DA": float(np.mean(np.sign(actual) == np.sign(pred))) if len(actual) else float("nan"),
        "pred_actual_q90_ratio": (
            float(np.quantile(np.abs(pred), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8))
            if len(actual) else float("nan")
        ),
    }


def build_model(num_features: int, target_scaler: TargetScaler, local_norm: LocalTargetNormalizer, args: argparse.Namespace) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=num_features,
        lstm_units=[int(x) for x in args.lstm_units.split(",")],
        dropout=args.dropout,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=target_scaler.mean,
        target_std=target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=local_norm.floor,
        high_quantile=0.85,
        high_weight=1.75,
        base_weight=1.0,
        tail_error_threshold=0.035,
        tail_penalty_weight=0.05,
    )
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=args.w_rel, w_nll=args.w_nll),
    )
    return model


def predict_mu(model: keras.Model, x: np.ndarray, scale: np.ndarray, target_scaler: TargetScaler, local_norm: LocalTargetNormalizer) -> np.ndarray:
    out = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    mu_scaled = out[:, 0] if out.ndim == 2 else out.reshape(-1)
    mu_local = inverse_target_scaler_values(mu_scaled, target_scaler)
    return inverse_local_target_normalizer(mu_local, scale, local_norm).reshape(-1).astype(np.float32)


def choose_scale(y_train: np.ndarray, pred_train: np.ndarray) -> tuple[float, float]:
    best_score = -1e9
    best_scale = 1.0
    for scale in np.round(np.arange(0.0, 1.26, 0.05), 2):
        score = rel_score(y_train, pred_train * scale)
        if np.isfinite(score) and score > best_score:
            best_score = float(score)
            best_scale = float(scale)
    return best_scale, best_score


def build_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = load_training_frame(args.data, stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["__tn__"] = frame[args.target_normalizer].astype(float)
    features = tuple(DEFAULT_FEATURE_COLUMNS)
    train_full = frame.loc[frame["Date"] <= "2020-03-31"].copy()
    scaler = fit_feature_scaler(train_full.dropna(subset=features), features)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, features, args.target_column, args.window_size,
        extra_meta_columns=("__tn__",), sequence_normalization="none"
    )
    return x_all, y_all.astype(np.float32), meta_all.reset_index(drop=True)


def make_folds(meta: pd.DataFrame, mode: str) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    val_days = pd.DatetimeIndex(sorted(set(meta.loc[(meta["Date"] >= VAL_START) & (meta["Date"] <= VAL_END), "Date"].values)))
    all_days = pd.DatetimeIndex(sorted(set(meta["Date"].values)))
    folds = []
    if mode == "w504":
        train_days = 504
    elif mode == "w756":
        train_days = 756
    elif mode == "expanding":
        train_days = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    start_idx = 0
    while start_idx + TEST_DAYS <= len(val_days):
        test_dates = val_days[start_idx:start_idx + TEST_DAYS]
        before = all_days[all_days < test_dates[0]]
        if train_days is None:
            train_dates = before
        else:
            if len(before) < train_days:
                start_idx += STEP_DAYS
                continue
            train_dates = before[-train_days:]
        if len(train_dates) > 0:
            folds.append((train_dates, test_dates))
        start_idx += STEP_DAYS
    return folds


def date_mask(meta: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(meta["Date"].values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"))


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    x_all, y_all, meta = build_inputs(args)
    seeds = parse_seeds(args.seeds)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    fold_rows = []
    daily_rows = []

    for mode in modes:
        folds = make_folds(meta, mode)
        if args.max_folds is not None:
            folds = folds[:args.max_folds]
        print(f"Mode={mode} folds={len(folds)}")
        for seed in seeds:
            for fold_id, (train_dates, test_dates) in enumerate(folds):
                cache = pred_dir / f"{mode}_seed{seed}_fold{fold_id:03d}.npz"
                train_mask = date_mask(meta, train_dates)
                test_mask = date_mask(meta, test_dates)
                meta_train = meta.loc[train_mask].reset_index(drop=True)
                meta_test = meta.loc[test_mask].reset_index(drop=True)
                if cache.exists() and not args.force:
                    data = np.load(cache)
                    pred_train = data["pred_train"]
                    pred_test = data["pred_test"]
                else:
                    print(f"train {mode} seed={seed} fold={fold_id} train_days={len(train_dates)} test={test_dates[0].date()}..{test_dates[-1].date()} n={train_mask.sum()}/{test_mask.sum()}")
                    train_scale = meta_train["__tn__"].to_numpy(dtype=np.float32)
                    test_scale = meta_test["__tn__"].to_numpy(dtype=np.float32)
                    local_norm = fit_local_target_normalizer(train_scale, args.target_normalizer)
                    y_train_local = apply_local_target_normalizer(y_all[train_mask], train_scale, local_norm)
                    target_scaler = fit_target_scaler(y_train_local)
                    y_train_scaled = apply_target_scaler(y_train_local, target_scaler).reshape(-1, 1)
                    y_train_model = np.concatenate([y_train_scaled, train_scale.reshape(-1, 1)], axis=1).astype(np.float32)
                    set_global_seed(seed)
                    model = build_model(x_all[train_mask].shape[2], target_scaler, local_norm, args)
                    model.fit(
                        x_all[train_mask], y_train_model,
                        epochs=args.epochs, batch_size=args.batch_size, verbose=0,
                        callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=args.patience, restore_best_weights=True)],
                    )
                    pred_train = predict_mu(model, x_all[train_mask], train_scale, target_scaler, local_norm)
                    pred_test = predict_mu(model, x_all[test_mask], test_scale, target_scaler, local_norm)
                    np.savez_compressed(cache, pred_train=pred_train, pred_test=pred_test, y_train=y_all[train_mask], y_test=y_all[test_mask])
                scale, train_rel = choose_scale(y_all[train_mask], pred_train)
                variants = {"raw": pred_test, "cal": pred_test * scale, "shrink_075": pred_test * 0.75, "zero": np.zeros_like(pred_test)}
                for variant, pred in variants.items():
                    row = metric(y_all[test_mask], pred)
                    row.update({
                        "mode": mode, "seed": seed, "fold_id": fold_id,
                        "test_start": str(test_dates[0].date()), "test_end": str(test_dates[-1].date()),
                        "train_days": len(train_dates), "variant": variant,
                        "cal_scale": scale, "train_rel_score": train_rel,
                    })
                    fold_rows.append(row)
                    tmp = pd.DataFrame({"Date": meta_test["Date"].values, "actual": y_all[test_mask], "pred": pred})
                    for date, group in tmp.groupby("Date", sort=True):
                        drow = metric(group["actual"].to_numpy(), group["pred"].to_numpy())
                        drow.update({"mode": mode, "seed": seed, "fold_id": fold_id, "variant": variant, "Date": date})
                        daily_rows.append(drow)
                pd.DataFrame(fold_rows).to_csv(args.output_dir / "per_fold_metrics.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    daily_df = pd.DataFrame(daily_rows)
    fold_df.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    daily_df.to_csv(args.output_dir / "daily_metric_series.csv", index=False)

    summary = fold_df.groupby(["mode", "variant"]).agg(
        mean_rel_score=("rel_score", "mean"),
        median_rel_score=("rel_score", "median"),
        positive_folds=("rel_score", lambda x: int((x > 0).sum())),
        n_folds=("rel_score", "size"),
        mean_absE_robust=("absE_robust", "mean"),
        p90_absE_robust=("absE_robust", lambda x: float(x.quantile(0.9))),
        mean_absE_q90=("absE_q90", "mean"),
        mean_DA=("DA", "mean"),
        mean_ratio=("pred_actual_q90_ratio", "mean"),
        mean_cal_scale=("cal_scale", "mean"),
        mean_train_rel=("train_rel_score", "mean"),
    ).reset_index().sort_values(["mean_rel_score"], ascending=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    summary.to_csv(args.gold_dir / "summary_metrics.csv", index=False)
    fold_df.to_csv(args.gold_dir / "per_fold_metrics.csv", index=False)

    cols = ["mode", "variant", "mean_rel_score", "median_rel_score", "positive_folds", "n_folds", "mean_absE_robust", "p90_absE_robust", "mean_absE_q90", "mean_DA", "mean_ratio", "mean_cal_scale", "mean_train_rel"]
    text = "\n".join([
        "# Long-Window Rolling RelScore Probe",
        "",
        "Validation only. Holdout/test not used.",
        "",
        summary[cols].round(5).to_markdown(index=False),
    ])
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--seeds", default="43,52,62")
    p.add_argument("--modes", default="w504,w756,expanding")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--lstm-units", default="64,32")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--target-column", default="target_next_return")
    p.add_argument("--target-normalizer", default="volatility_20")
    p.add_argument("--w-rel", type=float, default=0.7)
    p.add_argument("--w-nll", type=float, default=0.3)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
