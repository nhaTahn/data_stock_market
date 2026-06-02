"""P2: σ-aware shrinkage post-processing probe.

Retrains hetero_combined (best from P1) for each seed, predicts (mu, sigma) on
val once, then sweeps several post-processing rules offline.

Variants applied per (seed, prediction matrix):
  - raw mu (= hetero_combined baseline)
  - sigma_shrink_k1.0 / k1.5 / k2.0 / k2.5: p = mu * clip(1 - k*(sigma - sigma_med)/sigma_med, 0.2, 1.0)
  - sigma_clip_k1.5 / k2.0: p = clip(mu, -k*sigma, +k*sigma)
  - sigma_abstain_top10 / top25: zero out samples with sigma in top decile/quartile (per day)
  - selective_strong: keep mu where sigma <= sigma_q40 AND |mu| >= mu_q70 (sample-pool calibration on train)

Reports rel_score, daily_q90_max, spike_days_ge_8pct, DA, pred/actual q90 ratio,
plus coverage for abstention variants.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import (  # noqa: E402
    build_sequence_dataset, split_frame_by_date, split_sequence_dataset,
)
from src.models.training.scalers import (  # noqa: E402
    LocalTargetNormalizer, TargetScaler,
    apply_feature_scaler, apply_local_target_normalizer, apply_target_scaler,
    fit_feature_scaler, fit_local_target_normalizer, fit_target_scaler,
    inverse_local_target_normalizer, inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402

from experiments.training.run_hetero_nll_probe import (  # noqa: E402
    GaussianNLLLoss, CombinedRelScoreNLLLoss,
)

DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "sigma_shrinkage_probe_20260521"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "sigma_shrinkage_probe_20260521"


def robust_loss(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(np.abs(v), 0.5) + 0.5 * np.quantile(np.abs(v), 0.9))


def rel_score_fn(actual: np.ndarray, pred: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    return float(1.0 - robust_loss(actual - pred) / base)


@dataclass(frozen=True)
class PreparedData:
    feature_columns: tuple[str, ...]
    x_train: np.ndarray
    x_val: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_train_model: np.ndarray
    y_val_model: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    train_scale: np.ndarray
    val_scale: np.ndarray
    target_scaler: TargetScaler
    local_normalizer: LocalTargetNormalizer


def load_and_prepare(args) -> PreparedData:
    feature_columns = DEFAULT_FEATURE_COLUMNS
    frame = load_training_frame(args.data, stocks=None)
    required = {"Date", "code", args.target_column, args.target_normalizer, *feature_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    target_alias = "__tn__"
    frame[target_alias] = frame[args.target_normalizer].astype(float)
    train_df, _, _ = split_frame_by_date(frame, args.train_end_date, args.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=feature_columns), feature_columns)
    scaled = apply_feature_scaler(frame, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, feature_columns, args.target_column, args.window_size,
        extra_meta_columns=(target_alias,), sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, args.train_end_date, args.val_end_date)
    x_train, y_train_raw, meta_train = splits["train"]
    x_val, y_val_raw, meta_val = splits["val"]
    train_scale = meta_train[target_alias].to_numpy(dtype=np.float32)
    val_scale = meta_val[target_alias].to_numpy(dtype=np.float32)
    local_norm = fit_local_target_normalizer(train_scale, args.target_normalizer)
    y_train_local = apply_local_target_normalizer(y_train_raw, train_scale, local_norm)
    ts = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, ts).reshape(-1, 1)
    y_train_model = np.concatenate([y_train_scaled, train_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    y_val_local = apply_local_target_normalizer(y_val_raw, val_scale, local_norm)
    y_val_scaled = apply_target_scaler(y_val_local, ts).reshape(-1, 1)
    y_val_model = np.concatenate([y_val_scaled, val_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    return PreparedData(
        feature_columns=feature_columns,
        x_train=x_train, x_val=x_val,
        y_train_raw=y_train_raw.astype(np.float32), y_val_raw=y_val_raw.astype(np.float32),
        y_train_model=y_train_model, y_val_model=y_val_model,
        meta_train=meta_train, meta_val=meta_val,
        train_scale=train_scale, val_scale=val_scale,
        target_scaler=ts, local_normalizer=local_norm,
    )


def build_hetero_combined_model(data: PreparedData, lstm_units, lr, dropout, window_size) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size, num_features=data.x_train.shape[2],
        lstm_units=lstm_units, dropout=dropout,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    model = keras.Model(inputs=inputs, outputs=output)
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=data.target_scaler.mean, target_std=data.target_scaler.std,
        use_target_scaler=True, local_scale_floor=data.local_normalizer.floor,
        high_quantile=0.85, high_weight=1.75, base_weight=1.0,
        tail_error_threshold=0.035, tail_penalty_weight=0.05,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=0.7, w_nll=0.3),
    )
    return model


def predict_raw(model, data, x, scale):
    output = np.asarray(model.predict(x, verbose=0), dtype=np.float32)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    mu_scaled = output[:, 0]
    log_sigma_scaled = output[:, 1]
    sigma_scaled = np.log1p(np.exp(log_sigma_scaled)) + 1e-4
    mu_local = inverse_target_scaler_values(mu_scaled, data.target_scaler)
    mu_raw = inverse_local_target_normalizer(mu_local, scale, data.local_normalizer).reshape(-1)
    sigma_local = sigma_scaled * data.target_scaler.std
    sigma_raw = (sigma_local * np.maximum(np.abs(scale), data.local_normalizer.floor)).reshape(-1)
    return mu_raw.astype(np.float32), sigma_raw.astype(np.float32)


def evaluate(actual, pred, meta, coverage_mask=None):
    if coverage_mask is None:
        coverage_mask = np.ones_like(actual, dtype=bool)
    keep = coverage_mask
    if keep.sum() == 0:
        return {"coverage": 0.0}
    a = actual[keep]
    p = pred[keep]
    m = meta.iloc[keep]
    abs_err = np.abs(a - p)
    daily = pd.DataFrame({"Date": m["Date"].values, "abs_error": abs_err})
    daily_q90 = daily.groupby("Date")["abs_error"].quantile(0.90)
    return {
        "rel_score": rel_score_fn(a, p),
        "daily_q90_p90": float(daily_q90.quantile(0.90)),
        "daily_q90_max": float(daily_q90.max()),
        "median_abs_error": float(np.quantile(abs_err, 0.5)),
        "q90_abs_error": float(np.quantile(abs_err, 0.9)),
        "directional_accuracy": float(np.mean(np.sign(a) == np.sign(p))),
        "spike_days_ge_5pct": int(daily_q90.ge(0.05).sum()),
        "spike_days_ge_8pct": int(daily_q90.ge(0.08).sum()),
        "pred_actual_q90_ratio": float(
            np.quantile(np.abs(p), 0.9) / max(np.quantile(np.abs(a), 0.9), 1e-8)
        ),
        "coverage": float(keep.mean()),
    }


# ---- shrinkage rules ------------------------------------------------------

def shrink_sigma(mu, sigma, k, sigma_med):
    factor = np.clip(1.0 - k * (sigma - sigma_med) / max(sigma_med, 1e-6), 0.2, 1.0)
    return mu * factor


def clip_sigma(mu, sigma, k):
    return np.clip(mu, -k * sigma, k * sigma)


def apply_rules(mu_train, sigma_train, mu_val, sigma_val):
    """Calibrate thresholds on train, return dict of rule -> (pred_val, mask_val)."""
    sigma_med = float(np.median(sigma_train))
    sigma_q40 = float(np.quantile(sigma_train, 0.40))
    sigma_q75 = float(np.quantile(sigma_train, 0.75))
    sigma_q90 = float(np.quantile(sigma_train, 0.90))
    mu_q70 = float(np.quantile(np.abs(mu_train), 0.70))

    n = len(mu_val)
    full_mask = np.ones(n, dtype=bool)
    out = {
        "raw": (mu_val, full_mask),
        "sigma_shrink_k1.0": (shrink_sigma(mu_val, sigma_val, 1.0, sigma_med), full_mask),
        "sigma_shrink_k1.5": (shrink_sigma(mu_val, sigma_val, 1.5, sigma_med), full_mask),
        "sigma_shrink_k2.0": (shrink_sigma(mu_val, sigma_val, 2.0, sigma_med), full_mask),
        "sigma_shrink_k2.5": (shrink_sigma(mu_val, sigma_val, 2.5, sigma_med), full_mask),
        "sigma_clip_k1.5": (clip_sigma(mu_val, sigma_val, 1.5), full_mask),
        "sigma_clip_k2.0": (clip_sigma(mu_val, sigma_val, 2.0), full_mask),
        "abstain_sigma_top25": (mu_val, sigma_val <= sigma_q75),
        "abstain_sigma_top10": (mu_val, sigma_val <= sigma_q90),
        "selective_strong": (
            mu_val,
            (sigma_val <= sigma_q40) & (np.abs(mu_val) >= mu_q70),
        ),
        # combo: shrink+abstain
        "shrink_k1.5_abstain_top10": (
            shrink_sigma(mu_val, sigma_val, 1.5, sigma_med),
            sigma_val <= sigma_q90,
        ),
    }
    return out, {
        "sigma_med": sigma_med, "sigma_q40": sigma_q40,
        "sigma_q75": sigma_q75, "sigma_q90": sigma_q90,
        "mu_q70_abs": mu_q70,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--train-end-date", default="2020-03-31")
    p.add_argument("--val-end-date", default="2022-11-15")
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--lstm-units", default="64,32")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="43,52,71")
    p.add_argument("--target-column", default="target_next_return")
    p.add_argument("--target-normalizer", default="volatility_20")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    lstm_units = [int(s) for s in args.lstm_units.split(",")]
    print("Loading data...")
    data = load_and_prepare(args)
    print(f"Train={len(data.x_train)}, Val={len(data.x_val)}, Features={data.x_train.shape[2]}")

    all_rows = []
    cal_rows = []
    for seed in seeds:
        print(f"-- seed={seed} --")
        set_global_seed(seed)
        model = build_hetero_combined_model(data, lstm_units, args.lr, args.dropout, args.window_size)
        model.fit(
            data.x_train, data.y_train_model,
            validation_data=(data.x_val, data.y_val_model),
            epochs=args.epochs, batch_size=args.batch_size, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True)],
        )
        mu_train, sigma_train = predict_raw(model, data, data.x_train, data.train_scale)
        mu_val, sigma_val = predict_raw(model, data, data.x_val, data.val_scale)
        # Save predictions for this seed for further offline analysis
        np.savez_compressed(
            args.output_dir / f"predictions_seed_{seed}.npz",
            mu_train=mu_train, sigma_train=sigma_train,
            mu_val=mu_val, sigma_val=sigma_val,
            y_train=data.y_train_raw, y_val=data.y_val_raw,
        )
        rules, cal = apply_rules(mu_train, sigma_train, mu_val, sigma_val)
        cal_rows.append({"seed": seed, **cal,
                          "mean_sigma_val": float(np.mean(sigma_val)),
                          "median_sigma_val": float(np.median(sigma_val))})
        for name, (pred, mask) in rules.items():
            metrics = evaluate(data.y_val_raw, pred, data.meta_val, coverage_mask=mask)
            metrics.update({"seed": seed, "rule": name})
            all_rows.append(metrics)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(args.output_dir / "calibration_per_seed.csv", index=False)

    agg_rows = []
    for rule, g in df.groupby("rule"):
        row = {"rule": rule, "n_seeds": len(g)}
        for c in ["rel_score","daily_q90_max","daily_q90_p90","spike_days_ge_8pct",
                   "spike_days_ge_5pct","directional_accuracy","pred_actual_q90_ratio",
                   "coverage","median_abs_error","q90_abs_error"]:
            if c in g.columns:
                v = g[c].astype(float)
                row[f"{c}_mean"] = float(v.mean())
                row[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values("rel_score_mean", ascending=False)
    agg.to_csv(args.output_dir / "results_aggregate.csv", index=False)
    agg.to_csv(args.gold_dir / "results_aggregate.csv", index=False)

    md = agg.to_markdown(index=False)
    text = "\n".join([
        "# σ-Shrinkage Post-Processing Probe Readout",
        "",
        "Priority 2 from improvement plan: σ-aware shrinkage on top of hetero_combined.",
        "Scope: VN train/validation only.  Holdout/test not used.",
        "",
        "## Aggregate Validation (sorted by rel_score_mean)",
        "",
        md,
        "",
        "## Baselines for comparison",
        "- stressaux_w20 (advisor): rel_score 0.0248, spike_8pct 7.7, daily_q90_max 9.44%",
        "- in-script baseline (P1): rel_score 0.0195, spike_8pct 7.0, daily_q90_max 9.71%",
        "- hetero_combined raw (P1): rel_score 0.0372, spike_8pct 13.7, daily_q90_max 11.23%",
    ])
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
