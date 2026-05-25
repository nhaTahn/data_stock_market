"""Priority 1: Heteroscedastic NLL LSTM probe.

Architecture: same LSTM backbone [64,32], but with TWO output heads:
  - mu head: predict mean return (same as current)
  - sigma head: predict conditional std (softplus activation)

Loss: Gaussian NLL
  L = (1/N) sum [ (r - mu)^2 / (2*sigma^2) + 0.5*log(sigma^2) ]

This allows the model to learn WHEN to predict large (low sigma = confident)
and WHEN to shrink toward zero (high sigma = uncertain).

Comparison variants:
  - baseline: current stressaux_w20 (rel_score_weighted_tail loss)
  - hetero_nll: pure Gaussian NLL loss
  - hetero_combined: 0.7*rel_score_weighted + 0.3*NLL (keep rel_score signal + vol awareness)

Seeds: 43, 52, 71. Epochs: 18. Patience: 5.
Evaluate on raw return space using rel_score + spike metrics.
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
from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    LocalTargetNormalizer, TargetScaler,
    apply_feature_scaler, apply_local_target_normalizer, apply_target_scaler,
    fit_feature_scaler, fit_local_target_normalizer, fit_target_scaler,
    inverse_local_target_normalizer, inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "hetero_nll_probe_20260521"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "hetero_nll_probe_20260521"
MARKET_CONTEXT_FEATURES = (
    "market_close_return",
    "market_return_5",
    "market_return_20",
    "market_volatility_20",
    "market_breadth_pos_ratio",
    "market_abs_return",
    "cs_momentum_20_rank",
    "cs_volatility_20_rank",
    "cs_ma_20_gap_rank",
    "cs_volume_change_rank",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Comma-separated feature columns. Defaults to VN DEFAULT_FEATURE_COLUMNS.",
    )
    parser.add_argument(
        "--variants",
        default="baseline,hetero_nll,hetero_combined",
        help="Comma-separated variants to run: baseline,hetero_nll,hetero_combined.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save train/validation mu/sigma/y/meta arrays for downstream ensemble analysis.",
    )
    parser.add_argument(
        "--add-market-context-features",
        action="store_true",
        help="Generate market-level and cross-sectional context adapter features before training.",
    )
    return parser.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def parse_lstm_units(value: str) -> list[int]:
    return [int(s.strip()) for s in value.split(",") if s.strip()]

def parse_features(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_FEATURE_COLUMNS
    features = tuple(item.strip() for item in value.split(",") if item.strip())
    if not features:
        raise ValueError("--feature-columns was provided but no columns were parsed.")
    return features

def add_market_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add portable market context and cross-sectional adapter features."""
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    daily = out.groupby("Date").agg(
        market_close_return=("close_return", "mean"),
        market_breadth_pos_ratio=("close_return", lambda series: float((series > 0).mean())),
    ).sort_index()
    daily["market_return_5"] = daily["market_close_return"].rolling(5, min_periods=2).mean()
    daily["market_return_20"] = daily["market_close_return"].rolling(20, min_periods=5).mean()
    daily["market_volatility_20"] = daily["market_close_return"].rolling(20, min_periods=5).std()
    daily["market_abs_return"] = daily["market_close_return"].abs()
    for column in daily.columns:
        out[column] = out["Date"].map(daily[column])
    out["cs_momentum_20_rank"] = out.groupby("Date")["momentum_20"].rank(pct=True)
    out["cs_volatility_20_rank"] = out.groupby("Date")["volatility_20"].rank(pct=True)
    out["cs_ma_20_gap_rank"] = out.groupby("Date")["ma_20_gap"].rank(pct=True)
    out["cs_volume_change_rank"] = out.groupby("Date")["volume_change"].rank(pct=True)
    return out


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


# --- Custom Gaussian NLL loss for Keras ---

@keras.utils.register_keras_serializable(package="hetero")
class GaussianNLLLoss(keras.losses.Loss):
    """Heteroscedastic Gaussian negative log-likelihood.

    y_pred shape: (batch, 2) where [:, 0] = mu, [:, 1] = log_sigma.
    y_true shape: (batch, 1) or (batch, 2) with [:, 0] = target.
    """
    def __init__(self, min_sigma: float = 1e-4, name: str = "gaussian_nll"):
        super().__init__(name=name)
        self.min_sigma = float(min_sigma)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Handle rel_score-style 2-col y_true (col0=target, col1=scale)
        if y_true.shape.rank is not None and y_true.shape.rank >= 2:
            target = y_true[:, 0:1]
        else:
            target = tf.reshape(y_true, [-1, 1])
        mu = y_pred[:, 0:1]
        log_sigma = y_pred[:, 1:2]
        sigma = tf.nn.softplus(log_sigma) + self.min_sigma
        nll = 0.5 * tf.square((target - mu) / sigma) + tf.math.log(sigma)
        return tf.reduce_mean(nll)

    def get_config(self):
        return {"name": self.name, "min_sigma": self.min_sigma}


@keras.utils.register_keras_serializable(package="hetero")
class CombinedRelScoreNLLLoss(keras.losses.Loss):
    """Combined: w_rel * RelScoreWeightedTail + w_nll * GaussianNLL."""
    def __init__(self, rel_loss, w_rel: float = 0.7, w_nll: float = 0.3,
                 min_sigma: float = 1e-4, name: str = "combined_rel_nll"):
        super().__init__(name=name)
        self.rel_loss = rel_loss
        self.nll_loss = GaussianNLLLoss(min_sigma=min_sigma)
        self.w_rel = float(w_rel)
        self.w_nll = float(w_nll)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mu = y_pred[:, 0:1]
        rel_component = self.rel_loss(y_true, mu)
        nll_component = self.nll_loss(y_true, y_pred)
        return self.w_rel * rel_component + self.w_nll * nll_component

    def get_config(self):
        return {"name": self.name, "w_rel": self.w_rel, "w_nll": self.w_nll}


# --- Data preparation ---

@dataclass(frozen=True)
class PreparedData:
    feature_columns: tuple[str, ...]
    x_train: np.ndarray
    x_val: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_train_model: np.ndarray  # (N, 2): [scaled_target, local_scale]
    y_val_model: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    train_scale: np.ndarray
    val_scale: np.ndarray
    target_scaler: TargetScaler
    local_normalizer: LocalTargetNormalizer


def load_and_prepare(args: argparse.Namespace) -> PreparedData:
    feature_columns = parse_features(args.feature_columns)
    frame = load_training_frame(args.data, stocks=None)
    if args.add_market_context_features:
        frame = add_market_context_features(frame)
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


# --- Model builders ---

def build_baseline_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    """Current best: single output with rel_score_weighted_tail loss."""
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size, num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units), dropout=args.dropout,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    model = keras.Model(inputs=inputs, outputs=pred)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=RelScoreWeightedTailLoss(
            target_mean=data.target_scaler.mean, target_std=data.target_scaler.std,
            use_target_scaler=True, local_scale_floor=data.local_normalizer.floor,
            high_quantile=0.85, high_weight=1.75, base_weight=1.0,
            tail_error_threshold=0.035, tail_penalty_weight=0.05,
        ),
    )
    return model


def build_hetero_nll_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    """Heteroscedastic: output (mu, log_sigma), trained with Gaussian NLL."""
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size, num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units), dropout=args.dropout,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=GaussianNLLLoss(min_sigma=1e-4),
    )
    return model


def build_hetero_combined_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    """Combined: 0.7*rel_score_weighted_tail + 0.3*NLL."""
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size, num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units), dropout=args.dropout,
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
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=0.7, w_nll=0.3),
    )
    return model


# --- Training and evaluation ---

def train_model(model: keras.Model, data: PreparedData, args: argparse.Namespace, seed: int) -> dict:
    set_global_seed(seed)
    history = model.fit(
        data.x_train, data.y_train_model,
        validation_data=(data.x_val, data.y_val_model),
        epochs=args.epochs, batch_size=args.batch_size, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
    )
    return history.history


def predict_raw(model: keras.Model, data: PreparedData, x: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu_raw, sigma_raw). For baseline model sigma is zeros."""
    output = model.predict(x, verbose=0)
    output = np.asarray(output, dtype=np.float32)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    if output.shape[1] >= 2:
        mu_scaled = output[:, 0]
        log_sigma_scaled = output[:, 1]
        sigma_scaled = np.log1p(np.exp(log_sigma_scaled)) + 1e-4  # softplus
    else:
        mu_scaled = output[:, 0]
        sigma_scaled = np.zeros_like(mu_scaled)
    # Inverse transform mu
    mu_local = inverse_target_scaler_values(mu_scaled, data.target_scaler)
    mu_raw = inverse_local_target_normalizer(mu_local, scale, data.local_normalizer).reshape(-1)
    # Inverse transform sigma (same scale transform)
    sigma_local = sigma_scaled * data.target_scaler.std  # undo z-score std
    sigma_raw = (sigma_local * np.maximum(np.abs(scale), data.local_normalizer.floor)).reshape(-1)
    return mu_raw.astype(np.float32), sigma_raw.astype(np.float32)


def evaluate_predictions(actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray, meta: pd.DataFrame) -> dict:
    abs_error = np.abs(actual - mu)
    daily = pd.DataFrame({"Date": meta["Date"].values, "abs_error": abs_error})
    daily_q90 = daily.groupby("Date")["abs_error"].quantile(0.90)
    # Sigma-clipped prediction: clip mu at ±2*sigma
    if np.any(sigma > 0):
        mu_clipped = np.clip(mu, mu - 2 * sigma, mu + 2 * sigma)  # no-op for symmetric
        # Actually clip magnitude: |mu_clipped| <= 2*sigma
        mu_vol_clipped = np.where(sigma > 0, np.clip(mu, -2 * sigma, 2 * sigma), mu)
        abs_error_clipped = np.abs(actual - mu_vol_clipped)
        daily_clipped = pd.DataFrame({"Date": meta["Date"].values, "abs_error": abs_error_clipped})
        daily_q90_clipped = daily_clipped.groupby("Date")["abs_error"].quantile(0.90)
    else:
        mu_vol_clipped = mu
        daily_q90_clipped = daily_q90
    return {
        "rel_score": rel_score_fn(actual, mu),
        "rel_score_vol_clipped": rel_score_fn(actual, mu_vol_clipped),
        "median_abs_error": float(np.quantile(abs_error, 0.5)),
        "q90_abs_error": float(np.quantile(abs_error, 0.9)),
        "daily_q90_p90": float(daily_q90.quantile(0.90)),
        "daily_q90_max": float(daily_q90.max()),
        "daily_q90_clipped_p90": float(daily_q90_clipped.quantile(0.90)),
        "daily_q90_clipped_max": float(daily_q90_clipped.max()),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(mu))),
        "spike_days_ge_5pct": int(daily_q90.ge(0.05).sum()),
        "spike_days_ge_8pct": int(daily_q90.ge(0.08).sum()),
        "spike_days_clipped_ge_8pct": int(daily_q90_clipped.ge(0.08).sum()),
        "mean_sigma": float(np.mean(sigma)) if np.any(sigma > 0) else 0.0,
        "median_sigma": float(np.median(sigma)) if np.any(sigma > 0) else 0.0,
        "pred_actual_q90_ratio": float(np.quantile(np.abs(mu), 0.9) / max(np.quantile(np.abs(actual), 0.9), 1e-8)),
    }


VARIANTS = ["baseline", "hetero_nll", "hetero_combined"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; allowed={VARIANTS}")
    print(f"Loading data...")
    data = load_and_prepare(args)
    print(f"Train: {len(data.x_train)}, Val: {len(data.x_val)}, Features: {data.x_train.shape[2]}")

    results: list[dict] = []
    for seed in seeds:
        for variant in variants:
            print(f"  {variant} seed={seed}")
            set_global_seed(seed)
            if variant == "baseline":
                model = build_baseline_model(data, args)
            elif variant == "hetero_nll":
                model = build_hetero_nll_model(data, args)
            elif variant == "hetero_combined":
                model = build_hetero_combined_model(data, args)
            else:
                raise ValueError(f"Unknown variant: {variant}")
            train_model(model, data, args, seed)
            mu_val, sigma_val = predict_raw(model, data, data.x_val, data.val_scale)
            metrics = evaluate_predictions(data.y_val_raw, mu_val, sigma_val, data.meta_val)
            metrics.update({"variant": variant, "seed": seed, "split": "val"})
            results.append(metrics)
            # Also evaluate on train for overfit check
            mu_train, sigma_train = predict_raw(model, data, data.x_train, data.train_scale)
            metrics_train = evaluate_predictions(data.y_train_raw, mu_train, sigma_train, data.meta_train)
            metrics_train.update({"variant": variant, "seed": seed, "split": "train"})
            results.append(metrics_train)
            if args.save_predictions:
                pred_dir = args.output_dir / "predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    pred_dir / f"{variant}_seed_{seed}.npz",
                    mu_train=mu_train.astype(np.float32),
                    sigma_train=sigma_train.astype(np.float32),
                    y_train=data.y_train_raw.astype(np.float32),
                    train_dates=pd.to_datetime(data.meta_train["Date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
                    train_codes=data.meta_train["code"].astype(str).to_numpy(),
                    mu_val=mu_val.astype(np.float32),
                    sigma_val=sigma_val.astype(np.float32),
                    y_val=data.y_val_raw.astype(np.float32),
                    val_dates=pd.to_datetime(data.meta_val["Date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
                    val_codes=data.meta_val["code"].astype(str).to_numpy(),
                )

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    # Aggregate val only
    val_df = results_df[results_df["split"] == "val"]
    agg_rows = []
    for variant, group in val_df.groupby("variant"):
        row = {"variant": variant, "n_seeds": len(group)}
        for col in ["rel_score", "rel_score_vol_clipped", "daily_q90_max", "daily_q90_clipped_max",
                    "spike_days_ge_8pct", "spike_days_clipped_ge_8pct", "directional_accuracy",
                    "pred_actual_q90_ratio", "mean_sigma"]:
            if col in group.columns:
                vals = group[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).sort_values("rel_score_mean", ascending=False)
    agg_df.to_csv(args.output_dir / "results_aggregate.csv", index=False)
    agg_df.to_csv(args.gold_dir / "results_aggregate.csv", index=False)
    # Write readout
    lines = [
        "# Heteroscedastic NLL Probe Readout",
        "",
        "Priority 1 from advisor report §6: volatility forecasting head + NLL loss.",
        "Scope: VN train/validation only. Holdout/test not used.",
        "",
        "## Aggregate Validation (3 seeds)",
        "",
        agg_df.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "Pass if rel_score_mean >= baseline AND (spike_days_ge_8pct_mean < baseline OR daily_q90_clipped_max < baseline).",
        "",
        "If hetero_combined passes: promote as new candidate.",
        "If only hetero_nll passes: use sigma for post-processing clip only.",
        "If neither passes: move to Priority 2 (supervised gate).",
    ]
    readout = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(readout, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(readout, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
