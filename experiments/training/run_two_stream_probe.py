"""Step 3: Two-stream LSTM probe (market head + alpha head).

Architecture:
  Stock features (window=15) -> Stock LSTM [64,32] -> alpha_pred (Dense 1)
  Market features (window=15) -> Market LSTM [32]   -> market_pred (Dense 1)
  final_pred = market_pred + alpha_pred

Loss:
  L = w_pred * RelScoreWeighted(actual, final_pred)
    + w_market * Huber(market_actual, market_pred)

Market features are strictly lagged (no same-day leak).
Market target = cross-sectional mean of target_next_return (legitimate future target).

Variants:
- raw_baseline: single-stream LSTM on raw target (same as Step 2A baseline).
- two_stream_joint: market + alpha heads trained jointly.
- two_stream_frozen_market: train market head first, freeze, then train alpha.
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
from src.models.components.losses import RelScoreLoss, RelScoreWeightedLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset  # noqa: E402
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
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402


DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history"
    / "training_runs" / "reports" / "two_stream_probe_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "two_stream_probe_20260520"

MARKET_FEATURES = (
    "market_return_today_lag1",
    "market_return_today_lag2",
    "market_return_lag1_5",
    "market_return_lag1_20",
    "market_volatility_lag1_20",
    "market_negative_ratio_lag1",
    "market_abs_q90_lag1",
    "market_breadth_lag1",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stream LSTM probe: market head + alpha head.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--market-lstm-units", default="32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", default="43,52,71")
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--market-loss-weight", type=float, default=0.5)
    parser.add_argument("--spike-thresholds", default="0.05,0.07,0.08")
    return parser.parse_args(argv)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_lstm_units(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score_fn(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def add_market_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Add lagged market features and market target (cross-sectional mean of next-day return)."""
    work = df.sort_values(["code", "Date"], kind="stable").copy()
    work["Date"] = pd.to_datetime(work["Date"])
    work["return_today"] = work.groupby("code", sort=False)["adjust"].pct_change()
    daily = work.groupby("Date", sort=True).agg(
        market_target=("target_next_return", "mean"),
        market_return_today=("return_today", "mean"),
        market_q10_today=("return_today", lambda v: float(np.nanquantile(v, 0.10))),
        market_q90_today=("return_today", lambda v: float(np.nanquantile(v, 0.90))),
        market_negative_ratio_today=("return_today", lambda v: float(np.nanmean(v < 0.0))),
        market_abs_q90_today=("return_today", lambda v: float(np.nanquantile(np.abs(v), 0.90))),
        market_breadth_today=("return_today", lambda v: float(np.nanmean(v > 0.0))),
    ).reset_index()
    daily = daily.sort_values("Date", kind="stable")
    daily["market_return_today_lag1"] = daily["market_return_today"].shift(1)
    daily["market_return_today_lag2"] = daily["market_return_today"].shift(2)
    daily["market_return_lag1_5"] = daily["market_return_today"].shift(1).rolling(5, min_periods=3).mean()
    daily["market_return_lag1_20"] = daily["market_return_today"].shift(1).rolling(20, min_periods=10).mean()
    daily["market_volatility_lag1_20"] = daily["market_return_today"].shift(1).rolling(20, min_periods=10).std()
    daily["market_negative_ratio_lag1"] = daily["market_negative_ratio_today"].shift(1)
    daily["market_abs_q90_lag1"] = daily["market_abs_q90_today"].shift(1)
    daily["market_breadth_lag1"] = daily["market_breadth_today"].shift(1)
    work = work.drop(columns=["return_today"])
    merge_cols = ["Date", "market_target", *MARKET_FEATURES]
    work = work.merge(daily[merge_cols], on="Date", how="left")
    for col in [*MARKET_FEATURES, "market_target"]:
        work[col] = work[col].fillna(0.0)
    return work


@dataclass(frozen=True)
class PreparedData:
    stock_feature_columns: tuple[str, ...]
    x_stock_train: np.ndarray
    x_stock_val: np.ndarray
    x_market_train: np.ndarray
    x_market_val: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_train_model: np.ndarray
    y_val_model: np.ndarray
    market_target_train: np.ndarray
    market_target_val: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    train_scale_values: np.ndarray
    val_scale_values: np.ndarray
    target_scaler: TargetScaler
    local_target_normalizer: LocalTargetNormalizer


def prepare_data(raw: pd.DataFrame, stock_features: tuple[str, ...], args: argparse.Namespace) -> PreparedData:
    target_alias = f"__tn__{args.target_normalizer}"
    work = raw.copy()
    work[target_alias] = work[args.target_normalizer].astype(float)
    extra_meta = [target_alias, "market_target"]
    all_features = tuple(dict.fromkeys([*stock_features, *MARKET_FEATURES]))
    train_df, _, _ = split_frame_by_date(work, args.train_end_date, args.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=all_features), all_features)
    scaled = apply_feature_scaler(work, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, all_features, args.target_column, args.window_size,
        extra_meta_columns=tuple(extra_meta), sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, args.train_end_date, args.val_end_date)
    x_train, y_train_raw, meta_train = splits["train"]
    x_val, y_val_raw, meta_val = splits["val"]
    n_stock = len(stock_features)
    n_market = len(MARKET_FEATURES)
    stock_idx = list(range(n_stock))
    market_idx = list(range(len(all_features) - n_market, len(all_features)))
    x_stock_train = x_train[:, :, stock_idx]
    x_stock_val = x_val[:, :, stock_idx]
    x_market_train = x_train[:, :, market_idx]
    x_market_val = x_val[:, :, market_idx]
    train_scale = meta_train[target_alias].to_numpy(dtype=np.float32)
    val_scale = meta_val[target_alias].to_numpy(dtype=np.float32)
    local_norm = fit_local_target_normalizer(train_scale, args.target_normalizer)
    y_train_local = apply_local_target_normalizer(y_train_raw, train_scale, local_norm)
    target_scaler = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, target_scaler).reshape(-1, 1)
    y_train_model = np.concatenate([y_train_scaled, train_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    y_val_local = apply_local_target_normalizer(y_val_raw, val_scale, local_norm)
    y_val_scaled = apply_target_scaler(y_val_local, target_scaler).reshape(-1, 1)
    y_val_model = np.concatenate([y_val_scaled, val_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    market_target_train = meta_train["market_target"].to_numpy(dtype=np.float32)
    market_target_val = meta_val["market_target"].to_numpy(dtype=np.float32)
    return PreparedData(
        stock_feature_columns=stock_features,
        x_stock_train=x_stock_train, x_stock_val=x_stock_val,
        x_market_train=x_market_train, x_market_val=x_market_val,
        y_train_raw=y_train_raw.astype(np.float32), y_val_raw=y_val_raw.astype(np.float32),
        y_train_model=y_train_model, y_val_model=y_val_model,
        market_target_train=market_target_train, market_target_val=market_target_val,
        meta_train=meta_train, meta_val=meta_val,
        train_scale_values=train_scale, val_scale_values=val_scale,
        target_scaler=target_scaler, local_target_normalizer=local_norm,
    )


def build_baseline_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    """Single-stream baseline: stock features only, raw target."""
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_stock_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    model = keras.Model(inputs=inputs, outputs=pred)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=RelScoreWeightedLoss(
            target_mean=data.target_scaler.mean,
            target_std=data.target_scaler.std,
            use_target_scaler=True,
            local_scale_floor=data.local_target_normalizer.floor,
            high_quantile=0.85,
            high_weight=1.75,
            base_weight=1.0,
        ),
    )
    return model


def build_two_stream_model(data: PreparedData, args: argparse.Namespace) -> keras.Model:
    """Two-stream: market LSTM + stock LSTM, final = market_pred + alpha_pred."""
    stock_input = layers.Input(shape=(args.window_size, data.x_stock_train.shape[2]), name="stock_input")
    market_input = layers.Input(shape=(args.window_size, data.x_market_train.shape[2]), name="market_input")
    # Market stream (small)
    market_units = parse_lstm_units(args.market_lstm_units)
    x_m = market_input
    for idx, units in enumerate(market_units):
        return_seq = idx < len(market_units) - 1
        x_m = layers.LSTM(units, return_sequences=return_seq, name=f"market_lstm_{idx}")(x_m)
        if args.dropout > 0 and return_seq:
            x_m = layers.Dropout(args.dropout)(x_m)
    market_pred = layers.Dense(1, name="market_pred")(x_m)
    # Stock/alpha stream (standard)
    stock_units = parse_lstm_units(args.lstm_units)
    x_s = stock_input
    for idx, units in enumerate(stock_units):
        return_seq = idx < len(stock_units) - 1
        x_s = layers.LSTM(units, return_sequences=return_seq, name=f"stock_lstm_{idx}")(x_s)
        if args.dropout > 0 and return_seq:
            x_s = layers.Dropout(args.dropout)(x_s)
    alpha_pred = layers.Dense(1, name="alpha_pred")(x_s)
    # Combine
    final_pred = layers.Add(name="final_pred")([market_pred, alpha_pred])
    model = keras.Model(inputs=[stock_input, market_input], outputs=[final_pred, market_pred])
    pred_loss = RelScoreWeightedLoss(
        target_mean=data.target_scaler.mean,
        target_std=data.target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=data.local_target_normalizer.floor,
        high_quantile=0.85,
        high_weight=1.75,
        base_weight=1.0,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={"final_pred": pred_loss, "market_pred": keras.losses.Huber(delta=0.005)},
        loss_weights={"final_pred": 1.0, "market_pred": args.market_loss_weight},
    )
    return model


def fit_baseline(data: PreparedData, args: argparse.Namespace, seed: int) -> tuple[keras.Model, dict]:
    set_global_seed(seed)
    model = build_baseline_model(data, args)
    history = model.fit(
        data.x_stock_train, data.y_train_model,
        validation_data=(data.x_stock_val, data.y_val_model),
        epochs=args.epochs, batch_size=args.batch_size, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
    )
    return model, history.history


def fit_two_stream(data: PreparedData, args: argparse.Namespace, seed: int) -> tuple[keras.Model, dict]:
    set_global_seed(seed)
    model = build_two_stream_model(data, args)
    # Market target needs to be scaled same way as main target for the Huber loss
    # But market_pred output is in raw scale (no local normalizer), so use raw market target
    # Actually market_pred is in model-internal scale. Let's keep it simple: market target in raw scale.
    # The market_pred Dense(1) output is unconstrained, Huber(delta=0.005) on raw market return.
    train_inputs = [data.x_stock_train, data.x_market_train]
    val_inputs = [data.x_stock_val, data.x_market_val]
    train_targets = {"final_pred": data.y_train_model, "market_pred": data.market_target_train.reshape(-1, 1)}
    val_targets = {"final_pred": data.y_val_model, "market_pred": data.market_target_val.reshape(-1, 1)}
    history = model.fit(
        train_inputs, train_targets,
        validation_data=(val_inputs, val_targets),
        epochs=args.epochs, batch_size=args.batch_size, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)],
    )
    return model, history.history


def predict_raw(model: keras.Model, data: PreparedData, x, scale_values: np.ndarray, is_two_stream: bool) -> tuple[np.ndarray, np.ndarray]:
    """Returns (final_pred_raw, market_pred_raw)."""
    output = model.predict(x, verbose=0)
    if is_two_stream:
        final_scaled = np.asarray(output[0], dtype=np.float32).reshape(-1)
        market_raw = np.asarray(output[1], dtype=np.float32).reshape(-1)
    else:
        final_scaled = np.asarray(output, dtype=np.float32).reshape(-1)
        market_raw = np.zeros_like(final_scaled)
    final_local = inverse_target_scaler_values(final_scaled, data.target_scaler)
    final_raw = inverse_local_target_normalizer(final_local, scale_values, data.local_target_normalizer).reshape(-1)
    return final_raw.astype(np.float32), market_raw.astype(np.float32)


def evaluate_split(
    actual: np.ndarray,
    final_pred: np.ndarray,
    market_pred: np.ndarray,
    meta: pd.DataFrame,
    spike_thresholds: tuple[float, ...],
) -> dict[str, object]:
    abs_error = np.abs(actual - final_pred)
    daily = pd.DataFrame({"Date": meta["Date"].values, "abs_error": abs_error})
    daily_agg = daily.groupby("Date")["abs_error"].quantile(0.90).reset_index(name="daily_q90")
    row: dict[str, object] = {
        "rel_score": rel_score_fn(actual, final_pred),
        "median_abs_error": float(np.quantile(abs_error, 0.50)),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)),
        "daily_q90_p90": float(daily_agg["daily_q90"].quantile(0.90)),
        "daily_max": float(daily_agg["daily_q90"].max()),
        "pred_actual_q90_ratio": float(np.quantile(np.abs(final_pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8)),
        "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(final_pred))),
        "market_pred_r2": float(1.0 - np.sum((meta["market_target"].values - market_pred) ** 2) / max(np.sum((meta["market_target"].values - np.mean(meta["market_target"].values)) ** 2), 1e-12)),
        "market_pred_corr": float(np.corrcoef(meta["market_target"].values, market_pred)[0, 1]) if len(market_pred) > 5 else float("nan"),
    }
    for threshold in spike_thresholds:
        key = int(round(threshold * 100))
        row[f"spike_days_ge_{key}pct"] = int(daily_agg["daily_q90"].ge(threshold).sum())
    return row


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    spike_thresholds = tuple(float(t) for t in parse_csv(args.spike_thresholds))
    stock_features = DEFAULT_FEATURE_COLUMNS
    print(f"Loading data from {args.data}")
    raw = load_training_frame(args.data, stocks=None)
    required = {"Date", "code", args.target_column, args.target_normalizer, "adjust", *stock_features}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    raw = raw.loc[:, sorted(required)].copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw["code"] = raw["code"].astype(str).str.upper()
    raw = add_market_features(raw, args.target_column)
    print(f"Preparing data...")
    data = prepare_data(raw, stock_features, args)
    print(f"Stock features: {data.x_stock_train.shape[2]}, Market features: {data.x_market_train.shape[2]}")
    print(f"Train: {len(data.x_stock_train)}, Val: {len(data.x_stock_val)}")

    results: list[dict[str, object]] = []
    for seed in seeds:
        # Baseline
        print(f"Baseline seed={seed}")
        model_b, hist_b = fit_baseline(data, args, seed)
        pred_b, mkt_b = predict_raw(model_b, data, data.x_stock_val, data.val_scale_values, is_two_stream=False)
        row_b = evaluate_split(data.y_val_raw, pred_b, mkt_b, data.meta_val, spike_thresholds)
        row_b.update({"variant": "raw_baseline", "seed": seed, "split": "val"})
        results.append(row_b)
        # Two-stream joint
        print(f"Two-stream joint seed={seed}")
        model_ts, hist_ts = fit_two_stream(data, args, seed)
        pred_ts, mkt_ts = predict_raw(model_ts, data, [data.x_stock_val, data.x_market_val], data.val_scale_values, is_two_stream=True)
        row_ts = evaluate_split(data.y_val_raw, pred_ts, mkt_ts, data.meta_val, spike_thresholds)
        row_ts.update({"variant": "two_stream_joint", "seed": seed, "split": "val"})
        results.append(row_ts)
        # Save models
        model_b.save(args.output_dir / f"baseline_seed_{seed}.keras")
        model_ts.save(args.output_dir / f"two_stream_seed_{seed}.keras")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    results_df.to_csv(args.gold_dir / "results_per_seed.csv", index=False)
    # Aggregate
    agg_rows = []
    for variant, group in results_df.groupby("variant"):
        row = {"variant": variant, "n_seeds": len(group)}
        for col in ["rel_score", "daily_max", "spike_days_ge_8pct", "directional_accuracy", "market_pred_r2", "market_pred_corr", "pred_actual_q90_ratio"]:
            if col in group.columns:
                vals = group[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(args.output_dir / "results_aggregate.csv", index=False)
    agg_df.to_csv(args.gold_dir / "results_aggregate.csv", index=False)
    # Write readout
    lines = [
        "# Two-Stream Probe Readout",
        "",
        "Step 3 of input/target processing improvement plan.",
        "Scope: VN train/validation only. Holdout/test is not used.",
        "",
        "## Per-Seed Validation",
        "",
        results_df.to_markdown(index=False),
        "",
        "## Aggregate",
        "",
        agg_df.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "Two-stream passes if rel_score_mean > baseline AND spike_days_ge_8pct_mean < baseline.",
        "market_pred_r2 and market_pred_corr show how well the market head learned.",
    ]
    readout = "\n".join(lines)
    (args.output_dir / "summary.md").write_text(readout, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(readout, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
