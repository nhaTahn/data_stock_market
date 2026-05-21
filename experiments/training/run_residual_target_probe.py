"""Step 2A: Residual target probe.

Test whether training LSTM on a market-residual target improves rel_score and
reduces spike days, when reconstructed back to raw return space.

Target definitions:
- raw target (baseline): y_raw = target_next_return
- residual target: y_res = target_next_return - market_proxy_return_target,
  where market_proxy_return_target[t] = mean over universe of target_next_return[i, t].

Reconstruction at evaluation:
- raw model: prediction = direct LSTM output.
- residual model: prediction = LSTM_residual_output + market_component[t].

Two market components are tested:
- `oracle`: market_component = market_proxy_return_target[t] (true daily mean).
  This is the upper bound test. If even this cannot beat the raw baseline,
  two-stream architecture cannot help.
- `lagged_ar1`: market_component = AR(1) prediction trained on train days only.
  This is a realistic lower bound consistent with the diagnostic in Step 1.

Decision rule for proceeding to Step 2B / Step 3:
- if oracle reconstruction beats baseline by >= +0.005 rel_score AND reduces
  daily_max p90 by >= 0.5% absolute -> two-stream is worth the engineering.
- if oracle gain is positive but small or spike does not improve -> conditional
  gate (Step 2B) is the only remaining option.
- if oracle gain is negative -> abandon two-stream and switch to selective
  abstention path.
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


BASE_RUN_CONFIG = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "broad_signmag_portable_no_identity_20260428_allvn_r01"
    / "reports"
    / "core"
    / "config.json"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "residual_target_probe_20260519"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "residual_target_probe_20260519"


@dataclass(frozen=True)
class Variant:
    name: str
    target_mode: str  # "raw" or "residual"
    loss: str  # "rel_score" or "rel_score_weighted"
    weighted_high_quantile: float = 0.85
    weighted_high_weight: float = 1.75


@dataclass(frozen=True)
class PreparedData:
    feature_columns: tuple[str, ...]
    x_train: np.ndarray
    y_train_raw: np.ndarray  # raw target_next_return values
    y_train_for_loss: np.ndarray  # raw or residual depending on variant
    y_train_model: np.ndarray  # scaled+local-normalized target with scale appended for rel_score loss
    market_train: np.ndarray  # market component per train sample (per Date)
    meta_train: pd.DataFrame
    train_scale_values: np.ndarray
    x_val: np.ndarray
    y_val_raw: np.ndarray
    y_val_for_loss: np.ndarray
    y_val_model: np.ndarray
    market_val: np.ndarray
    meta_val: pd.DataFrame
    val_scale_values: np.ndarray
    target_scaler: TargetScaler
    local_target_normalizer: LocalTargetNormalizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe residual target reconstruction for VN LSTM.")
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
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument(
        "--variants",
        default="raw_baseline,residual_oracle,residual_lagged_ar1",
    )
    parser.add_argument("--spike-thresholds", default="0.05,0.07,0.08")
    return parser.parse_args(argv)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_lstm_units(value: str) -> list[int]:
    units = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not units:
        raise ValueError("lstm_units must not be empty.")
    return units


def load_base_feature_columns() -> tuple[str, ...]:
    """Use the canonical default feature set from configs/lstm_config.json.

    The all-VN portable run config saves an internally-normalized feature list
    ("open_level_20", etc.) which is not directly applicable to fresh sequence
    builds. Use DEFAULT_FEATURE_COLUMNS to match the same feature pipeline as
    `run_tail_aware_lstm_probe.py`.
    """
    return DEFAULT_FEATURE_COLUMNS


def all_variants() -> dict[str, Variant]:
    return {
        "raw_baseline": Variant("raw_baseline", "raw", "rel_score_weighted"),
        "residual_oracle": Variant("residual_oracle", "residual", "rel_score_weighted"),
        "residual_lagged_ar1": Variant("residual_lagged_ar1", "residual", "rel_score_weighted"),
    }


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, prediction: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - prediction) / base)


def add_market_components(df: pd.DataFrame, target_column: str, train_end: str) -> pd.DataFrame:
    """Add per-day market components: oracle (mean of target) and lagged AR(1)."""
    work = df.sort_values(["Date", "code"], kind="stable").copy()
    work["Date"] = pd.to_datetime(work["Date"])
    daily_market = (
        work.groupby("Date", sort=True)[target_column]
        .mean()
        .rename("market_oracle")
        .reset_index()
    )
    work["return_today"] = work.groupby("code", sort=False)["adjust"].pct_change()
    daily_today = (
        work.groupby("Date", sort=True)["return_today"]
        .mean()
        .rename("market_today")
        .reset_index()
    )
    daily = daily_market.merge(daily_today, on="Date", how="left")
    daily["market_today_lag1"] = daily["market_today"].shift(1)
    train_mask = daily["Date"] <= pd.to_datetime(train_end)
    train_panel = daily.loc[train_mask].dropna(subset=["market_oracle", "market_today_lag1"])
    if len(train_panel) >= 30:
        x = train_panel["market_today_lag1"].to_numpy(dtype=float)
        y = train_panel["market_oracle"].to_numpy(dtype=float)
        X_design = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        intercept = float(coef[0])
        slope = float(coef[1])
    else:
        intercept = 0.0
        slope = 0.0
    daily["market_lagged_ar1"] = intercept + slope * daily["market_today_lag1"]
    daily["market_lagged_ar1"] = daily["market_lagged_ar1"].fillna(0.0)
    daily["market_oracle"] = daily["market_oracle"].fillna(0.0)
    work = work.drop(columns=["return_today"])
    work = work.merge(daily[["Date", "market_oracle", "market_lagged_ar1"]], on="Date", how="left")
    return work


def load_frame(
    data_path: Path,
    feature_columns: tuple[str, ...],
    target_column: str,
    target_normalizer: str,
    train_end: str,
) -> pd.DataFrame:
    frame = load_training_frame(data_path, stocks=None)
    required = {"Date", "code", target_column, target_normalizer, "adjust", *feature_columns}
    missing = sorted(required.difference(set(frame.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    frame = add_market_components(frame, target_column, train_end)
    frame.attrs["feature_columns"] = feature_columns
    return frame


def select_market_column(variant: Variant) -> str:
    if variant.target_mode == "raw":
        return "market_oracle"  # not used for loss but kept for evaluation alignment
    if variant.name == "residual_oracle":
        return "market_oracle"
    if variant.name == "residual_lagged_ar1":
        return "market_lagged_ar1"
    raise ValueError(f"Unknown variant for market column selection: {variant.name}")


def build_model_target(
    y_for_loss: np.ndarray,
    scale_values: np.ndarray,
    local_target_normalizer: LocalTargetNormalizer,
    target_scaler: TargetScaler | None = None,
) -> tuple[np.ndarray, TargetScaler]:
    y_local = apply_local_target_normalizer(y_for_loss, scale_values, local_target_normalizer)
    fitted_scaler = target_scaler or fit_target_scaler(y_local)
    y_scaled = apply_target_scaler(y_local, fitted_scaler).reshape(-1, 1)
    return (
        np.concatenate(
            [y_scaled, np.asarray(scale_values, dtype=np.float32).reshape(-1, 1)], axis=1
        ).astype(np.float32),
        fitted_scaler,
    )


def prepare_data(
    raw: pd.DataFrame,
    feature_columns: tuple[str, ...],
    variant: Variant,
    args: argparse.Namespace,
) -> PreparedData:
    market_column = select_market_column(variant)
    target_alias = f"__target_normalizer__{args.target_normalizer}"
    work = raw.copy()
    work[target_alias] = work[args.target_normalizer].astype(float)
    extra_meta_columns = [target_alias, market_column, "market_oracle"]
    train_df, _, _ = split_frame_by_date(work, args.train_end_date, args.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=feature_columns), feature_columns)
    scaled = apply_feature_scaler(work, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        feature_columns,
        args.target_column,
        args.window_size,
        extra_meta_columns=tuple(extra_meta_columns),
        sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, args.train_end_date, args.val_end_date)
    x_train, y_train_raw, meta_train = splits["train"]
    x_val, y_val_raw, meta_val = splits["val"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(f"Not enough sequences for {variant.name}.")

    train_scale = meta_train[target_alias].to_numpy(dtype=np.float32)
    val_scale = meta_val[target_alias].to_numpy(dtype=np.float32)
    market_train = meta_train[market_column].to_numpy(dtype=np.float32)
    market_val = meta_val[market_column].to_numpy(dtype=np.float32)
    if variant.target_mode == "residual":
        y_train_for_loss = (y_train_raw - market_train).astype(np.float32)
        y_val_for_loss = (y_val_raw - market_val).astype(np.float32)
    else:
        y_train_for_loss = y_train_raw.astype(np.float32)
        y_val_for_loss = y_val_raw.astype(np.float32)
    local_normalizer = fit_local_target_normalizer(train_scale, args.target_normalizer)
    y_train_model, target_scaler = build_model_target(y_train_for_loss, train_scale, local_normalizer)
    y_val_model, _ = build_model_target(y_val_for_loss, val_scale, local_normalizer, target_scaler)
    return PreparedData(
        feature_columns=feature_columns,
        x_train=x_train,
        y_train_raw=y_train_raw.astype(np.float32),
        y_train_for_loss=y_train_for_loss,
        y_train_model=y_train_model,
        market_train=market_train,
        meta_train=meta_train,
        train_scale_values=train_scale,
        x_val=x_val,
        y_val_raw=y_val_raw.astype(np.float32),
        y_val_for_loss=y_val_for_loss,
        y_val_model=y_val_model,
        market_val=market_val,
        meta_val=meta_val,
        val_scale_values=val_scale,
        target_scaler=target_scaler,
        local_target_normalizer=local_normalizer,
    )


def build_pred_loss(variant: Variant, target_scaler: TargetScaler, local_norm: LocalTargetNormalizer) -> keras.losses.Loss:
    kwargs = {
        "target_mean": target_scaler.mean,
        "target_std": target_scaler.std,
        "use_target_scaler": True,
        "local_scale_floor": local_norm.floor,
    }
    if variant.loss == "rel_score":
        return RelScoreLoss(**kwargs)
    if variant.loss == "rel_score_weighted":
        return RelScoreWeightedLoss(
            **kwargs,
            high_quantile=variant.weighted_high_quantile,
            high_weight=variant.weighted_high_weight,
            base_weight=1.0,
        )
    raise ValueError(f"Unsupported loss: {variant.loss}")


def build_model(data: PreparedData, args: argparse.Namespace, variant: Variant) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    model = keras.Model(inputs=inputs, outputs=pred)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
    )
    return model


def fit_variant(variant: Variant, data: PreparedData, args: argparse.Namespace, seed: int) -> tuple[keras.Model, keras.callbacks.History]:
    set_global_seed(seed)
    model = build_model(data, args, variant)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        data.x_train,
        data.y_train_model,
        validation_data=(data.x_val, data.y_val_model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def predict_in_target_space(
    model: keras.Model,
    data: PreparedData,
    x: np.ndarray,
    scale_values: np.ndarray,
) -> np.ndarray:
    output = model.predict(x, verbose=0)
    pred_scaled = np.asarray(output, dtype=np.float32).reshape(-1)
    pred_local = inverse_target_scaler_values(pred_scaled, data.target_scaler)
    pred_target_space = inverse_local_target_normalizer(
        pred_local,
        scale_values,
        data.local_target_normalizer,
    ).reshape(-1)
    return pred_target_space.astype(np.float32)


def reconstruct_raw_prediction(
    pred_target_space: np.ndarray,
    market_component: np.ndarray,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "raw":
        return pred_target_space.astype(np.float32)
    return (pred_target_space + market_component).astype(np.float32)


def prediction_frame(
    variant: Variant,
    seed: int,
    model: keras.Model,
    data: PreparedData,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split, x, y_raw, meta, scale_values, market_component in [
        ("train", data.x_train, data.y_train_raw, data.meta_train, data.train_scale_values, data.market_train),
        ("val", data.x_val, data.y_val_raw, data.meta_val, data.val_scale_values, data.market_val),
    ]:
        pred_target_space = predict_in_target_space(model, data, x, scale_values)
        pred_raw = reconstruct_raw_prediction(pred_target_space, market_component, variant.target_mode)
        part = meta.loc[:, ["code", "Date"]].copy()
        part["split"] = split
        part["variant"] = variant.name
        part["seed"] = seed
        part["target_mode"] = variant.target_mode
        part["prediction_raw"] = pred_raw
        part["prediction_target_space"] = pred_target_space
        part["market_component"] = market_component
        part["actual"] = y_raw
        part["error_raw"] = part["actual"] - part["prediction_raw"]
        part["abs_error_raw"] = part["error_raw"].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize_predictions(frame: pd.DataFrame, spike_thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, seed, split), group in frame.groupby(["variant", "seed", "split"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        pred = group["prediction_raw"].to_numpy(dtype=float)
        abs_error = np.abs(actual - pred)
        daily = (
            group.groupby("Date", sort=True)
            .agg(daily_q90_abs_error=("abs_error_raw", lambda values: float(np.quantile(values, 0.90))))
            .reset_index()
        )
        row: dict[str, object] = {
            "variant": variant,
            "seed": int(seed),
            "split": split,
            "n_obs": int(len(group)),
            "n_days": int(daily.shape[0]),
            "rel_score": rel_score(actual, pred),
            "median_abs_error": float(np.quantile(abs_error, 0.50)),
            "q90_abs_error": float(np.quantile(abs_error, 0.90)),
            "daily_q90_abs_error_median": float(daily["daily_q90_abs_error"].median()),
            "daily_q90_abs_error_q90": float(daily["daily_q90_abs_error"].quantile(0.90)),
            "daily_q90_abs_error_max": float(daily["daily_q90_abs_error"].max()),
            "actual_abs_q90": float(np.quantile(np.abs(actual), 0.90)),
            "prediction_abs_q90": float(np.quantile(np.abs(pred), 0.90)),
            "prediction_actual_abs_q90_ratio": float(
                np.quantile(np.abs(pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8)
            ),
            "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        }
        for threshold in spike_thresholds:
            key = int(round(threshold * 100))
            row[f"spike_days_ge_{key}pct"] = int(daily["daily_q90_abs_error"].ge(threshold).sum())
            row[f"spike_rate_ge_{key}pct"] = float(daily["daily_q90_abs_error"].ge(threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_variant(per_seed: pd.DataFrame) -> pd.DataFrame:
    val = per_seed.loc[per_seed["split"] == "val"].copy()
    rows: list[dict[str, object]] = []
    metric_cols = [
        "rel_score",
        "median_abs_error",
        "q90_abs_error",
        "daily_q90_abs_error_q90",
        "daily_q90_abs_error_max",
        "prediction_actual_abs_q90_ratio",
        "directional_accuracy",
        "spike_days_ge_5pct",
        "spike_days_ge_7pct",
        "spike_days_ge_8pct",
    ]
    for variant, group in val.groupby("variant", sort=True):
        row = {"variant": variant, "n_seeds": int(group.shape[0])}
        for column in metric_cols:
            if column not in group.columns:
                continue
            values = group[column].astype(float).to_numpy()
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_min"] = float(np.nanmin(values))
            row[f"{column}_max"] = float(np.nanmax(values))
        rows.append(row)
    return pd.DataFrame(rows)


def write_readout(output_dir: Path, gold_dir: Path, per_seed: pd.DataFrame, aggregate: pd.DataFrame, args: argparse.Namespace) -> None:
    val_per_seed = per_seed.loc[per_seed["split"] == "val"].copy()
    metric_cols = [
        "rel_score",
        "median_abs_error",
        "q90_abs_error",
        "daily_q90_abs_error_q90",
        "daily_q90_abs_error_max",
        "prediction_actual_abs_q90_ratio",
        "directional_accuracy",
        "spike_days_ge_8pct",
    ]
    display_per_seed = val_per_seed[["variant", "seed", *metric_cols]].copy()
    for column in metric_cols:
        if column in display_per_seed.columns:
            display_per_seed[column] = display_per_seed[column].astype(float).map(
                lambda value: f"{value:.5f}" if np.isfinite(value) else "n/a"
            )
    aggregate_display = aggregate.copy()
    for column in aggregate.columns:
        if column == "variant" or column == "n_seeds":
            continue
        aggregate_display[column] = aggregate_display[column].astype(float).map(
            lambda value: f"{value:.5f}" if np.isfinite(value) else "n/a"
        )

    lines: list[str] = []
    lines.append("# Residual Target Probe Readout")
    lines.append("")
    lines.append("Step 2A of input/target processing improvement plan.")
    lines.append("")
    lines.append("Scope: VN train (<= 2020-03-31) and validation (2020-04-01 .. 2022-11-15).")
    lines.append("Holdout/test is not used.")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| name | target_mode | market_component (eval) |")
    lines.append("| --- | --- | --- |")
    lines.append("| `raw_baseline` | raw | n/a (direct LSTM output) |")
    lines.append("| `residual_oracle` | residual | true cross-sectional mean of next-day return |")
    lines.append("| `residual_lagged_ar1` | residual | AR(1) on lagged market return |")
    lines.append("")
    lines.append("## Per-Seed Validation")
    lines.append("")
    lines.append(display_per_seed.to_markdown(index=False))
    lines.append("")
    lines.append("## Aggregate Validation (mean / std / min)")
    lines.append("")
    lines.append(aggregate_display.to_markdown(index=False))
    lines.append("")
    lines.append("## Decision Rule")
    lines.append("")
    lines.append("Step 2A passes if BOTH conditions hold versus `raw_baseline`:")
    lines.append("- residual_oracle rel_score mean improvement >= +0.005")
    lines.append("- residual_oracle daily_q90_abs_error_q90 mean improvement <= -0.005 (lower is better)")
    lines.append("")
    lines.append("If oracle passes but lagged_ar1 fails, two-stream needs a strong market_pred head.")
    lines.append("If oracle fails, the upper bound is exhausted: switch to selective abstention.")
    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    gold_dir = args.gold_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = parse_csv(args.feature_columns) if args.feature_columns else load_base_feature_columns()
    selected = parse_csv(args.variants)
    variant_map = all_variants()
    unknown = sorted(set(selected).difference(variant_map))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    variants = [variant_map[name] for name in selected]
    seeds = parse_seeds(args.seeds)
    spike_thresholds = tuple(float(item) for item in parse_csv(args.spike_thresholds))

    raw = load_frame(
        args.data,
        feature_columns,
        args.target_column,
        args.target_normalizer,
        args.train_end_date,
    )
    feature_columns = tuple(raw.attrs.get("feature_columns", feature_columns))
    all_predictions: list[pd.DataFrame] = []
    manifest: dict[str, object] = {
        "data": str(args.data),
        "feature_columns": list(feature_columns),
        "variants": [variant.name for variant in variants],
        "seeds": seeds,
        "holdout_test_used": False,
    }
    for variant in variants:
        data = prepare_data(raw, feature_columns, variant, args)
        for seed in seeds:
            print(f"Run {variant.name} seed={seed}")
            model, history = fit_variant(variant, data, args, seed=seed)
            tag = f"{variant.name}_seed_{seed}"
            model.save(output_dir / f"{tag}.keras")
            pd.DataFrame(history.history).to_csv(output_dir / f"history_{tag}.csv", index=False)
            pred = prediction_frame(variant, seed, model, data)
            pred.to_csv(output_dir / f"predictions_{tag}.csv", index=False)
            all_predictions.append(pred)

    predictions = pd.concat(all_predictions, ignore_index=True)
    per_seed = summarize_predictions(predictions, spike_thresholds)
    aggregate = aggregate_by_variant(per_seed)
    predictions.to_csv(output_dir / "predictions_all.csv", index=False)
    per_seed.to_csv(output_dir / "summary_per_seed.csv", index=False)
    aggregate.to_csv(output_dir / "summary_aggregate.csv", index=False)
    aggregate.to_csv(gold_dir / "summary_aggregate.csv", index=False)
    write_readout(output_dir, gold_dir, per_seed, aggregate, args)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (gold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gold_dir": str(gold_dir)}, indent=2))


if __name__ == "__main__":
    main()
