from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_tail_aware_lstm_probe import (  # noqa: E402
    DEFAULT_DATA,
    Variant,
    balanced_binary_weights,
    build_pred_loss,
    load_base_feature_columns,
    load_frame,
    parse_lstm_units,
    prepare_data,
    rel_score,
)
from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "distributional_scale_lstm_probe_20260520"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "distributional_scale_lstm_probe_20260520"
DEFAULT_REPORT_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs" / "reports"

PAST_TAIL_FEATURES = (
    "market_return_q10_lag1",
    "market_return_q25_lag1",
    "market_negative_ratio_lag1",
    "market_left_tail_4pct_ratio_lag1",
    "market_left_tail_6pct_ratio_lag1",
    "market_abs_return_q90_lag1",
    "market_negative_ratio_ewm5_lag1",
    "market_abs_return_q90_ewm5_lag1",
)


@dataclass(frozen=True)
class ScaleVariant:
    name: str
    variant: Variant
    scale_weight: float
    tail_weight: float
    market_abs_weight: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train distributional scale/tail LSTM probes.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seeds", default="52")
    parser.add_argument("--variants", default="dist_scale_global,dist_scale_past_tail")
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-daily-n", type=int, default=20)
    parser.add_argument("--target-error", type=float, default=0.035)
    parser.add_argument("--tail-threshold", type=float, default=0.035)
    parser.add_argument("--init-from-stressaux", action="store_true")
    parser.add_argument("--freeze-copied-layers", action="store_true")
    return parser.parse_args(argv)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scale_variants() -> dict[str, ScaleVariant]:
    base = {
        "normalization": "global",
        "model_type": "marketaux",
        "loss": "rel_score_weighted_tail",
        "weighted_high_quantile": 0.85,
        "weighted_high_weight": 1.75,
        "tail_error_threshold": 0.035,
        "tail_penalty_weight": 0.05,
        "market_aux_return_column": "future_market_return_mean",
        "market_aux_abs_column": "future_market_abs_return_q90",
    }
    return {
        "dist_scale_global": ScaleVariant(
            name="dist_scale_global",
            variant=Variant(
                name="dist_scale_global",
                extra_feature_columns=(),
                **base,
            ),
            scale_weight=0.25,
            tail_weight=0.12,
            market_abs_weight=0.12,
        ),
        "dist_scale_past_tail": ScaleVariant(
            name="dist_scale_past_tail",
            variant=Variant(
                name="dist_scale_past_tail",
                extra_feature_columns=PAST_TAIL_FEATURES,
                **base,
            ),
            scale_weight=0.25,
            tail_weight=0.12,
            market_abs_weight=0.12,
        ),
        "dist_scale_past_tail_strong": ScaleVariant(
            name="dist_scale_past_tail_strong",
            variant=Variant(
                name="dist_scale_past_tail_strong",
                extra_feature_columns=PAST_TAIL_FEATURES,
                **base,
            ),
            scale_weight=0.40,
            tail_weight=0.20,
            market_abs_weight=0.20,
        ),
    }


def build_distributional_scale_model(data: object, args: argparse.Namespace, spec: ScaleVariant) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    scale = layers.Dense(1, activation="softplus", name="scale_aux")(encoded)
    tail = layers.Dense(1, activation="sigmoid", name="tail35_prob")(encoded)
    market_abs = layers.Dense(1, activation="softplus", name="market_abs_aux")(encoded)
    model = keras.Model(
        inputs=inputs,
        outputs={
            "pred": pred,
            "scale_aux": scale,
            "tail35_prob": tail,
            "market_abs_aux": market_abs,
        },
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(spec.variant, data.target_scaler, data.local_target_normalizer),
            "scale_aux": keras.losses.Huber(delta=0.01),
            "tail35_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "market_abs_aux": keras.losses.Huber(delta=0.01),
        },
        loss_weights={
            "pred": 1.0,
            "scale_aux": spec.scale_weight,
            "tail35_prob": spec.tail_weight,
            "market_abs_aux": spec.market_abs_weight,
        },
    )
    return model


def stressaux_model_path(report_root: Path, seed: int) -> Path:
    return (
        report_root
        / "stressaux_lstm_probe_20260519"
        / f"seed_{seed}"
        / "plain_global_weighted_mild_tail35_stressaux_w20.keras"
    )


def initialize_from_stressaux(model: keras.Model, source_path: Path, *, freeze_copied_layers: bool) -> list[str]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = keras.models.load_model(source_path, compile=False)
    source_layers = {layer.name: layer for layer in source.layers}
    copied: list[str] = []
    used_source: set[str] = set()
    for layer in model.layers:
        source_layer = source_layers.get(layer.name)
        if source_layer is None:
            continue
        source_weights = source_layer.get_weights()
        target_weights = layer.get_weights()
        if not source_weights or len(source_weights) != len(target_weights):
            continue
        if all(a.shape == b.shape for a, b in zip(source_weights, target_weights)):
            layer.set_weights(source_weights)
            if freeze_copied_layers:
                layer.trainable = False
            copied.append(layer.name)
            used_source.add(source_layer.name)
    for layer in model.layers:
        if layer.name in copied or not layer.get_weights():
            continue
        if "LSTM" not in layer.__class__.__name__:
            continue
        target_weights = layer.get_weights()
        for source_layer in source.layers:
            if source_layer.name in used_source or source_layer.__class__ is not layer.__class__:
                continue
            source_weights = source_layer.get_weights()
            if not source_weights or len(source_weights) != len(target_weights):
                continue
            if all(a.shape == b.shape for a, b in zip(source_weights, target_weights)):
                layer.set_weights(source_weights)
                if freeze_copied_layers:
                    layer.trainable = False
                copied.append(f"{layer.name}<-{source_layer.name}")
                used_source.add(source_layer.name)
                break
    return copied


def scale_targets(y_raw: np.ndarray, market_abs: np.ndarray, tail_threshold: float) -> dict[str, np.ndarray]:
    y = np.asarray(y_raw, dtype=np.float32).reshape(-1, 1)
    market = np.asarray(market_abs, dtype=np.float32).reshape(-1, 1)
    return {
        "scale_aux": np.abs(y).astype(np.float32),
        "tail35_prob": (np.abs(y) >= tail_threshold).astype(np.float32),
        "market_abs_aux": np.nan_to_num(market, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }


def fit_scale_variant(spec: ScaleVariant, data: object, args: argparse.Namespace, seed: int) -> tuple[keras.Model, keras.callbacks.History]:
    set_global_seed(seed)
    model = build_distributional_scale_model(data, args, spec)
    copied_layers: list[str] = []
    if args.init_from_stressaux:
        copied_layers = initialize_from_stressaux(
            model,
            stressaux_model_path(args.report_root, seed),
            freeze_copied_layers=args.freeze_copied_layers,
        )
        if copied_layers:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
                loss={
                    "pred": build_pred_loss(spec.variant, data.target_scaler, data.local_target_normalizer),
                    "scale_aux": keras.losses.Huber(delta=0.01),
                    "tail35_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
                    "market_abs_aux": keras.losses.Huber(delta=0.01),
                },
                loss_weights={
                    "pred": 1.0,
                    "scale_aux": spec.scale_weight,
                    "tail35_prob": spec.tail_weight,
                    "market_abs_aux": spec.market_abs_weight,
                },
            )
    train_targets = scale_targets(
        data.y_train_raw,
        data.meta_train[spec.variant.market_aux_abs_column].to_numpy(dtype=np.float32),
        args.tail_threshold,
    )
    val_targets = scale_targets(
        data.y_val_raw,
        data.meta_val[spec.variant.market_aux_abs_column].to_numpy(dtype=np.float32),
        args.tail_threshold,
    )
    y_train = {"pred": data.y_train_model, **train_targets}
    y_val = {"pred": data.y_val_model, **val_targets}
    sample_weight = {
        "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
        "scale_aux": np.ones(len(data.y_train_raw), dtype=np.float32),
        "tail35_prob": balanced_binary_weights(train_targets["tail35_prob"], max_weight=6.0),
        "market_abs_aux": np.ones(len(data.y_train_raw), dtype=np.float32),
    }
    history = model.fit(
        data.x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(data.x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if args.freeze_copied_layers else "val_pred_loss",
                mode="min",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    return model, history


def predict_outputs(model: keras.Model, data: object, x: np.ndarray, scale_values: np.ndarray) -> dict[str, np.ndarray]:
    output = model.predict(x, verbose=0)
    if isinstance(output, dict):
        out = output
    else:
        out = {name: value for name, value in zip(model.output_names, output)}
    pred_scaled = np.asarray(out["pred"], dtype=np.float32).reshape(-1)
    pred_local = inverse_target_scaler_values(pred_scaled, data.target_scaler)
    pred_raw = inverse_local_target_normalizer(pred_local, scale_values, data.local_target_normalizer).reshape(-1)
    return {
        "prediction": pred_raw.astype(np.float32),
        "scale_aux": np.asarray(out["scale_aux"], dtype=np.float32).reshape(-1),
        "tail35_prob": np.asarray(out["tail35_prob"], dtype=np.float32).reshape(-1),
        "market_abs_aux": np.asarray(out["market_abs_aux"], dtype=np.float32).reshape(-1),
    }


def prediction_frame(spec: ScaleVariant, model: keras.Model, data: object, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split, x, y, meta, scale_values in [
        ("train", data.x_train, data.y_train_raw, data.meta_train, data.train_scale_values),
        ("val", data.x_val, data.y_val_raw, data.meta_val, data.val_scale_values),
    ]:
        pred = predict_outputs(model, data, x, scale_values)
        part = meta.loc[:, ["code", "Date"]].copy()
        part["seed"] = seed
        part["split"] = split
        part["variant"] = spec.name
        part["actual"] = y
        part["prediction_raw"] = pred["prediction"]
        part["scale_aux"] = pred["scale_aux"]
        part["tail35_prob"] = pred["tail35_prob"]
        part["market_abs_aux"] = pred["market_abs_aux"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def robust_daily_q90(frame: pd.DataFrame, pred_col: str, min_daily_n: int) -> pd.Series:
    work = frame.loc[:, ["Date", "code", "actual", pred_col]].copy()
    work["abs_error"] = (work["actual"].astype(float) - work[pred_col].astype(float)).abs()
    counts = work.groupby("Date", sort=True)["code"].nunique()
    daily = work.groupby("Date", sort=True)["abs_error"].quantile(0.90)
    return daily[counts >= min_daily_n]


def summarize(frame: pd.DataFrame, pred_col: str, *, seed: int, variant: str, policy: str, min_daily_n: int) -> dict[str, object]:
    actual = frame["actual"].to_numpy(dtype=float)
    pred = frame[pred_col].to_numpy(dtype=float)
    abs_error = np.abs(actual - pred)
    daily = robust_daily_q90(frame, pred_col, min_daily_n)
    return {
        "seed": seed,
        "variant": variant,
        "policy": policy,
        "n_obs": int(len(frame)),
        "n_days": int(frame["Date"].nunique()),
        "rel_score": rel_score(actual, pred),
        "median_abs_error": float(np.quantile(abs_error, 0.50)),
        "q90_abs_error": float(np.quantile(abs_error, 0.90)),
        "daily_q90_median": float(daily.median()) if not daily.empty else float("nan"),
        "daily_q90_p90": float(daily.quantile(0.90)) if not daily.empty else float("nan"),
        "daily_q90_max": float(daily.max()) if not daily.empty else float("nan"),
        "days_ge_3p5": int(daily.ge(0.035).sum()) if not daily.empty else 0,
        "days_ge_5": int(daily.ge(0.050).sum()) if not daily.empty else 0,
        "days_ge_7": int(daily.ge(0.070).sum()) if not daily.empty else 0,
        "days_ge_8": int(daily.ge(0.080).sum()) if not daily.empty else 0,
        "prediction_abs_q90": float(np.quantile(np.abs(pred), 0.90)),
        "actual_abs_q90": float(np.quantile(np.abs(actual), 0.90)),
    }


def rank_from_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    clean = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    if clean.size == 0:
        return np.full(len(values), 0.5, dtype=float)
    return np.searchsorted(clean, np.asarray(values, dtype=float), side="right") / clean.size


def daily_scale_prediction(
    frame: pd.DataFrame,
    pred_col: str,
    score_col: str,
    *,
    multiplier: float,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    daily = (
        frame.groupby("Date", sort=True)
        .agg(
            pred_abs_q90=(pred_col, lambda values: float(np.nanquantile(np.abs(values), 0.90))),
            score_q90=(score_col, lambda values: float(np.nanquantile(values, 0.90))),
        )
        .reset_index()
    )
    daily["scale"] = multiplier * daily["score_q90"] / daily["pred_abs_q90"].clip(lower=1e-4)
    daily["scale"] = daily["scale"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(min_scale, max_scale)
    scale_map = dict(zip(daily["Date"], daily["scale"]))
    return frame[pred_col].to_numpy(dtype=float) * frame["Date"].map(scale_map).fillna(1.0).to_numpy(dtype=float)


def add_policy_predictions(cal: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    cal = cal.copy()
    val = val.copy()
    cal["pred_raw"] = cal["prediction_raw"].astype(float)
    val["pred_raw"] = val["prediction_raw"].astype(float)
    rows: list[dict[str, object]] = [{"policy": "raw", "params": {}}]

    scale_ref = cal["scale_aux"].to_numpy(dtype=float)
    tail_ref = cal["tail35_prob"].to_numpy(dtype=float)
    for frame, suffix in [(cal, "cal"), (val, "val")]:
        frame["scale_rank"] = rank_from_reference(scale_ref, frame["scale_aux"].to_numpy(dtype=float))
        frame["tail_rank"] = rank_from_reference(tail_ref, frame["tail35_prob"].to_numpy(dtype=float))

    for shrink in (0.10, 0.20, 0.30, 0.40, 0.50):
        for min_scale in (0.50, 0.65, 0.80):
            name = f"tail_shrink_s{int(shrink * 100)}_m{int(min_scale * 100)}"
            cal[name] = cal["pred_raw"] * np.clip(1.0 - shrink * cal["tail_rank"], min_scale, 1.0)
            val[name] = val["pred_raw"] * np.clip(1.0 - shrink * val["tail_rank"], min_scale, 1.0)
            rows.append({"policy": name, "params": {"shrink": shrink, "min_scale": min_scale}})

    for multiplier in (0.25, 0.35, 0.50, 0.75, 1.0):
        for min_scale in (0.50, 0.75, 1.0):
            for max_scale in (1.25, 1.50, 2.0):
                name = f"daily_scale_x{int(multiplier * 100)}_lo{int(min_scale * 100)}_hi{int(max_scale * 100)}"
                cal[name] = daily_scale_prediction(
                    cal,
                    "pred_raw",
                    "scale_aux",
                    multiplier=multiplier,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                val[name] = daily_scale_prediction(
                    val,
                    "pred_raw",
                    "scale_aux",
                    multiplier=multiplier,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                rows.append(
                    {
                        "policy": name,
                        "params": {"multiplier": multiplier, "min_scale": min_scale, "max_scale": max_scale},
                    }
                )
    return cal, val, rows


def objective(row: dict[str, object], target_error: float) -> float:
    return (
        float(row["rel_score"])
        - 2.5 * max(0.0, float(row["daily_q90_p90"]) - target_error)
        - 1.0 * max(0.0, float(row["daily_q90_max"]) - 0.06)
        - 0.003 * float(row["days_ge_5"])
        - 0.006 * float(row["days_ge_7"])
    )


def split_calibration(train: pd.DataFrame, calibration_fraction: float) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(sorted(train["Date"].dropna().unique())))
    cutoff_idx = max(1, int(len(dates) * (1.0 - calibration_fraction)))
    cutoff = dates.iloc[cutoff_idx - 1]
    return train[train["Date"].gt(cutoff)].copy()


def evaluate_policies(frame: pd.DataFrame, args: argparse.Namespace, seed: int, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["split"].eq("train")].copy()
    cal = split_calibration(train, args.calibration_fraction)
    val = frame[frame["split"].eq("val")].copy()
    cal, val, policy_rows = add_policy_predictions(cal, val)
    selected_policy = "raw"
    selected_score = -np.inf
    cal_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    for item in policy_rows:
        policy = str(item["policy"])
        row = summarize(cal, policy if policy != "raw" else "pred_raw", seed=seed, variant=variant, policy=policy, min_daily_n=args.min_daily_n)
        row["objective"] = objective(row, args.target_error)
        row["params_json"] = json.dumps(item["params"], sort_keys=True)
        cal_rows.append(row)
        if float(row["objective"]) > selected_score:
            selected_score = float(row["objective"])
            selected_policy = policy

    cal_frame = pd.DataFrame(cal_rows)
    raw_cal = cal_frame[cal_frame["policy"].eq("raw")].iloc[0]
    stability_pool = cal_frame[cal_frame["rel_score"].ge(float(raw_cal["rel_score"]) - 0.010)].copy()
    if stability_pool.empty:
        stability_pool = cal_frame.copy()
    stability_policy = str(
        stability_pool.sort_values(["daily_q90_max", "days_ge_7", "daily_q90_p90"], ascending=[True, True, True]).iloc[0][
            "policy"
        ]
    )

    for mode, policy in [("balanced", selected_policy), ("stability", stability_policy)]:
        pred_col = policy if policy != "raw" else "pred_raw"
        out_col = f"prediction_{mode}"
        val[out_col] = val[pred_col].astype(float)
        selected = summarize(
            val,
            out_col,
            seed=seed,
            variant=variant,
            policy=f"{mode}:{policy}",
            min_daily_n=args.min_daily_n,
        )
        selected["selected_on"] = f"late_train_{mode}"
        val_rows.append(selected)
    raw_val = summarize(val.assign(prediction_raw=val["pred_raw"]), "prediction_raw", seed=seed, variant=variant, policy="raw", min_daily_n=args.min_daily_n)
    raw_val["selected_on"] = "none"
    val_rows.append(raw_val)
    keep = val.loc[
        :,
        [
            "seed",
            "variant",
            "code",
            "Date",
            "actual",
            "prediction_raw",
            "prediction_balanced",
            "prediction_stability",
            "scale_aux",
            "tail35_prob",
            "market_abs_aux",
        ],
    ].copy()
    keep["balanced_policy"] = selected_policy
    keep["stability_policy"] = stability_policy
    return pd.concat([pd.DataFrame(cal_rows), pd.DataFrame(val_rows)], ignore_index=True), keep


def load_baseline_predictions(report_root: Path, seeds: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for seed in seeds:
        path = (
            report_root
            / "stressaux_lstm_probe_20260519"
            / f"seed_{seed}"
            / "predictions_plain_global_weighted_mild_tail35_stressaux_w20.csv"
        )
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"])
        val = frame[frame["split"].eq("val")].copy()
        val["seed"] = seed
        val["variant"] = "baseline_stressaux_w20"
        val["policy"] = "raw"
        val = val.rename(columns={"prediction": "prediction_selected"})
        parts.append(val.loc[:, ["seed", "variant", "policy", "code", "Date", "actual", "prediction_selected"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    selected_on = results["selected_on"].fillna("")
    selected = results[selected_on.eq("none") | selected_on.str.startswith("late_train")].copy()
    selected["selection_mode"] = selected["selected_on"].replace(
        {
            "none": "raw_or_baseline",
            "late_train_balanced": "balanced",
            "late_train_stability": "stability",
        }
    )
    return (
        selected.groupby(["variant", "selection_mode"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            policies=("policy", lambda values: ", ".join(sorted(set(map(str, values))))),
            rel_score_mean=("rel_score", "mean"),
            q90_abs_error_mean=("q90_abs_error", "mean"),
            daily_q90_median_mean=("daily_q90_median", "mean"),
            daily_q90_p90_mean=("daily_q90_p90", "mean"),
            daily_q90_max_mean=("daily_q90_max", "mean"),
            days_ge_3p5_mean=("days_ge_3p5", "mean"),
            days_ge_5_mean=("days_ge_5", "mean"),
            days_ge_7_mean=("days_ge_7", "mean"),
            days_ge_8_mean=("days_ge_8", "mean"),
            prediction_abs_q90_mean=("prediction_abs_q90", "mean"),
            actual_abs_q90_mean=("actual_abs_q90", "mean"),
        )
        .reset_index()
    )


def write_frontier(gold_dir: Path, aggregate_frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.scatter(
        100 * aggregate_frame["daily_q90_p90_mean"],
        100 * aggregate_frame["daily_q90_max_mean"],
        s=85,
        alpha=0.85,
    )
    for _, row in aggregate_frame.iterrows():
        label = f"{row['variant']} / {row['policy']}"
        ax.annotate(label, (100 * row["daily_q90_p90_mean"], 100 * row["daily_q90_max_mean"]), fontsize=8)
    ax.axvline(3.5, color="#dc2626", linestyle="--", linewidth=1.0)
    ax.axhline(5.0, color="#f97316", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Daily q90(|E|) p90 (%)")
    ax.set_ylabel("Daily q90(|E|) max (%)")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(gold_dir / "distributional_scale_train_frontier.png", dpi=180)
    plt.close(fig)


def pct(value: object) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def write_summary(gold_dir: Path, aggregate_frame: pd.DataFrame, args: argparse.Namespace) -> None:
    display = aggregate_frame.sort_values(["daily_q90_p90_mean", "daily_q90_max_mean"], ascending=[True, True])
    lines = [
        "# Distributional Scale LSTM Train Probe",
        "",
        "Scope: train new LSTM heads on VN train, tune output policy on late-train, evaluate on validation. Holdout/test is not used.",
        "",
        f"- seeds: `{args.seeds}`",
        f"- epochs: `{args.epochs}`",
        f"- variants: `{args.variants}`",
        "",
        "| variant | mode | policies | seeds | rel_score | q90 abs E | daily median | daily p90 | daily max | days >=5% | days >=7% |",
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| `{row.variant}` | `{row.selection_mode}` | `{row.policies}` | {int(row.seeds)} | {float(row.rel_score_mean):.5f} | "
            f"{pct(row.q90_abs_error_mean)} | {pct(row.daily_q90_median_mean)} | {pct(row.daily_q90_p90_mean)} | "
            f"{pct(row.daily_q90_max_mean)} | {float(row.days_ge_5_mean):.1f} | {float(row.days_ge_7_mean):.1f} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- A trained distributional head is useful only if it improves both rel_score and daily q90 spike metrics versus `baseline_stressaux_w20`.",
        "- If rel_score improves but max spike rises, the head is learning residual direction but not stress-day uncertainty.",
        "- If max spike falls but rel_score drops, the head is mostly shrinking forecasts rather than learning better timing.",
    ]
    (gold_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    selected_variants = parse_csv(args.variants)
    variants = scale_variants()
    unknown = sorted(set(selected_variants).difference(variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    feature_columns = load_base_feature_columns()
    raw = load_frame(args.data, feature_columns, args.target_column, args.target_normalizer)
    feature_columns = tuple(raw.attrs.get("feature_columns", feature_columns))

    result_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    baseline = load_baseline_predictions(args.report_root, seeds)
    if not baseline.empty:
        for seed, group in baseline.groupby("seed", sort=True):
            result_parts.append(
                pd.DataFrame(
                    [
                        summarize(
                            group,
                            "prediction_selected",
                            seed=int(seed),
                            variant="baseline_stressaux_w20",
                            policy="raw",
                            min_daily_n=args.min_daily_n,
                        )
                        | {"selected_on": "none"}
                    ]
                )
            )

    for seed in seeds:
        for name in selected_variants:
            spec = variants[name]
            print(f"Train seed={seed} variant={name}")
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            data = prepare_data(raw, feature_columns, spec.variant, seed_args)
            model, history = fit_scale_variant(spec, data, seed_args, seed)
            model_path = args.output_dir / f"{name}_seed_{seed}.keras"
            model.save(model_path)
            pd.DataFrame(history.history).to_csv(args.output_dir / f"history_{name}_seed_{seed}.csv", index=False)
            predictions = prediction_frame(spec, model, data, seed)
            predictions.to_csv(args.output_dir / f"predictions_{name}_seed_{seed}.csv", index=False)
            result, val_keep = evaluate_policies(predictions, args, seed, name)
            result_parts.append(result)
            prediction_parts.append(val_keep)

    results = pd.concat(result_parts, ignore_index=True)
    aggregate_frame = aggregate(results)
    val_predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    results.to_csv(args.output_dir / "distributional_scale_train_by_seed.csv", index=False)
    aggregate_frame.to_csv(args.output_dir / "distributional_scale_train_aggregate.csv", index=False)
    if not val_predictions.empty:
        val_predictions.to_csv(args.output_dir / "distributional_scale_val_predictions.csv", index=False)
    manifest = {
        "seeds": seeds,
        "variants": list(selected_variants),
        "feature_columns": list(feature_columns),
        "past_tail_features": list(PAST_TAIL_FEATURES),
        "holdout_test_used": False,
        "target_error": args.target_error,
        "tail_threshold": args.tail_threshold,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for file_name in [
        "distributional_scale_train_by_seed.csv",
        "distributional_scale_train_aggregate.csv",
        "manifest.json",
    ]:
        (args.gold_dir / file_name).write_bytes((args.output_dir / file_name).read_bytes())
    write_frontier(args.gold_dir, aggregate_frame)
    write_summary(args.gold_dir, aggregate_frame, args)
    print((args.gold_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
