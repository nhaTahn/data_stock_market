from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

from src.models.training.pipeline import (
    build_training_target_array,
    compute_metric_details,
    load_frame,
    validate_columns,
)
from src.evaluation.metric import directional_accuracy
from src.models.architectures.panel import build_panel_model
from src.models.config import DEFAULT_FEATURE_COLUMNS, get_config
from src.models.components.callbacks import build_training_callbacks
from src.models.training import (
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    build_magnitude_sample_weights,
    build_sequence_dataset,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
    set_global_seed,
    split_frame_by_date,
    split_sequence_dataset,
)


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
VN30_PATH = ROOT / "market_lists" / "vn30.txt"


def parse_lstm_units(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal VN30 panel LSTM probe with stock-specific head.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--lstm-units", type=parse_lstm_units, default=parse_lstm_units("64,32"))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--head-units", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lstm-seeds", type=parse_seed_list, default=parse_seed_list("42,52,62"))
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--val-end-date", default="2024-12-31")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default="magnitude")
    parser.add_argument("--use-all-features", action="store_true")
    return parser.parse_args()


def build_run_dir(run_name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"vn30_panel_probe_{stamp}"
    run_dir = RUN_BASE / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "core").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_vn30_stocks() -> str:
    text = VN30_PATH.read_text(encoding="utf-8")
    stocks = [token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip()]
    return ",".join(stocks)


def build_stock_ids(meta: pd.DataFrame, stock_to_idx: dict[str, int]) -> np.ndarray:
    return np.asarray([stock_to_idx[str(code)] for code in meta["code"].astype(str)], dtype=np.int32).reshape(-1, 1)


def score_split(actual: np.ndarray, prediction: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    group_ids = meta["code"].to_numpy()
    details = compute_metric_details(actual, prediction, group_ids=group_ids)
    return {
        "mse": float(np.mean((actual - prediction) ** 2)),
        "mae": float(np.mean(np.abs(actual - prediction))),
        "directional_accuracy": directional_accuracy(prediction, actual, group_ids=group_ids),
        "base_loss": float(details["base_loss"]),
        "abs_loss": float(details["abs_loss"]),
        "rel_score": float(details["rel_score"]),
    }


def main() -> None:
    args = parse_args()
    config = get_config(target_mode="return")
    config.window_size = args.window_size
    config.lstm_units = args.lstm_units
    config.dropout = args.dropout
    config.lr = args.lr
    config.loss = "rel_score"
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.patience = args.patience
    config.lstm_seeds = args.lstm_seeds
    config.train_end_date = args.train_end_date
    config.val_end_date = args.val_end_date
    config.target_normalizer = args.target_normalizer
    config.sample_weight_mode = args.sample_weight_mode
    config.feature_columns = DEFAULT_FEATURE_COLUMNS if not args.use_all_features else tuple(config.feature_columns)

    run_dir = build_run_dir(args.run_name)
    stocks_arg = load_vn30_stocks()
    df = load_frame(config.data_path, stocks_arg)
    validate_columns(df, config.feature_columns, config.target_column, config.target_normalizer)

    target_normalizer_alias = f"__target_normalizer__{config.target_normalizer}"
    df[target_normalizer_alias] = df[config.target_normalizer].astype(float)
    train_df, val_df, test_df = split_frame_by_date(df, config.train_end_date, config.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=config.feature_columns), config.feature_columns)
    scaled_df = apply_feature_scaler(df, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        config.feature_columns,
        config.target_column,
        config.window_size,
        extra_meta_columns=(target_normalizer_alias,),
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, config.train_end_date, config.val_end_date)
    x_train, y_train, meta_train = splits["train"]
    x_val, y_val, meta_val = splits["val"]
    x_test, y_test, meta_test = splits["test"]
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Not enough panel sequences for train/val/test.")

    stock_codes = sorted({str(code) for code in pd.concat([meta_train["code"], meta_val["code"], meta_test["code"]]).unique()})
    stock_to_idx = {code: idx for idx, code in enumerate(stock_codes)}
    stock_train = build_stock_ids(meta_train, stock_to_idx)
    stock_val = build_stock_ids(meta_val, stock_to_idx)
    stock_test = build_stock_ids(meta_test, stock_to_idx)

    train_target_norm_values = meta_train[target_normalizer_alias].to_numpy(dtype=np.float32)
    val_target_norm_values = meta_val[target_normalizer_alias].to_numpy(dtype=np.float32)
    test_target_norm_values = meta_test[target_normalizer_alias].to_numpy(dtype=np.float32)
    local_target_normalizer = fit_local_target_normalizer(train_target_norm_values, config.target_normalizer)

    y_train_local = apply_local_target_normalizer(y_train, train_target_norm_values, local_target_normalizer)
    y_val_local = apply_local_target_normalizer(y_val, val_target_norm_values, local_target_normalizer)
    target_scaler = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, target_scaler)
    y_val_scaled = apply_target_scaler(y_val_local, target_scaler)
    y_train_model_target = build_training_target_array(y_train_scaled, config.loss, local_scale_values=train_target_norm_values)
    y_val_model_target = build_training_target_array(y_val_scaled, config.loss, local_scale_values=val_target_norm_values)

    train_sample_weight = None
    val_sample_weight = None
    if config.sample_weight_mode == "magnitude":
        train_sample_weight = build_magnitude_sample_weights(
            y_train_local,
            strength=config.sample_weight_strength,
            reference_quantile=config.sample_weight_quantile,
            clip_multiple=config.sample_weight_clip,
        )
        val_sample_weight = build_magnitude_sample_weights(
            y_val_local,
            strength=config.sample_weight_strength,
            reference_quantile=config.sample_weight_quantile,
            clip_multiple=config.sample_weight_clip,
        )

    prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    ranked_models: list[dict[str, float | str]] = []
    first_history_df = None
    first_model = None
    for seed_idx, seed in enumerate(config.lstm_seeds):
        set_global_seed(seed)
        model = build_panel_model(
            window_size=config.window_size,
            num_features=x_train.shape[2],
            num_stocks=len(stock_to_idx),
            lstm_units=config.lstm_units,
            lr=config.lr,
            dropout=config.dropout,
            embedding_dim=args.embedding_dim,
            head_units=args.head_units,
            loss=config.loss,
            huber_delta=config.huber_delta,
            rel_score_large_move_quantile=config.rel_score_large_move_quantile,
            rel_score_directional_penalty=config.rel_score_directional_penalty,
            rel_score_confidence_penalty=config.rel_score_confidence_penalty,
            rel_score_confidence_ratio=config.rel_score_confidence_ratio,
            rel_score_weighted_high_quantile=config.rel_score_weighted_high_quantile,
            rel_score_weighted_high_weight=config.rel_score_weighted_high_weight,
            rel_score_weighted_base_weight=config.rel_score_weighted_base_weight,
            target_scaler=target_scaler,
            local_target_normalizer=local_target_normalizer,
        )
        callbacks = build_training_callbacks(
            [x_val, stock_val],
            y_val_model_target,
            meta_val["code"].to_numpy(),
            config.patience,
            "val_rel_score",
            target_scaler=target_scaler,
            metric_y_val=y_val,
            local_target_normalizer=local_target_normalizer,
            local_target_scale_values=val_target_norm_values,
        )
        history = model.fit(
            [x_train, stock_train],
            y_train_model_target,
            sample_weight=train_sample_weight,
            validation_data=([x_val, stock_val], y_val_model_target, val_sample_weight)
            if val_sample_weight is not None
            else ([x_val, stock_val], y_val_model_target),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if seed_idx == 0:
            first_history_df = pd.DataFrame(history.history)
            first_model = model
        pred_train = model.predict([x_train, stock_train], verbose=0).reshape(-1)
        pred_val = model.predict([x_val, stock_val], verbose=0).reshape(-1)
        pred_test = model.predict([x_test, stock_test], verbose=0).reshape(-1)
        pred_map = {
            "train": inverse_local_target_normalizer(
                inverse_target_scaler_values(pred_train, target_scaler),
                train_target_norm_values,
                local_target_normalizer,
            ),
            "val": inverse_local_target_normalizer(
                inverse_target_scaler_values(pred_val, target_scaler),
                val_target_norm_values,
                local_target_normalizer,
            ),
            "test": inverse_local_target_normalizer(
                inverse_target_scaler_values(pred_test, target_scaler),
                test_target_norm_values,
                local_target_normalizer,
            ),
        }
        model_name = f"panel_lstm_seed_{seed}"
        prediction_maps[model_name] = pred_map
        ranked_models.append(
            {
                "model": model_name,
                "val_score": score_split(y_val, pred_map["val"], meta_val)["rel_score"],
                "test_score": score_split(y_test, pred_map["test"], meta_test)["rel_score"],
            }
        )

    ranked_models.sort(key=lambda item: (float(item["val_score"]), float(item["test_score"])), reverse=True)
    best_model_name = str(ranked_models[0]["model"])
    prediction_maps["panel_lstm_best_by_val"] = prediction_maps[best_model_name]
    prediction_maps["panel_lstm_ensemble"] = {
        split_name: np.mean([prediction_maps[f"panel_lstm_seed_{seed}"][split_name] for seed in config.lstm_seeds], axis=0).astype(np.float32)
        for split_name in ("train", "val", "test")
    }

    metrics = {}
    split_payloads = {
        "train": (y_train, meta_train),
        "val": (y_val, meta_val),
        "test": (y_test, meta_test),
    }
    for model_name, pred_map in prediction_maps.items():
        metrics[model_name] = {
            split_name: score_split(actual, pred_map[split_name], meta)
            for split_name, (actual, meta) in split_payloads.items()
        }

    frames = []
    for model_name, pred_map in prediction_maps.items():
        for split_name, (actual, meta) in split_payloads.items():
            frames.append(
                meta.assign(
                    split=split_name,
                    model=model_name,
                    prediction=pred_map[split_name],
                    actual=actual,
                )
            )
    prediction_df = pd.concat(frames, ignore_index=True)

    core_dir = run_dir / "reports" / "core"
    if first_model is not None:
        first_model.save(run_dir / "model_panel.keras")
    if first_history_df is not None:
        first_history_df.to_csv(core_dir / "history_panel.csv", index=False)
    prediction_df.to_csv(core_dir / "predictions.csv", index=False)
    (core_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    family_selection_summary = {
        "panel_lstm": {
            "ranked_models": ranked_models,
            "best_by_val": best_model_name,
            "top2_by_val": [item["model"] for item in ranked_models[:2]],
        }
    }
    (core_dir / "family_selection_summary.json").write_text(json.dumps(family_selection_summary, indent=2), encoding="utf-8")
    payload = {
        "run_name": run_dir.name,
        "stocks": stocks_arg,
        "stock_count": len(stock_to_idx),
        "feature_columns": list(config.feature_columns),
        "window_size": config.window_size,
        "lstm_units": config.lstm_units,
        "dropout": config.dropout,
        "lr": config.lr,
        "embedding_dim": args.embedding_dim,
        "head_units": args.head_units,
        "lstm_seeds": config.lstm_seeds,
        "train_end_date": config.train_end_date,
        "val_end_date": config.val_end_date,
        "target_normalizer": config.target_normalizer,
        "sample_weight_mode": config.sample_weight_mode,
        "best_model": best_model_name,
        "best_val_rel_score": metrics["panel_lstm_best_by_val"]["val"]["rel_score"],
        "best_test_rel_score": metrics["panel_lstm_best_by_val"]["test"]["rel_score"],
    }
    (core_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
