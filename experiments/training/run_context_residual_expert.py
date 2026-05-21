from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from src.models.training.pipeline import (
    SPLIT_NAMES,
    augment_sequence_with_stock_identity,
    build_config_payload,
    build_prediction_frame,
    build_prediction_map,
    build_run_dir,
    build_stock_to_idx,
    build_training_target_array,
    compute_metrics_bundle,
    load_frame,
    override_config,
    parse_lstm_units,
    parse_seed_list,
    resolve_monitor_metric,
    save_scaler,
    save_target_scaler,
    validate_columns,
)
from src.reporting import cleanup_report_noise, mirror_run_artifacts, report_core_path, resolve_run_artifact
from src.models.training import (
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_model,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
    predict,
    set_global_seed,
    split_frame_by_date,
    build_sequence_dataset,
    split_sequence_dataset,
)
from experiments.analysis.committee_relscore_experiment import load_prediction_frame, merge_prediction_frames


RESIDUAL_TARGET_COLUMN = "__residual_target__"
MARKET_PREDICTION_COLUMN = "__market_prediction__"
ACTUAL_TARGET_COLUMN = "__actual_target__"
EXPERT_PREDICTION_COLUMN = "__expert_prediction__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a residual expert LSTM on actual-market residuals and evaluate combined predictions."
    )
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--market-run", type=Path, required=True)
    parser.add_argument("--expert-model", default="lstm_best_by_val")
    parser.add_argument("--market-model", default="lstm_signmag_best_by_val")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--lstm-seeds", type=parse_seed_list, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lstm-units", type=parse_lstm_units, default=None)
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict[str, object]:
    path = resolve_run_artifact(run_dir, "config.json", bucket="core")
    return json.loads(path.read_text(encoding="utf-8"))


def build_residual_frame(
    *,
    df: pd.DataFrame,
    target_column: str,
    market_run: Path,
    market_model: str,
    expert_run: Path,
    expert_model: str,
) -> pd.DataFrame:
    market_pred_df = load_prediction_frame(market_run, market_model).rename(columns={"prediction": MARKET_PREDICTION_COLUMN})
    expert_pred_df = load_prediction_frame(expert_run, expert_model).rename(columns={"prediction": EXPERT_PREDICTION_COLUMN})

    market_cols = ["split", "code", "Date", "actual", MARKET_PREDICTION_COLUMN]
    expert_cols = ["split", "code", "Date", EXPERT_PREDICTION_COLUMN]
    merged = market_pred_df[market_cols].merge(expert_pred_df[expert_cols], on=["split", "code", "Date"], how="left")
    merged = merged.rename(columns={"actual": ACTUAL_TARGET_COLUMN})

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    merged["Date"] = pd.to_datetime(merged["Date"])
    work = work.merge(
        merged[["split", "code", "Date", ACTUAL_TARGET_COLUMN, MARKET_PREDICTION_COLUMN, EXPERT_PREDICTION_COLUMN]],
        on=["code", "Date"],
        how="left",
    )
    work[ACTUAL_TARGET_COLUMN] = work[ACTUAL_TARGET_COLUMN].where(work[ACTUAL_TARGET_COLUMN].notna(), work[target_column])
    work[RESIDUAL_TARGET_COLUMN] = work[ACTUAL_TARGET_COLUMN] - work[MARKET_PREDICTION_COLUMN]
    return work


def build_seed_summary_row(
    *,
    model_name: str,
    combined_metrics: dict[str, dict[str, float]],
    residual_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "model": model_name,
        "combined_val_rel_score": combined_metrics["val"].get("rel_score"),
        "combined_test_rel_score": combined_metrics["test"].get("rel_score"),
        "combined_val_directional_accuracy": combined_metrics["val"].get("directional_accuracy"),
        "combined_test_directional_accuracy": combined_metrics["test"].get("directional_accuracy"),
        "residual_val_rel_score": residual_metrics["val"].get("rel_score"),
        "residual_test_rel_score": residual_metrics["test"].get("rel_score"),
    }


def pick_best_model(rows: list[dict[str, object]]) -> str:
    ordered = sorted(
        rows,
        key=lambda item: (
            float(item["combined_val_rel_score"]),
            float(item["combined_test_rel_score"]),
        ),
        reverse=True,
    )
    return str(ordered[0]["model"])


def average_prediction_maps(maps: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    model_names = sorted(maps)
    return {
        split_name: np.mean([maps[name][split_name] for name in model_names], axis=0).astype(np.float32)
        for split_name in SPLIT_NAMES
    }


def main() -> None:
    args = parse_args()
    expert_config = load_run_config(args.expert_run)

    base_args = argparse.Namespace(
        data_path=Path(str(expert_config["data_path"])),
        target_mode=str(expert_config["target_mode"]),
        train_end_date=expert_config["train_end_date"],
        val_end_date=expert_config["val_end_date"],
        window_size=args.window_size if args.window_size is not None else expert_config["window_size"],
        lstm_units=args.lstm_units if args.lstm_units is not None else expert_config["lstm_units"],
        dropout=args.dropout if args.dropout is not None else expert_config["dropout"],
        lr=args.lr if args.lr is not None else expert_config["lr"],
        loss=str(expert_config["loss"]),
        huber_delta=expert_config.get("huber_delta"),
        batch_size=args.batch_size if args.batch_size is not None else expert_config["batch_size"],
        epochs=args.epochs if args.epochs is not None else expert_config["epochs"],
        patience=args.patience if args.patience is not None else expert_config["patience"],
        stocks=expert_config.get("stocks"),
        sector=expert_config.get("sector"),
        feature_columns=",".join(expert_config["feature_columns"]),
        use_all_features=False,
        feature_selection_mode="sector_config",
        stock_search_summary=None,
        min_stock_val_rel_score=0.03,
        max_stocks=None,
        feature_top_k=10,
        extra_context_features="",
        target_normalizer=expert_config.get("target_normalizer"),
        lstm_seeds=args.lstm_seeds if args.lstm_seeds is not None else expert_config["lstm_seeds"],
        signmag_signed_loss_weight=None,
        signmag_sign_loss_weight=None,
        signmag_magnitude_loss_weight=None,
        no_signmag_log_magnitude=False,
        sample_weight_mode="none",
        sample_weight_strength=None,
        sample_weight_quantile=None,
        sample_weight_clip=None,
        enable_attention_family=False,
        attention_heads=None,
        attention_key_dim=None,
        enable_quantile_family=False,
        enable_event_family=False,
        event_threshold=None,
        event_signed_loss_weight=None,
        event_prob_loss_weight=None,
        event_sign_loss_weight=None,
        event_magnitude_loss_weight=None,
        no_event_log_magnitude=False,
        enable_fk_benchmark=False,
        fk_window_size=None,
        fk_hidden_units=None,
        fk_dropout=None,
        fk_learning_rate=None,
        fk_batch_size=None,
        fk_epochs=None,
        fk_patience=None,
        fk_train_fraction=None,
        fk_top_k=None,
        run_name=args.run_name,
    )
    config = override_config(base_args)
    run_name = args.run_name or f"residual_{args.expert_run.name}_plus_{args.market_run.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = build_run_dir(config.output_dir, run_name, config.target_mode)

    raw_df = load_frame(config.data_path, base_args.stocks)
    validate_columns(raw_df, config.feature_columns, config.target_column, config.target_normalizer)
    residual_df = build_residual_frame(
        df=raw_df,
        target_column=config.target_column,
        market_run=args.market_run,
        market_model=args.market_model,
        expert_run=args.expert_run,
        expert_model=args.expert_model,
    )

    extra_meta_columns = (ACTUAL_TARGET_COLUMN, MARKET_PREDICTION_COLUMN, EXPERT_PREDICTION_COLUMN)
    target_normalizer_alias = None
    if config.target_normalizer:
        target_normalizer_alias = f"__target_normalizer__{config.target_normalizer}"
        residual_df[target_normalizer_alias] = residual_df[config.target_normalizer].astype(float)
        extra_meta_columns = (*extra_meta_columns, target_normalizer_alias)

    train_df, val_df, test_df = split_frame_by_date(residual_df, config.train_end_date, config.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=config.feature_columns), config.feature_columns)
    scaled_df = apply_feature_scaler(residual_df, scaler)

    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        config.feature_columns,
        RESIDUAL_TARGET_COLUMN,
        config.window_size,
        extra_meta_columns=extra_meta_columns,
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, config.train_end_date, config.val_end_date)
    x_train, y_train_resid, meta_train = splits["train"]
    x_val, y_val_resid, meta_val = splits["val"]
    x_test, y_test_resid, meta_test = splits["test"]
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Not enough residual sequences for train/val/test.")

    stock_to_idx = build_stock_to_idx([meta_train, meta_val, meta_test])
    if stock_to_idx:
        x_train_lstm = augment_sequence_with_stock_identity(x_train, meta_train, stock_to_idx)
        x_val_lstm = augment_sequence_with_stock_identity(x_val, meta_val, stock_to_idx)
        x_test_lstm = augment_sequence_with_stock_identity(x_test, meta_test, stock_to_idx)
    else:
        x_train_lstm, x_val_lstm, x_test_lstm = x_train, x_val, x_test

    local_target_normalizer = None
    train_norm = val_norm = test_norm = None
    if target_normalizer_alias is not None:
        train_norm = meta_train[target_normalizer_alias].to_numpy(dtype=np.float32)
        val_norm = meta_val[target_normalizer_alias].to_numpy(dtype=np.float32)
        test_norm = meta_test[target_normalizer_alias].to_numpy(dtype=np.float32)
        local_target_normalizer = fit_local_target_normalizer(train_norm, config.target_normalizer)

    y_train_local = apply_local_target_normalizer(y_train_resid, train_norm, local_target_normalizer)
    y_val_local = apply_local_target_normalizer(y_val_resid, val_norm, local_target_normalizer)
    target_scaler = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, target_scaler)
    y_val_scaled = apply_target_scaler(y_val_local, target_scaler)
    y_train_model = build_training_target_array(
        y_train_scaled,
        config.loss,
        local_scale_values=train_norm if local_target_normalizer is not None else None,
    )
    y_val_model = build_training_target_array(
        y_val_scaled,
        config.loss,
        local_scale_values=val_norm if local_target_normalizer is not None else None,
    )

    actual_target_map = {
        "train": meta_train[ACTUAL_TARGET_COLUMN].to_numpy(dtype=np.float32),
        "val": meta_val[ACTUAL_TARGET_COLUMN].to_numpy(dtype=np.float32),
        "test": meta_test[ACTUAL_TARGET_COLUMN].to_numpy(dtype=np.float32),
    }
    market_target_map = {
        "train": meta_train[MARKET_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
        "val": meta_val[MARKET_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
        "test": meta_test[MARKET_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
    }
    expert_target_map = {
        "train": meta_train[EXPERT_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
        "val": meta_val[EXPERT_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
        "test": meta_test[EXPERT_PREDICTION_COLUMN].to_numpy(dtype=np.float32),
    }
    residual_target_map = {
        "train": y_train_resid,
        "val": y_val_resid,
        "test": y_test_resid,
    }
    meta_map = {"train": meta_train, "val": meta_val, "test": meta_test}
    split_arrays = {
        "train": (x_train_lstm, y_train_resid, meta_train),
        "val": (x_val_lstm, y_val_resid, meta_val),
        "test": (x_test_lstm, y_test_resid, meta_test),
    }

    monitor_metric = resolve_monitor_metric(config.target_mode)
    seed_residual_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    seed_combined_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    seed_rows: list[dict[str, object]] = []
    best_model_name: str | None = None
    best_model = None
    best_history = None
    best_val_score = float("-inf")

    for seed in config.lstm_seeds:
        set_global_seed(int(seed))
        model, history = fit_model(
            x_train_lstm,
            y_train_model,
            x_val_lstm,
            y_val_model,
            window_size=config.window_size,
            num_features=x_train_lstm.shape[2],
            lstm_units=config.lstm_units,
            dropout=config.dropout,
            lr=config.lr,
            loss=config.loss,
            huber_delta=config.huber_delta,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
            monitor_metric=monitor_metric,
            val_group_ids=meta_val["code"].to_numpy(),
            target_scaler=target_scaler,
            metric_y_val=y_val_resid,
            local_target_normalizer=local_target_normalizer,
            local_target_scale_values=val_norm,
        )

        residual_prediction_map = build_prediction_map(model, split_arrays, predict)
        residual_prediction_map = {
            split_name: inverse_target_scaler_values(values, target_scaler)
            for split_name, values in residual_prediction_map.items()
        }
        if local_target_normalizer is not None:
            residual_prediction_map = {
                split_name: inverse_local_target_normalizer(
                    values,
                    {"train": train_norm, "val": val_norm, "test": test_norm}[split_name],
                    local_target_normalizer,
                )
                for split_name, values in residual_prediction_map.items()
            }

        combined_prediction_map = {
            split_name: (market_target_map[split_name] + residual_prediction_map[split_name]).astype(np.float32)
            for split_name in SPLIT_NAMES
        }
        residual_metrics, _ = compute_metrics_bundle(residual_prediction_map, residual_target_map, config.target_mode, meta_map)
        combined_metrics, _ = compute_metrics_bundle(combined_prediction_map, actual_target_map, config.target_mode, meta_map)
        model_name = f"residual_seed_{seed}"
        seed_residual_prediction_maps[model_name] = residual_prediction_map
        seed_combined_prediction_maps[model_name] = combined_prediction_map
        seed_rows.append(
            build_seed_summary_row(
                model_name=model_name,
                combined_metrics=combined_metrics,
                residual_metrics=residual_metrics,
            )
        )

        if float(combined_metrics["val"]["rel_score"]) > best_val_score:
            best_val_score = float(combined_metrics["val"]["rel_score"])
            best_model_name = model_name
            best_model = model
            best_history = pd.DataFrame(history.history)

    if best_model_name is None or best_model is None or best_history is None:
        raise RuntimeError("Residual training did not produce any model.")

    ensemble_residual_map = average_prediction_maps(seed_residual_prediction_maps)
    ensemble_combined_map = {
        split_name: (market_target_map[split_name] + ensemble_residual_map[split_name]).astype(np.float32)
        for split_name in SPLIT_NAMES
    }

    base_prediction_maps = {
        "market_component": market_target_map,
        "expert_component": expert_target_map,
        "residual_combined_best_by_val": seed_combined_prediction_maps[best_model_name],
        "residual_combined_ensemble": ensemble_combined_map,
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metric_details: dict[str, dict[str, dict[str, float | list[float]]]] = {}
    for model_name, prediction_map in base_prediction_maps.items():
        model_metrics, model_details = compute_metrics_bundle(prediction_map, actual_target_map, config.target_mode, meta_map)
        metrics[model_name] = model_metrics
        metric_details[model_name] = model_details

    output_prediction_maps = {
        "market_component": market_target_map,
        "expert_component": expert_target_map,
        "residual_combined_best_by_val": seed_combined_prediction_maps[best_model_name],
        "residual_combined_ensemble": ensemble_combined_map,
    }
    extra_prediction_maps = {
        "market_component": {
            split_name: {
                "prediction_market": market_target_map[split_name],
                "prediction_expert": expert_target_map[split_name],
                "prediction_residual": np.zeros_like(market_target_map[split_name], dtype=np.float32),
                "prediction_combined": market_target_map[split_name],
                "actual_residual": residual_target_map[split_name],
            }
            for split_name in SPLIT_NAMES
        },
        "expert_component": {
            split_name: {
                "prediction_market": market_target_map[split_name],
                "prediction_expert": expert_target_map[split_name],
                "prediction_residual": np.zeros_like(market_target_map[split_name], dtype=np.float32),
                "prediction_combined": expert_target_map[split_name],
                "actual_residual": residual_target_map[split_name],
            }
            for split_name in SPLIT_NAMES
        },
        "residual_combined_best_by_val": {
            split_name: {
                "prediction_market": market_target_map[split_name],
                "prediction_expert": expert_target_map[split_name],
                "prediction_residual": seed_residual_prediction_maps[best_model_name][split_name],
                "prediction_combined": seed_combined_prediction_maps[best_model_name][split_name],
                "actual_residual": residual_target_map[split_name],
            }
            for split_name in SPLIT_NAMES
        },
        "residual_combined_ensemble": {
            split_name: {
                "prediction_market": market_target_map[split_name],
                "prediction_expert": expert_target_map[split_name],
                "prediction_residual": ensemble_residual_map[split_name],
                "prediction_combined": ensemble_combined_map[split_name],
                "actual_residual": residual_target_map[split_name],
            }
            for split_name in SPLIT_NAMES
        },
    }
    prediction_frame = build_prediction_frame(meta_map, actual_target_map, output_prediction_maps, extra_prediction_maps)

    seed_summary_df = pd.DataFrame(seed_rows).sort_values(
        ["combined_val_rel_score", "combined_test_rel_score"],
        ascending=[False, False],
        kind="stable",
    )

    summary = {
        "expert_run": str(args.expert_run),
        "expert_model": args.expert_model,
        "market_run": str(args.market_run),
        "market_model": args.market_model,
        "best_residual_model": best_model_name,
        "best_combined_val_rel_score": metrics["residual_combined_best_by_val"]["val"]["rel_score"],
        "best_combined_test_rel_score": metrics["residual_combined_best_by_val"]["test"]["rel_score"],
        "ensemble_combined_test_rel_score": metrics["residual_combined_ensemble"]["test"]["rel_score"],
        "expert_test_rel_score": metrics["expert_component"]["test"]["rel_score"],
        "market_test_rel_score": metrics["market_component"]["test"]["rel_score"],
        "stocks": base_args.stocks,
        "feature_columns": list(config.feature_columns),
    }

    config_payload = build_config_payload(config, base_args, train_df, val_df, test_df, splits, monitor_metric)
    config_payload.update(
        {
            "experiment_type": "context_residual_expert",
            "expert_run": str(args.expert_run),
            "expert_model": args.expert_model,
            "market_run": str(args.market_run),
            "market_model": args.market_model,
            "residual_target_column": RESIDUAL_TARGET_COLUMN,
        }
    )

    best_model.save(run_dir / "model_residual_best_by_val.keras")
    best_history.to_csv(run_dir / "history_residual_best_by_val.csv", index=False)
    save_scaler(run_dir, scaler)
    save_target_scaler(run_dir, target_scaler)
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "metric_details.json").write_text(json.dumps(metric_details, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
    seed_summary_df.to_csv(run_dir / "residual_seed_summary.csv", index=False)
    mirror_run_artifacts(run_dir)
    cleanup_report_noise(run_dir)

    payload = {
        "run_dir": str(run_dir),
        "summary_path": str(report_core_path(run_dir, "summary.json")),
        "seed_summary_path": str(report_core_path(run_dir, "residual_seed_summary.csv")),
        "metrics_path": str(report_core_path(run_dir, "metrics.json")),
        "predictions_path": str(report_core_path(run_dir, "predictions.csv")),
        "best_combined_test_rel_score": summary["best_combined_test_rel_score"],
        "ensemble_combined_test_rel_score": summary["ensemble_combined_test_rel_score"],
        "expert_test_rel_score": summary["expert_test_rel_score"],
        "market_test_rel_score": summary["market_test_rel_score"],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
