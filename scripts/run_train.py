from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.metric import directional_accuracy, evaluate
from src.models.baseline import fit_linear_regression, predict_linear_regression
from src.models.config import get_config
from src.utils.vn_sector import load_industry_reference
from src.models.lstm import (
    apply_feature_scaler,
    build_sequence_dataset,
    fit_feature_scaler,
    fit_model,
    predict,
    split_frame_by_date,
    split_sequence_dataset,
)
from src.visualization.model_plots import save_actual_vs_prediction_plot

SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM for stock forecasting.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["price", "growth", "return", "return_3d", "return_5d"], default="return_3d")
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--lstm-units", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", choices=["mse", "huber"], default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def override_config(args: argparse.Namespace):
    config = get_config(target_mode=args.target_mode)
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.train_end_date is not None:
        config.train_end_date = args.train_end_date
    if args.val_end_date is not None:
        config.val_end_date = args.val_end_date
    if args.window_size is not None:
        config.window_size = args.window_size
    if args.lstm_units is not None:
        config.lstm_units = args.lstm_units
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.lr is not None:
        config.lr = args.lr
    if args.loss is not None:
        config.loss = args.loss
    if args.huber_delta is not None:
        config.huber_delta = args.huber_delta
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.patience is not None:
        config.patience = args.patience
    if args.feature_columns:
        config.feature_columns = tuple(item.strip() for item in args.feature_columns.split(",") if item.strip())
    return config


def load_frame(path: Path, stocks: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if stocks:
        selected = {item.strip() for item in stocks.split(",") if item.strip()}
        df = df[df["code"].isin(selected)].copy()
    return df.sort_values(["code", "Date"]).reset_index(drop=True)


def validate_columns(df: pd.DataFrame, feature_columns: tuple[str, ...], target_column: str) -> None:
    missing = [col for col in list(feature_columns) + [target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def build_run_dir(base_dir: Path, run_name: str | None, target_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"lstm_{target_mode}_{stamp}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_scaler(run_dir: Path, scaler) -> None:
    np.savez(
        run_dir / "feature_scaler.npz",
        mean=scaler.mean,
        std=scaler.std,
        feature_columns=np.asarray(scaler.feature_columns, dtype=object),
    )


def resolve_monitor_metric(target_mode: str) -> str:
    return "val_rel_score" if target_mode.startswith("return") else "val_loss"


def compute_basic_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(np.mean((actual - prediction) ** 2)),
        "mae": float(np.mean(np.abs(actual - prediction))),
        "directional_accuracy": directional_accuracy(prediction, actual),
    }


def compute_metric_details(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | list[float]]:
    result = evaluate(prediction, actual)
    return {
        "base_loss": float(result["base_loss"]),
        "abs_loss": float(result["abs_loss"]),
        "rel_score": float(result["rel_score"]),
        "directional_accuracy": float(result["directional_accuracy"]),
        "error": result["error"].tolist(),
        "base": result["base"].tolist(),
    }


def save_metric_series(run_dir: Path, model_name: str, split_name: str, details: dict[str, float | list[float]]) -> None:
    pd.DataFrame({"error": details["error"], "base": details["base"]}).to_csv(
        run_dir / f"metric_series_{model_name}_{split_name}.csv",
        index=False,
    )


def enrich_prediction_frame(meta: pd.DataFrame, split: str, model_name: str, prediction: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    return meta.assign(split=split, model=model_name, prediction=prediction, actual=actual)


def build_prediction_map(model, split_arrays: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]], predictor) -> dict[str, np.ndarray]:
    return {
        split_name: predictor(model, split_arrays[split_name][0])
        for split_name in SPLIT_NAMES
    }


def compute_metrics_bundle(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    target_mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float | list[float]]]]:
    metrics: dict[str, dict[str, float]] = {}
    metric_details: dict[str, dict[str, float | list[float]]] = {}

    for split_name in SPLIT_NAMES:
        split_metrics = compute_basic_metrics(targets[split_name], predictions[split_name])
        metrics[split_name] = split_metrics
        if target_mode.startswith("return"):
            detail = compute_metric_details(targets[split_name], predictions[split_name])
            metric_details[split_name] = detail
            split_metrics["base_loss"] = detail["base_loss"]
            split_metrics["abs_loss"] = detail["abs_loss"]
            split_metrics["rel_score"] = detail["rel_score"]
            split_metrics["directional_accuracy"] = detail["directional_accuracy"]
    return metrics, metric_details


def build_prediction_frame(
    meta_map: dict[str, pd.DataFrame],
    target_map: dict[str, np.ndarray],
    prediction_maps: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    frames = []
    for model_name, pred_map in prediction_maps.items():
        for split_name in SPLIT_NAMES:
            frames.append(
                enrich_prediction_frame(
                    meta_map[split_name],
                    split_name,
                    model_name,
                    pred_map[split_name],
                    target_map[split_name],
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_config_payload(config, args: argparse.Namespace, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, splits) -> dict[str, object]:
    x_train, _, _ = splits["train"]
    x_val, _, _ = splits["val"]
    x_test, _, _ = splits["test"]
    return {
        "data_path": str(config.data_path),
        "target_mode": config.target_mode,
        "target_column": config.target_column,
        "feature_columns": list(config.feature_columns),
        "train_end_date": config.train_end_date,
        "val_end_date": config.val_end_date,
        "window_size": config.window_size,
        "lstm_units": config.lstm_units,
        "dropout": config.dropout,
        "lr": config.lr,
        "loss": config.loss,
        "huber_delta": config.huber_delta,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "monitor_metric": resolve_monitor_metric(config.target_mode),
        "stocks": args.stocks,
        "raw_rows_train": int(len(train_df)),
        "raw_rows_val": int(len(val_df)),
        "raw_rows_test": int(len(test_df)),
        "seq_rows_train": int(len(x_train)),
        "seq_rows_val": int(len(x_val)),
        "seq_rows_test": int(len(x_test)),
        "baseline_model": "linear_regression(flatten_sequence)",
    }


def resolve_features_for_df(df: pd.DataFrame, config) -> tuple[str, ...]:
    codes = df["code"].unique()
    try:
        ind_df = load_industry_reference()
        sector_map = ind_df.set_index("code")["sector"].to_dict()
        sectors = {sector_map.get(c) for c in codes if sector_map.get(c) is not None}
        
        if len(sectors) == 1:
            sector_name = list(sectors)[0]
            if sector_name in config.sector_features_map:
                print(f"Info: Detected single sector '{sector_name}'. Using {len(config.sector_features_map[sector_name])} optimized features.")
                return config.sector_features_map[sector_name]
            else:
                print(f"Info: Detected single sector '{sector_name}' but no specific features mapped. Using default features.")
        elif len(sectors) > 1:
            print(f"Info: Detected multiple sectors. Using {len(config.feature_columns)} default features.")
    except Exception as e:
        print(f"Warning: Could not auto-detect sector: {e}")
    
    return config.feature_columns


def main() -> None:
    args = parse_args()
    config = override_config(args)
    run_dir = build_run_dir(config.output_dir, args.run_name, config.target_mode)

    df = load_frame(config.data_path, args.stocks)
    
    if not args.feature_columns:
        auto_features = resolve_features_for_df(df, config)
        config.feature_columns = auto_features

    validate_columns(df, config.feature_columns, config.target_column)

    train_df, val_df, test_df = split_frame_by_date(df, config.train_end_date, config.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=config.feature_columns), config.feature_columns)
    scaled_df = apply_feature_scaler(df, scaler)

    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        config.feature_columns,
        config.target_column,
        config.window_size,
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, config.train_end_date, config.val_end_date)
    x_train, y_train, meta_train = splits["train"]
    x_val, y_val, meta_val = splits["val"]
    x_test, y_test, meta_test = splits["test"]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Not enough sequences for train/val/test. Adjust date split or window size.")

    monitor_metric = resolve_monitor_metric(config.target_mode)
    model, history = fit_model(
        x_train,
        y_train,
        x_val,
        y_val,
        window_size=config.window_size,
        num_features=len(config.feature_columns),
        lstm_units=config.lstm_units,
        dropout=config.dropout,
        lr=config.lr,
        loss=config.loss,
        huber_delta=config.huber_delta,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        monitor_metric=monitor_metric,
    )
    linear_model = fit_linear_regression(x_train, y_train)

    split_arrays = {
        "train": (x_train, y_train, meta_train),
        "val": (x_val, y_val, meta_val),
        "test": (x_test, y_test, meta_test),
    }
    targets = {split_name: split_arrays[split_name][1] for split_name in SPLIT_NAMES}
    meta_map = {split_name: split_arrays[split_name][2] for split_name in SPLIT_NAMES}

    prediction_maps = {
        "lstm": build_prediction_map(model, split_arrays, predict),
        "linear_regression": build_prediction_map(linear_model, split_arrays, predict_linear_regression),
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metric_details: dict[str, dict[str, dict[str, float | list[float]]]] = {}

    for model_name, pred_map in prediction_maps.items():
        metrics[model_name], metric_details[model_name] = compute_metrics_bundle(pred_map, targets, config.target_mode)
        if config.target_mode.startswith("return"):
            for split_name, detail in metric_details[model_name].items():
                save_metric_series(run_dir, model_name, split_name, detail)

    history_df = pd.DataFrame(history.history)
    prediction_df = build_prediction_frame(meta_map, targets, prediction_maps)

    model.save(run_dir / "model.keras")
    joblib.dump(linear_model, run_dir / "linear_regression.joblib")
    save_scaler(run_dir, scaler)
    history_df.to_csv(run_dir / "history.csv", index=False)
    prediction_df.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if config.target_mode.startswith("return"):
        with (run_dir / "metric_details.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    model_name: {
                        split_name: {
                            "base_loss": detail["base_loss"],
                            "abs_loss": detail["abs_loss"],
                            "rel_score": detail["rel_score"],
                            "directional_accuracy": detail["directional_accuracy"],
                            "error_len": len(detail["error"]),
                            "base_len": len(detail["base"]),
                        }
                        for split_name, detail in split_map.items()
                    }
                    for model_name, split_map in metric_details.items()
                },
                f,
                indent=2,
            )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(build_config_payload(config, args, train_df, val_df, test_df, splits), f, indent=2)

    for model_name in prediction_maps:
        try:
            save_actual_vs_prediction_plot(run_dir, prediction_df, model_name)
        except Exception:
            pass

    print("Saved run to:", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
