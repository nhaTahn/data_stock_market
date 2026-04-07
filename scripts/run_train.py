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
from src.models.baseline import fit_arima, fit_linear_regression, predict_arima, predict_linear_regression
from src.models.config import ALL_FEATURE_COLUMNS, get_config
from src.utils.vn_sector import load_industry_reference
from src.models.lstm import (
    apply_feature_scaler,
    apply_target_scaler,
    build_sequence_dataset,
    fit_feature_scaler,
    fit_model,
    fit_target_scaler,
    inverse_target_scaler_values,
    predict,
    split_frame_by_date,
    split_sequence_dataset,
)
from src.visualization.model_plots import save_actual_vs_prediction_plot, save_rel_score_hist_plot

SPLIT_NAMES = ("train", "val", "test")


def parse_lstm_units(value: str) -> int | list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("lstm_units must not be empty.")
    units = [int(item) for item in items]
    return units[0] if len(units) == 1 else units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LSTM for stock forecasting.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--target-mode", choices=["price", "growth", "return", "return_3d", "return_5d"], default="return_3d")
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--lstm-units", type=parse_lstm_units, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss", choices=["mse", "huber", "directional_huber"], default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--use-all-features", action="store_true")
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
    if args.use_all_features:
        config.feature_columns = ALL_FEATURE_COLUMNS
    if args.feature_columns:
        config.feature_columns = tuple(item.strip() for item in args.feature_columns.split(",") if item.strip())
    return config


def load_frame(path: Path, stocks: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["code", "Date"]).reset_index(drop=True)

    price_column = "adjust" if "adjust" in df.columns else "close" if "close" in df.columns else None
    if price_column is None:
        raise ValueError("Dataset must contain either 'adjust' or 'close' column to compute macro features.")

    df["temp_return"] = df.groupby("code")[price_column].pct_change()
    vnindex_return = (
        df.groupby("Date", sort=False)["temp_return"]
        .mean()
        .rename("vnindex_return")
        .reset_index()
    )

    vingroup_df = df[df["code"].isin({"VIC", "VHM", "VRE"})]
    if vingroup_df.empty:
        vingroup_momentum = pd.DataFrame(
            {
                "Date": vnindex_return["Date"],
                "vingroup_momentum": np.zeros(len(vnindex_return), dtype=np.float32),
            }
        )
    else:
        vingroup_momentum = (
            vingroup_df.groupby("Date", sort=False)["temp_return"]
            .mean()
            .rename("vingroup_momentum")
            .reset_index()
        )

    df = df.merge(vnindex_return, on="Date", how="left")
    df = df.merge(vingroup_momentum, on="Date", how="left")
    df[["vnindex_return", "vingroup_momentum"]] = df[["vnindex_return", "vingroup_momentum"]].fillna(0.0)
    df = df.drop(columns="temp_return")

    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["Date"].dt.dayofweek.astype(np.float32)

    if "rsi_14" not in df.columns:
        delta = df.groupby("code")[price_column].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
        avg_loss = loss.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

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


def save_target_scaler(run_dir: Path, scaler) -> None:
    np.savez(
        run_dir / "target_scaler.npz",
        mean=np.asarray([scaler.mean], dtype=np.float32),
        std=np.asarray([scaler.std], dtype=np.float32),
    )


def resolve_monitor_metric(target_mode: str) -> str:
    return "val_rel_score" if target_mode.startswith("return") else "val_loss"


def compute_basic_metrics(actual: np.ndarray, prediction: np.ndarray, group_ids: np.ndarray | None = None) -> dict[str, float]:
    return {
        "mse": float(np.mean((actual - prediction) ** 2)),
        "mae": float(np.mean(np.abs(actual - prediction))),
        "directional_accuracy": directional_accuracy(prediction, actual, group_ids=group_ids),
    }


def compute_metric_details(
    actual: np.ndarray,
    prediction: np.ndarray,
    group_ids: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    result = evaluate(prediction, actual, group_ids=group_ids)
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


def build_stock_to_idx(meta_frames: list[pd.DataFrame]) -> dict[str, int]:
    codes = sorted(
        {
            str(code)
            for meta in meta_frames
            if "code" in meta.columns
            for code in meta["code"].dropna().unique().tolist()
        }
    )
    return {code: idx for idx, code in enumerate(codes)}


def augment_sequence_with_stock_identity(
    x: np.ndarray,
    meta: pd.DataFrame,
    stock_to_idx: dict[str, int],
) -> np.ndarray:
    if x.size == 0 or not stock_to_idx:
        return x
    if "code" not in meta.columns:
        return x

    identity = np.zeros((len(meta), len(stock_to_idx)), dtype=np.float32)
    for row_idx, code in enumerate(meta["code"].astype(str).tolist()):
        stock_idx = stock_to_idx.get(code)
        if stock_idx is not None:
            identity[row_idx, stock_idx] = 1.0

    identity = np.repeat(identity[:, None, :], x.shape[1], axis=1)
    return np.concatenate([x, identity], axis=2)


def compute_metrics_bundle(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    target_mode: str,
    meta_map: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float | list[float]]]]:
    metrics: dict[str, dict[str, float]] = {}
    metric_details: dict[str, dict[str, float | list[float]]] = {}

    for split_name in SPLIT_NAMES:
        group_ids = meta_map[split_name]["code"].to_numpy() if "code" in meta_map[split_name].columns else None
        split_metrics = compute_basic_metrics(targets[split_name], predictions[split_name], group_ids=group_ids)
        metrics[split_name] = split_metrics
        if target_mode.startswith("return"):
            detail = compute_metric_details(targets[split_name], predictions[split_name], group_ids=group_ids)
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


def build_config_payload(
    config,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits,
    monitor_metric: str,
) -> dict[str, object]:
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
        "lstm_units": config.lstm_units if isinstance(config.lstm_units, int) else list(config.lstm_units),
        "dropout": config.dropout,
        "lr": config.lr,
        "loss": config.loss,
        "huber_delta": config.huber_delta,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "patience": config.patience,
        "monitor_metric": monitor_metric,
        "stocks": args.stocks,
        "use_all_features": bool(args.use_all_features),
        "raw_rows_train": int(len(train_df)),
        "raw_rows_val": int(len(val_df)),
        "raw_rows_test": int(len(test_df)),
        "seq_rows_train": int(len(x_train)),
        "seq_rows_val": int(len(x_val)),
        "seq_rows_test": int(len(x_test)),
        "baseline_model": "linear_regression(flatten_sequence)",
        "baseline_models": [
            "linear_regression(flatten_sequence)",
            "arima_proxy(ar1_fast)",
        ],
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
    
    if not args.feature_columns and not args.use_all_features:
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

    stock_to_idx = build_stock_to_idx([meta_train, meta_val, meta_test])
    use_stock_identity = len(stock_to_idx) > 1
    if use_stock_identity:
        x_train_lstm = augment_sequence_with_stock_identity(x_train, meta_train, stock_to_idx)
        x_val_lstm = augment_sequence_with_stock_identity(x_val, meta_val, stock_to_idx)
        x_test_lstm = augment_sequence_with_stock_identity(x_test, meta_test, stock_to_idx)
    else:
        x_train_lstm = x_train
        x_val_lstm = x_val
        x_test_lstm = x_test

    target_scaler = fit_target_scaler(y_train)
    y_train_scaled = apply_target_scaler(y_train, target_scaler)
    y_val_scaled = apply_target_scaler(y_val, target_scaler)

    monitor_metric = resolve_monitor_metric(config.target_mode)
    model, history = fit_model(
        x_train_lstm,
        y_train_scaled,
        x_val_lstm,
        y_val_scaled,
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
        val_group_ids=meta_val["code"].to_numpy() if "code" in meta_val.columns else None,
        target_scaler=target_scaler,
    )
    linear_model = fit_linear_regression(x_train, y_train)
    arima_model = fit_arima(x_train, y_train)

    split_arrays = {
        "train": (x_train, y_train, meta_train),
        "val": (x_val, y_val, meta_val),
        "test": (x_test, y_test, meta_test),
    }
    lstm_split_arrays = {
        "train": (x_train_lstm, y_train, meta_train),
        "val": (x_val_lstm, y_val, meta_val),
        "test": (x_test_lstm, y_test, meta_test),
    }
    targets = {split_name: split_arrays[split_name][1] for split_name in SPLIT_NAMES}
    meta_map = {split_name: split_arrays[split_name][2] for split_name in SPLIT_NAMES}

    lstm_prediction_map = build_prediction_map(model, lstm_split_arrays, predict)
    lstm_prediction_map = {
        split_name: inverse_target_scaler_values(pred_values, target_scaler)
        for split_name, pred_values in lstm_prediction_map.items()
    }
    prediction_maps = {
        "lstm": lstm_prediction_map,
        "linear_regression": build_prediction_map(linear_model, split_arrays, predict_linear_regression),
        "arima": build_prediction_map(arima_model, split_arrays, predict_arima),
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metric_details: dict[str, dict[str, dict[str, float | list[float]]]] = {}

    for model_name, pred_map in prediction_maps.items():
        metrics[model_name], metric_details[model_name] = compute_metrics_bundle(pred_map, targets, config.target_mode, meta_map)
        if config.target_mode.startswith("return"):
            for split_name, detail in metric_details[model_name].items():
                save_metric_series(run_dir, model_name, split_name, detail)

    history_df = pd.DataFrame(history.history)
    prediction_df = build_prediction_frame(meta_map, targets, prediction_maps)

    model.save(run_dir / "model.keras")
    joblib.dump(linear_model, run_dir / "linear_regression.joblib")
    save_scaler(run_dir, scaler)
    save_target_scaler(run_dir, target_scaler)
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
        payload = build_config_payload(config, args, train_df, val_df, test_df, splits, monitor_metric)
        payload["lstm_use_stock_identity"] = use_stock_identity
        payload["lstm_stock_identity_codes"] = list(stock_to_idx.keys()) if use_stock_identity else []
        payload["lstm_input_feature_count"] = int(x_train_lstm.shape[2])
        json.dump(payload, f, indent=2)

    for model_name in prediction_maps:
        try:
            save_actual_vs_prediction_plot(run_dir, prediction_df, model_name)
            save_rel_score_hist_plot(run_dir, model_name)
        except Exception:
            pass

    print("Saved run to:", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
