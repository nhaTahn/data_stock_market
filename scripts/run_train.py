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
from src.models.baselines import fit_arima, fit_linear_regression, predict_arima, predict_linear_regression
from src.models.config import ALL_FEATURE_COLUMNS, get_config
from src.models.fischer_krauss import (
    apply_fischer_krauss_scaler,
    build_fischer_krauss_sequences,
    build_long_short_portfolio_returns,
    calibrate_probability_score_to_return_proxy,
    compute_fischer_krauss_metrics,
    fit_fischer_krauss_model,
    fit_fischer_krauss_scaler,
    predict_fischer_krauss_probabilities,
    prepare_fischer_krauss_frame,
    probability_to_score,
    resolve_fk_train_end_date,
    summarize_long_short_portfolio,
    resolve_price_column,
    split_fischer_krauss_sequences,
)
from src.models.reporting import (
    cleanup_legacy_report_artifacts,
    cleanup_report_noise,
    mirror_run_artifacts,
    report_benchmark_path,
    report_core_path,
    report_metric_series_path,
)
from src.models.training import (
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    build_magnitude_sample_weights,
    build_sequence_dataset,
    fit_attention_model,
    fit_event_gated_model,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_model,
    fit_quantile_model,
    fit_sign_magnitude_model,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
    predict,
    set_global_seed,
    split_frame_by_date,
    split_sequence_dataset,
)
from src.models.training_recipe import (
    DEFAULT_CONTEXT_FEATURES,
    DEFAULT_SEARCH_SUMMARY_PATH,
    TrainingRecipe,
    build_training_recipe,
)
from src.utils.vn_sector import load_industry_reference
from src.visualization.model_plots import (
    save_actual_vs_prediction_plot,
    save_equity_curve_plot,
    save_rel_score_hist_plot,
)

SPLIT_NAMES = ("train", "val", "test")


def parse_lstm_units(value: str) -> int | list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("lstm_units must not be empty.")
    units = [int(item) for item in items]
    return units[0] if len(units) == 1 else units


def parse_seed_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("lstm_seeds must not be empty.")
    return [int(item) for item in items]


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    parser.add_argument("--loss", choices=["mse", "huber", "directional_huber", "rel_score"], default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--sector", default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--use-all-features", action="store_true")
    parser.add_argument(
        "--feature-selection-mode",
        choices=["auto", "sector_config", "search_summary"],
        default="auto",
    )
    parser.add_argument("--stock-search-summary", type=Path, default=None)
    parser.add_argument("--min-stock-val-rel-score", type=float, default=0.03)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--feature-top-k", type=int, default=10)
    parser.add_argument(
        "--extra-context-features",
        default=",".join(DEFAULT_CONTEXT_FEATURES),
    )
    parser.add_argument("--target-normalizer", default=None)
    parser.add_argument("--lstm-seeds", type=parse_seed_list, default=None)
    parser.add_argument("--signmag-signed-loss-weight", type=float, default=None)
    parser.add_argument("--signmag-sign-loss-weight", type=float, default=None)
    parser.add_argument("--signmag-magnitude-loss-weight", type=float, default=None)
    parser.add_argument("--no-signmag-log-magnitude", action="store_true")
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default=None)
    parser.add_argument("--sample-weight-strength", type=float, default=None)
    parser.add_argument("--sample-weight-quantile", type=float, default=None)
    parser.add_argument("--sample-weight-clip", type=float, default=None)
    parser.add_argument("--enable-attention-family", action="store_true")
    parser.add_argument("--attention-heads", type=int, default=None)
    parser.add_argument("--attention-key-dim", type=int, default=None)
    parser.add_argument("--enable-quantile-family", action="store_true")
    parser.add_argument("--enable-event-family", action="store_true")
    parser.add_argument("--event-threshold", type=float, default=None)
    parser.add_argument("--event-signed-loss-weight", type=float, default=None)
    parser.add_argument("--event-prob-loss-weight", type=float, default=None)
    parser.add_argument("--event-sign-loss-weight", type=float, default=None)
    parser.add_argument("--event-magnitude-loss-weight", type=float, default=None)
    parser.add_argument("--no-event-log-magnitude", action="store_true")
    parser.add_argument("--enable-fk-benchmark", action="store_true")
    parser.add_argument("--fk-window-size", type=int, default=None)
    parser.add_argument("--fk-hidden-units", type=int, default=None)
    parser.add_argument("--fk-dropout", type=float, default=None)
    parser.add_argument("--fk-learning-rate", type=float, default=None)
    parser.add_argument("--fk-batch-size", type=int, default=None)
    parser.add_argument("--fk-epochs", type=int, default=None)
    parser.add_argument("--fk-patience", type=int, default=None)
    parser.add_argument("--fk-train-fraction", type=float, default=None)
    parser.add_argument("--fk-top-k", type=int, default=None)
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
    if args.target_normalizer is not None:
        config.target_normalizer = args.target_normalizer.strip() or None
    if args.lstm_seeds is not None:
        config.lstm_seeds = list(args.lstm_seeds)
    if args.signmag_signed_loss_weight is not None:
        config.signmag_signed_loss_weight = args.signmag_signed_loss_weight
    if args.signmag_sign_loss_weight is not None:
        config.signmag_sign_loss_weight = args.signmag_sign_loss_weight
    if args.signmag_magnitude_loss_weight is not None:
        config.signmag_magnitude_loss_weight = args.signmag_magnitude_loss_weight
    if args.no_signmag_log_magnitude:
        config.signmag_log_magnitude = False
    if args.sample_weight_mode is not None:
        config.sample_weight_mode = args.sample_weight_mode
    if args.sample_weight_strength is not None:
        config.sample_weight_strength = args.sample_weight_strength
    if args.sample_weight_quantile is not None:
        config.sample_weight_quantile = args.sample_weight_quantile
    if args.sample_weight_clip is not None:
        config.sample_weight_clip = args.sample_weight_clip
    if args.enable_attention_family:
        config.attention_enabled = True
    if args.attention_heads is not None:
        config.attention_heads = args.attention_heads
    if args.attention_key_dim is not None:
        config.attention_key_dim = args.attention_key_dim
    if args.enable_quantile_family:
        config.quantile_enabled = True
    if args.enable_event_family:
        config.event_enabled = True
    if args.event_threshold is not None:
        config.event_threshold = args.event_threshold
    if args.event_signed_loss_weight is not None:
        config.event_signed_loss_weight = args.event_signed_loss_weight
    if args.event_prob_loss_weight is not None:
        config.event_prob_loss_weight = args.event_prob_loss_weight
    if args.event_sign_loss_weight is not None:
        config.event_sign_loss_weight = args.event_sign_loss_weight
    if args.event_magnitude_loss_weight is not None:
        config.event_magnitude_loss_weight = args.event_magnitude_loss_weight
    if args.no_event_log_magnitude:
        config.event_log_magnitude = False
    if args.enable_fk_benchmark:
        config.fk_benchmark_enabled = True
    if args.fk_window_size is not None:
        config.fk_window_size = args.fk_window_size
    if args.fk_hidden_units is not None:
        config.fk_hidden_units = args.fk_hidden_units
    if args.fk_dropout is not None:
        config.fk_dropout = args.fk_dropout
    if args.fk_learning_rate is not None:
        config.fk_learning_rate = args.fk_learning_rate
    if args.fk_batch_size is not None:
        config.fk_batch_size = args.fk_batch_size
    if args.fk_epochs is not None:
        config.fk_epochs = args.fk_epochs
    if args.fk_patience is not None:
        config.fk_patience = args.fk_patience
    if args.fk_train_fraction is not None:
        config.fk_train_fraction = args.fk_train_fraction
    if args.fk_top_k is not None:
        config.fk_top_k = args.fk_top_k
    return config


def load_frame(path: Path, stocks: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["code", "Date"]).reset_index(drop=True)

    price_column = "adjust" if "adjust" in df.columns else "close" if "close" in df.columns else None
    if price_column is None:
        raise ValueError("Dataset must contain either 'adjust' or 'close' column to compute macro features.")
    close_column = "close" if "close" in df.columns else price_column

    df["temp_return"] = df.groupby("code")[price_column].pct_change()
    df["return_daily"] = df.groupby("code")[close_column].pct_change()
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

    breadth = (
        df.assign(
            advancing=(df["return_daily"] > 0).astype(np.int32),
            declining=(df["return_daily"] < 0).astype(np.int32),
        )
        .groupby("Date", sort=False)[["advancing", "declining"]]
        .sum()
        .reset_index()
    )
    breadth["a_d_ratio"] = breadth["advancing"] / (breadth["declining"] + 1.0)

    df = df.merge(vnindex_return, on="Date", how="left")
    df = df.merge(vingroup_momentum, on="Date", how="left")
    df = df.merge(breadth[["Date", "a_d_ratio"]], on="Date", how="left")
    df[["vnindex_return", "vingroup_momentum"]] = df[["vnindex_return", "vingroup_momentum"]].fillna(0.0)
    df["a_d_ratio"] = df["a_d_ratio"].fillna(1.0)
    df = df.drop(columns=["temp_return", "return_daily"])

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


def filter_frame_by_sector(df: pd.DataFrame, sector: str | None) -> pd.DataFrame:
    if not sector:
        return df
    if "sector" not in df.columns:
        raise ValueError("Dataset does not contain a 'sector' column for sector filtering.")
    filtered = df[df["sector"] == sector].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for sector '{sector}'.")
    return filtered.sort_values(["code", "Date"]).reset_index(drop=True)


def validate_columns(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    target_normalizer: str | None = None,
) -> None:
    required = list(feature_columns) + [target_column]
    if target_normalizer is not None:
        required.append(target_normalizer)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def build_run_dir(base_dir: Path, run_name: str | None, target_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"lstm_{target_mode}_{stamp}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_training_target_array(
    target_values: np.ndarray,
    loss_name: str,
    local_scale_values: np.ndarray | None = None,
) -> np.ndarray:
    target_values = np.asarray(target_values, dtype=np.float32).reshape(-1, 1)
    if loss_name != "rel_score":
        return target_values.reshape(-1)
    if local_scale_values is None:
        return target_values
    local_scale_values = np.asarray(local_scale_values, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([target_values, local_scale_values], axis=1).astype(np.float32)


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
    metric_series_df = pd.DataFrame({"error": details["error"], "base": details["base"]})
    filename = f"metric_series_{model_name}_{split_name}.csv"
    metric_series_df.to_csv(run_dir / filename, index=False)
    metric_series_df.to_csv(report_metric_series_path(run_dir, filename), index=False)


def select_report_model_names(model_names: list[str]) -> list[str]:
    available = set(model_names)
    selected: list[str] = []

    for baseline_name in ("linear_regression", "arima", "fischer_krauss"):
        if baseline_name in available:
            selected.append(baseline_name)

    family_prefixes = ("lstm", "lstm_quantile", "lstm_signmag", "lstm_attention", "lstm_event")
    for prefix in family_prefixes:
        preferred = [f"{prefix}_best_by_val", f"{prefix}_ensemble", prefix]
        added = False
        for candidate in preferred:
            if candidate in available:
                selected.append(candidate)
                added = True
        if not added:
            fallback = sorted(name for name in available if name.startswith(prefix) and "_seed_" not in name)
            selected.extend(fallback[:1])

    seen: set[str] = set()
    ordered: list[str] = []
    for name in selected:
        if name in available and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def enrich_prediction_frame(
    meta: pd.DataFrame,
    split: str,
    model_name: str,
    prediction: np.ndarray,
    actual: np.ndarray,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    payload: dict[str, object] = {
        "split": split,
        "model": model_name,
        "prediction": prediction,
        "actual": actual,
    }
    if extra_columns:
        payload.update(extra_columns)
    return meta.assign(**payload)


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


def score_prediction_map_for_split(
    prediction_map: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    meta_map: dict[str, pd.DataFrame],
    split_name: str,
    target_mode: str,
) -> float:
    prediction = prediction_map[split_name]
    actual = targets[split_name]
    group_ids = meta_map[split_name]["code"].to_numpy() if "code" in meta_map[split_name].columns else None
    if target_mode.startswith("return"):
        return float(evaluate(prediction, actual, group_ids=group_ids)["rel_score"])
    return float(-np.mean((actual - prediction) ** 2))


def build_family_selection_maps(
    family_name: str,
    seed_prediction_maps: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    meta_map: dict[str, pd.DataFrame],
    target_mode: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    if not seed_prediction_maps:
        return {}, {}

    ranked = sorted(
        [
            {
                "model": model_name,
                "val_score": score_prediction_map_for_split(pred_map, targets, meta_map, "val", target_mode),
                "test_score": score_prediction_map_for_split(pred_map, targets, meta_map, "test", target_mode),
            }
            for model_name, pred_map in seed_prediction_maps.items()
        ],
        key=lambda item: (item["val_score"], item["test_score"]),
        reverse=True,
    )

    selection_maps: dict[str, dict[str, np.ndarray]] = {}
    selection_summary: dict[str, object] = {
        "ranked_models": ranked,
    }

    best_model_name = ranked[0]["model"]
    selection_maps[f"{family_name}_best_by_val"] = seed_prediction_maps[best_model_name]
    selection_summary["best_by_val"] = best_model_name

    top2 = ranked[:2]
    selection_summary["top2_by_val"] = [item["model"] for item in top2]
    if len(top2) == 2:
        selection_maps[f"{family_name}_top2_by_val"] = {
            split_name: np.mean(
                [seed_prediction_maps[item["model"]][split_name] for item in top2],
                axis=0,
            ).astype(np.float32)
            for split_name in SPLIT_NAMES
        }

    return selection_maps, selection_summary


def build_quantile_aux_maps(
    family_name: str,
    family_selection_summary: dict[str, object],
    seed_maps: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    if not seed_maps:
        return {}

    ordered_seed_keys = sorted(seed_maps.keys())
    aux_maps: dict[str, dict[str, np.ndarray]] = {
        family_name: seed_maps[ordered_seed_keys[0]],
    }
    if len(ordered_seed_keys) > 1:
        aux_maps[f"{family_name}_ensemble"] = {
            split_name: np.mean(
                [seed_maps[model_name][split_name] for model_name in ordered_seed_keys],
                axis=0,
            ).astype(np.float32)
            for split_name in SPLIT_NAMES
        }

    best_by_val = family_selection_summary.get("best_by_val")
    if isinstance(best_by_val, str) and best_by_val in seed_maps:
        aux_maps[f"{family_name}_best_by_val"] = seed_maps[best_by_val]

    top2_by_val = family_selection_summary.get("top2_by_val")
    if isinstance(top2_by_val, list) and len(top2_by_val) == 2 and all(name in seed_maps for name in top2_by_val):
        aux_maps[f"{family_name}_top2_by_val"] = {
            split_name: np.mean(
                [seed_maps[model_name][split_name] for model_name in top2_by_val],
                axis=0,
            ).astype(np.float32)
            for split_name in SPLIT_NAMES
        }
    return aux_maps


def build_quantile_extra_prediction_maps(
    q50_maps: dict[str, dict[str, np.ndarray]],
    q90_maps: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    extra_maps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model_name, q50_map in q50_maps.items():
        q90_map = q90_maps.get(model_name)
        if q90_map is None:
            continue
        extra_maps[model_name] = {}
        for split_name in SPLIT_NAMES:
            q50 = np.asarray(q50_map[split_name], dtype=np.float32)
            q90 = np.asarray(q90_map[split_name], dtype=np.float32)
            extra_maps[model_name][split_name] = {
                "prediction_q50": q50,
                "prediction_q90": q90,
                "prediction_uncertainty": (q90 - q50).astype(np.float32),
            }
    return extra_maps


def build_prediction_frame(
    meta_map: dict[str, pd.DataFrame],
    target_map: dict[str, np.ndarray],
    prediction_maps: dict[str, dict[str, np.ndarray]],
    extra_prediction_maps: dict[str, dict[str, dict[str, np.ndarray]]] | None = None,
) -> pd.DataFrame:
    frames = []
    for model_name, pred_map in prediction_maps.items():
        for split_name in SPLIT_NAMES:
            extra_columns = None
            if extra_prediction_maps is not None:
                extra_columns = extra_prediction_maps.get(model_name, {}).get(split_name)
            frames.append(
                enrich_prediction_frame(
                    meta_map[split_name],
                    split_name,
                    model_name,
                    pred_map[split_name],
                    target_map[split_name],
                    extra_columns=extra_columns,
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_fischer_krauss_prediction_frame(
    split_map: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    probability_map: dict[str, np.ndarray],
    return_proxy_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, (_, y_split, meta_split) in split_map.items():
        if len(meta_split) == 0:
            continue
        probabilities = probability_map[split_name]
        frame = meta_split.copy()
        frame["split"] = split_name
        frame["actual_class"] = y_split.astype(int)
        frame["predicted_class"] = (probabilities[:, 1] >= 0.5).astype(int)
        frame["prob_class_0"] = probabilities[:, 0]
        frame["prob_class_1"] = probabilities[:, 1]
        frame["fk_score_raw"] = probability_to_score(probabilities)
        frame["fk_return_proxy"] = return_proxy_map[split_name]
        frame["model"] = "fischer_krauss"
        frame["prediction"] = frame["fk_return_proxy"].astype(np.float32)
        frame["actual"] = frame["actual_return"].astype(np.float32)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_fischer_krauss_scaler(run_dir: Path, mean: float, std: float) -> None:
    np.savez(
        report_benchmark_path(run_dir, "benchmark_fischer_krauss_scaler.npz"),
        mean=np.asarray([mean], dtype=np.float32),
        std=np.asarray([std], dtype=np.float32),
    )


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
        "target_normalizer": config.target_normalizer,
        "lstm_seeds": list(config.lstm_seeds),
        "signmag_signed_loss_weight": config.signmag_signed_loss_weight,
        "signmag_sign_loss_weight": config.signmag_sign_loss_weight,
        "signmag_magnitude_loss_weight": config.signmag_magnitude_loss_weight,
        "signmag_log_magnitude": bool(config.signmag_log_magnitude),
        "sample_weight_mode": config.sample_weight_mode,
        "sample_weight_strength": config.sample_weight_strength,
        "sample_weight_quantile": config.sample_weight_quantile,
        "sample_weight_clip": config.sample_weight_clip,
        "attention_enabled": bool(config.attention_enabled),
        "attention_heads": config.attention_heads,
        "attention_key_dim": config.attention_key_dim,
        "quantile_enabled": bool(config.quantile_enabled),
        "event_enabled": bool(config.event_enabled),
        "event_threshold": config.event_threshold,
        "event_signed_loss_weight": config.event_signed_loss_weight,
        "event_prob_loss_weight": config.event_prob_loss_weight,
        "event_sign_loss_weight": config.event_sign_loss_weight,
        "event_magnitude_loss_weight": config.event_magnitude_loss_weight,
        "event_log_magnitude": bool(config.event_log_magnitude),
        "fk_benchmark_enabled": bool(config.fk_benchmark_enabled),
        "fk_window_size": config.fk_window_size,
        "fk_hidden_units": config.fk_hidden_units,
        "fk_dropout": config.fk_dropout,
        "fk_learning_rate": config.fk_learning_rate,
        "fk_batch_size": config.fk_batch_size,
        "fk_epochs": config.fk_epochs,
        "fk_patience": config.fk_patience,
        "fk_train_fraction": config.fk_train_fraction,
        "fk_top_k": config.fk_top_k,
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


def resolve_recipe_stocks(
    args: argparse.Namespace,
) -> tuple[str | None, TrainingRecipe | None]:
    if args.stocks:
        return args.stocks, None
    if not args.sector:
        return None, None

    summary_path = args.stock_search_summary or DEFAULT_SEARCH_SUMMARY_PATH
    recipe = build_training_recipe(
        summary_path=summary_path,
        sector=args.sector,
        stocks=None,
        min_best_val_rel_score=args.min_stock_val_rel_score,
        max_stocks=args.max_stocks,
        top_features=args.feature_top_k,
        available_columns=None,
        extra_features=parse_csv_list(args.extra_context_features),
    )
    selected_stocks = ",".join(recipe.selected_stocks)
    print(
        "Info: Selected stocks from search summary:",
        selected_stocks,
    )
    return selected_stocks, recipe


def main() -> None:
    args = parse_args()
    config = override_config(args)
    run_dir = build_run_dir(config.output_dir, args.run_name, config.target_mode)

    selected_stocks_arg, stock_recipe = resolve_recipe_stocks(args)
    stocks_arg = selected_stocks_arg if selected_stocks_arg is not None else args.stocks

    df = load_frame(config.data_path, stocks_arg)
    df = filter_frame_by_sector(df, args.sector)
    
    feature_recipe = stock_recipe
    search_summary_applied = False
    if not args.feature_columns and not args.use_all_features:
        if args.feature_selection_mode in {"auto", "search_summary"}:
            try:
                summary_path = args.stock_search_summary or DEFAULT_SEARCH_SUMMARY_PATH
                feature_recipe = build_training_recipe(
                    summary_path=summary_path,
                    sector=args.sector,
                    stocks=tuple(sorted(df["code"].astype(str).unique().tolist())),
                    min_best_val_rel_score=args.min_stock_val_rel_score,
                    max_stocks=args.max_stocks,
                    top_features=args.feature_top_k,
                    available_columns=set(df.columns),
                    extra_features=parse_csv_list(args.extra_context_features),
                )
                config.feature_columns = feature_recipe.feature_columns
                print(
                    "Info: Using search-summary features:",
                    ",".join(config.feature_columns),
                )
                search_summary_applied = True
            except Exception as exc:
                if args.feature_selection_mode == "search_summary":
                    raise
                print(f"Info: Search-summary feature selection unavailable: {exc}")

        if args.feature_selection_mode == "sector_config" or not config.feature_columns:
            auto_features = resolve_features_for_df(df, config)
            config.feature_columns = auto_features
        elif args.feature_selection_mode == "auto" and not search_summary_applied:
            auto_features = resolve_features_for_df(df, config)
            config.feature_columns = auto_features

    validate_columns(df, config.feature_columns, config.target_column, config.target_normalizer)

    extra_meta_columns: tuple[str, ...] = ()
    target_normalizer_alias = None
    if config.target_normalizer:
        target_normalizer_alias = f"__target_normalizer__{config.target_normalizer}"
        df[target_normalizer_alias] = df[config.target_normalizer].astype(float)
        extra_meta_columns = (target_normalizer_alias,)

    train_df, val_df, test_df = split_frame_by_date(df, config.train_end_date, config.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=config.feature_columns), config.feature_columns)
    scaled_df = apply_feature_scaler(df, scaler)

    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        config.feature_columns,
        config.target_column,
        config.window_size,
        extra_meta_columns=extra_meta_columns,
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

    local_target_normalizer = None
    train_target_norm_values = None
    val_target_norm_values = None
    test_target_norm_values = None
    if target_normalizer_alias is not None:
        train_target_norm_values = meta_train[target_normalizer_alias].to_numpy(dtype=np.float32)
        val_target_norm_values = meta_val[target_normalizer_alias].to_numpy(dtype=np.float32)
        test_target_norm_values = meta_test[target_normalizer_alias].to_numpy(dtype=np.float32)
        local_target_normalizer = fit_local_target_normalizer(
            train_target_norm_values,
            config.target_normalizer,
        )

    y_train_local = apply_local_target_normalizer(y_train, train_target_norm_values, local_target_normalizer)
    y_val_local = apply_local_target_normalizer(y_val, val_target_norm_values, local_target_normalizer)
    target_scaler = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, target_scaler)
    y_val_scaled = apply_target_scaler(y_val_local, target_scaler)
    y_train_model_target = build_training_target_array(
        y_train_scaled,
        config.loss,
        local_scale_values=train_target_norm_values if local_target_normalizer is not None else None,
    )
    y_val_model_target = build_training_target_array(
        y_val_scaled,
        config.loss,
        local_scale_values=val_target_norm_values if local_target_normalizer is not None else None,
    )
    y_train_signed_target = build_training_target_array(
        y_train_local,
        config.loss,
        local_scale_values=train_target_norm_values if local_target_normalizer is not None else None,
    )
    y_val_signed_target = build_training_target_array(
        y_val_local,
        config.loss,
        local_scale_values=val_target_norm_values if local_target_normalizer is not None else None,
    )
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
    local_scale_map = {
        "train": train_target_norm_values,
        "val": val_target_norm_values,
        "test": test_target_norm_values,
    }

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

    monitor_metric = resolve_monitor_metric(config.target_mode)
    if config.loss == "rel_score" and config.sample_weight_mode != "none":
        print(
            "Info: rel_score loss is batch-level; magnitude sample weights remain active for auxiliary heads "
            "but have limited effect on the main signed/regression loss."
        )
    seed_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    first_model = None
    first_history_df = None
    for seed_idx, seed in enumerate(config.lstm_seeds):
        set_global_seed(seed)
        model, history = fit_model(
            x_train_lstm,
            y_train_model_target,
            x_val_lstm,
            y_val_model_target,
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
            metric_y_val=y_val,
            local_target_normalizer=local_target_normalizer,
            local_target_scale_values=val_target_norm_values,
            sample_weight=train_sample_weight,
            val_sample_weight=val_sample_weight,
        )
        if seed_idx == 0:
            first_model = model
            first_history_df = pd.DataFrame(history.history)
        model.save(run_dir / f"model_seed_{seed}.keras")
        pd.DataFrame(history.history).to_csv(run_dir / f"history_seed_{seed}.csv", index=False)

        split_prediction_map = build_prediction_map(model, lstm_split_arrays, predict)
        split_prediction_map = {
            split_name: inverse_target_scaler_values(pred_values, target_scaler)
            for split_name, pred_values in split_prediction_map.items()
        }
        if local_target_normalizer is not None:
            split_prediction_map = {
                split_name: inverse_local_target_normalizer(
                    pred_values,
                    local_scale_map[split_name],
                    local_target_normalizer,
                )
                for split_name, pred_values in split_prediction_map.items()
        }
        seed_prediction_maps[f"lstm_seed_{seed}"] = split_prediction_map

    quantile_seed_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    quantile_seed_upper_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    first_quantile_model = None
    first_quantile_history_df = None
    if config.quantile_enabled:
        for seed_idx, seed in enumerate(config.lstm_seeds):
            set_global_seed(seed)
            quantile_model, quantile_history = fit_quantile_model(
                x_train_lstm,
                y_train_scaled,
                x_val_lstm,
                y_val_scaled,
                window_size=config.window_size,
                num_features=x_train_lstm.shape[2],
                lstm_units=config.lstm_units,
                dropout=config.dropout,
                lr=config.lr,
                batch_size=config.batch_size,
                epochs=config.epochs,
                patience=config.patience,
                monitor_metric=monitor_metric,
                val_group_ids=meta_val["code"].to_numpy() if "code" in meta_val.columns else None,
                target_scaler=target_scaler,
                metric_y_val=y_val,
                local_target_normalizer=local_target_normalizer,
                local_target_scale_values=val_target_norm_values,
            )
            if seed_idx == 0:
                first_quantile_model = quantile_model
                first_quantile_history_df = pd.DataFrame(quantile_history.history)
            quantile_model.save(run_dir / f"model_quantile_seed_{seed}.keras")
            pd.DataFrame(quantile_history.history).to_csv(
                run_dir / f"history_quantile_seed_{seed}.csv",
                index=False,
            )

            split_prediction_map = build_prediction_map(
                quantile_model,
                lstm_split_arrays,
                lambda model, x: predict(model, x, prediction_key=0),
            )
            split_upper_prediction_map = build_prediction_map(
                quantile_model,
                lstm_split_arrays,
                lambda model, x: predict(model, x, prediction_key=1),
            )
            split_prediction_map = {
                split_name: inverse_target_scaler_values(pred_values, target_scaler)
                for split_name, pred_values in split_prediction_map.items()
            }
            split_upper_prediction_map = {
                split_name: inverse_target_scaler_values(pred_values, target_scaler)
                for split_name, pred_values in split_upper_prediction_map.items()
            }
            if local_target_normalizer is not None:
                split_prediction_map = {
                    split_name: inverse_local_target_normalizer(
                        pred_values,
                        local_scale_map[split_name],
                        local_target_normalizer,
                    )
                    for split_name, pred_values in split_prediction_map.items()
                }
                split_upper_prediction_map = {
                    split_name: inverse_local_target_normalizer(
                        pred_values,
                        local_scale_map[split_name],
                        local_target_normalizer,
                    )
                    for split_name, pred_values in split_upper_prediction_map.items()
                }
            quantile_seed_prediction_maps[f"lstm_quantile_seed_{seed}"] = split_prediction_map
            quantile_seed_upper_prediction_maps[f"lstm_quantile_seed_{seed}"] = split_upper_prediction_map

    signmag_seed_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    first_signmag_model = None
    first_signmag_history_df = None
    enable_sign_magnitude = config.target_mode.startswith("return") and local_target_normalizer is not None
    if enable_sign_magnitude:
        for seed_idx, seed in enumerate(config.lstm_seeds):
            set_global_seed(seed)
            signmag_model, signmag_history = fit_sign_magnitude_model(
                x_train_lstm,
                y_train_signed_target,
                x_val_lstm,
                y_val_signed_target,
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
                metric_y_val=y_val,
                local_target_normalizer=local_target_normalizer,
                local_target_scale_values=val_target_norm_values,
                sign_loss_weight=config.signmag_sign_loss_weight,
                magnitude_loss_weight=config.signmag_magnitude_loss_weight,
                signed_loss_weight=config.signmag_signed_loss_weight,
                use_log_magnitude=config.signmag_log_magnitude,
                sample_weight=train_sample_weight,
                val_sample_weight=val_sample_weight,
            )
            if seed_idx == 0:
                first_signmag_model = signmag_model
                first_signmag_history_df = pd.DataFrame(signmag_history.history)
            signmag_model.save(run_dir / f"model_signmag_seed_{seed}.keras")
            pd.DataFrame(signmag_history.history).to_csv(run_dir / f"history_signmag_seed_{seed}.csv", index=False)

            split_prediction_map = build_prediction_map(
                signmag_model,
                lstm_split_arrays,
                lambda model, x: predict(model, x, prediction_key="signed_prediction"),
            )
            split_prediction_map = {
                split_name: inverse_local_target_normalizer(
                    pred_values,
                    local_scale_map[split_name],
                    local_target_normalizer,
                )
                for split_name, pred_values in split_prediction_map.items()
            }
            signmag_seed_prediction_maps[f"lstm_signmag_seed_{seed}"] = split_prediction_map

    attention_seed_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    first_attention_model = None
    first_attention_history_df = None
    if config.attention_enabled:
        for seed_idx, seed in enumerate(config.lstm_seeds):
            set_global_seed(seed)
            attention_model, attention_history = fit_attention_model(
                x_train_lstm,
                y_train_model_target,
                x_val_lstm,
                y_val_model_target,
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
                metric_y_val=y_val,
                local_target_normalizer=local_target_normalizer,
                local_target_scale_values=val_target_norm_values,
                attention_heads=config.attention_heads,
                attention_key_dim=config.attention_key_dim,
                sample_weight=train_sample_weight,
                val_sample_weight=val_sample_weight,
            )
            if seed_idx == 0:
                first_attention_model = attention_model
                first_attention_history_df = pd.DataFrame(attention_history.history)
            attention_model.save(run_dir / f"model_attention_seed_{seed}.keras")
            pd.DataFrame(attention_history.history).to_csv(
                run_dir / f"history_attention_seed_{seed}.csv",
                index=False,
            )

            split_prediction_map = build_prediction_map(attention_model, lstm_split_arrays, predict)
            split_prediction_map = {
                split_name: inverse_target_scaler_values(pred_values, target_scaler)
                for split_name, pred_values in split_prediction_map.items()
            }
            if local_target_normalizer is not None:
                split_prediction_map = {
                    split_name: inverse_local_target_normalizer(
                        pred_values,
                        local_scale_map[split_name],
                        local_target_normalizer,
                    )
                    for split_name, pred_values in split_prediction_map.items()
            }
            attention_seed_prediction_maps[f"lstm_attention_seed_{seed}"] = split_prediction_map

    event_seed_prediction_maps: dict[str, dict[str, np.ndarray]] = {}
    first_event_model = None
    first_event_history_df = None
    enable_event_family = config.target_mode.startswith("return") and local_target_normalizer is not None and config.event_enabled
    if enable_event_family:
        for seed_idx, seed in enumerate(config.lstm_seeds):
            set_global_seed(seed)
            event_model, event_history = fit_event_gated_model(
                x_train_lstm,
                y_train_signed_target,
                x_val_lstm,
                y_val_signed_target,
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
                metric_y_val=y_val,
                local_target_normalizer=local_target_normalizer,
                local_target_scale_values=val_target_norm_values,
                attention_heads=config.attention_heads,
                attention_key_dim=config.attention_key_dim,
                event_threshold=config.event_threshold,
                event_loss_weight=config.event_prob_loss_weight,
                sign_loss_weight=config.event_sign_loss_weight,
                magnitude_loss_weight=config.event_magnitude_loss_weight,
                signed_loss_weight=config.event_signed_loss_weight,
                use_log_magnitude=config.event_log_magnitude,
                sample_weight=train_sample_weight,
                val_sample_weight=val_sample_weight,
            )
            if seed_idx == 0:
                first_event_model = event_model
                first_event_history_df = pd.DataFrame(event_history.history)
            event_model.save(run_dir / f"model_event_seed_{seed}.keras")
            pd.DataFrame(event_history.history).to_csv(
                run_dir / f"history_event_seed_{seed}.csv",
                index=False,
            )

            split_prediction_map = build_prediction_map(
                event_model,
                lstm_split_arrays,
                lambda model, x: predict(model, x, prediction_key="signed_prediction"),
            )
            split_prediction_map = {
                split_name: inverse_local_target_normalizer(
                    pred_values,
                    local_scale_map[split_name],
                    local_target_normalizer,
                )
                for split_name, pred_values in split_prediction_map.items()
            }
            event_seed_prediction_maps[f"lstm_event_seed_{seed}"] = split_prediction_map

    linear_model = fit_linear_regression(x_train, y_train)
    arima_model = fit_arima(x_train, y_train)
    family_selection_summary: dict[str, dict[str, object]] = {}

    lstm_seed_keys = sorted(seed_prediction_maps.keys())
    lstm_prediction_map = seed_prediction_maps[lstm_seed_keys[0]]
    prediction_maps = {
        "lstm": lstm_prediction_map,
        "linear_regression": build_prediction_map(linear_model, split_arrays, predict_linear_regression),
        "arima": build_prediction_map(arima_model, split_arrays, predict_arima),
    }
    if len(lstm_seed_keys) > 1:
        prediction_maps["lstm_ensemble"] = {
            split_name: np.mean(
                [seed_prediction_maps[model_name][split_name] for model_name in lstm_seed_keys],
                axis=0,
            ).astype(np.float32)
            for split_name in SPLIT_NAMES
        }
    prediction_maps.update(seed_prediction_maps)
    lstm_selection_maps, lstm_selection_summary = build_family_selection_maps(
        "lstm",
        seed_prediction_maps,
        targets,
        meta_map,
        config.target_mode,
    )
    prediction_maps.update(lstm_selection_maps)
    family_selection_summary["lstm"] = lstm_selection_summary
    if quantile_seed_prediction_maps:
        quantile_seed_keys = sorted(quantile_seed_prediction_maps.keys())
        prediction_maps["lstm_quantile"] = quantile_seed_prediction_maps[quantile_seed_keys[0]]
        if len(quantile_seed_keys) > 1:
            prediction_maps["lstm_quantile_ensemble"] = {
                split_name: np.mean(
                    [quantile_seed_prediction_maps[model_name][split_name] for model_name in quantile_seed_keys],
                    axis=0,
                ).astype(np.float32)
                for split_name in SPLIT_NAMES
            }
        prediction_maps.update(quantile_seed_prediction_maps)
        quantile_selection_maps, quantile_selection_summary = build_family_selection_maps(
            "lstm_quantile",
            quantile_seed_prediction_maps,
            targets,
            meta_map,
            config.target_mode,
        )
        prediction_maps.update(quantile_selection_maps)
        family_selection_summary["lstm_quantile"] = quantile_selection_summary
    if attention_seed_prediction_maps:
        attention_seed_keys = sorted(attention_seed_prediction_maps.keys())
        prediction_maps["lstm_attention"] = attention_seed_prediction_maps[attention_seed_keys[0]]
        if len(attention_seed_keys) > 1:
            prediction_maps["lstm_attention_ensemble"] = {
                split_name: np.mean(
                    [attention_seed_prediction_maps[model_name][split_name] for model_name in attention_seed_keys],
                    axis=0,
                ).astype(np.float32)
                for split_name in SPLIT_NAMES
            }
        prediction_maps.update(attention_seed_prediction_maps)
        attention_selection_maps, attention_selection_summary = build_family_selection_maps(
            "lstm_attention",
            attention_seed_prediction_maps,
            targets,
            meta_map,
            config.target_mode,
        )
        prediction_maps.update(attention_selection_maps)
        family_selection_summary["lstm_attention"] = attention_selection_summary
    if event_seed_prediction_maps:
        event_seed_keys = sorted(event_seed_prediction_maps.keys())
        prediction_maps["lstm_event"] = event_seed_prediction_maps[event_seed_keys[0]]
        if len(event_seed_keys) > 1:
            prediction_maps["lstm_event_ensemble"] = {
                split_name: np.mean(
                    [event_seed_prediction_maps[model_name][split_name] for model_name in event_seed_keys],
                    axis=0,
                ).astype(np.float32)
                for split_name in SPLIT_NAMES
            }
        prediction_maps.update(event_seed_prediction_maps)
        event_selection_maps, event_selection_summary = build_family_selection_maps(
            "lstm_event",
            event_seed_prediction_maps,
            targets,
            meta_map,
            config.target_mode,
        )
        prediction_maps.update(event_selection_maps)
        family_selection_summary["lstm_event"] = event_selection_summary
    if signmag_seed_prediction_maps:
        signmag_seed_keys = sorted(signmag_seed_prediction_maps.keys())
        prediction_maps["lstm_signmag"] = signmag_seed_prediction_maps[signmag_seed_keys[0]]
        if len(signmag_seed_keys) > 1:
            prediction_maps["lstm_signmag_ensemble"] = {
                split_name: np.mean(
                    [signmag_seed_prediction_maps[model_name][split_name] for model_name in signmag_seed_keys],
                    axis=0,
                ).astype(np.float32)
                for split_name in SPLIT_NAMES
            }
        prediction_maps.update(signmag_seed_prediction_maps)
        signmag_selection_maps, signmag_selection_summary = build_family_selection_maps(
            "lstm_signmag",
            signmag_seed_prediction_maps,
            targets,
            meta_map,
            config.target_mode,
        )
        prediction_maps.update(signmag_selection_maps)
        family_selection_summary["lstm_signmag"] = signmag_selection_summary
    quantile_extra_prediction_maps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    if quantile_seed_prediction_maps:
        quantile_aux_maps = build_quantile_aux_maps(
            "lstm_quantile",
            family_selection_summary["lstm_quantile"],
            quantile_seed_upper_prediction_maps,
        )
        quantile_q50_all_maps = {**quantile_seed_prediction_maps}
        quantile_q50_all_maps.update(
            {
                model_name: prediction_maps[model_name]
                for model_name in prediction_maps
                if model_name.startswith("lstm_quantile")
            }
        )
        quantile_q90_all_maps = {**quantile_seed_upper_prediction_maps, **quantile_aux_maps}
        quantile_extra_prediction_maps = build_quantile_extra_prediction_maps(
            quantile_q50_all_maps,
            quantile_q90_all_maps,
        )

    report_model_names = set(select_report_model_names(sorted(prediction_maps.keys()) + (["fischer_krauss"] if config.fk_benchmark_enabled else [])))
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    metric_details: dict[str, dict[str, dict[str, float | list[float]]]] = {}

    for model_name, pred_map in prediction_maps.items():
        metrics[model_name], metric_details[model_name] = compute_metrics_bundle(pred_map, targets, config.target_mode, meta_map)
        if config.target_mode.startswith("return"):
            if model_name in report_model_names:
                for split_name, detail in metric_details[model_name].items():
                    save_metric_series(run_dir, model_name, split_name, detail)

    history_df = first_history_df if first_history_df is not None else pd.DataFrame()
    prediction_df = build_prediction_frame(meta_map, targets, prediction_maps, extra_prediction_maps=quantile_extra_prediction_maps)
    fk_summary_payload: dict[str, object] | None = None
    if config.fk_benchmark_enabled:
        fk_price_column = resolve_price_column(df)
        fk_frame = prepare_fischer_krauss_frame(df, price_column=fk_price_column)
        fk_train_end_date = resolve_fk_train_end_date(
            fk_frame,
            config.val_end_date,
            train_fraction=config.fk_train_fraction,
        )
        fk_scaler = fit_fischer_krauss_scaler(fk_frame, fk_train_end_date)
        fk_frame = apply_fischer_krauss_scaler(fk_frame, fk_scaler)
        fk_x, fk_y, fk_meta = build_fischer_krauss_sequences(
            fk_frame,
            window_size=config.fk_window_size,
        )
        fk_splits = split_fischer_krauss_sequences(
            fk_x,
            fk_y,
            fk_meta,
            train_end_date=fk_train_end_date,
            validation_end_date=config.val_end_date,
        )
        fk_x_train, fk_y_train, _ = fk_splits["train"]
        fk_x_val, fk_y_val, _ = fk_splits["val"]
        fk_x_test, fk_y_test, fk_meta_test = fk_splits["test"]
        if len(fk_x_train) == 0 or len(fk_x_val) == 0 or len(fk_x_test) == 0:
            raise ValueError("Not enough Fischer-Krauss benchmark sequences for train/val/test.")

        fk_model, fk_history = fit_fischer_krauss_model(
            fk_x_train,
            fk_y_train,
            fk_x_val,
            fk_y_val,
            window_size=config.fk_window_size,
            hidden_units=config.fk_hidden_units,
            dropout=config.fk_dropout,
            learning_rate=config.fk_learning_rate,
            batch_size=config.fk_batch_size,
            epochs=config.fk_epochs,
            patience=config.fk_patience,
        )
        fk_probability_map = {
            split_name: predict_fischer_krauss_probabilities(fk_model, fk_splits[split_name][0])
            for split_name in SPLIT_NAMES
        }
        fk_metrics = {
            split_name: compute_fischer_krauss_metrics(fk_splits[split_name][1], fk_probability_map[split_name])
            for split_name in SPLIT_NAMES
        }
        fk_return_proxy_map = {
            split_name: calibrate_probability_score_to_return_proxy(
                fk_probability_map[split_name],
                fk_splits["train"][2]["actual_return"].to_numpy(dtype=np.float32),
            )
            for split_name in SPLIT_NAMES
        }
        fk_prediction_df = build_fischer_krauss_prediction_frame(
            fk_splits,
            fk_probability_map,
            fk_return_proxy_map,
        )
        fk_rel_prediction_map = {
            split_name: fk_return_proxy_map[split_name]
            for split_name in SPLIT_NAMES
        }
        fk_rel_targets = {
            split_name: fk_splits[split_name][2]["actual_return"].to_numpy(dtype=np.float32)
            for split_name in SPLIT_NAMES
        }
        fk_rel_meta = {
            split_name: fk_splits[split_name][2][["code", "Date"]].copy()
            for split_name in SPLIT_NAMES
        }
        metrics["fischer_krauss"], metric_details["fischer_krauss"] = compute_metrics_bundle(
            fk_rel_prediction_map,
            fk_rel_targets,
            "return",
            fk_rel_meta,
        )
        for split_name in SPLIT_NAMES:
            fk_metrics[split_name]["rel_score"] = metrics["fischer_krauss"][split_name]["rel_score"]
            fk_metrics[split_name]["base_loss"] = metrics["fischer_krauss"][split_name]["base_loss"]
            fk_metrics[split_name]["abs_loss"] = metrics["fischer_krauss"][split_name]["abs_loss"]
            fk_metrics[split_name]["directional_accuracy"] = metrics["fischer_krauss"][split_name]["directional_accuracy"]
        if "fischer_krauss" in report_model_names:
            for split_name, detail in metric_details["fischer_krauss"].items():
                save_metric_series(run_dir, "fischer_krauss", split_name, detail)
        prediction_df = pd.concat(
            [
                prediction_df,
                fk_prediction_df[
                    [
                        "code",
                        "Date",
                        "split",
                        "model",
                        "prediction",
                        "actual",
                        "actual_class",
                        "predicted_class",
                        "prob_class_0",
                        "prob_class_1",
                        "fk_score_raw",
                        "fk_return_proxy",
                    ]
                ],
            ],
            ignore_index=True,
        )
        fk_long_short_df = build_long_short_portfolio_returns(
            fk_meta_test,
            fk_probability_map["test"],
            top_k=config.fk_top_k,
        )
        fk_long_short_summary = summarize_long_short_portfolio(fk_long_short_df)
        fk_model.save(report_benchmark_path(run_dir, "benchmark_fischer_krauss_model.keras"))
        pd.DataFrame(fk_history.history).to_csv(
            report_benchmark_path(run_dir, "benchmark_fischer_krauss_history.csv"),
            index=False,
        )
        fk_prediction_df.to_csv(report_benchmark_path(run_dir, "benchmark_fischer_krauss_predictions.csv"), index=False)
        fk_long_short_df.to_csv(
            report_benchmark_path(run_dir, "benchmark_fischer_krauss_long_short_daily_returns.csv"),
            index=False,
        )
        with report_benchmark_path(run_dir, "benchmark_fischer_krauss_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(fk_metrics, f, indent=2)
        with report_benchmark_path(run_dir, "benchmark_fischer_krauss_long_short_summary.json").open("w", encoding="utf-8") as f:
            json.dump(fk_long_short_summary, f, indent=2)
        save_fischer_krauss_scaler(run_dir, fk_scaler.mean, fk_scaler.std)
        if not fk_long_short_df.empty:
            fk_equity_df = fk_long_short_df[["Date", "equity_curve"]].rename(columns={"equity_curve": "equity"})
            fk_equity_df["label"] = "FK LongShort"
            save_equity_curve_plot(
                fk_equity_df[["Date", "label", "equity"]],
                report_benchmark_path(run_dir, "benchmark_fischer_krauss_equity.png"),
                "Benchmark Fischer-Krauss Long/Short Equity",
            )
        fk_summary_payload = {
            "train_end_date": str(fk_train_end_date.date()),
            "validation_end_date": config.val_end_date,
            "window_size": config.fk_window_size,
            "hidden_units": config.fk_hidden_units,
            "dropout": config.fk_dropout,
            "learning_rate": config.fk_learning_rate,
            "batch_size": config.fk_batch_size,
            "epochs": config.fk_epochs,
            "patience": config.fk_patience,
            "train_fraction": config.fk_train_fraction,
            "top_k": config.fk_top_k,
            "metrics": fk_metrics,
            "long_short_summary": fk_long_short_summary,
        }

    if first_model is not None:
        first_model.save(run_dir / "model.keras")
    if first_quantile_model is not None:
        first_quantile_model.save(run_dir / "model_quantile.keras")
    if first_attention_model is not None:
        first_attention_model.save(run_dir / "model_attention.keras")
    if first_event_model is not None:
        first_event_model.save(run_dir / "model_event.keras")
    if first_signmag_model is not None:
        first_signmag_model.save(run_dir / "model_signmag.keras")
    joblib.dump(linear_model, run_dir / "linear_regression.joblib")
    save_scaler(run_dir, scaler)
    save_target_scaler(run_dir, target_scaler)
    history_df.to_csv(report_core_path(run_dir, "history.csv"), index=False)
    if first_quantile_history_df is not None:
        first_quantile_history_df.to_csv(report_core_path(run_dir, "history_quantile.csv"), index=False)
    if first_attention_history_df is not None:
        first_attention_history_df.to_csv(report_core_path(run_dir, "history_attention.csv"), index=False)
    if first_event_history_df is not None:
        first_event_history_df.to_csv(report_core_path(run_dir, "history_event.csv"), index=False)
    if first_signmag_history_df is not None:
        first_signmag_history_df.to_csv(report_core_path(run_dir, "history_signmag.csv"), index=False)
    prediction_df.to_csv(report_core_path(run_dir, "predictions.csv"), index=False)
    with report_core_path(run_dir, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if config.target_mode.startswith("return"):
        with report_core_path(run_dir, "metric_details.json").open("w", encoding="utf-8") as f:
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
    with report_core_path(run_dir, "config.json").open("w", encoding="utf-8") as f:
        payload = build_config_payload(config, args, train_df, val_df, test_df, splits, monitor_metric)
        payload["stocks"] = stocks_arg
        payload["sector"] = args.sector
        payload["lstm_use_stock_identity"] = use_stock_identity
        payload["lstm_stock_identity_codes"] = list(stock_to_idx.keys()) if use_stock_identity else []
        payload["lstm_input_feature_count"] = int(x_train_lstm.shape[2])
        payload["feature_selection_mode"] = args.feature_selection_mode
        payload["recipe_sector"] = args.sector
        payload["recipe_selected_stocks"] = sorted(df["code"].astype(str).unique().tolist())
        payload["recipe_source"] = feature_recipe.source if feature_recipe is not None else None
        payload["recipe_feature_summary"] = feature_recipe.feature_summary if feature_recipe is not None else []
        payload["recipe_stock_summary"] = feature_recipe.stock_summary if feature_recipe is not None else []
        payload["lstm_attention_enabled"] = bool(config.attention_enabled)
        payload["lstm_quantile_enabled"] = bool(config.quantile_enabled)
        payload["lstm_event_enabled"] = enable_event_family
        payload["lstm_signmag_enabled"] = enable_sign_magnitude
        payload["fischer_krauss_benchmark"] = fk_summary_payload
        payload["family_selection_summary"] = family_selection_summary
        if local_target_normalizer is not None:
            payload["target_normalizer_floor"] = float(local_target_normalizer.floor)
        json.dump(payload, f, indent=2)

    if family_selection_summary:
        with report_core_path(run_dir, "family_selection_summary.json").open("w", encoding="utf-8") as f:
            json.dump(family_selection_summary, f, indent=2)

    for model_name in select_report_model_names(sorted(prediction_df["model"].dropna().unique().tolist())):
        try:
            save_actual_vs_prediction_plot(run_dir, prediction_df, model_name)
            save_rel_score_hist_plot(run_dir, model_name)
        except Exception:
            pass

    mirror_run_artifacts(run_dir)
    cleanup_report_noise(run_dir)
    cleanup_legacy_report_artifacts(run_dir)

    print("Saved run to:", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
