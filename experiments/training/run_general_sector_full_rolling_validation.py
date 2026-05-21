from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_current_best_signmag_feature_pruning import (  # noqa: E402
    FAST_OVERLAP_BLOCK,
    SECTOR_BREADTH_FEATURES,
    SECTOR_MOMENTUM_RELATIVE_FEATURES,
    SECTOR_RETURN_FEATURES,
    SOURCE_CONFIG_PATH,
    VINGROUP_MOMENTUM_BLOCK,
    ordered_remove,
    ordered_replace,
)
from src.evaluation.metric import evaluate  # noqa: E402
from src.models.training import (  # noqa: E402
    apply_feature_scaler,
    apply_local_target_normalizer,
    build_magnitude_sample_weights,
    build_sequence_dataset,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_sign_magnitude_model,
    inverse_local_target_normalizer,
    predict,
    set_global_seed,
    split_sequence_dataset,
)
from src.models.training.pipeline import (  # noqa: E402
    augment_sequence_with_stock_identity,
    build_stock_to_idx,
    build_training_target_array,
    load_frame,
    resolve_monitor_metric,
    validate_columns,
)
from src.utils.features import ensure_paper_features  # noqa: E402


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "rolling_validation"


@dataclass(frozen=True)
class RollingWindow:
    name: str
    train_end_date: str
    val_end_date: str

    @property
    def val_start_date(self) -> str:
        return (pd.Timestamp(self.train_end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


DEFAULT_WINDOWS: tuple[RollingWindow, ...] = (
    RollingWindow("val_2019", "2018-12-31", "2019-12-31"),
    RollingWindow("val_2020", "2019-12-31", "2020-12-31"),
    RollingWindow("val_2021", "2020-12-31", "2021-12-31"),
    RollingWindow("val_2022_partial", "2021-12-31", "2022-11-15"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train/validation-only rolling checks for the portable general_sector_full candidate."
    )
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override. Defaults to source config seeds.")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_seed_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return [int(seed) for seed in default]
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed.")
    return seeds


def build_general_sector_full_features(source_config: dict) -> tuple[str, ...]:
    base_features = tuple(str(item) for item in source_config["feature_columns"])
    no_fast_overlap = ordered_remove(base_features, FAST_OVERLAP_BLOCK)
    return ordered_replace(
        no_fast_overlap,
        VINGROUP_MOMENTUM_BLOCK,
        (
            *SECTOR_MOMENTUM_RELATIVE_FEATURES,
            *SECTOR_RETURN_FEATURES,
            *SECTOR_BREADTH_FEATURES,
        ),
        insert_after="vnindex_return",
    )


def build_quartile_equity(meta: pd.DataFrame, prediction: np.ndarray, actual: np.ndarray, quantile: float = 0.25) -> float:
    work = meta.loc[:, ["Date", "code"]].copy()
    work["prediction"] = np.asarray(prediction, dtype=float)
    work["actual"] = np.asarray(actual, dtype=float)
    daily_returns: list[float] = []
    for _, date_df in work.groupby("Date", sort=True):
        date_work = date_df.dropna(subset=["prediction", "actual"]).sort_values("prediction", kind="stable")
        count = len(date_work)
        selection_size = min(count // 2, max(1, int(np.floor(count * quantile))))
        if selection_size <= 0:
            continue
        longs = date_work.tail(selection_size)
        shorts = date_work.head(selection_size)
        daily_returns.append(float(longs["actual"].mean() - shorts["actual"].mean()))
    if not daily_returns:
        return 1.0
    return float(np.prod(1.0 + np.asarray(daily_returns, dtype=float)))


def score_prediction(actual: np.ndarray, prediction: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    group_ids = meta["code"].to_numpy() if "code" in meta.columns else None
    metric = evaluate(prediction, actual, group_ids=group_ids)
    error = np.asarray(metric["error"], dtype=float)
    return {
        "rel_score": float(metric["rel_score"]),
        "directional_accuracy": float(metric["directional_accuracy"]),
        "base_loss": float(metric["base_loss"]),
        "abs_loss": float(metric["abs_loss"]),
        "error_q2": float(np.quantile(error, 0.2)),
        "error_q8": float(np.quantile(error, 0.8)),
        "quartile_equity": build_quartile_equity(meta, prediction, actual),
    }


def prepare_frame(source_config: dict, feature_columns: tuple[str, ...], max_date: str) -> pd.DataFrame:
    df = load_frame(Path(source_config["data_path"]), str(source_config["stocks"]))
    if source_config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)
    df = df[pd.to_datetime(df["Date"]) <= pd.Timestamp(max_date)].copy()
    validate_columns(df, feature_columns, str(source_config["target_column"]), str(source_config["target_normalizer"]))
    return df


def run_window(
    window: RollingWindow,
    source_config: dict,
    feature_columns: tuple[str, ...],
    seeds: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    df = prepare_frame(source_config, feature_columns, window.val_end_date)
    target_normalizer = str(source_config["target_normalizer"])
    target_normalizer_alias = f"__target_normalizer__{target_normalizer}"
    df[target_normalizer_alias] = df[target_normalizer].astype(float)

    train_df = df[pd.to_datetime(df["Date"]) <= pd.Timestamp(window.train_end_date)].copy()
    scaler = fit_feature_scaler(train_df.dropna(subset=feature_columns), feature_columns)
    scaled_df = apply_feature_scaler(df, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        feature_columns,
        str(source_config["target_column"]),
        int(source_config["window_size"]),
        extra_meta_columns=(target_normalizer_alias,),
        sequence_normalization=str(source_config.get("sequence_normalization", "none")),
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, window.train_end_date, window.val_end_date)
    x_train, y_train, meta_train = splits["train"]
    x_val, y_val, meta_val = splits["val"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(f"Window {window.name} has empty train or validation sequences.")

    stock_to_idx = build_stock_to_idx([meta_train, meta_val])
    x_train_lstm = augment_sequence_with_stock_identity(x_train, meta_train, stock_to_idx)
    x_val_lstm = augment_sequence_with_stock_identity(x_val, meta_val, stock_to_idx)

    train_target_norm_values = meta_train[target_normalizer_alias].to_numpy(dtype=np.float32)
    val_target_norm_values = meta_val[target_normalizer_alias].to_numpy(dtype=np.float32)
    local_target_normalizer = fit_local_target_normalizer(train_target_norm_values, target_normalizer)
    y_train_local = apply_local_target_normalizer(y_train, train_target_norm_values, local_target_normalizer)
    y_val_local = apply_local_target_normalizer(y_val, val_target_norm_values, local_target_normalizer)
    y_train_signed_target = build_training_target_array(
        y_train_local,
        str(source_config["loss"]),
        local_scale_values=train_target_norm_values,
    )
    y_val_signed_target = build_training_target_array(
        y_val_local,
        str(source_config["loss"]),
        local_scale_values=val_target_norm_values,
    )
    train_sample_weight = build_magnitude_sample_weights(
        y_train_local,
        strength=float(source_config["sample_weight_strength"]),
        reference_quantile=float(source_config["sample_weight_quantile"]),
        clip_multiple=float(source_config["sample_weight_clip"]),
    )
    val_sample_weight = build_magnitude_sample_weights(
        y_val_local,
        strength=float(source_config["sample_weight_strength"]),
        reference_quantile=float(source_config["sample_weight_quantile"]),
        clip_multiple=float(source_config["sample_weight_clip"]),
    )

    seed_rows: list[dict[str, object]] = []
    seed_predictions: dict[int, np.ndarray] = {}
    for seed in seeds:
        set_global_seed(seed)
        model, _ = fit_sign_magnitude_model(
            x_train_lstm,
            y_train_signed_target,
            x_val_lstm,
            y_val_signed_target,
            window_size=int(source_config["window_size"]),
            num_features=int(x_train_lstm.shape[2]),
            lstm_units=source_config["lstm_units"],
            dropout=float(source_config["dropout"]),
            lr=float(source_config["lr"]),
            loss=str(source_config["loss"]),
            huber_delta=float(source_config["huber_delta"]),
            rel_score_large_move_quantile=float(source_config["rel_score_large_move_quantile"]),
            rel_score_directional_penalty=float(source_config["rel_score_directional_penalty"]),
            rel_score_confidence_penalty=float(source_config["rel_score_confidence_penalty"]),
            rel_score_confidence_ratio=float(source_config["rel_score_confidence_ratio"]),
            rel_score_weighted_high_quantile=float(source_config["rel_score_weighted_high_quantile"]),
            rel_score_weighted_high_weight=float(source_config["rel_score_weighted_high_weight"]),
            rel_score_weighted_base_weight=float(source_config["rel_score_weighted_base_weight"]),
            batch_size=int(source_config["batch_size"]),
            epochs=int(source_config["epochs"]),
            patience=int(source_config["patience"]),
            monitor_metric=resolve_monitor_metric(str(source_config["target_mode"])),
            val_group_ids=meta_val["code"].to_numpy(),
            metric_y_val=y_val,
            local_target_normalizer=local_target_normalizer,
            local_target_scale_values=val_target_norm_values,
            sign_loss_weight=float(source_config["signmag_sign_loss_weight"]),
            magnitude_loss_weight=float(source_config["signmag_magnitude_loss_weight"]),
            signed_loss_weight=float(source_config["signmag_signed_loss_weight"]),
            use_log_magnitude=bool(source_config.get("signmag_log_magnitude", True)),
            sample_weight=train_sample_weight,
            val_sample_weight=val_sample_weight,
        )
        pred_local = predict(model, x_val_lstm, prediction_key="signed_prediction")
        prediction = inverse_local_target_normalizer(pred_local, val_target_norm_values, local_target_normalizer)
        seed_predictions[seed] = prediction
        score = score_prediction(y_val, prediction, meta_val)
        seed_rows.append(
            {
                "window": window.name,
                "train_end_date": window.train_end_date,
                "val_start_date": window.val_start_date,
                "val_end_date": window.val_end_date,
                "seed": seed,
                "train_rows": int(len(y_train)),
                "val_rows": int(len(y_val)),
                **score,
            }
        )

    best_row = max(seed_rows, key=lambda row: float(row["rel_score"]))
    top2 = sorted(seed_rows, key=lambda row: float(row["rel_score"]), reverse=True)[:2]
    ensemble_prediction = np.mean([seed_predictions[int(row["seed"])] for row in top2], axis=0)
    ensemble_score = score_prediction(y_val, ensemble_prediction, meta_val)
    window_summary = {
        "window": window.name,
        "train_end_date": window.train_end_date,
        "val_start_date": window.val_start_date,
        "val_end_date": window.val_end_date,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "best_seed": int(best_row["seed"]),
        "best_seed_rel_score": float(best_row["rel_score"]),
        "best_seed_directional_accuracy": float(best_row["directional_accuracy"]),
        "best_seed_quartile_equity": float(best_row["quartile_equity"]),
        "top2_seeds": ",".join(str(row["seed"]) for row in top2),
        "top2_ensemble_rel_score": float(ensemble_score["rel_score"]),
        "top2_ensemble_directional_accuracy": float(ensemble_score["directional_accuracy"]),
        "top2_ensemble_error_q2": float(ensemble_score["error_q2"]),
        "top2_ensemble_error_q8": float(ensemble_score["error_q8"]),
        "top2_ensemble_quartile_equity": float(ensemble_score["quartile_equity"]),
    }
    return seed_rows, window_summary


def write_outputs(
    output_dir: Path,
    manifest: dict[str, object],
    seed_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    seed_columns = list(seed_rows[0].keys()) if seed_rows else []
    with (output_dir / "per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_columns)
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_columns = list(summary_rows[0].keys()) if summary_rows else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# General Sector Full Rolling Validation",
        "",
        "| Window | Train end | Validation | Rows | Best seed rel_score | Top2 rel_score | Top2 q2/q8 | Top2 quartile equity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"`{row['window']}` | `{row['train_end_date']}` | "
            f"`{row['val_start_date']}..{row['val_end_date']}` | "
            f"{row['val_rows']} | "
            f"{float(row['best_seed_rel_score']):+.4f} | "
            f"{float(row['top2_ensemble_rel_score']):+.4f} | "
            f"{float(row['top2_ensemble_error_q2']):+.4f} / {float(row['top2_ensemble_error_q8']):+.4f} | "
            f"{float(row['top2_ensemble_quartile_equity']):.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_config = load_json(SOURCE_CONFIG_PATH)
    feature_columns = build_general_sector_full_features(source_config)
    seeds = parse_seed_list(args.seeds, [int(seed) for seed in source_config["lstm_seeds"]])
    output_dir = REPORT_ROOT / f"general_sector_full_rolling_{args.stamp}"
    manifest = {
        "candidate": "general_sector_full",
        "stamp": args.stamp,
        "source_config_path": str(SOURCE_CONFIG_PATH),
        "output_dir": str(output_dir),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "seeds": seeds,
        "windows": [window.__dict__ | {"val_start_date": window.val_start_date} for window in DEFAULT_WINDOWS],
        "holdout_policy": "train/validation-only script; source rows are filtered to each val_end_date before sequence building",
    }
    if args.print_only:
        print(json.dumps(manifest, indent=2))
        return

    all_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for window in DEFAULT_WINDOWS:
        print(f"[ROLLING] {window.name}: train <= {window.train_end_date}, val <= {window.val_end_date}")
        seed_rows, window_summary = run_window(window, source_config, feature_columns, seeds)
        all_seed_rows.extend(seed_rows)
        summary_rows.append(window_summary)
        print(json.dumps(window_summary, indent=2))

    write_outputs(output_dir, manifest, all_seed_rows, summary_rows)
    print(json.dumps({"output_dir": str(output_dir), "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
