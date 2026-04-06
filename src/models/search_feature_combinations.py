from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.metric import evaluate
from src.models.baseline import fit_linear_regression, predict_linear_regression
from src.models.config import FEATURE_COLUMNS, get_config
from src.models.lstm import (
    apply_feature_scaler,
    build_sequence_dataset,
    fit_feature_scaler,
    split_frame_by_date,
    split_sequence_dataset,
)
from scripts.run_train import load_frame, validate_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search feature combinations with linear regression baseline.")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--target-mode", choices=["price", "growth", "return", "return_3d", "return_5d"], default="return")
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument("--min-combo-size", type=int, default=1)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--min-rel-score", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=50)
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
    if args.feature_columns:
        config.feature_columns = tuple(item.strip() for item in args.feature_columns.split(",") if item.strip())
    return config


def build_run_dir(base_dir: Path, run_name: str | None, target_mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"feature_search_{target_mode}_{stamp}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def score_predictions(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    metric = evaluate(prediction, actual)
    return {
        "mse": float(np.mean((actual - prediction) ** 2)),
        "mae": float(np.mean(np.abs(actual - prediction))),
        "base_loss": float(metric["base_loss"]),
        "abs_loss": float(metric["abs_loss"]),
        "rel_score": float(metric["rel_score"]),
    }


def prepare_split_data(df: pd.DataFrame, feature_columns: tuple[str, ...], target_column: str, train_end_date: str, val_end_date: str, window_size: int):
    validate_columns(df, feature_columns, target_column)
    train_df, _, _ = split_frame_by_date(df, train_end_date, val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=feature_columns), feature_columns)
    scaled_df = apply_feature_scaler(df, scaler)
    x_all, y_all, meta_all = build_sequence_dataset(scaled_df, feature_columns, target_column, window_size)
    return split_sequence_dataset(x_all, y_all, meta_all, train_end_date, val_end_date)


def run_feature_search(
    df: pd.DataFrame,
    *,
    stocks: str | None,
    config,
    min_combo_size: int,
    max_combo_size: int,
    min_rel_score: float,
    top_k: int,
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    feature_pool = tuple(col for col in config.feature_columns if col in df.columns)
    if not feature_pool:
        raise ValueError("No valid feature columns found in dataset.")

    rows = []
    for combo_size in range(min_combo_size, max_combo_size + 1):
        for feature_combo in combinations(feature_pool, combo_size):
            splits = prepare_split_data(
                df,
                feature_combo,
                config.target_column,
                config.train_end_date,
                config.val_end_date,
                config.window_size,
            )
            x_train, y_train, _ = splits["train"]
            x_val, y_val, _ = splits["val"]
            x_test, y_test, _ = splits["test"]
            if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
                continue

            try:
                model = fit_linear_regression(x_train, y_train)
                val_pred = predict_linear_regression(model, x_val)
                test_pred = predict_linear_regression(model, x_test)
                train_pred = predict_linear_regression(model, x_train)
            except LinAlgError:
                continue

            train_score = score_predictions(y_train, train_pred)
            val_score = score_predictions(y_val, val_pred)
            test_score = score_predictions(y_test, test_pred)

            rows.append(
                {
                    "features": ",".join(feature_combo),
                    "feature_count": combo_size,
                    "train_rel_score": train_score["rel_score"],
                    "val_rel_score": val_score["rel_score"],
                    "test_rel_score": test_score["rel_score"],
                    "train_mse": train_score["mse"],
                    "val_mse": val_score["mse"],
                    "test_mse": test_score["mse"],
                    "train_mae": train_score["mae"],
                    "val_mae": val_score["mae"],
                    "test_mae": test_score["mae"],
                    "train_abs_loss": train_score["abs_loss"],
                    "val_abs_loss": val_score["abs_loss"],
                    "test_abs_loss": test_score["abs_loss"],
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No valid feature combinations were evaluated.")

    results = results.sort_values(
        ["val_rel_score", "test_rel_score", "feature_count"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results.to_csv(run_dir / "feature_search_results.csv", index=False)

    good = results[results["val_rel_score"] > min_rel_score].copy()
    good = good.sort_values(["val_rel_score", "test_rel_score"], ascending=[False, False]).head(top_k)
    good.to_csv(run_dir / "feature_search_good.csv", index=False)

    summary = {
        "stocks": stocks,
        "target_mode": config.target_mode,
        "target_column": config.target_column,
        "window_size": config.window_size,
        "feature_pool": list(feature_pool),
        "min_combo_size": min_combo_size,
        "max_combo_size": max_combo_size,
        "min_rel_score": min_rel_score,
        "tested_combinations": int(len(results)),
        "good_combinations": int(len(good)),
        "best_val_rel_score": float(results.iloc[0]["val_rel_score"]),
        "best_test_rel_score": float(results.iloc[0]["test_rel_score"]),
        "best_features": results.iloc[0]["features"],
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return results, good, summary


def main() -> None:
    args = parse_args()
    config = override_config(args)
    run_dir = build_run_dir(config.output_dir, args.run_name, config.target_mode)

    if not config.target_mode.startswith("return"):
        raise ValueError("Feature search currently expects a return-style target_mode because it ranks by rel_score.")

    df = load_frame(config.data_path, args.stocks)
    _, _, summary = run_feature_search(
        df,
        stocks=args.stocks,
        config=config,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        min_rel_score=args.min_rel_score,
        top_k=args.top_k,
        run_dir=run_dir,
    )

    print("Saved search to:", run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
