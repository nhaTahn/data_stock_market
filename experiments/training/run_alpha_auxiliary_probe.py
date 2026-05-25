"""Alpha-target heteroscedastic probe for the multi-market framework.

This is a lightweight next step after the context-adapter smoke:
- train the heteroscedastic combined model on date-demeaned returns (alpha),
- evaluate both raw-return rel_score (using alpha prediction as zero-market pred)
  and alpha_rel_score,
- keep holdout/test closed.

The script intentionally reuses the hetero runner utilities to avoid adding a
second model stack.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_hetero_nll_probe import (  # noqa: E402
    MARKET_CONTEXT_FEATURES,
    PreparedData,
    build_hetero_combined_model,
    evaluate_predictions,
    load_and_prepare,
    parse_seeds,
    predict_raw,
    train_model,
)
from src.models.training.scalers import (  # noqa: E402
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_target_scaler,
)

DEFAULT_FEATURES = (
    "close_return",
    "adjust_return",
    "range_pct",
    "body_pct",
    "volume_change",
    "momentum_5",
    "momentum_20",
    "volatility_5",
    "volatility_20",
    "ma_5_gap",
    "ma_20_gap",
    *MARKET_CONTEXT_FEATURES,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", default="43,52,62")
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--add-market-context-features", action="store_true", default=True)
    return parser.parse_args(argv)


def date_demean(values: np.ndarray, dates: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"Date": pd.to_datetime(dates), "value": values})
    return (frame["value"] - frame.groupby("Date")["value"].transform("mean")).to_numpy(dtype=np.float32)


def with_alpha_target(data: PreparedData) -> tuple[PreparedData, np.ndarray, np.ndarray]:
    alpha_train = date_demean(data.y_train_raw, data.meta_train["Date"])
    alpha_val = date_demean(data.y_val_raw, data.meta_val["Date"])

    alpha_train_local = apply_local_target_normalizer(alpha_train, data.train_scale, data.local_normalizer)
    alpha_val_local = apply_local_target_normalizer(alpha_val, data.val_scale, data.local_normalizer)
    alpha_scaler = fit_target_scaler(alpha_train_local)
    y_train_scaled = apply_target_scaler(alpha_train_local, alpha_scaler).reshape(-1, 1)
    y_val_scaled = apply_target_scaler(alpha_val_local, alpha_scaler).reshape(-1, 1)
    y_train_model = np.concatenate([y_train_scaled, data.train_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    y_val_model = np.concatenate([y_val_scaled, data.val_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    alpha_data = replace(
        data,
        y_train_model=y_train_model,
        y_val_model=y_val_model,
        target_scaler=alpha_scaler,
    )
    return alpha_data, alpha_train, alpha_val


def alpha_metrics(raw_actual: np.ndarray, alpha_actual: np.ndarray, alpha_pred: np.ndarray, meta: pd.DataFrame) -> dict[str, float]:
    raw_metrics = evaluate_predictions(raw_actual, alpha_pred, np.zeros_like(alpha_pred), meta)
    alpha_eval = evaluate_predictions(alpha_actual, alpha_pred, np.zeros_like(alpha_pred), meta)
    return {
        "raw_rel_score": raw_metrics["rel_score"],
        "raw_directional_accuracy": raw_metrics["directional_accuracy"],
        "raw_pred_actual_q90_ratio": raw_metrics["pred_actual_q90_ratio"],
        "alpha_rel_score": alpha_eval["rel_score"],
        "alpha_directional_accuracy": alpha_eval["directional_accuracy"],
        "alpha_pred_actual_q90_ratio": alpha_eval["pred_actual_q90_ratio"],
        "alpha_median_abs_error": alpha_eval["median_abs_error"],
        "alpha_q90_abs_error": alpha_eval["q90_abs_error"],
        "daily_q90_max": alpha_eval["daily_q90_max"],
        "spike_days_ge_8pct": alpha_eval["spike_days_ge_8pct"],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    raw_data = load_and_prepare(args)
    data, alpha_train, alpha_val = with_alpha_target(raw_data)
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        model = build_hetero_combined_model(data, args)
        train_model(model, data, args, seed)
        pred_val, sigma_val = predict_raw(model, data, data.x_val, data.val_scale)
        metrics = alpha_metrics(data.y_val_raw, alpha_val, pred_val, data.meta_val)
        metrics.update({"seed": seed, "split": "val", "mean_sigma": float(np.mean(sigma_val))})
        rows.append(metrics)
        pred_train, sigma_train = predict_raw(model, data, data.x_train, data.train_scale)
        train_metrics = alpha_metrics(data.y_train_raw, alpha_train, pred_train, data.meta_train)
        train_metrics.update({"seed": seed, "split": "train", "mean_sigma": float(np.mean(sigma_train))})
        rows.append(train_metrics)
        np.savez_compressed(
            args.output_dir / f"alpha_aux_seed_{seed}.npz",
            pred_val=pred_val.astype(np.float32),
            sigma_val=sigma_val.astype(np.float32),
            y_val=data.y_val_raw.astype(np.float32),
            alpha_val=alpha_val.astype(np.float32),
            val_dates=pd.to_datetime(data.meta_val["Date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=str),
            val_codes=data.meta_val["code"].astype(str).to_numpy(),
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    result.to_csv(args.gold_dir / "results_per_seed.csv", index=False)
    val = result[result["split"] == "val"]
    aggregate = val.agg(
        {
            "raw_rel_score": ["mean", "std"],
            "alpha_rel_score": ["mean", "std"],
            "raw_directional_accuracy": ["mean", "std"],
            "alpha_directional_accuracy": ["mean", "std"],
            "alpha_pred_actual_q90_ratio": ["mean", "std"],
            "daily_q90_max": ["mean", "std"],
        }
    )
    aggregate.to_csv(args.output_dir / "aggregate.csv")
    aggregate.to_csv(args.gold_dir / "aggregate.csv")
    text = "\n".join(
        [
            "# Alpha Auxiliary Hetero Probe",
            "",
            "Trains on date-demeaned returns (alpha), evaluates raw and alpha metrics. Holdout/test not used.",
            "",
            "## Validation Per Seed",
            "",
            val.round(6).to_markdown(index=False),
            "",
            "## Aggregate",
            "",
            aggregate.round(6).to_markdown(),
            "",
            json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2),
        ]
    )
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
