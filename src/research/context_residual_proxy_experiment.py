from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.research.committee_relscore_experiment import (
    DEFAULT_OUTPUT_DIR,
    evaluate_split,
    load_prediction_frame,
    merge_prediction_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple context+residual proxy combinations using existing expert and market predictions."
    )
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--market-run", type=Path, required=True)
    parser.add_argument("--expert-model", default="lstm_best_by_val")
    parser.add_argument("--market-model", default="lstm_best_by_val")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_output_dir(args: argparse.Namespace) -> Path:
    output_name = args.output_name or f"{args.expert_run.name}__residual_proxy__{args.market_run.name}"
    output_dir = args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def fit_residual_affine(val_df: pd.DataFrame) -> tuple[float, float]:
    x = val_df["prediction_expert"].to_numpy(dtype=np.float64)
    y = (val_df["actual"] - val_df["prediction_market"]).to_numpy(dtype=np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_two_signal_ols(val_df: pd.DataFrame) -> tuple[float, float, float]:
    expert = val_df["prediction_expert"].to_numpy(dtype=np.float64)
    market = val_df["prediction_market"].to_numpy(dtype=np.float64)
    actual = val_df["actual"].to_numpy(dtype=np.float64)
    design = np.column_stack([expert, market, np.ones_like(expert)])
    coef, _, _, _ = np.linalg.lstsq(design, actual, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def fit_two_signal_ridge(val_df: pd.DataFrame, alpha: float) -> tuple[float, float, float]:
    expert = val_df["prediction_expert"].to_numpy(dtype=np.float64)
    market = val_df["prediction_market"].to_numpy(dtype=np.float64)
    actual = val_df["actual"].to_numpy(dtype=np.float64)
    features = np.column_stack([expert, market])
    mean_x = features.mean(axis=0)
    std_x = features.std(axis=0)
    std_x[std_x == 0.0] = 1.0
    features_z = (features - mean_x) / std_x
    mean_y = actual.mean()
    centered_y = actual - mean_y
    gram = features_z.T @ features_z + alpha * np.eye(features_z.shape[1], dtype=np.float64)
    weights_z = np.linalg.solve(gram, features_z.T @ centered_y)
    weights = weights_z / std_x
    intercept = mean_y - float(mean_x @ weights)
    return float(weights[0]), float(weights[1]), float(intercept)


def evaluate_candidate(
    merged: pd.DataFrame,
    *,
    candidate_name: str,
    prediction: np.ndarray,
    details: dict[str, float],
) -> dict[str, object]:
    frame = merged.copy()
    frame["prediction_candidate"] = prediction.astype(np.float32)
    val = evaluate_split(frame[frame["split"] == "val"].copy(), "prediction_candidate")
    test = evaluate_split(frame[frame["split"] == "test"].copy(), "prediction_candidate")
    return {
        "candidate": candidate_name,
        "val_rel_score": val["rel_score"],
        "test_rel_score": test["rel_score"],
        "val_abs_loss": val["abs_loss"],
        "test_abs_loss": test["abs_loss"],
        **details,
    }


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)

    expert_df = load_prediction_frame(args.expert_run, args.expert_model)
    market_df = load_prediction_frame(args.market_run, args.market_model)
    merged = merge_prediction_frames(expert_df, market_df)
    val_df = merged[merged["split"] == "val"].copy()

    rows: list[dict[str, object]] = []
    rows.append(
        evaluate_candidate(
            merged,
            candidate_name="expert_only",
            prediction=merged["prediction_expert"].to_numpy(dtype=np.float32),
            details={},
        )
    )
    rows.append(
        evaluate_candidate(
            merged,
            candidate_name="market_only",
            prediction=merged["prediction_market"].to_numpy(dtype=np.float32),
            details={},
        )
    )

    residual_coef, residual_bias = fit_residual_affine(val_df)
    residual_prediction = (
        merged["prediction_market"].to_numpy(dtype=np.float32)
        + residual_coef * merged["prediction_expert"].to_numpy(dtype=np.float32)
        + residual_bias
    )
    rows.append(
        evaluate_candidate(
            merged,
            candidate_name="market_plus_residual_affine",
            prediction=residual_prediction,
            details={
                "coef_expert": residual_coef,
                "coef_market": 1.0,
                "intercept": residual_bias,
            },
        )
    )

    ols_expert, ols_market, ols_bias = fit_two_signal_ols(val_df)
    ols_prediction = (
        ols_expert * merged["prediction_expert"].to_numpy(dtype=np.float32)
        + ols_market * merged["prediction_market"].to_numpy(dtype=np.float32)
        + ols_bias
    )
    rows.append(
        evaluate_candidate(
            merged,
            candidate_name="two_signal_ols",
            prediction=ols_prediction,
            details={
                "coef_expert": ols_expert,
                "coef_market": ols_market,
                "intercept": ols_bias,
            },
        )
    )

    for alpha in parse_float_list(args.ridge_alphas):
        ridge_expert, ridge_market, ridge_bias = fit_two_signal_ridge(val_df, alpha)
        ridge_prediction = (
            ridge_expert * merged["prediction_expert"].to_numpy(dtype=np.float32)
            + ridge_market * merged["prediction_market"].to_numpy(dtype=np.float32)
            + ridge_bias
        )
        rows.append(
            evaluate_candidate(
                merged,
                candidate_name=f"two_signal_ridge_{alpha:g}",
                prediction=ridge_prediction,
                details={
                    "ridge_alpha": alpha,
                    "coef_expert": ridge_expert,
                    "coef_market": ridge_market,
                    "intercept": ridge_bias,
                },
            )
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["val_rel_score", "test_rel_score"],
        ascending=[False, False],
        kind="stable",
    )
    results_path = output_dir / "context_residual_proxy_results.csv"
    results_df.to_csv(results_path, index=False)

    best = results_df.iloc[0].to_dict()
    summary = {
        "expert_run": str(args.expert_run),
        "expert_model": args.expert_model,
        "market_run": str(args.market_run),
        "market_model": args.market_model,
        "results_path": str(results_path),
        "best_candidate": best,
    }
    summary_path = output_dir / "context_residual_proxy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
