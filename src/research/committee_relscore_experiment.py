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

from src.evaluation.metric import evaluate
from src.models.reporting import resolve_run_artifact

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "committee_experiments"
)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a two-model committee using existing prediction artifacts.")
    parser.add_argument("--expert-run", type=Path, required=True)
    parser.add_argument("--market-run", type=Path, required=True)
    parser.add_argument("--expert-models", default="lstm_best_by_val")
    parser.add_argument("--market-models", default="lstm_best_by_val")
    parser.add_argument("--methods", default="avg,agree_only")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--stable-weight-tolerance", type=float, default=0.001)
    parser.add_argument("--selection-mode", choices=["best_val", "stable_band"], default="best_val")
    parser.add_argument("--stable-selection-val-gap", type=float, default=0.006)
    parser.add_argument("--stable-selection-min-weight-count", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def load_prediction_frame(run_dir: Path, model_name: str) -> pd.DataFrame:
    predictions_path = resolve_run_artifact(run_dir, "predictions.csv", bucket="core")
    df = pd.read_csv(predictions_path)
    df = df[df["model"] == model_name].copy()
    if df.empty:
        raise ValueError(f"Model '{model_name}' not found in {predictions_path}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["split", "code", "Date", "actual", "prediction"]]


def merge_prediction_frames(
    expert_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = expert_df.merge(
        market_df,
        on=["split", "code", "Date"],
        how="inner",
        suffixes=("_expert", "_market"),
    )
    if merged.empty:
        raise ValueError("No overlapping predictions found between expert and market runs.")

    actual_gap = np.abs(merged["actual_expert"] - merged["actual_market"])
    if float(actual_gap.max()) > 1e-6:
        raise ValueError("Expert and market predictions disagree on aligned actual values.")

    merged["actual"] = merged["actual_expert"].astype(np.float32)
    merged["agreement"] = (
        np.sign(merged["prediction_expert"].astype(float))
        == np.sign(merged["prediction_market"].astype(float))
    )
    return merged


def build_committee_prediction(merged: pd.DataFrame, method: str, weight_expert: float) -> np.ndarray:
    expert = merged["prediction_expert"].to_numpy(dtype=np.float32)
    market = merged["prediction_market"].to_numpy(dtype=np.float32)
    combined = weight_expert * expert + (1.0 - weight_expert) * market

    if method == "avg":
        return combined.astype(np.float32)
    if method == "agree_only":
        agree = merged["agreement"].to_numpy(dtype=bool)
        return np.where(agree, combined, expert).astype(np.float32)
    raise ValueError(f"Unsupported committee method: {method}")


def evaluate_split(df: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    aligned = df.sort_values(["code", "Date"], kind="stable")
    result = evaluate(
        aligned[prediction_column].to_numpy(dtype=np.float32),
        aligned["actual"].to_numpy(dtype=np.float32),
        group_ids=aligned["code"].to_numpy(),
    )
    return {
        "rel_score": float(result["rel_score"]),
        "base_loss": float(result["base_loss"]),
        "abs_loss": float(result["abs_loss"]),
    }


def summarize_component(merged: pd.DataFrame, prediction_column: str) -> dict[str, object]:
    val_df = merged[merged["split"] == "val"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    return {
        "val": evaluate_split(val_df, prediction_column),
        "test": evaluate_split(test_df, prediction_column),
    }


def build_output_dir(args: argparse.Namespace) -> Path:
    output_name = args.output_name or f"{args.expert_run.name}__plus__{args.market_run.name}"
    output_dir = args.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_stability_summary(
    results_df: pd.DataFrame,
    *,
    stable_weight_tolerance: float,
) -> pd.DataFrame:
    group_cols = ["expert_run", "expert_model", "market_run", "market_model", "method"]
    rows: list[dict[str, object]] = []

    for _, group_df in results_df.groupby(group_cols, sort=False):
        ranked = group_df.sort_values(
            ["committee_val_rel_score", "committee_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        )
        best = ranked.iloc[0]
        threshold = float(best["committee_val_rel_score"]) - stable_weight_tolerance
        stable = ranked[ranked["committee_val_rel_score"] >= threshold].copy()
        rows.append(
            {
                "expert_run": best["expert_run"],
                "expert_model": best["expert_model"],
                "market_run": best["market_run"],
                "market_model": best["market_model"],
                "method": best["method"],
                "code_count": int(best["code_count"]),
                "overlap_codes": best["overlap_codes"],
                "best_weight_expert": float(best["weight_expert"]),
                "best_val_rel_score": float(best["committee_val_rel_score"]),
                "best_test_rel_score": float(best["committee_test_rel_score"]),
                "stable_weight_tolerance": float(stable_weight_tolerance),
                "stable_weight_min": float(stable["weight_expert"].min()),
                "stable_weight_max": float(stable["weight_expert"].max()),
                "stable_weight_count": int(len(stable)),
                "stable_weight_center": float(stable["weight_expert"].mean()),
                "stable_test_rel_score_mean": float(stable["committee_test_rel_score"].mean()),
                "stable_test_rel_score_median": float(stable["committee_test_rel_score"].median()),
                "stable_test_rel_score_min": float(stable["committee_test_rel_score"].min()),
                "stable_test_rel_score_max": float(stable["committee_test_rel_score"].max()),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["best_val_rel_score", "best_test_rel_score", "stable_weight_count"],
        ascending=[False, False, False],
        kind="stable",
    )
    return summary_df


def select_best_committee_row(
    results_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    *,
    selection_mode: str,
    stable_selection_val_gap: float,
    stable_selection_min_weight_count: int,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if results_df.empty:
        raise ValueError("No committee rows available for selection.")

    if selection_mode == "best_val" or stability_df.empty:
        best = results_df.iloc[0].to_dict()
        return best, None

    global_best_val = float(stability_df["best_val_rel_score"].max())
    candidates = stability_df[
        stability_df["best_val_rel_score"] >= global_best_val - max(0.0, stable_selection_val_gap)
    ].copy()
    if stable_selection_min_weight_count > 1:
        filtered = candidates[candidates["stable_weight_count"] >= stable_selection_min_weight_count].copy()
        if not filtered.empty:
            candidates = filtered
    candidates["stable_weight_span"] = candidates["stable_weight_max"] - candidates["stable_weight_min"]
    candidates = candidates.sort_values(
        ["stable_weight_count", "stable_weight_span", "best_val_rel_score", "best_test_rel_score"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    chosen_stability = candidates.iloc[0].to_dict()
    matched = results_df[
        (results_df["expert_model"] == chosen_stability["expert_model"])
        & (results_df["market_model"] == chosen_stability["market_model"])
        & (results_df["method"] == chosen_stability["method"])
        & (results_df["weight_expert"] == chosen_stability["best_weight_expert"])
    ]
    if matched.empty:
        matched = results_df[
            (results_df["expert_model"] == chosen_stability["expert_model"])
            & (results_df["market_model"] == chosen_stability["market_model"])
            & (results_df["method"] == chosen_stability["method"])
        ].sort_values(
            ["committee_val_rel_score", "committee_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        )
    selected = matched.iloc[0].to_dict()
    return selected, chosen_stability


def main() -> None:
    args = parse_args()
    output_dir = build_output_dir(args)

    expert_models = parse_csv_list(args.expert_models)
    market_models = parse_csv_list(args.market_models)
    methods = parse_csv_list(args.methods)
    if args.weight_step <= 0 or args.weight_step > 1:
        raise ValueError("--weight-step must be in (0, 1].")

    weights = np.round(np.arange(0.0, 1.0 + 1e-9, args.weight_step), 10)
    all_rows: list[dict[str, object]] = []

    for expert_model in expert_models:
        expert_df = load_prediction_frame(args.expert_run, expert_model)
        for market_model in market_models:
            market_df = load_prediction_frame(args.market_run, market_model)
            merged = merge_prediction_frames(expert_df, market_df)

            component_summary = {
                "expert": summarize_component(merged, "prediction_expert"),
                "market": summarize_component(merged, "prediction_market"),
            }
            overlap_codes = ",".join(sorted(merged["code"].drop_duplicates().tolist()))
            overlap_counts = {
                "val": int((merged["split"] == "val").sum()),
                "test": int((merged["split"] == "test").sum()),
                "code_count": int(merged["code"].nunique()),
            }
            agreement_rate = float(merged["agreement"].mean())

            for method in methods:
                for weight_expert in weights:
                    trial = merged.copy()
                    trial["prediction_committee"] = build_committee_prediction(trial, method, float(weight_expert))
                    val_metrics = evaluate_split(trial[trial["split"] == "val"].copy(), "prediction_committee")
                    test_metrics = evaluate_split(trial[trial["split"] == "test"].copy(), "prediction_committee")
                    row = {
                        "expert_run": args.expert_run.name,
                        "expert_model": expert_model,
                        "market_run": args.market_run.name,
                        "market_model": market_model,
                        "method": method,
                        "weight_expert": float(weight_expert),
                        "agreement_rate": agreement_rate,
                        "overlap_codes": overlap_codes,
                        "code_count": overlap_counts["code_count"],
                        "val_rows": overlap_counts["val"],
                        "test_rows": overlap_counts["test"],
                        "expert_val_rel_score_overlap": component_summary["expert"]["val"]["rel_score"],
                        "expert_test_rel_score_overlap": component_summary["expert"]["test"]["rel_score"],
                        "market_val_rel_score_overlap": component_summary["market"]["val"]["rel_score"],
                        "market_test_rel_score_overlap": component_summary["market"]["test"]["rel_score"],
                        "committee_val_rel_score": val_metrics["rel_score"],
                        "committee_test_rel_score": test_metrics["rel_score"],
                        "committee_val_abs_loss": val_metrics["abs_loss"],
                        "committee_test_abs_loss": test_metrics["abs_loss"],
                    }
                    all_rows.append(row)
    if not all_rows:
        raise ValueError("No committee candidate could be evaluated.")

    results_df = pd.DataFrame(all_rows).sort_values(
        ["committee_val_rel_score", "committee_test_rel_score"],
        ascending=[False, False],
    )
    results_path = output_dir / "committee_grid_results.csv"
    results_df.to_csv(results_path, index=False)

    stability_df = build_stability_summary(
        results_df,
        stable_weight_tolerance=max(0.0, float(args.stable_weight_tolerance)),
    )
    stability_path = output_dir / "committee_stability_summary.csv"
    stability_df.to_csv(stability_path, index=False)

    best_row, selected_stability_row = select_best_committee_row(
        results_df,
        stability_df,
        selection_mode=args.selection_mode,
        stable_selection_val_gap=float(args.stable_selection_val_gap),
        stable_selection_min_weight_count=int(args.stable_selection_min_weight_count),
    )

    expert_df = load_prediction_frame(args.expert_run, str(best_row["expert_model"]))
    market_df = load_prediction_frame(args.market_run, str(best_row["market_model"]))
    best_merged = merge_prediction_frames(expert_df, market_df)
    best_merged["prediction_committee"] = build_committee_prediction(
        best_merged,
        str(best_row["method"]),
        float(best_row["weight_expert"]),
    )

    best_predictions = best_merged[
        [
            "split",
            "code",
            "Date",
            "actual",
            "prediction_expert",
            "prediction_market",
            "prediction_committee",
            "agreement",
        ]
    ].copy()
    best_predictions_path = output_dir / "best_committee_predictions.csv"
    best_predictions.to_csv(best_predictions_path, index=False)

    best_stability_row = None
    if selected_stability_row is not None:
        best_stability_row = selected_stability_row
    elif not stability_df.empty:
        matched = stability_df[
            (stability_df["expert_model"] == best_row["expert_model"])
            & (stability_df["market_model"] == best_row["market_model"])
            & (stability_df["method"] == best_row["method"])
        ]
        if not matched.empty:
            best_stability_row = matched.iloc[0].to_dict()

    summary = {
        "expert_run": str(args.expert_run),
        "market_run": str(args.market_run),
        "selection_mode": args.selection_mode,
        "best_committee": best_row,
        "results_path": str(results_path),
        "stability_path": str(stability_path),
        "best_committee_stability": best_stability_row,
        "best_predictions_path": str(best_predictions_path),
    }
    summary_path = output_dir / "best_committee_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {results_path}")
    print(f"Saved: {stability_path}")
    print(f"Saved: {best_predictions_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
