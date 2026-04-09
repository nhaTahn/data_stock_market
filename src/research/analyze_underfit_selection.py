from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze amplitude bias, split gaps, and selection instability for a run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-prefix", default="underfit_selection")
    return parser.parse_args()


def resolve_core(run_dir: Path, filename: str) -> Path:
    candidate = run_dir / "reports" / "core" / filename
    if candidate.exists():
        return candidate
    return run_dir / filename


def split_bias_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, model_df in df.groupby("model"):
        for split_name, split_df in model_df.groupby("split"):
            actual_abs = float(split_df["actual"].abs().mean())
            pred_abs = float(split_df["prediction"].abs().mean())
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "rows": int(len(split_df)),
                    "pred_pos_rate": float((split_df["prediction"] > 0).mean()),
                    "actual_pos_rate": float((split_df["actual"] > 0).mean()),
                    "pred_mean": float(split_df["prediction"].mean()),
                    "actual_mean": float(split_df["actual"].mean()),
                    "pred_abs_mean": pred_abs,
                    "actual_abs_mean": actual_abs,
                    "pred_abs_over_actual_abs": float(pred_abs / actual_abs) if actual_abs > 0 else np.nan,
                    "corr": float(split_df["prediction"].corr(split_df["actual"])) if len(split_df) > 1 else np.nan,
                }
            )
    return rows


def build_gap_rows(metrics: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, split_map in metrics.items():
        train = split_map.get("train", {})
        val = split_map.get("val", {})
        test = split_map.get("test", {})
        train_rel = train.get("rel_score")
        val_rel = val.get("rel_score")
        test_rel = test.get("rel_score")
        rows.append(
            {
                "model": model_name,
                "train_rel_score": train_rel,
                "val_rel_score": val_rel,
                "test_rel_score": test_rel,
                "train_minus_val": None if train_rel is None or val_rel is None else float(train_rel - val_rel),
                "val_minus_test": None if val_rel is None or test_rel is None else float(val_rel - test_rel),
                "train_minus_test": None if train_rel is None or test_rel is None else float(train_rel - test_rel),
                "train_directional_accuracy": train.get("directional_accuracy"),
                "val_directional_accuracy": val.get("directional_accuracy"),
                "test_directional_accuracy": test.get("directional_accuracy"),
            }
        )
    return rows


def build_family_instability_rows(family_selection: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_name, summary in family_selection.items():
        ranked = summary.get("ranked_models", [])
        if not ranked:
            continue
        best = ranked[0]
        top2 = ranked[:2]
        val_scores = [float(item["val_score"]) for item in ranked if item.get("val_score") is not None]
        test_scores = [float(item["test_score"]) for item in ranked if item.get("test_score") is not None]
        rows.append(
            {
                "family": family_name,
                "best_by_val": summary.get("best_by_val"),
                "best_val_score": best.get("val_score"),
                "best_test_score": best.get("test_score"),
                "second_by_val": top2[1]["model"] if len(top2) > 1 else None,
                "second_val_score": top2[1].get("val_score") if len(top2) > 1 else None,
                "second_test_score": top2[1].get("test_score") if len(top2) > 1 else None,
                "top2_val_gap": None
                if len(top2) < 2
                else float(top2[0]["val_score"] - top2[1]["val_score"]),
                "top2_test_gap": None
                if len(top2) < 2
                else float(top2[0]["test_score"] - top2[1]["test_score"]),
                "best_val_minus_test": None
                if best.get("val_score") is None or best.get("test_score") is None
                else float(best["val_score"] - best["test_score"]),
                "val_score_std": float(np.std(val_scores)) if val_scores else None,
                "test_score_std": float(np.std(test_scores)) if test_scores else None,
                "positive_test_ratio": float(np.mean(np.array(test_scores) > 0.0)) if test_scores else None,
            }
        )
    return rows


def classify_run(gap_df: pd.DataFrame, instability_df: pd.DataFrame, bias_df: pd.DataFrame) -> dict[str, object]:
    focus_models = gap_df[gap_df["model"].isin(["lstm_best_by_val", "lstm_signmag_best_by_val", "lstm_seed_52", "lstm_signmag_seed_52"])]
    focus_bias = bias_df[(bias_df["split"] == "test") & (bias_df["model"].isin(["lstm_best_by_val", "lstm_signmag_best_by_val", "lstm_seed_52", "lstm_signmag_seed_52"]))]
    underfit_signals = []
    for _, row in focus_bias.iterrows():
        if pd.notna(row["pred_abs_over_actual_abs"]) and float(row["pred_abs_over_actual_abs"]) < 0.2:
            underfit_signals.append(f"{row['model']} amplitude_ratio={row['pred_abs_over_actual_abs']:.3f}")
        if pd.notna(row["pred_pos_rate"]) and pd.notna(row["actual_pos_rate"]) and abs(float(row["pred_pos_rate"]) - float(row["actual_pos_rate"])) > 0.2:
            underfit_signals.append(
                f"{row['model']} pos_rate_gap={float(row['pred_pos_rate']) - float(row['actual_pos_rate']):.3f}"
            )
    instability_signals = []
    for _, row in instability_df.iterrows():
        if pd.notna(row["top2_test_gap"]) and abs(float(row["top2_test_gap"])) > 0.02:
            instability_signals.append(f"{row['family']} top2_test_gap={row['top2_test_gap']:.3f}")
        if pd.notna(row["positive_test_ratio"]) and float(row["positive_test_ratio"]) < 0.5:
            instability_signals.append(f"{row['family']} positive_test_ratio={row['positive_test_ratio']:.3f}")
    return {
        "underfit_signals": underfit_signals,
        "selection_instability_signals": instability_signals,
        "focus_models": focus_models.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    predictions = pd.read_csv(resolve_core(run_dir, "predictions.csv"))
    metrics = json.load(open(resolve_core(run_dir, "metrics.json"), "r", encoding="utf-8"))
    family_selection = json.load(open(resolve_core(run_dir, "family_selection_summary.json"), "r", encoding="utf-8"))

    bias_df = pd.DataFrame(split_bias_rows(predictions))
    gap_df = pd.DataFrame(build_gap_rows(metrics))
    instability_df = pd.DataFrame(build_family_instability_rows(family_selection))
    summary = classify_run(gap_df, instability_df, bias_df)

    output_dir = run_dir / "reports" / "core"
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_df.to_csv(output_dir / f"{args.output_prefix}_bias.csv", index=False)
    gap_df.to_csv(output_dir / f"{args.output_prefix}_gaps.csv", index=False)
    instability_df.to_csv(output_dir / f"{args.output_prefix}_instability.csv", index=False)
    with (output_dir / f"{args.output_prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(output_dir / f"{args.output_prefix}_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
