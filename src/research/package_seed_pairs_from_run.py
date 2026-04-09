from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(ROOT))

from src.evaluation.metric import evaluate

RUN_BASE = ROOT / "data/processed/assets/data_info_vn/history/training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package selected seed pairs from an existing run as reusable models.")
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="Format: label=model_a,model_b",
    )
    return parser.parse_args()


def parse_pair_specs(values: list[str]) -> list[tuple[str, list[str]]]:
    pairs: list[tuple[str, list[str]]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid pair spec: {item}")
        label, models_csv = item.split("=", 1)
        models = [x.strip() for x in models_csv.split(",") if x.strip()]
        if len(models) != 2:
            raise ValueError(f"Pair '{label}' must contain exactly two models.")
        pairs.append((label.strip(), models))
    return pairs


def eval_split(df: pd.DataFrame) -> dict[str, float]:
    aligned = df.sort_values(["code", "Date"], kind="stable")
    result = evaluate(
        aligned["prediction"].to_numpy(dtype=np.float32),
        aligned["actual"].to_numpy(dtype=np.float32),
        group_ids=aligned["code"].to_numpy(),
    )
    return {
        "rel_score": float(result["rel_score"]),
        "abs_loss": float(result["abs_loss"]),
        "base_loss": float(result["base_loss"]),
    }


def main() -> None:
    args = parse_args()
    source_run = args.source_run.resolve()
    output_run = RUN_BASE / args.output_run_name
    if output_run.exists():
        shutil.rmtree(output_run)

    pair_specs = parse_pair_specs(args.pair)
    source_core = source_run / "reports/core"
    predictions = pd.read_csv(source_core / "predictions.csv")
    predictions["Date"] = pd.to_datetime(predictions["Date"])

    rows = []
    packaged_frames = []
    metrics: dict[str, dict[str, dict[str, float]]] = {}

    for label, models in pair_specs:
        packaged_model_name = f"lstm_pair_{label}"
        pair_df = (
            predictions[predictions["model"].isin(models)]
            .groupby(["split", "code", "Date", "actual"], as_index=False)["prediction"]
            .mean()
        )
        val_df = pair_df[pair_df["split"] == "val"].copy()
        test_df = pair_df[pair_df["split"] == "test"].copy()
        val_metrics = eval_split(val_df)
        test_metrics = eval_split(test_df)
        rows.append(
            {
                "label": label,
                "packaged_model_name": packaged_model_name,
                "seed_a": models[0],
                "seed_b": models[1],
                "val_rel_score": val_metrics["rel_score"],
                "test_rel_score": test_metrics["rel_score"],
                "val_abs_loss": val_metrics["abs_loss"],
                "test_abs_loss": test_metrics["abs_loss"],
            }
        )
        pair_df["model"] = packaged_model_name
        packaged_frames.append(pair_df[["code", "Date", "actual", "split", "model", "prediction"]])
        metrics[packaged_model_name] = {
            "val": {
                "rel_score": val_metrics["rel_score"],
                "abs_loss": val_metrics["abs_loss"],
            },
            "test": {
                "rel_score": test_metrics["rel_score"],
                "abs_loss": test_metrics["abs_loss"],
            },
        }

    output_core = output_run / "reports/core"
    output_core.mkdir(parents=True, exist_ok=True)
    pd.concat(packaged_frames, ignore_index=True).to_csv(output_core / "predictions.csv", index=False)
    (output_core / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pair_df = pd.DataFrame(rows).sort_values(["val_rel_score", "test_rel_score"], ascending=[False, False], kind="stable")
    pair_df.to_csv(output_core / "pair_frontier_summary.csv", index=False)
    family_selection = {
        "lstm_pair": {
            "ranked_models": [
                {
                    "model": row["packaged_model_name"],
                    "val_score": row["val_rel_score"],
                    "test_score": row["test_rel_score"],
                }
                for row in pair_df.to_dict(orient="records")
            ],
            "best_by_val": pair_df.iloc[0]["packaged_model_name"],
            "top2_by_val": pair_df["packaged_model_name"].head(2).tolist(),
        }
    }
    (output_core / "family_selection_summary.json").write_text(json.dumps(family_selection, indent=2), encoding="utf-8")
    config = {
        "source_run": source_run.name,
        "packaged_seed_pairs": rows,
    }
    (output_core / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = {
        "source_run": source_run.name,
        "output_run": str(output_run),
        "pairs": rows,
    }
    (output_core / "pair_frontier_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
