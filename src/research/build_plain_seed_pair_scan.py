from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(ROOT))

from src.evaluation.metric import evaluate

RUN_BASE = ROOT / "data/processed/assets/data_info_vn/history/training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan pair ensembles across plain LSTM seeds and package selected pair models.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-run-name", required=True)
    parser.add_argument("--score-amp-weight", type=float, default=0.08)
    parser.add_argument("--score-posgap-weight", type=float, default=0.08)
    return parser.parse_args()


def eval_rel(df: pd.DataFrame, prediction_column: str = "prediction") -> dict[str, float]:
    aligned = df.sort_values(["code", "Date"], kind="stable")
    result = evaluate(
        aligned[prediction_column].to_numpy(dtype=np.float32),
        aligned["actual"].to_numpy(dtype=np.float32),
        group_ids=aligned["code"].to_numpy(),
    )
    return {
        "rel_score": float(result["rel_score"]),
        "abs_loss": float(result["abs_loss"]),
        "base_loss": float(result["base_loss"]),
    }


def split_bias(df: pd.DataFrame) -> dict[str, float]:
    actual_abs = float(df["actual"].abs().mean())
    pred_abs = float(df["prediction"].abs().mean())
    return {
        "pred_pos_rate": float((df["prediction"] > 0.0).mean()),
        "actual_pos_rate": float((df["actual"] > 0.0).mean()),
        "pos_rate_gap": float((df["prediction"] > 0.0).mean() - (df["actual"] > 0.0).mean()),
        "pred_abs_over_actual_abs": float(pred_abs / actual_abs) if actual_abs else 0.0,
        "corr": float(df["prediction"].corr(df["actual"])) if len(df) > 1 else 0.0,
    }


def package_pair_models(
    output_run_dir: Path,
    grouped_predictions: dict[str, pd.DataFrame],
    summary_rows: list[dict[str, object]],
    source_run_name: str,
) -> None:
    core = output_run_dir / "reports/core"
    core.mkdir(parents=True, exist_ok=True)

    frames = []
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for row in summary_rows:
        model_name = str(row["packaged_model_name"])
        df = grouped_predictions[model_name].copy()
        df["model"] = model_name
        frames.append(df[["code", "Date", "actual", "split", "model", "prediction"]])
        metrics[model_name] = {
            "val": {
                "rel_score": float(row["val_rel_score"]),
                "abs_loss": float(row["val_abs_loss"]),
            },
            "test": {
                "rel_score": float(row["test_rel_score"]),
                "abs_loss": float(row["test_abs_loss"]),
            },
        }

    pd.concat(frames, ignore_index=True).to_csv(core / "predictions.csv", index=False)
    (core / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config = {
        "source_run": source_run_name,
        "phase2_pair_packaged": True,
        "models": [row["packaged_model_name"] for row in summary_rows],
    }
    (core / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    family_summary = {
        "lstm_pair": {
            "ranked_models": [
                {
                    "model": row["packaged_model_name"],
                    "val_score": row["val_rel_score"],
                    "test_score": row["test_rel_score"],
                }
                for row in summary_rows
            ],
            "best_by_val": max(summary_rows, key=lambda item: float(item["val_rel_score"]))["packaged_model_name"],
            "top2_by_val": [row["packaged_model_name"] for row in sorted(summary_rows, key=lambda item: float(item["val_rel_score"]), reverse=True)[:2]],
        }
    }
    (core / "family_selection_summary.json").write_text(json.dumps(family_summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    core = run_dir / "reports/core"
    predictions = pd.read_csv(core / "predictions.csv")
    predictions["Date"] = pd.to_datetime(predictions["Date"])
    seed_models = sorted([model for model in predictions["model"].unique() if model.startswith("lstm_seed_")])
    if len(seed_models) < 2:
        raise ValueError("Need at least two plain LSTM seed models.")

    rows: list[dict[str, object]] = []
    grouped_outputs: dict[str, pd.DataFrame] = {}
    for i, seed_a in enumerate(seed_models):
        for seed_b in seed_models[i + 1 :]:
            pair_df = (
                predictions[predictions["model"].isin([seed_a, seed_b])]
                .groupby(["split", "code", "Date", "actual"], as_index=False)["prediction"]
                .mean()
            )
            val_df = pair_df[pair_df["split"] == "val"].copy()
            test_df = pair_df[pair_df["split"] == "test"].copy()
            val_metrics = eval_rel(val_df)
            test_metrics = eval_rel(test_df)
            val_bias = split_bias(val_df)
            test_bias = split_bias(test_df)
            selection_score = (
                val_metrics["rel_score"]
                + float(args.score_amp_weight) * val_bias["pred_abs_over_actual_abs"]
                - float(args.score_posgap_weight) * abs(val_bias["pos_rate_gap"])
            )
            pair_name = f"{seed_a}__{seed_b}"
            rows.append(
                {
                    "pair_name": pair_name,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "val_rel_score": val_metrics["rel_score"],
                    "test_rel_score": test_metrics["rel_score"],
                    "val_abs_loss": val_metrics["abs_loss"],
                    "test_abs_loss": test_metrics["abs_loss"],
                    "val_pred_abs_over_actual_abs": val_bias["pred_abs_over_actual_abs"],
                    "val_pos_rate_gap": val_bias["pos_rate_gap"],
                    "val_corr": val_bias["corr"],
                    "test_pred_abs_over_actual_abs": test_bias["pred_abs_over_actual_abs"],
                    "test_pos_rate_gap": test_bias["pos_rate_gap"],
                    "test_corr": test_bias["corr"],
                    "selection_score": selection_score,
                }
            )
            grouped_outputs[pair_name] = pair_df

    pair_grid = pd.DataFrame(rows).sort_values(
        ["selection_score", "val_rel_score", "test_rel_score"],
        ascending=[False, False, False],
        kind="stable",
    )
    best_by_val = pair_grid.sort_values(["val_rel_score", "test_rel_score"], ascending=[False, False], kind="stable").iloc[0].to_dict()
    best_by_score = pair_grid.iloc[0].to_dict()

    output_run_dir = RUN_BASE / args.output_run_name
    if output_run_dir.exists():
        import shutil

        shutil.rmtree(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "reports/core").mkdir(parents=True, exist_ok=True)

    pair_grid.to_csv(output_run_dir / "reports/core/pair_scan_grid.csv", index=False)

    packaged_rows = []
    for source_row, packaged_name in (
        (best_by_val, "lstm_pair_best_by_val"),
        (best_by_score, "lstm_pair_best_by_score"),
    ):
        row = dict(source_row)
        row["packaged_model_name"] = packaged_name
        grouped_outputs[packaged_name] = grouped_outputs[row["pair_name"]].copy()
        packaged_rows.append(row)

    package_pair_models(output_run_dir, grouped_outputs, packaged_rows, run_dir.name)

    summary = {
        "source_run": run_dir.name,
        "seed_count": len(seed_models),
        "best_pair_by_val": best_by_val,
        "best_pair_by_score": best_by_score,
        "pair_grid_path": str(output_run_dir / "reports/core/pair_scan_grid.csv"),
        "packaged_run": str(output_run_dir),
    }
    (output_run_dir / "reports/core/pair_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
