from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize targeted overnight rel_score runs.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_best_model(metrics: dict[str, object]) -> tuple[str, dict[str, object]]:
    rows: list[tuple[str, float, float, dict[str, object]]] = []
    for model_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        val = float(payload.get("val", {}).get("rel_score", float("-inf")))
        test = float(payload.get("test", {}).get("rel_score", float("-inf")))
        rows.append((model_name, val, test, payload))
    rows.sort(key=lambda item: (item[1], item[2]), reverse=True)
    model_name, _, _, payload = rows[0]
    return model_name, payload


def compute_bias_stats(predictions_path: Path, model_name: str) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    df = df[df["model"] == model_name].copy()
    if df.empty:
        return {}
    rows: dict[str, float] = {}
    for split in ("val", "test"):
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        pred_abs_mean = float(split_df["prediction"].abs().mean())
        actual_abs_mean = float(split_df["actual"].abs().mean())
        rows[f"{split}_pred_mean"] = float(split_df["prediction"].mean())
        rows[f"{split}_actual_mean"] = float(split_df["actual"].mean())
        rows[f"{split}_pred_pos_rate"] = float((split_df["prediction"] > 0).mean())
        rows[f"{split}_actual_pos_rate"] = float((split_df["actual"] > 0).mean())
        rows[f"{split}_pred_abs_mean"] = pred_abs_mean
        rows[f"{split}_actual_abs_mean"] = actual_abs_mean
        rows[f"{split}_pred_abs_over_actual_abs"] = (
            pred_abs_mean / actual_abs_mean if actual_abs_mean > 0 else float("nan")
        )
    return rows


def summarize_train_run(run_name: str) -> dict[str, object]:
    run_dir = RUN_BASE / run_name
    core_dir = run_dir / "reports" / "core"
    metrics = load_json(core_dir / "metrics.json")
    config = load_json(core_dir / "config.json")
    best_model_name, best_payload = find_best_model(metrics)
    result = {
        "run_name": run_name,
        "kind": "train",
        "best_model_by_val": best_model_name,
        "best_val_rel_score": float(best_payload["val"]["rel_score"]),
        "best_test_rel_score": float(best_payload["test"]["rel_score"]),
        "best_test_directional_accuracy": float(best_payload["test"]["directional_accuracy"]),
        "stocks": config.get("stocks"),
        "window_size": config.get("window_size"),
        "lstm_units": config.get("lstm_units"),
        "lr": config.get("lr"),
        "dropout": config.get("dropout"),
        "sample_weight_mode": config.get("sample_weight_mode"),
    }
    result.update(compute_bias_stats(core_dir / "predictions.csv", best_model_name))
    return result


def summarize_shared_committee(run_name: str, preset: str) -> dict[str, object]:
    run_dir = RUN_BASE / run_name
    core_dir = run_dir / "reports" / "core"
    summary_path = (
        RUN_BASE
        / "reports"
        / "committee_experiments"
        / f"{run_name}__committee__{preset}"
        / "best_committee_summary.json"
    )
    summary = load_json(summary_path)
    committee = summary["best_committee"]
    stability = summary.get("best_committee_stability") or {}
    metrics = load_json(core_dir / "metrics.json")
    best_model_name, best_payload = find_best_model(metrics)
    return {
        "run_name": run_name,
        "kind": "shared_committee",
        "preset": preset,
        "context_best_model_by_val": best_model_name,
        "context_best_val_rel_score": float(best_payload["val"]["rel_score"]),
        "context_best_test_rel_score": float(best_payload["test"]["rel_score"]),
        "committee_expert_model": committee.get("expert_model"),
        "committee_market_model": committee.get("market_model"),
        "committee_method": committee.get("method"),
        "committee_weight_expert": committee.get("weight_expert"),
        "committee_code_count": committee.get("code_count"),
        "committee_val_rel_score": committee.get("committee_val_rel_score"),
        "committee_test_rel_score": committee.get("committee_test_rel_score"),
        "stable_weight_min": stability.get("stable_weight_min"),
        "stable_weight_max": stability.get("stable_weight_max"),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    run_rows: list[dict[str, object]] = []
    committee_rows: list[dict[str, object]] = []

    for row in rows:
        kind = row["kind"]
        run_name = row["run_name"]
        if kind == "train":
            payload = summarize_train_run(run_name)
            payload["label"] = row.get("label")
            run_rows.append(payload)
        elif kind == "shared_committee":
            payload = summarize_shared_committee(run_name, row["preset"])
            payload["label"] = row.get("label")
            committee_rows.append(payload)

    run_summary_path = args.manifest.parent / "relscore_push_runs_summary.csv"
    committee_summary_path = args.manifest.parent / "relscore_push_committee_summary.csv"

    if run_rows:
        pd.DataFrame(run_rows).sort_values(
            ["best_val_rel_score", "best_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        ).to_csv(run_summary_path, index=False)
    if committee_rows:
        pd.DataFrame(committee_rows).sort_values(
            ["committee_val_rel_score", "committee_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        ).to_csv(committee_summary_path, index=False)

    payload = {
        "manifest": str(args.manifest),
        "run_summary_path": str(run_summary_path) if run_rows else None,
        "committee_summary_path": str(committee_summary_path) if committee_rows else None,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
