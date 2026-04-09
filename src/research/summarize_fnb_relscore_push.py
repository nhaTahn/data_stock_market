from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize focused F&B rel_score push runs.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_best_model(metrics: dict[str, object]) -> tuple[str, dict[str, object]]:
    ranked: list[tuple[str, float, float, dict[str, object]]] = []
    for model_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        val = float(payload.get("val", {}).get("rel_score", float("-inf")))
        test = float(payload.get("test", {}).get("rel_score", float("-inf")))
        ranked.append((model_name, val, test, payload))
    ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
    name, _, _, payload = ranked[0]
    return name, payload


def compute_bias_stats(predictions_path: Path, model_name: str) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    df = df[df["model"] == model_name].copy()
    out: dict[str, float] = {}
    for split in ("val", "test"):
        part = df[df["split"] == split].copy()
        if part.empty:
            continue
        pred_abs = float(part["prediction"].abs().mean())
        actual_abs = float(part["actual"].abs().mean())
        out[f"{split}_pred_pos_rate"] = float((part["prediction"] > 0).mean())
        out[f"{split}_actual_pos_rate"] = float((part["actual"] > 0).mean())
        out[f"{split}_pred_abs_over_actual_abs"] = pred_abs / actual_abs if actual_abs > 0 else float("nan")
        out[f"{split}_pred_mean"] = float(part["prediction"].mean())
        out[f"{split}_actual_mean"] = float(part["actual"].mean())
    return out


def summarize_train(run_name: str, label: str) -> dict[str, object]:
    core_dir = RUN_BASE / run_name / "reports" / "core"
    metrics = load_json(core_dir / "metrics.json")
    config = load_json(core_dir / "config.json")
    best_model_name, best_payload = find_best_model(metrics)
    row = {
        "run_name": run_name,
        "label": label,
        "best_model_by_val": best_model_name,
        "best_val_rel_score": float(best_payload["val"]["rel_score"]),
        "best_test_rel_score": float(best_payload["test"]["rel_score"]),
        "best_test_directional_accuracy": float(best_payload["test"]["directional_accuracy"]),
        "window_size": config.get("window_size"),
        "lstm_units": config.get("lstm_units"),
        "lr": config.get("lr"),
        "dropout": config.get("dropout"),
        "sample_weight_mode": config.get("sample_weight_mode"),
    }
    row.update(compute_bias_stats(core_dir / "predictions.csv", best_model_name))
    return row


def summarize_committee(output_name: str, label: str, committee_kind: str) -> dict[str, object]:
    path = RUN_BASE / "reports" / "committee_experiments" / output_name / "best_committee_summary.json"
    summary = load_json(path)
    best = summary["best_committee"]
    stability = summary.get("best_committee_stability") or {}
    return {
        "output_name": output_name,
        "label": label,
        "committee_kind": committee_kind,
        "expert_model": best.get("expert_model"),
        "market_model": best.get("market_model"),
        "method": best.get("method"),
        "weight_expert": best.get("weight_expert"),
        "code_count": best.get("code_count"),
        "committee_val_rel_score": best.get("committee_val_rel_score"),
        "committee_test_rel_score": best.get("committee_test_rel_score"),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
        "summary_path": str(path),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    train_rows: list[dict[str, object]] = []
    committee_rows: list[dict[str, object]] = []

    for row in rows:
        kind = row["kind"]
        if kind == "train":
            train_rows.append(summarize_train(row["run_name"], row.get("label", "")))
        elif kind == "committee":
            committee_rows.append(
                summarize_committee(
                    output_name=row["output_name"],
                    label=row.get("label", ""),
                    committee_kind=row.get("committee_kind", ""),
                )
            )

    train_summary_path = args.manifest.parent / "fnb_runs_summary.csv"
    committee_summary_path = args.manifest.parent / "fnb_committee_summary.csv"

    if train_rows:
        pd.DataFrame(train_rows).sort_values(
            ["best_val_rel_score", "best_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        ).to_csv(train_summary_path, index=False)
    if committee_rows:
        pd.DataFrame(committee_rows).sort_values(
            ["committee_val_rel_score", "committee_test_rel_score"],
            ascending=[False, False],
            kind="stable",
        ).to_csv(committee_summary_path, index=False)

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "train_summary_path": str(train_summary_path) if train_rows else None,
                "committee_summary_path": str(committee_summary_path) if committee_rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
