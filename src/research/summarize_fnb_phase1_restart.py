from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data/processed/assets/data_info_vn/history/training_runs"
COMMITTEE_BASE = RUN_BASE / "reports/committee_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize F&B phase 1 restart runs.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_train_candidate(run_name: str, label: str, model_name: str) -> dict[str, object] | None:
    core = RUN_BASE / run_name / "reports/core"
    metrics_path = core / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    if model_name not in metrics:
        return None

    bias = pd.read_csv(core / "underfit_selection_bias.csv")
    gaps = pd.read_csv(core / "underfit_selection_gaps.csv")
    backtest = load_json(RUN_BASE / run_name / "reports/backtests/threshold_backtest_summary_phase1.json")

    bias_row = bias[(bias["model"] == model_name) & (bias["split"] == "test")]
    gap_row = gaps[gaps["model"] == model_name]
    if bias_row.empty or gap_row.empty:
        return None

    bias_row = bias_row.iloc[0]
    gap_row = gap_row.iloc[0]
    bt = backtest.get(model_name, {})
    val_rel = float(metrics[model_name]["val"]["rel_score"])
    test_rel = float(metrics[model_name]["test"]["rel_score"])
    selection_score = (
        test_rel
        + 0.35 * val_rel
        + 0.06 * float(bias_row["pred_abs_over_actual_abs"])
        - 0.08 * abs(float(bias_row["pred_pos_rate"]) - float(bias_row["actual_pos_rate"]))
        - 0.15 * abs(float(gap_row["val_minus_test"]))
    )
    return {
        "candidate_kind": "standalone",
        "source_label": label,
        "run_name": run_name,
        "model_name": model_name,
        "val_rel_score": val_rel,
        "test_rel_score": test_rel,
        "pred_pos_rate": float(bias_row["pred_pos_rate"]),
        "actual_pos_rate": float(bias_row["actual_pos_rate"]),
        "pos_rate_gap": float(bias_row["pred_pos_rate"] - bias_row["actual_pos_rate"]),
        "pred_abs_over_actual_abs": float(bias_row["pred_abs_over_actual_abs"]),
        "test_corr": float(bias_row["corr"]),
        "train_minus_val": float(gap_row["train_minus_val"]),
        "val_minus_test": float(gap_row["val_minus_test"]),
        "trade_count": bt.get("trade_count"),
        "directional_accuracy": bt.get("directional_accuracy"),
        "final_equity": bt.get("final_equity"),
        "selection_score": float(selection_score),
    }


def summarize_committee_candidate(output_name: str, label: str) -> dict[str, object]:
    summary = load_json(COMMITTEE_BASE / output_name / "best_committee_summary.json")
    best = summary["best_committee"]
    stability = summary.get("best_committee_stability") or {}
    selection_score = (
        float(best["committee_test_rel_score"])
        + 0.35 * float(best["committee_val_rel_score"])
        + 0.01 * float(stability.get("stable_weight_count") or 0)
        - 0.05 * abs(float(best["committee_val_rel_score"]) - float(best["committee_test_rel_score"]))
    )
    return {
        "candidate_kind": "committee",
        "source_label": label,
        "run_name": f'{best["expert_run"]} + {best["market_run"]}',
        "model_name": f'{best["expert_model"]} + {best["market_model"]}',
        "val_rel_score": float(best["committee_val_rel_score"]),
        "test_rel_score": float(best["committee_test_rel_score"]),
        "pred_pos_rate": np.nan,
        "actual_pos_rate": np.nan,
        "pos_rate_gap": np.nan,
        "pred_abs_over_actual_abs": np.nan,
        "test_corr": np.nan,
        "train_minus_val": np.nan,
        "val_minus_test": float(best["committee_val_rel_score"] - best["committee_test_rel_score"]),
        "trade_count": np.nan,
        "directional_accuracy": np.nan,
        "final_equity": np.nan,
        "selection_score": float(selection_score),
        "method": best["method"],
        "weight_expert": float(best["weight_expert"]),
        "code_count": int(best["code_count"]),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))

    standalone_rows: list[dict[str, object]] = []
    committee_rows: list[dict[str, object]] = []

    for row in rows:
        kind = row.get("kind")
        run_name = row.get("run_name", "")
        label = row.get("label", "")
        if kind == "train":
            for model_name in ("lstm_best_by_val", "lstm_top2_by_val", "lstm_ensemble"):
                payload = summarize_train_candidate(run_name, label, model_name)
                if payload is not None:
                    standalone_rows.append(payload)
        elif kind == "committee":
            committee_rows.append(summarize_committee_candidate(run_name, label))

    out_dir = args.manifest.parent
    if standalone_rows:
        pd.DataFrame(standalone_rows).sort_values(
            ["selection_score", "test_rel_score", "val_rel_score"],
            ascending=[False, False, False],
            kind="stable",
        ).to_csv(out_dir / "phase1_train_summary.csv", index=False)
    if committee_rows:
        pd.DataFrame(committee_rows).sort_values(
            ["selection_score", "test_rel_score", "val_rel_score"],
            ascending=[False, False, False],
            kind="stable",
        ).to_csv(out_dir / "phase1_committee_summary.csv", index=False)

    ranking_df = pd.DataFrame([*standalone_rows, *committee_rows])
    if not ranking_df.empty:
        ranking_df.sort_values(
            ["selection_score", "test_rel_score", "val_rel_score"],
            ascending=[False, False, False],
            kind="stable",
        ).to_csv(out_dir / "phase1_candidate_ranking.csv", index=False)

    summary = {
        "manifest": str(args.manifest),
        "train_rows": len(standalone_rows),
        "committee_rows": len(committee_rows),
        "best_candidate": None if ranking_df.empty else ranking_df.sort_values(
            ["selection_score", "test_rel_score", "val_rel_score"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0].to_dict(),
    }
    (out_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
