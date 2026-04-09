from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS = ROOT / "data/processed/assets/data_info_vn/history/training_runs"
OUTPUT_DIR = TRAINING_RUNS / "reports/research_restarts/fnb_restart_20260409"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_selection_stability(summary: dict, family_prefix: str) -> float:
    hits = 0
    for item in summary.get("selection_instability_signals", []):
        if item.startswith(family_prefix):
            hits += 1
    return float(hits)


def load_standalone_candidate(run_name: str, model_name: str, label: str, candidate_type: str) -> dict:
    core_dir = TRAINING_RUNS / run_name / "reports/core"
    metrics = load_json(core_dir / "metrics.json")
    underfit = load_json(core_dir / "underfit_selection_summary.json")
    bias = pd.read_csv(core_dir / "underfit_selection_bias.csv")
    gaps = pd.read_csv(core_dir / "underfit_selection_gaps.csv")

    bias_row = bias[(bias["model"] == model_name) & (bias["split"] == "test")].iloc[0]
    gap_row = gaps[gaps["model"] == model_name].iloc[0]
    metric_block = metrics[model_name]["test"]
    val_block = metrics[model_name]["val"]

    family_prefix = "lstm_signmag" if "signmag" in model_name else "lstm"
    return {
        "candidate_label": label,
        "candidate_type": candidate_type,
        "run_name": run_name,
        "model_name": model_name,
        "selection_mode": "best_by_val",
        "code_count": 4,
        "val_rel_score": float(val_block["rel_score"]),
        "test_rel_score": float(metric_block["rel_score"]),
        "pred_pos_rate": float(bias_row["pred_pos_rate"]),
        "actual_pos_rate": float(bias_row["actual_pos_rate"]),
        "pos_rate_gap": float(bias_row["pred_pos_rate"] - bias_row["actual_pos_rate"]),
        "pred_abs_over_actual_abs": float(bias_row["pred_abs_over_actual_abs"]),
        "test_corr": float(bias_row["corr"]),
        "train_minus_val": float(gap_row["train_minus_val"]),
        "val_minus_test": float(gap_row["val_minus_test"]),
        "train_minus_test": float(gap_row["train_minus_test"]),
        "selection_instability_count": score_selection_stability(underfit, family_prefix),
    }


def compute_committee_bias(predictions_path: Path) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    pred_col = "prediction_committee" if "prediction_committee" in df.columns else "prediction"
    pred = df[pred_col]
    actual = df["actual"]
    actual_abs_mean = float(actual.abs().mean())
    pred_abs_mean = float(pred.abs().mean())
    return {
        "pred_pos_rate": float((pred > 0.0).mean()),
        "actual_pos_rate": float((actual > 0.0).mean()),
        "pos_rate_gap": float((pred > 0.0).mean() - (actual > 0.0).mean()),
        "pred_abs_over_actual_abs": float(pred_abs_mean / actual_abs_mean) if actual_abs_mean else 0.0,
        "test_corr": float(pred.corr(actual)),
    }


def load_committee_candidate(summary_path: Path, label: str) -> dict:
    summary = load_json(summary_path)
    best = summary["best_committee"]
    stability = summary.get("best_committee_stability") or {}
    bias = compute_committee_bias(Path(summary["best_predictions_path"]))
    return {
        "candidate_label": label,
        "candidate_type": "committee",
        "run_name": f'{best["expert_run"]} + {best["market_run"]}',
        "model_name": f'{best["expert_model"]} + {best["market_model"]}',
        "selection_mode": summary["selection_mode"],
        "code_count": int(best["code_count"]),
        "val_rel_score": float(best["committee_val_rel_score"]),
        "test_rel_score": float(best["committee_test_rel_score"]),
        "pred_pos_rate": bias["pred_pos_rate"],
        "actual_pos_rate": bias["actual_pos_rate"],
        "pos_rate_gap": bias["pos_rate_gap"],
        "pred_abs_over_actual_abs": bias["pred_abs_over_actual_abs"],
        "test_corr": bias["test_corr"],
        "train_minus_val": None,
        "val_minus_test": float(best["committee_val_rel_score"] - best["committee_test_rel_score"]),
        "train_minus_test": None,
        "selection_instability_count": 0.0,
        "stable_weight_min": stability.get("stable_weight_min"),
        "stable_weight_max": stability.get("stable_weight_max"),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
    }


def classify_candidate(row: pd.Series) -> str:
    if (
        row["candidate_type"] == "committee"
        and row["test_rel_score"] >= 0.05
        and row["code_count"] >= 4
    ):
        return "keep_main"
    if row["test_rel_score"] >= 0.025 and row["pred_abs_over_actual_abs"] >= 0.06:
        return "keep_aux"
    return "archive"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        load_standalone_candidate(
            "overnight_fnb_w5_mag_base_20260409_101741",
            "lstm_best_by_val",
            "plain_fnb_baseline",
            "standalone",
        ),
        load_standalone_candidate(
            "overnight_fnb_w5_mag_base_20260409_101741",
            "lstm_signmag_best_by_val",
            "plainrun_signmag_best",
            "standalone",
        ),
        load_standalone_candidate(
            "biaspush_signmag_sector_base_20260409_111710",
            "lstm_signmag_best_by_val",
            "signmag_sector_base",
            "standalone",
        ),
        load_standalone_candidate(
            "biaspush_signmag_sector_rawmag_20260409_111710",
            "lstm_signmag_best_by_val",
            "signmag_sector_rawmag",
            "standalone",
        ),
        load_committee_candidate(
            TRAINING_RUNS / "reports/committee_experiments/confirm_vn100_fnb_committee_20260408_235445_r01__committee__fnb/best_committee_summary.json",
            "shared_context_committee_old",
        ),
        load_committee_candidate(
            TRAINING_RUNS / "reports/committee_experiments/biaspush_sectorbase__committee__plain_expert/best_committee_summary.json",
            "sectorbase_committee_new",
        ),
        load_committee_candidate(
            TRAINING_RUNS / "reports/committee_experiments/biaspush_sectorraw__committee__plain_expert/best_committee_summary.json",
            "sectorraw_committee_alt",
        ),
    ]

    df = pd.DataFrame(rows)
    df["recommendation"] = df.apply(classify_candidate, axis=1)
    df = df.sort_values(["test_rel_score", "val_rel_score"], ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "candidate_matrix.csv", index=False)

    summary = {
        "best_candidate": df.iloc[0]["candidate_label"],
        "best_test_rel_score": float(df.iloc[0]["test_rel_score"]),
        "main_candidates": df[df["recommendation"] == "keep_main"]["candidate_label"].tolist(),
        "aux_candidates": df[df["recommendation"] == "keep_aux"]["candidate_label"].tolist(),
        "archive_candidates": df[df["recommendation"] == "archive"]["candidate_label"].tolist(),
    }
    with (OUTPUT_DIR / "candidate_matrix_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# F&B Candidate Matrix",
        "",
        "## Short conclusion",
        f'- Best candidate hiện tại: `{summary["best_candidate"]}`',
        f'- Best test rel_score: `{summary["best_test_rel_score"]:.6f}`',
        "",
        "## Main candidates",
    ]
    for name in summary["main_candidates"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Auxiliary candidates"])
    for name in summary["aux_candidates"]:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Archive candidates"])
    for name in summary["archive_candidates"]:
        lines.append(f"- `{name}`")
    lines.append("")
    (OUTPUT_DIR / "candidate_matrix_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
