from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize focused F&B signmag bias push runs.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_split_stats(df: pd.DataFrame, model_name: str, split_name: str) -> dict[str, float]:
    part = df[(df["model"] == model_name) & (df["split"] == split_name)].copy()
    if part.empty:
        return {}
    pred_abs = float(part["prediction"].abs().mean())
    actual_abs = float(part["actual"].abs().mean())
    pred_pos = float((part["prediction"] > 0).mean())
    actual_pos = float((part["actual"] > 0).mean())
    return {
        f"{split_name}_pred_pos_rate": pred_pos,
        f"{split_name}_actual_pos_rate": actual_pos,
        f"{split_name}_pos_gap": pred_pos - actual_pos,
        f"{split_name}_pred_abs_over_actual_abs": pred_abs / actual_abs if actual_abs > 0 else float("nan"),
        f"{split_name}_pred_mean": float(part["prediction"].mean()),
        f"{split_name}_actual_mean": float(part["actual"].mean()),
    }


def build_selection_score(val_rel: float, val_amp_ratio: float, val_pos_gap: float, train_val_gap: float) -> float:
    amp_penalty = max(0.0, 0.2 - val_amp_ratio) * 0.4
    pos_penalty = abs(val_pos_gap) * 0.25
    gap_penalty = abs(train_val_gap) * 0.2
    return float(val_rel - amp_penalty - pos_penalty - gap_penalty)


def summarize_train(run_name: str, label: str) -> dict[str, object]:
    core_dir = RUN_BASE / run_name / "reports" / "core"
    metrics = load_json(core_dir / "metrics.json")
    config = load_json(core_dir / "config.json")
    predictions = pd.read_csv(core_dir / "predictions.csv")
    family = load_json(core_dir / "family_selection_summary.json")
    signmag_summary = family.get("lstm_signmag", {})
    best_model = str(signmag_summary.get("best_by_val", "lstm_signmag_best_by_val"))
    metric_payload = metrics.get("lstm_signmag_best_by_val") or metrics.get(best_model) or {}
    train_rel = float(metric_payload.get("train", {}).get("rel_score", np.nan))
    val_rel = float(metric_payload.get("val", {}).get("rel_score", np.nan))
    test_rel = float(metric_payload.get("test", {}).get("rel_score", np.nan))
    row: dict[str, object] = {
        "run_name": run_name,
        "label": label,
        "best_signmag_by_val": best_model,
        "train_rel_score": train_rel,
        "val_rel_score": val_rel,
        "test_rel_score": test_rel,
        "train_minus_val": train_rel - val_rel,
        "val_minus_test": val_rel - test_rel,
        "window_size": config.get("window_size"),
        "lstm_units": config.get("lstm_units"),
        "lr": config.get("lr"),
        "dropout": config.get("dropout"),
        "feature_count": len(config.get("feature_columns", [])),
        "signmag_sign_loss_weight": config.get("signmag_sign_loss_weight"),
        "signmag_magnitude_loss_weight": config.get("signmag_magnitude_loss_weight"),
        "signmag_signed_loss_weight": config.get("signmag_signed_loss_weight"),
        "signmag_log_magnitude": config.get("signmag_log_magnitude"),
        "sample_weight_mode": config.get("sample_weight_mode"),
    }
    for split_name in ("train", "val", "test"):
        row.update(compute_split_stats(predictions, "lstm_signmag_best_by_val", split_name))
    row["selection_score"] = build_selection_score(
        float(row.get("val_rel_score", np.nan)),
        float(row.get("val_pred_abs_over_actual_abs", np.nan)),
        float(row.get("val_pos_gap", np.nan)),
        float(row.get("train_minus_val", np.nan)),
    )
    return row


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))
    train_rows: list[dict[str, object]] = []
    for row in rows:
        if row.get("kind") != "train":
            continue
        train_rows.append(summarize_train(row["run_name"], row.get("label", "")))

    summary_path = args.manifest.parent / "signmag_bias_push_summary.csv"
    detail_path = args.manifest.parent / "signmag_bias_push_summary_by_val.csv"
    if train_rows:
        df = pd.DataFrame(train_rows)
        df.sort_values(
            ["selection_score", "val_rel_score", "test_rel_score"],
            ascending=[False, False, False],
            kind="stable",
        ).to_csv(summary_path, index=False)
        df.sort_values(
            ["val_rel_score", "test_rel_score"],
            ascending=[False, False],
            kind="stable",
        ).to_csv(detail_path, index=False)

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "summary_path": str(summary_path) if train_rows else None,
                "by_val_path": str(detail_path) if train_rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
