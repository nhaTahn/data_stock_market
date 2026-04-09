from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect overnight summaries for shared-context committee experiments."
    )
    parser.add_argument("--run-base", type=Path, default=RUN_BASE)
    parser.add_argument("--run-names", nargs="+", required=True)
    parser.add_argument("--min-code-count", type=int, default=3)
    parser.add_argument("--all-csv", type=Path, required=True)
    parser.add_argument("--stable-csv", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def best_context_row(metrics: dict[str, object]) -> dict[str, object]:
    candidate_names = [name for name in metrics if name.startswith("lstm")]
    ranked = sorted(
        candidate_names,
        key=lambda name: (
            float(metrics.get(name, {}).get("val", {}).get("rel_score", float("-inf"))),
            float(metrics.get(name, {}).get("test", {}).get("rel_score", float("-inf"))),
        ),
        reverse=True,
    )
    best_name = ranked[0] if ranked else None
    best_metrics = metrics.get(best_name, {}) if best_name else {}
    return {
        "context_best_model": best_name,
        "context_best_val_rel_score": best_metrics.get("val", {}).get("rel_score"),
        "context_best_test_rel_score": best_metrics.get("test", {}).get("rel_score"),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for run_name in args.run_names:
        run_dir = args.run_base / run_name
        config_path = run_dir / "reports" / "core" / "config.json"
        metrics_path = run_dir / "reports" / "core" / "metrics.json"
        suite_path = run_dir / "reports" / "core" / "committee_suite_summary.csv"

        if not config_path.exists() or not metrics_path.exists() or not suite_path.exists():
            continue

        config = load_json(config_path)
        metrics = load_json(metrics_path)
        context_summary = best_context_row(metrics)
        suite_df = pd.read_csv(suite_path)
        if suite_df.empty:
            continue

        for row in suite_df.to_dict(orient="records"):
            merged = {
                "context_run": run_name,
                "stocks": config.get("stocks"),
                "stock_count_config": len(str(config.get("stocks", "")).split(",")) if config.get("stocks") else None,
                "window_size": config.get("window_size"),
                "lstm_units": json.dumps(config.get("lstm_units")),
                "dropout": config.get("dropout"),
                "lr": config.get("lr"),
                "target_normalizer": config.get("target_normalizer"),
                "lstm_use_stock_identity": config.get("lstm_use_stock_identity"),
                **context_summary,
                **row,
            }
            merged["committee_gain_vs_expert_test"] = (
                float(merged["committee_test_rel_score"]) - float(merged["expert_test_rel_score_overlap"])
            )
            merged["committee_gain_vs_market_test"] = (
                float(merged["committee_test_rel_score"]) - float(merged["market_test_rel_score_overlap"])
            )
            merged["stable_committee"] = int(float(merged.get("code_count", 0)) >= args.min_code_count)
            rows.append(merged)

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise ValueError("No committee suite rows were found.")

    sort_cols = [
        "stable_committee",
        "committee_val_rel_score",
        "committee_test_rel_score",
        "code_count",
        "committee_gain_vs_expert_test",
    ]
    all_df = all_df.sort_values(sort_cols, ascending=[False, False, False, False, False], kind="stable")
    args.all_csv.parent.mkdir(parents=True, exist_ok=True)
    args.stable_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.all_csv, index=False)
    stable_df = all_df[all_df["stable_committee"] == 1].copy()
    stable_df.to_csv(args.stable_csv, index=False)

    print(all_df.to_string(index=False))
    print(f"Saved: {args.all_csv}")
    print(f"Saved: {args.stable_csv}")


if __name__ == "__main__":
    main()
