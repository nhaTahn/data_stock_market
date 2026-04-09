from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize whole-market watchlist overnight runs.")
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


def summarize_run(run_name: str, label: str, kind: str, sector: str | None, window: str | None) -> dict[str, object]:
    core_dir = RUN_BASE / run_name / "reports" / "core"
    metrics = load_json(core_dir / "metrics.json")
    config = load_json(core_dir / "config.json")
    best_model_name, best_payload = find_best_model(metrics)
    return {
        "run_name": run_name,
        "label": label,
        "kind": kind,
        "sector": sector,
        "window": window,
        "best_model_by_val": best_model_name,
        "best_val_rel_score": float(best_payload["val"]["rel_score"]),
        "best_test_rel_score": float(best_payload["test"]["rel_score"]),
        "best_test_directional_accuracy": float(best_payload["test"]["directional_accuracy"]),
        "stocks": config.get("stocks"),
    }


def summarize_committee(
    context_run: str,
    expert_run: str,
    output_name: str,
    label: str,
    preset: str,
    sector: str | None,
    window: str | None,
) -> dict[str, object]:
    path = RUN_BASE / "reports" / "committee_experiments" / output_name / "best_committee_summary.json"
    summary = load_json(path)
    best = summary["best_committee"]
    stability = summary.get("best_committee_stability") or {}
    return {
        "label": label,
        "preset": preset,
        "sector": sector,
        "window": window,
        "context_run": context_run,
        "expert_run": expert_run,
        "committee_expert_model": best.get("expert_model"),
        "committee_market_model": best.get("market_model"),
        "committee_method": best.get("method"),
        "committee_weight_expert": best.get("weight_expert"),
        "committee_code_count": best.get("code_count"),
        "committee_val_rel_score": best.get("committee_val_rel_score"),
        "committee_test_rel_score": best.get("committee_test_rel_score"),
        "stable_weight_count": stability.get("stable_weight_count"),
        "stable_test_rel_score_median": stability.get("stable_test_rel_score_median"),
        "summary_path": str(path),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8")))

    run_rows: list[dict[str, object]] = []
    committee_rows: list[dict[str, object]] = []
    for row in rows:
        kind = row["kind"]
        if kind in {"shared_run", "sector_run"}:
            run_rows.append(
                summarize_run(
                    run_name=row["run_name"],
                    label=row.get("label", ""),
                    kind=kind,
                    sector=row.get("sector") or None,
                    window=row.get("window") or None,
                )
            )
        elif kind == "committee":
            committee_rows.append(
                summarize_committee(
                    context_run=row["context_run"],
                    expert_run=row["expert_run"],
                    output_name=row["output_name"],
                    label=row.get("label", ""),
                    preset=row.get("preset", ""),
                    sector=row.get("sector") or None,
                    window=row.get("window") or None,
                )
            )

    run_summary_path = args.manifest.parent / "whole_market_runs_summary.csv"
    committee_summary_path = args.manifest.parent / "whole_market_committee_summary.csv"

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

    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "run_summary_path": str(run_summary_path) if run_rows else None,
                "committee_summary_path": str(committee_summary_path) if committee_rows else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
