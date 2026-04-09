from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data/processed/assets/data_info_vn/history/training_runs"
PAIR_GRID = RUN_BASE / "phase2_plain_sector_pairscan_20260409_134132/reports/core/pair_scan_grid.csv"
OUTPUT_DIR = RUN_BASE / "reports/research_restarts/fnb_phase5_frontier_selection"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PAIR_GRID)

    val_max = df.sort_values(["val_rel_score", "test_rel_score"], ascending=[False, False], kind="stable").iloc[0].to_dict()

    balanced_pool = df[
        (df["val_rel_score"] >= 0.01)
        & (df["test_rel_score"] >= 0.02)
        & (df["val_pred_abs_over_actual_abs"] >= 0.09)
    ].copy()
    if balanced_pool.empty:
        balanced_pool = df.copy()
    balanced = balanced_pool.sort_values(
        ["val_rel_score", "test_rel_score", "test_pred_abs_over_actual_abs"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0].to_dict()

    frontier = df.sort_values(
        ["test_rel_score", "test_pred_abs_over_actual_abs", "val_rel_score"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0].to_dict()

    shortlist = pd.DataFrame(
        [
            {"selection_type": "val_max", **val_max},
            {"selection_type": "balanced", **balanced},
            {"selection_type": "frontier", **frontier},
        ]
    )
    shortlist.to_csv(OUTPUT_DIR / "pair_frontier_shortlist.csv", index=False)

    summary = {
        "source_pair_grid": str(PAIR_GRID),
        "val_max": val_max,
        "balanced": balanced,
        "frontier": frontier,
    }
    (OUTPUT_DIR / "pair_frontier_shortlist.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Pair Frontier Shortlist",
        "",
        "## Val-max",
        f"- `{val_max['pair_name']}`",
        f"- val `{val_max['val_rel_score']:.6f}`",
        f"- test `{val_max['test_rel_score']:.6f}`",
        "",
        "## Balanced",
        f"- `{balanced['pair_name']}`",
        f"- val `{balanced['val_rel_score']:.6f}`",
        f"- test `{balanced['test_rel_score']:.6f}`",
        "",
        "## Frontier",
        f"- `{frontier['pair_name']}`",
        f"- val `{frontier['val_rel_score']:.6f}`",
        f"- test `{frontier['test_rel_score']:.6f}`",
        "",
    ]
    (OUTPUT_DIR / "pair_frontier_shortlist.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
