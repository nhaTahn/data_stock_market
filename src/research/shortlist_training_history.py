from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.reporting import resolve_run_artifact


DEFAULT_RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
IGNORE_DIRS = {
    "active",
    "feature_correlation",
    "overnight_logs",
    "reports",
    "representative_runs",
    "search_runs",
    "sector_logs",
}


@dataclass
class RunSummary:
    run_name: str
    run_dir: Path
    size_mb: float
    target_mode: str | None
    stocks_count: int
    best_model: str | None
    best_test_rel_score: float | None
    best_val_rel_score: float | None
    val_test_gap: float | None
    committee_test_rel_score: float | None
    committee_val_rel_score: float | None
    stable_test_rel_score_median: float | None
    retention_status: str = "review"
    retention_reason: str = ""
    duplicate_of: str = ""
    fingerprint: str = ""

    @property
    def effective_score(self) -> float:
        committee_score = self.committee_test_rel_score or float("-inf")
        standalone_score = self.best_test_rel_score or float("-inf")
        return max(committee_score, standalone_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shortlist training history runs into keep/review/delete buckets.",
    )
    parser.add_argument("--run-base", type=Path, default=DEFAULT_RUN_BASE)
    parser.add_argument(
        "--keep-csv",
        type=Path,
        default=DEFAULT_RUN_BASE / "reports" / "training_history_retention_shortlist.csv",
    )
    parser.add_argument(
        "--keep-md",
        type=Path,
        default=DEFAULT_RUN_BASE / "reports" / "training_history_retention_shortlist.md",
    )
    parser.add_argument(
        "--committee-keep-threshold",
        type=float,
        default=0.04,
        help="Keep committee runs at or above this test rel_score.",
    )
    parser.add_argument(
        "--standalone-keep-threshold",
        type=float,
        default=0.03,
        help="Keep standalone runs at or above this test rel_score when validation gap is acceptable.",
    )
    parser.add_argument(
        "--stable-gap-threshold",
        type=float,
        default=0.02,
        help="Maximum |val-test| gap for automatic keep on standalone runs.",
    )
    parser.add_argument(
        "--heavy-run-threshold-mb",
        type=float,
        default=50.0,
        help="Runs above this size with weak score are recommended for deletion.",
    )
    parser.add_argument(
        "--weak-score-threshold",
        type=float,
        default=0.02,
        help="Heavy runs below this standalone score are recommended for deletion.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_run_dirs(run_base: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_base.iterdir())
        if path.is_dir()
        and path.name not in IGNORE_DIRS
        and resolve_run_artifact(path, "metrics.json", "core").exists()
    ]


def compute_size_mb(run_dir: Path) -> float:
    total_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    return round(total_bytes / 1024 / 1024, 2)


def load_best_model(metrics: dict[str, Any]) -> tuple[str | None, float | None, float | None, float | None]:
    best_model: str | None = None
    best_test: float | None = None
    best_val: float | None = None

    for model_name, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        test_rel = payload.get("test", {}).get("rel_score")
        val_rel = payload.get("val", {}).get("rel_score")
        if test_rel is None or val_rel is None:
            continue
        test_value = float(test_rel)
        val_value = float(val_rel)
        if best_model is None or (test_value, val_value) > (best_test or float("-inf"), best_val or float("-inf")):
            best_model = model_name
            best_test = test_value
            best_val = val_value

    if best_model is None or best_test is None or best_val is None:
        return None, None, None, None
    return best_model, best_test, best_val, abs(best_val - best_test)


def load_committee_scores(run_dir: Path) -> tuple[float | None, float | None, float | None]:
    rotation_path = resolve_run_artifact(run_dir, "committee_rotation_active.csv", "core")
    if not rotation_path.exists():
        return None, None, None

    rows = list(csv.DictReader(rotation_path.open("r", encoding="utf-8")))
    if not rows:
        return None, None, None

    rows.sort(
        key=lambda row: (
            float(row.get("committee_test_rel_score") or float("-inf")),
            float(row.get("committee_val_rel_score") or float("-inf")),
        ),
        reverse=True,
    )
    best_row = rows[0]
    stable_median_raw = best_row.get("stable_test_rel_score_median")
    return (
        float(best_row["committee_test_rel_score"]),
        float(best_row["committee_val_rel_score"]),
        None if stable_median_raw in {"", None} else float(stable_median_raw),
    )


def build_fingerprint(config: dict[str, Any], metrics: dict[str, Any], committee_scores: tuple[float | None, float | None, float | None]) -> str:
    fingerprint_payload = {
        "target_mode": config.get("target_mode"),
        "stocks": config.get("stocks"),
        "window_size": config.get("window_size"),
        "lstm_units": config.get("lstm_units"),
        "dropout": config.get("dropout"),
        "lr": config.get("lr"),
        "loss": config.get("loss"),
        "feature_columns": config.get("feature_columns"),
        "family_selection_summary": config.get("family_selection_summary"),
        "metrics": metrics,
        "committee_scores": committee_scores,
    }
    encoded = json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def summarize_run(run_dir: Path) -> RunSummary:
    config = load_json(resolve_run_artifact(run_dir, "config.json", "core"))
    metrics = load_json(resolve_run_artifact(run_dir, "metrics.json", "core"))
    best_model, best_test, best_val, gap = load_best_model(metrics)
    committee_scores = load_committee_scores(run_dir)
    fingerprint = build_fingerprint(config, metrics, committee_scores)
    stocks = config.get("stocks")
    if isinstance(stocks, list):
        stocks_count = len(stocks)
    elif isinstance(stocks, str) and stocks:
        stocks_count = len([item for item in stocks.split(",") if item])
    else:
        stocks_count = 0

    return RunSummary(
        run_name=run_dir.name,
        run_dir=run_dir,
        size_mb=compute_size_mb(run_dir),
        target_mode=config.get("target_mode"),
        stocks_count=stocks_count,
        best_model=best_model,
        best_test_rel_score=best_test,
        best_val_rel_score=best_val,
        val_test_gap=gap,
        committee_test_rel_score=committee_scores[0],
        committee_val_rel_score=committee_scores[1],
        stable_test_rel_score_median=committee_scores[2],
        fingerprint=fingerprint,
    )


def pick_duplicate_canonical(candidates: list[RunSummary]) -> RunSummary:
    return sorted(
        candidates,
        key=lambda item: (
            item.committee_test_rel_score is None,
            -(item.committee_test_rel_score or float("-inf")),
            -(item.best_test_rel_score or float("-inf")),
            item.size_mb,
            item.run_name,
        ),
    )[0]


def classify_runs(
    runs: list[RunSummary],
    committee_keep_threshold: float,
    standalone_keep_threshold: float,
    stable_gap_threshold: float,
    heavy_run_threshold_mb: float,
    weak_score_threshold: float,
) -> None:
    grouped: dict[str, list[RunSummary]] = {}
    for run in runs:
        grouped.setdefault(run.fingerprint, []).append(run)

    canonical_by_group = {
        fingerprint: pick_duplicate_canonical(items)
        for fingerprint, items in grouped.items()
    }

    for run in runs:
        canonical = canonical_by_group[run.fingerprint]
        if run is not canonical:
            run.retention_status = "delete"
            run.duplicate_of = canonical.run_name
            run.retention_reason = "duplicate config+metrics; keep canonical run only"
            continue

        if (run.committee_test_rel_score or float("-inf")) >= committee_keep_threshold:
            run.retention_status = "keep"
            run.retention_reason = "top committee result"
            continue

        if (
            (run.best_test_rel_score or float("-inf")) >= standalone_keep_threshold
            and (run.val_test_gap or float("inf")) <= stable_gap_threshold
        ):
            run.retention_status = "keep"
            run.retention_reason = "strong standalone result with acceptable val/test gap"
            continue

        if (run.best_test_rel_score or float("-inf")) >= standalone_keep_threshold:
            run.retention_status = "review"
            run.retention_reason = "high test score but unstable val/test gap"
            continue

        if run.size_mb >= heavy_run_threshold_mb and (run.best_test_rel_score or float("-inf")) < weak_score_threshold:
            run.retention_status = "delete"
            run.retention_reason = "heavy run with weak standalone score"
            continue

        run.retention_status = "review"
        run.retention_reason = "not clearly strong enough to keep and not large enough to force-delete"


def sort_runs(runs: list[RunSummary]) -> list[RunSummary]:
    status_rank = {"keep": 0, "review": 1, "delete": 2}
    return sorted(
        runs,
        key=lambda item: (
            status_rank[item.retention_status],
            -(item.committee_test_rel_score or float("-inf")),
            -(item.stable_test_rel_score_median or float("-inf")),
            -(item.best_test_rel_score or float("-inf")),
            item.val_test_gap or float("inf"),
            item.run_name,
        ),
    )


def write_csv(path: Path, runs: list[RunSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "retention_status",
                "retention_reason",
                "duplicate_of",
                "run_name",
                "size_mb",
                "target_mode",
                "stocks_count",
                "best_model",
                "best_test_rel_score",
                "best_val_rel_score",
                "val_test_gap",
                "committee_test_rel_score",
                "committee_val_rel_score",
                "stable_test_rel_score_median",
                "run_dir",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "retention_status": run.retention_status,
                    "retention_reason": run.retention_reason,
                    "duplicate_of": run.duplicate_of,
                    "run_name": run.run_name,
                    "size_mb": run.size_mb,
                    "target_mode": run.target_mode,
                    "stocks_count": run.stocks_count,
                    "best_model": run.best_model,
                    "best_test_rel_score": run.best_test_rel_score,
                    "best_val_rel_score": run.best_val_rel_score,
                    "val_test_gap": run.val_test_gap,
                    "committee_test_rel_score": run.committee_test_rel_score,
                    "committee_val_rel_score": run.committee_val_rel_score,
                    "stable_test_rel_score_median": run.stable_test_rel_score_median,
                    "run_dir": str(run.run_dir),
                }
            )


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_markdown(path: Path, runs: list[RunSummary]) -> None:
    keep_runs = [run for run in runs if run.retention_status == "keep"]
    review_runs = [run for run in runs if run.retention_status == "review"]
    delete_runs = [run for run in runs if run.retention_status == "delete"]
    reclaimable_mb = sum(run.size_mb for run in delete_runs)

    lines = [
        "# Training History Retention Shortlist",
        "",
        f"- Keep: `{len(keep_runs)}`",
        f"- Review: `{len(review_runs)}`",
        f"- Delete-recommended: `{len(delete_runs)}`",
        f"- Potential reclaim if delete-recommended is removed: `{reclaimable_mb:.2f} MB`",
        "",
        "## Keep",
        "",
    ]

    if not keep_runs:
        lines.append("- None")
    else:
        for run in keep_runs:
            lines.append(
                "- "
                f"`{run.run_name}` | reason: {run.retention_reason} | "
                f"best_test={format_float(run.best_test_rel_score)} | "
                f"best_val={format_float(run.best_val_rel_score)} | "
                f"committee_test={format_float(run.committee_test_rel_score)} | "
                f"stable_median={format_float(run.stable_test_rel_score_median)} | "
                f"size={run.size_mb:.2f}MB"
            )

    lines.extend(["", "## Review", ""])
    if not review_runs:
        lines.append("- None")
    else:
        for run in review_runs:
            lines.append(
                "- "
                f"`{run.run_name}` | reason: {run.retention_reason} | "
                f"best_test={format_float(run.best_test_rel_score)} | "
                f"best_val={format_float(run.best_val_rel_score)} | "
                f"gap={format_float(run.val_test_gap)} | "
                f"size={run.size_mb:.2f}MB"
            )

    lines.extend(["", "## Delete-Recommended", ""])
    if not delete_runs:
        lines.append("- None")
    else:
        for run in delete_runs:
            duplicate_suffix = f" | duplicate_of={run.duplicate_of}" if run.duplicate_of else ""
            lines.append(
                "- "
                f"`{run.run_name}` | reason: {run.retention_reason}{duplicate_suffix} | "
                f"best_test={format_float(run.best_test_rel_score)} | "
                f"size={run.size_mb:.2f}MB"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = [summarize_run(run_dir) for run_dir in iter_run_dirs(args.run_base)]
    classify_runs(
        runs=runs,
        committee_keep_threshold=args.committee_keep_threshold,
        standalone_keep_threshold=args.standalone_keep_threshold,
        stable_gap_threshold=args.stable_gap_threshold,
        heavy_run_threshold_mb=args.heavy_run_threshold_mb,
        weak_score_threshold=args.weak_score_threshold,
    )
    sorted_runs = sort_runs(runs)
    write_csv(args.keep_csv, sorted_runs)
    write_markdown(args.keep_md, sorted_runs)
    print(f"Saved CSV: {args.keep_csv}")
    print(f"Saved MD: {args.keep_md}")


if __name__ == "__main__":
    main()
