from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_BASE = ROOT / "data/processed/assets/data_info_vn/history/training_runs"

HEAVY_DIR_NAMES = {"plots", "metric_series", "diagnostics", "holdout_private"}
HEAVY_FILE_PATTERNS = (
    "history*.csv",
    "model*.keras",
    "metric_details.json",
    "predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove bulky regenerable artifacts from training runs.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--keep-run",
        action="append",
        default=[],
        help="Run directory name to fully preserve. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="append",
        default=["reports"],
        help="Top-level run dir prefix to preserve.",
    )
    return parser.parse_args()


def should_preserve(path: Path, keep_runs: set[str], keep_prefixes: tuple[str, ...]) -> bool:
    rel = path.relative_to(RUN_BASE)
    top = rel.parts[0]
    if top in keep_runs:
        return True
    return any(top.startswith(prefix) for prefix in keep_prefixes)


def collect_targets(keep_runs: set[str], keep_prefixes: tuple[str, ...]) -> list[Path]:
    targets: list[Path] = []
    for path in RUN_BASE.rglob("*"):
        if should_preserve(path, keep_runs, keep_prefixes):
            continue
        if path.is_dir() and path.name in HEAVY_DIR_NAMES:
            targets.append(path)
            continue
        if not path.is_file():
            continue
        if any(path.match(f"**/{pattern}") for pattern in HEAVY_FILE_PATTERNS):
            targets.append(path)
    return sorted(set(targets))


def compute_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    if path.exists():
        path.rmdir()


def main() -> None:
    args = parse_args()
    keep_runs = set(args.keep_run)
    keep_prefixes = tuple(args.keep_prefix)
    targets = collect_targets(keep_runs, keep_prefixes)

    rows: list[dict[str, object]] = []
    total_bytes = 0
    for path in targets:
        size_bytes = compute_size_bytes(path)
        total_bytes += size_bytes
        rows.append(
            {
                "path": str(path),
                "size_bytes": size_bytes,
                "is_dir": path.is_dir(),
            }
        )
        if args.apply:
            remove_path(path)

    report = {
        "apply": bool(args.apply),
        "target_count": len(rows),
        "total_bytes": total_bytes,
        "kept_runs": sorted(keep_runs),
        "kept_prefixes": list(keep_prefixes),
        "rows": rows[:200],
    }
    out = RUN_BASE / "reports" / "info" / ("cleanup_applied.json" if args.apply else "cleanup_dry_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out)
    print(f"targets={len(rows)} total_bytes={total_bytes}")


if __name__ == "__main__":
    main()
