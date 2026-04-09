from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_BASE = RUN_BASE / "reports" / "committee_confirmations"
DEFAULT_LOG_BASE = RUN_BASE / "shared_context_logs"
PYTHON_BIN = ROOT / "venv" / "bin" / "python"
SHARED_RUNNER = ROOT / "scripts" / "run_shared_vn30_committee.py"
WINNER_SUMMARY = (
    RUN_BASE
    / "reports"
    / "committee_experiments"
    / "overnight_shared_vn100_w20_u64_32_relscore_20260408_173407__committee__fnb"
    / "best_committee_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confirmation jobs for the current VN100 + F&B committee baseline."
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-name-prefix", default="confirm_vn100_fnb_committee")
    parser.add_argument("--log-base", type=Path, default=DEFAULT_LOG_BASE)
    parser.add_argument("--summary-dir", type=Path, default=REPORT_BASE)
    parser.add_argument("--winner-summary", type=Path, default=WINNER_SUMMARY)
    return parser.parse_args()


def run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(cmd)}\nLog: {log_path}")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_command(run_name: str) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(SHARED_RUNNER),
        "--context-run-dir",
        str(RUN_BASE / run_name),
        "--universe-path",
        "market_lists/vn100.txt",
        "--committee-preset",
        "fnb",
        "--committee-selection-mode",
        "stable_band",
        "--committee-stable-selection-val-gap",
        "0.006",
        "--committee-stable-selection-min-weight-count",
        "2",
        "--window-size",
        "20",
        "--lstm-units",
        "64,32",
        "--dropout",
        "0.05",
        "--lr",
        "0.0005",
        "--batch-size",
        "64",
        "--epochs",
        "30",
        "--patience",
        "8",
        "--target-normalizer",
        "volatility_20",
        "--lstm-seeds",
        "42,52,62",
        "--sample-weight-mode",
        "none",
        "--rotation-min-val-rel-score",
        "0.015",
        "--rotation-min-stable-weight-count",
        "2",
    ]


def collect_row(run_name: str, winner_summary: dict[str, object]) -> dict[str, object]:
    run_dir = RUN_BASE / run_name / "reports" / "core"
    committee_path = run_dir / "committee_suite_summary.csv"
    metrics_path = run_dir / "metrics.json"
    best_summary_path = (
        RUN_BASE
        / "reports"
        / "committee_experiments"
        / f"{run_name}__committee__fnb"
        / "best_committee_summary.json"
    )
    if not committee_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing confirmation artifacts in {run_dir}")
    if not best_summary_path.exists():
        raise FileNotFoundError(f"Missing committee summary: {best_summary_path}")

    metrics = load_json(metrics_path)
    committee_rows = list(csv.DictReader(committee_path.open("r", encoding="utf-8")))
    if not committee_rows:
        raise ValueError(f"No committee rows found in {committee_path}")
    committee = committee_rows[0]
    best_summary = load_json(best_summary_path)
    best_committee = best_summary.get("best_committee", {})
    best_stability = best_summary.get("best_committee_stability", {})
    winner = winner_summary["best_committee"]

    return {
        "context_run": run_name,
        "context_best_val_rel_score": metrics.get("lstm_signmag_best_by_val", {}).get("val", {}).get("rel_score"),
        "context_best_test_rel_score": metrics.get("lstm_signmag_best_by_val", {}).get("test", {}).get("rel_score"),
        "expert_model": best_committee.get("expert_model"),
        "market_model": best_committee.get("market_model"),
        "committee_method": best_committee.get("method"),
        "committee_weight_expert": best_committee.get("weight_expert"),
        "committee_code_count": best_committee.get("code_count", committee.get("code_count")),
        "committee_overlap_codes": best_committee.get("overlap_codes", committee.get("overlap_codes")),
        "committee_val_rel_score": best_committee.get("committee_val_rel_score", committee.get("committee_val_rel_score")),
        "committee_test_rel_score": best_committee.get("committee_test_rel_score", committee.get("committee_test_rel_score")),
        "stable_weight_min": best_stability.get("stable_weight_min"),
        "stable_weight_max": best_stability.get("stable_weight_max"),
        "stable_weight_count": best_stability.get("stable_weight_count"),
        "stable_test_rel_score_median": best_stability.get("stable_test_rel_score_median"),
        "winner_val_rel_score": winner.get("committee_val_rel_score"),
        "winner_test_rel_score": winner.get("committee_test_rel_score"),
        "delta_vs_winner_test": float(best_committee.get("committee_test_rel_score")) - float(winner.get("committee_test_rel_score")),
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = args.summary_dir / f"{args.run_name_prefix}_{stamp}"
    summary_dir.mkdir(parents=True, exist_ok=True)
    winner_summary = load_json(args.winner_summary)

    rows: list[dict[str, object]] = []
    for idx in range(1, args.repeats + 1):
        run_name = f"{args.run_name_prefix}_{stamp}_r{idx:02d}"
        log_path = args.log_base / f"{run_name}.log"
        run_and_log(build_command(run_name), log_path)
        rows.append(collect_row(run_name, winner_summary))

    fieldnames = [
        "context_run",
        "context_best_val_rel_score",
        "context_best_test_rel_score",
        "expert_model",
        "market_model",
        "committee_method",
        "committee_weight_expert",
        "committee_code_count",
        "committee_overlap_codes",
        "committee_val_rel_score",
        "committee_test_rel_score",
        "stable_weight_min",
        "stable_weight_max",
        "stable_weight_count",
        "stable_test_rel_score_median",
        "winner_val_rel_score",
        "winner_test_rel_score",
        "delta_vs_winner_test",
    ]
    summary_path = summary_dir / "confirmation_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "summary_path": str(summary_path),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
