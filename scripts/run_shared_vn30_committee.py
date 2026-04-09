from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
DEFAULT_UNIVERSE_PATH = ROOT / "market_lists" / "vn30.txt"
DEFAULT_LOG_BASE = RUN_BASE / "shared_context_logs"
TRAIN_SCRIPT = ROOT / "scripts" / "run_train.py"
COMMITTEE_SCRIPT = ROOT / "src" / "research" / "committee_relscore_experiment.py"
PYTHON_BIN = ROOT / "venv" / "bin" / "python"


@dataclass(frozen=True)
class ExpertPreset:
    name: str
    expert_run_name: str
    expert_models: str
    market_models: str = "lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble"

    @property
    def expert_run_dir(self) -> Path:
        return RUN_BASE / self.expert_run_name


EXPERT_PRESETS: dict[str, ExpertPreset] = {
    "fnb": ExpertPreset(
        name="fnb",
        expert_run_name="mini_tpdouong_g06_uncertainty_sidecar",
        expert_models="lstm_best_by_val,lstm_ensemble",
    ),
    "bds": ExpertPreset(
        name="bds",
        expert_run_name="mini_bat_ong_san_g01_return_w20_pruned_v2",
        expert_models="lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble",
    ),
    "bank": ExpertPreset(
        name="bank",
        expert_run_name="mini_ngan_hang_g02_return_w20_pruned_v2",
        expert_models="lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble",
    ),
    "chung": ExpertPreset(
        name="chung",
        expert_run_name="sector_dich_vu_tai_chinh_return_w5_relscore",
        expert_models="lstm_best_by_val,lstm_signmag_best_by_val,lstm_ensemble",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a shared VN30 context model and evaluate committee combinations against expert runs."
    )
    parser.add_argument("--context-run-dir", type=Path, default=None)
    parser.add_argument("--run-name-prefix", default="shared_vn30_return_w20_relscore")
    parser.add_argument("--universe-path", type=Path, default=DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--target-mode", choices=["return", "return_3d", "return_5d"], default="return")
    parser.add_argument("--loss", choices=["mse", "huber", "directional_huber", "rel_score"], default="rel_score")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--lstm-seeds", default="42,52,62")
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default="none")
    parser.add_argument("--feature-selection-mode", choices=["auto", "sector_config", "search_summary"], default="sector_config")
    parser.add_argument("--committee-preset", action="append", choices=sorted(EXPERT_PRESETS), default=None)
    parser.add_argument("--committee-methods", default="avg,agree_only")
    parser.add_argument("--committee-weight-step", type=float, default=0.05)
    parser.add_argument("--committee-stable-weight-tolerance", type=float, default=0.001)
    parser.add_argument("--committee-selection-mode", choices=["best_val", "stable_band"], default="stable_band")
    parser.add_argument("--committee-stable-selection-val-gap", type=float, default=0.006)
    parser.add_argument("--committee-stable-selection-min-weight-count", type=int, default=2)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-committee", action="store_true")
    parser.add_argument("--rotation-min-code-count", type=int, default=3)
    parser.add_argument("--rotation-min-val-rel-score", type=float, default=0.015)
    parser.add_argument("--rotation-min-stable-weight-count", type=int, default=2)
    parser.add_argument("--rotation-min-stable-test-median", type=float, default=0.0)
    parser.add_argument("--log-base", type=Path, default=DEFAULT_LOG_BASE)
    return parser.parse_args()


def load_stocks(universe_path: Path) -> list[str]:
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    raw_text = universe_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        raise ValueError(f"Universe file is empty: {universe_path}")
    tokens = [item.strip() for item in re.split(r"[\s,;]+", raw_text) if item.strip()]
    if tokens and tokens[0].lower() in {"symbol", "code"}:
        tokens = tokens[1:]
    stocks = list(dict.fromkeys(tokens))
    if not stocks:
        raise ValueError(f"No stocks parsed from: {universe_path}")
    return stocks


def build_context_run_dir(args: argparse.Namespace) -> Path:
    if args.context_run_dir is not None:
        return args.context_run_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUN_BASE / f"{args.run_name_prefix}_{stamp}"


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


def build_train_command(args: argparse.Namespace, context_run_dir: Path, stocks: list[str]) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(TRAIN_SCRIPT),
        "--target-mode",
        args.target_mode,
        "--loss",
        args.loss,
        "--stocks",
        ",".join(stocks),
        "--feature-selection-mode",
        args.feature_selection_mode,
        "--window-size",
        str(args.window_size),
        "--lstm-units",
        args.lstm_units,
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--target-normalizer",
        args.target_normalizer,
        "--lstm-seeds",
        args.lstm_seeds,
        "--sample-weight-mode",
        args.sample_weight_mode,
        "--run-name",
        context_run_dir.name,
    ]


def build_committee_command(
    args: argparse.Namespace,
    *,
    context_run_dir: Path,
    preset: ExpertPreset,
) -> tuple[list[str], str]:
    output_name = f"{context_run_dir.name}__committee__{preset.name}"
    cmd = [
        str(PYTHON_BIN),
        str(COMMITTEE_SCRIPT),
        "--expert-run",
        str(preset.expert_run_dir),
        "--expert-models",
        preset.expert_models,
        "--market-run",
        str(context_run_dir),
        "--market-models",
        preset.market_models,
        "--methods",
        args.committee_methods,
        "--weight-step",
        str(args.committee_weight_step),
        "--stable-weight-tolerance",
        str(args.committee_stable_weight_tolerance),
        "--selection-mode",
        args.committee_selection_mode,
        "--stable-selection-val-gap",
        str(args.committee_stable_selection_val_gap),
        "--stable-selection-min-weight-count",
        str(args.committee_stable_selection_min_weight_count),
        "--output-name",
        output_name,
    ]
    return cmd, output_name


def ensure_context_ready(context_run_dir: Path) -> None:
    metrics_path = context_run_dir / "reports" / "core" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Context run artifacts not found in {context_run_dir}. Expected {metrics_path}"
        )


def save_manifest(
    context_run_dir: Path,
    *,
    stocks: list[str],
    train_command: list[str] | None,
    committee_rows: list[dict[str, object]],
) -> Path:
    manifest = {
        "context_run_dir": str(context_run_dir),
        "stocks": stocks,
        "train_command": train_command,
        "committee_rows": committee_rows,
    }
    manifest_path = context_run_dir / "reports" / "core" / "committee_suite_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def save_committee_summary(context_run_dir: Path, rows: list[dict[str, object]]) -> Path:
    summary_path = context_run_dir / "reports" / "core" / "committee_suite_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "expert_run",
        "expert_model",
        "market_model",
        "method",
        "weight_expert",
        "code_count",
        "overlap_codes",
        "agreement_rate",
        "val_rows",
        "test_rows",
        "committee_val_rel_score",
        "committee_test_rel_score",
        "expert_val_rel_score_overlap",
        "expert_test_rel_score_overlap",
        "market_val_rel_score_overlap",
        "market_test_rel_score_overlap",
        "summary_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return summary_path


def save_rotation_summary(
    context_run_dir: Path,
    rows: list[dict[str, object]],
    *,
    min_code_count: int,
    min_val_rel_score: float,
    min_stable_weight_count: int,
    min_stable_test_median: float,
) -> tuple[Path, Path]:
    all_path = context_run_dir / "reports" / "core" / "committee_rotation_all.csv"
    active_path = context_run_dir / "reports" / "core" / "committee_rotation_active.csv"
    all_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_rows = []
    for row in rows:
        item = dict(row)
        item["committee_gain_vs_expert_test"] = (
            float(item["committee_test_rel_score"]) - float(item["expert_test_rel_score_overlap"])
        )
        item["committee_gain_vs_market_test"] = (
            float(item["committee_test_rel_score"]) - float(item["market_test_rel_score_overlap"])
        )
        stability = item.get("best_committee_stability") or {}
        item["stable_weight_min"] = stability.get("stable_weight_min")
        item["stable_weight_max"] = stability.get("stable_weight_max")
        item["stable_weight_count"] = stability.get("stable_weight_count")
        item["stable_test_rel_score_median"] = stability.get("stable_test_rel_score_median")
        item["rotation_active"] = int(
            int(item.get("code_count", 0)) >= min_code_count
            and float(item.get("committee_val_rel_score", float("-inf"))) > min_val_rel_score
            and int(item.get("stable_weight_count") or 0) >= min_stable_weight_count
            and float(item.get("stable_test_rel_score_median") or float("-inf")) >= min_stable_test_median
        )
        ordered_rows.append(item)

    ordered_rows.sort(
        key=lambda item: (
            item["rotation_active"],
            float(item["committee_val_rel_score"]),
            float(item["committee_test_rel_score"]),
            int(item["code_count"]),
        ),
        reverse=True,
    )
    fieldnames = [
        "preset",
        "expert_run",
        "expert_model",
        "market_model",
        "method",
        "weight_expert",
        "code_count",
        "overlap_codes",
        "committee_val_rel_score",
        "committee_test_rel_score",
        "stable_weight_min",
        "stable_weight_max",
        "stable_weight_count",
        "stable_test_rel_score_median",
        "expert_test_rel_score_overlap",
        "market_test_rel_score_overlap",
        "committee_gain_vs_expert_test",
        "committee_gain_vs_market_test",
        "rotation_active",
        "summary_path",
    ]
    with all_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with active_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            if int(row["rotation_active"]) == 1:
                writer.writerow({key: row.get(key) for key in fieldnames})
    return all_path, active_path


def main() -> None:
    args = parse_args()
    if args.skip_train and args.context_run_dir is None:
        raise ValueError("--skip-train requires --context-run-dir.")
    selected_presets = args.committee_preset or ["fnb", "bds"]
    presets = [EXPERT_PRESETS[name] for name in selected_presets]
    context_run_dir = build_context_run_dir(args)
    stocks = load_stocks(args.universe_path)

    if not args.skip_train:
        if context_run_dir.exists():
            raise FileExistsError(f"Context run dir already exists: {context_run_dir}")
        train_log = args.log_base / f"{context_run_dir.name}.log"
        train_command = build_train_command(args, context_run_dir, stocks)
        run_and_log(train_command, train_log)
    else:
        train_command = None

    ensure_context_ready(context_run_dir)

    committee_rows: list[dict[str, object]] = []
    if not args.skip_committee:
        for preset in presets:
            if not preset.expert_run_dir.exists():
                raise FileNotFoundError(f"Expert run not found: {preset.expert_run_dir}")
            committee_cmd, output_name = build_committee_command(
                args,
                context_run_dir=context_run_dir,
                preset=preset,
            )
            committee_log = args.log_base / f"{output_name}.log"
            run_and_log(committee_cmd, committee_log)
            summary_path = (
                RUN_BASE
                / "reports"
                / "committee_experiments"
                / output_name
                / "best_committee_summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            best = dict(summary.get("best_committee", {}))
            best["best_committee_stability"] = summary.get("best_committee_stability")
            best["preset"] = preset.name
            best["summary_path"] = str(summary_path)
            committee_rows.append(best)

    manifest_path = save_manifest(
        context_run_dir,
        stocks=stocks,
        train_command=train_command,
        committee_rows=committee_rows,
    )
    summary_path = None
    if committee_rows:
        summary_path = save_committee_summary(context_run_dir, committee_rows)
        rotation_all_path, rotation_active_path = save_rotation_summary(
            context_run_dir,
            committee_rows,
            min_code_count=args.rotation_min_code_count,
            min_val_rel_score=args.rotation_min_val_rel_score,
            min_stable_weight_count=args.rotation_min_stable_weight_count,
            min_stable_test_median=args.rotation_min_stable_test_median,
        )
    else:
        rotation_all_path = None
        rotation_active_path = None
    payload = {
        "context_run_dir": str(context_run_dir),
        "stock_count": len(stocks),
        "committee_summary_path": str(summary_path) if summary_path is not None else None,
        "committee_rotation_all_path": str(rotation_all_path) if rotation_all_path is not None else None,
        "committee_rotation_active_path": str(rotation_active_path) if rotation_active_path is not None else None,
        "manifest_path": str(manifest_path),
        "committee_rows": committee_rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
