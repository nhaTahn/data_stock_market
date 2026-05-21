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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SOURCE_RUN_DIR = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "broad_signmag_portable_no_identity_20260428_allvn_r01"
)
US_DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_us" / "history" / "us_gold_recommended.csv"
US_RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_us" / "history" / "training_runs"
US_REPORT_ROOT = US_RUN_ROOT / "reports" / "us_native"
US_CODES_PATH = ROOT / "market_lists" / "us100.txt"
SECTOR_CONTEXT_FEATURES = (
    "sector_momentum_rank",
    "sector_momentum_rank_pct",
    "sector_momentum_20",
    "relative_sector_momentum_20",
    "sector_return",
    "alpha_sector",
    "sector_positive_ratio",
    "sector_ad_ratio",
)


@dataclass(frozen=True)
class BatchCase:
    name: str
    notes: str
    feature_columns: tuple[str, ...]
    disable_stock_identity: bool
    config_overrides: dict[str, object] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact US-native baseline batch using the current portable architecture."
    )
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_r%H%M"))
    parser.add_argument("--python-bin", type=Path, default=ROOT / "venv" / "bin" / "python")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument(
        "--case-set",
        choices=["baseline", "followup"],
        default="baseline",
        help="Baseline compares identity on the portable US feature set; followup runs small hparam checks around the current US winner.",
    )
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-patience", type=int, default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args(argv)


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict | list[dict]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_lstm_units_arg(value: int | list[int]) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def parse_codes(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip())


def ordered_remove(base_features: tuple[str, ...], removed_features: tuple[str, ...]) -> tuple[str, ...]:
    removed = set(removed_features)
    return tuple(feature for feature in base_features if feature not in removed)


def resolve_cases(source_config: dict, case_filter: tuple[str, ...]) -> list[BatchCase]:
    base_features = tuple(str(feature) for feature in source_config["feature_columns"])
    no_sector_context = ordered_remove(base_features, SECTOR_CONTEXT_FEATURES)
    if "sector_" in ",".join(base_features) or "alpha_sector" in base_features:
        effective_base = no_sector_context
    else:
        effective_base = base_features
    cases = [
        BatchCase(
            name="us100_no_sector_no_identity",
            notes="US100-native baseline with stock identity disabled and no sector features.",
            feature_columns=effective_base,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="us100_no_sector_with_identity",
            notes="US100-native baseline with stock identity enabled and no sector features.",
            feature_columns=effective_base,
            disable_stock_identity=False,
        ),
    ]
    if not case_filter:
        return cases
    allowed = set(case_filter)
    return [case for case in cases if case.name in allowed]


def resolve_followup_cases(source_config: dict, case_filter: tuple[str, ...]) -> list[BatchCase]:
    base_features = tuple(str(feature) for feature in source_config["feature_columns"])
    cases = [
        BatchCase(
            name="us100_no_identity_units48_24",
            notes="Smaller capacity follow-up around the current US no-identity winner.",
            feature_columns=base_features,
            disable_stock_identity=True,
            config_overrides={"lstm_units": [48, 24]},
        ),
        BatchCase(
            name="us100_no_identity_dropout08",
            notes="Slightly stronger dropout to test variance control on US.",
            feature_columns=base_features,
            disable_stock_identity=True,
            config_overrides={"dropout": 0.08},
        ),
        BatchCase(
            name="us100_no_identity_sample_w2",
            notes="Stronger magnitude weighting to see if larger US moves improve rel_score.",
            feature_columns=base_features,
            disable_stock_identity=True,
            config_overrides={"sample_weight_strength": 2.0},
        ),
    ]
    if not case_filter:
        return cases
    allowed = set(case_filter)
    return [case for case in cases if case.name in allowed]


def build_train_command(
    python_bin: Path,
    source_config: dict,
    case: BatchCase,
    run_name: str,
    args: argparse.Namespace,
    us_codes: tuple[str, ...],
) -> list[str]:
    effective_config = dict(source_config)
    effective_config.update(case.config_overrides or {})
    command = [
        str(python_bin),
        str(ROOT / "main.py"),
        "train",
        "--run-name",
        run_name,
        "--market",
        "US",
        "--data-path",
        str(US_DATA_PATH),
        "--output-dir",
        str(US_RUN_ROOT),
        "--target-mode",
        str(effective_config["target_mode"]),
        "--train-end-date",
        str(args.train_end_date or effective_config["train_end_date"]),
        "--val-end-date",
        str(args.val_end_date or effective_config["val_end_date"]),
        "--allow-nonstandard-time",
        "--stocks",
        ",".join(us_codes),
        "--feature-columns",
        ",".join(case.feature_columns),
        "--window-size",
        str(effective_config["window_size"]),
        "--lstm-units",
        build_lstm_units_arg(effective_config["lstm_units"]),
        "--dropout",
        str(effective_config["dropout"]),
        "--lr",
        str(effective_config["lr"]),
        "--loss",
        str(effective_config["loss"]),
        "--batch-size",
        str(effective_config["batch_size"]),
        "--epochs",
        str(args.override_epochs or effective_config["epochs"]),
        "--patience",
        str(args.override_patience or effective_config["patience"]),
        "--sequence-normalization",
        str(effective_config.get("sequence_normalization", "none")),
        "--lstm-seeds",
        ",".join(str(seed) for seed in effective_config["lstm_seeds"]),
        "--sample-weight-mode",
        str(effective_config["sample_weight_mode"]),
        "--sample-weight-strength",
        str(effective_config["sample_weight_strength"]),
        "--sample-weight-quantile",
        str(effective_config["sample_weight_quantile"]),
        "--sample-weight-clip",
        str(effective_config["sample_weight_clip"]),
        "--signmag-signed-loss-weight",
        str(effective_config["signmag_signed_loss_weight"]),
        "--signmag-sign-loss-weight",
        str(effective_config["signmag_sign_loss_weight"]),
        "--signmag-magnitude-loss-weight",
        str(effective_config["signmag_magnitude_loss_weight"]),
    ]
    if effective_config.get("target_normalizer"):
        command.extend(["--target-normalizer", str(effective_config["target_normalizer"])])
    if effective_config.get("feature_phase") not in {None, "", "none"}:
        command.extend(["--feature-phase", str(effective_config["feature_phase"])])
    optional_float_args = {
        "huber_delta": "--huber-delta",
        "rel_score_large_move_quantile": "--rel-score-large-move-quantile",
        "rel_score_directional_penalty": "--rel-score-directional-penalty",
        "rel_score_confidence_penalty": "--rel-score-confidence-penalty",
        "rel_score_confidence_ratio": "--rel-score-confidence-ratio",
        "rel_score_weighted_high_quantile": "--rel-score-weighted-high-quantile",
        "rel_score_weighted_high_weight": "--rel-score-weighted-high-weight",
        "rel_score_weighted_base_weight": "--rel-score-weighted-base-weight",
    }
    for config_key, cli_arg in optional_float_args.items():
        if effective_config.get(config_key) is not None:
            command.extend([cli_arg, str(effective_config[config_key])])
    if not bool(effective_config.get("signmag_log_magnitude", True)):
        command.append("--no-signmag-log-magnitude")
    if case.disable_stock_identity:
        command.append("--disable-stock-identity")
    return command


def run_logged_command(command: list[str], log_path: Path, *, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


def collect_run_row(batch_name: str, case: BatchCase, run_dir: Path) -> dict[str, object]:
    config = load_json(run_dir / "reports" / "core" / "config.json")
    metrics = load_json(run_dir / "reports" / "core" / "metrics.json")
    family = load_json(run_dir / "reports" / "core" / "family_selection_summary.json")
    best_model = str((family.get("lstm_signmag") or {}).get("best_by_val", ""))
    best_metrics = metrics[best_model]["val"]
    candidate_models = [
        name for name, payload in metrics.items()
        if isinstance(payload, dict) and "val" in payload and isinstance(payload["val"], dict)
    ]
    overall_best_model = max(
        candidate_models,
        key=lambda name: float(metrics[name]["val"].get("rel_score", float("-inf"))),
    )
    overall_best_metrics = metrics[overall_best_model]["val"]
    return {
        "batch_name": batch_name,
        "case_name": case.name,
        "run_name": run_dir.name,
        "feature_count": len(config["feature_columns"]),
        "trained_code_count": len(config.get("recipe_selected_stocks", [])),
        "disable_stock_identity": bool(config.get("lstm_use_stock_identity") is False),
        "overall_best_model": overall_best_model,
        "overall_val_rel_score": float(overall_best_metrics.get("rel_score", float("nan"))),
        "overall_val_directional_accuracy": float(overall_best_metrics.get("directional_accuracy", float("nan"))),
        "overall_val_mse": float(overall_best_metrics.get("mse", float("nan"))),
        "best_signmag_model": best_model,
        "val_rel_score": float(best_metrics.get("rel_score", float("nan"))),
        "val_directional_accuracy": float(best_metrics.get("directional_accuracy", float("nan"))),
        "val_mse": float(best_metrics.get("mse", float("nan"))),
        "window_size": int(config["window_size"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(batch_dir: Path, batch_name: str, run_rows: list[dict[str, object]]) -> None:
    lines = [
        "# US Native Baseline Batch",
        "",
        f"- Batch: `{batch_name}`",
        f"- Run summary CSV: `run_summary.csv`",
        "",
        "## Validation",
    ]
    ranked_rows = sorted(run_rows, key=lambda row: float(row["val_rel_score"]), reverse=True)
    for row in ranked_rows:
        lines.append(
            f"- `{row['case_name']}`: overall best `{row['overall_best_model']}` at rel_score `{float(row['overall_val_rel_score']):+.5f}`, "
            f"signmag best `{row['best_signmag_model']}` at rel_score `{float(row['val_rel_score']):+.5f}`, "
            f"trained codes `{int(row['trained_code_count'])}`"
        )
    batch_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    batch_name: str,
    batch_dir: Path,
    source_config: dict,
    cases: list[BatchCase],
    us_codes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "batch_name": batch_name,
        "batch_dir": str(batch_dir),
        "source_run_dir": str(args.source_run_dir.resolve()),
        "source_run_name": args.source_run_dir.resolve().name,
        "source_config_path": str(args.source_run_dir.resolve() / "reports" / "core" / "config.json"),
        "python_bin": str(args.python_bin.resolve()),
        "market": "US",
        "data_path": str(US_DATA_PATH),
        "output_dir": str(US_RUN_ROOT),
        "codes_path": str(US_CODES_PATH),
        "requested_codes": int(len(us_codes)),
        "train_end_date": args.train_end_date or source_config["train_end_date"],
        "val_end_date": args.val_end_date or source_config["val_end_date"],
        "override_epochs": args.override_epochs,
        "override_patience": args.override_patience,
        "cases": [
            {
                "name": case.name,
                "notes": case.notes,
                "feature_columns": list(case.feature_columns),
                "feature_count": len(case.feature_columns),
                "disable_stock_identity": case.disable_stock_identity,
                "config_overrides": case.config_overrides or {},
            }
            for case in cases
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_run_dir = args.source_run_dir.resolve()
    source_config = load_json(source_run_dir / "reports" / "core" / "config.json")
    us_codes = parse_codes(US_CODES_PATH)
    if args.case_set == "baseline":
        cases = resolve_cases(source_config, parse_csv_list(args.case_filter))
    else:
        cases = resolve_followup_cases(source_config, parse_csv_list(args.case_filter))
    if not cases:
        raise ValueError("No US-native cases selected.")

    batch_name = f"us_native_{args.case_set}_{args.stamp}"
    batch_dir = US_REPORT_ROOT / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    dump_json(batch_dir / "manifest.json", build_manifest(args, batch_name, batch_dir, source_config, cases, us_codes))

    planned_commands: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for case in cases:
        run_name = f"{case.name}_{args.stamp}"
        command = build_train_command(args.python_bin, source_config, case, run_name, args, us_codes)
        planned_commands.append({"stage": "train", "case_name": case.name, "command": command})
        if args.print_only:
            continue
        run_logged_command(command, batch_dir / "logs" / f"{run_name}.log", cwd=ROOT)
        run_dir = US_RUN_ROOT / run_name
        run_rows.append(collect_run_row(batch_name, case, run_dir))

    dump_json(batch_dir / "planned_commands.json", planned_commands)
    if args.print_only:
        print(json.dumps({"batch_dir": str(batch_dir), "planned_commands": planned_commands}, indent=2))
        return

    write_csv(batch_dir / "run_summary.csv", run_rows)
    write_summary_markdown(batch_dir, batch_name, run_rows)
    print(json.dumps({"batch_name": batch_name, "batch_dir": str(batch_dir), "run_summary_rows": len(run_rows)}, indent=2))


if __name__ == "__main__":
    main()
