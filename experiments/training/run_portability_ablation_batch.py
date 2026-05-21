from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUN_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
REPORT_ROOT = RUN_ROOT / "reports" / "portability_ablation"
DEFAULT_SOURCE_RUN_DIR = RUN_ROOT / "broad_signmag_prune_general_sector_full_20260424_r04"
DEFAULT_MARKETS = ("JP", "KR", "US")
MARKET_CODES_PATHS = {
    "JP": ROOT / "market_lists" / "jp50.txt",
    "KR": ROOT / "market_lists" / "kr50.txt",
    "US": ROOT / "market_lists" / "us100.txt",
}
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
ICHIMOKU_TK_GAP_FEATURES = (
    "ichi_8_21_42_tenkan_kijun_gap",
    "ichi_8_22_44_tenkan_kijun_gap",
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
        description="Run portability-focused VN signmag ablations and OOD readiness checks."
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y%m%d_r%H%M"),
        help="Batch stamp used in run names and report paths.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=ROOT / "venv" / "bin" / "python",
        help="Python executable used to launch training and analysis jobs.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=DEFAULT_SOURCE_RUN_DIR,
        help="Trusted VN anchor run used as the hyperparameter source.",
    )
    parser.add_argument(
        "--markets",
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated OOD markets to evaluate after training. Default: JP,KR,US.",
    )
    parser.add_argument(
        "--train-end-date",
        default=None,
        help="Override the training end date. Defaults to the source run boundary.",
    )
    parser.add_argument(
        "--val-end-date",
        default=None,
        help="Override the validation end date. Defaults to the source run boundary.",
    )
    parser.add_argument(
        "--override-epochs",
        type=int,
        default=None,
        help="Optional epoch override, mainly for quick smoke checks.",
    )
    parser.add_argument(
        "--override-patience",
        type=int,
        default=None,
        help="Optional patience override, mainly for quick smoke checks.",
    )
    parser.add_argument(
        "--case-filter",
        default=None,
        help="Optional comma-separated case names to run.",
    )
    parser.add_argument(
        "--stocks-mode",
        choices=["anchor", "all"],
        default="anchor",
        help="Use the source run stock list or the full cleaned dataset universe.",
    )
    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help="Train the VN cases only and skip JP/KR/US readiness evaluation.",
    )
    parser.add_argument(
        "--allow-nonstandard-time",
        action="store_true",
        help="Allow explicit date overrides outside the locked reporting window.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Write the manifest and print planned commands without launching jobs.",
    )
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


def ordered_remove(base_features: tuple[str, ...], removed_features: tuple[str, ...]) -> tuple[str, ...]:
    removed = set(removed_features)
    return tuple(feature for feature in base_features if feature not in removed)


def ordered_insert_after(
    base_features: tuple[str, ...],
    added_features: tuple[str, ...],
    *,
    insert_after: str,
) -> tuple[str, ...]:
    added_set = set(added_features)
    out: list[str] = []
    inserted = False
    for feature in base_features:
        if feature not in added_set:
            out.append(feature)
        if feature == insert_after:
            out.extend(item for item in added_features if item not in out)
            inserted = True
    if not inserted:
        out.extend(item for item in added_features if item not in out)
    return tuple(out)


def resolve_cases(source_config: dict, case_filter: tuple[str, ...]) -> list[BatchCase]:
    anchor_features = tuple(str(feature) for feature in source_config["feature_columns"])
    no_sector_context = ordered_remove(anchor_features, SECTOR_CONTEXT_FEATURES)
    ichi_8_21_features = ordered_insert_after(
        anchor_features,
        ("ichi_8_21_42_tenkan_kijun_gap",),
        insert_after="momentum_20",
    )
    ichi_8_22_features = ordered_insert_after(
        anchor_features,
        ("ichi_8_22_44_tenkan_kijun_gap",),
        insert_after="momentum_20",
    )
    ichi_both_features = ordered_insert_after(
        anchor_features,
        ICHIMOKU_TK_GAP_FEATURES,
        insert_after="momentum_20",
    )
    cases = [
        BatchCase(
            name="portable_no_identity",
            notes="Keep general_sector_full features, remove stock identity to measure code-memorization dependence.",
            feature_columns=anchor_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="portable_no_identity_ichi_8_21_tk",
            notes="Add the strongest causal Ichimoku 8/21/42 Tenkan-Kijun gap from validation IC screening.",
            feature_columns=ichi_8_21_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="portable_no_identity_ichi_8_22_tk",
            notes="Add the user's causal Ichimoku 8/22/44 Tenkan-Kijun gap candidate from validation IC screening.",
            feature_columns=ichi_8_22_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="portable_no_identity_ichi_tk_both",
            notes="Add both stable causal Ichimoku Tenkan-Kijun gap variants to test incremental redundancy.",
            feature_columns=ichi_both_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="portable_no_identity_units48_24",
            notes="Portable no-identity candidate with slightly lower capacity to reduce variance after removing code one-hots.",
            feature_columns=anchor_features,
            disable_stock_identity=True,
            config_overrides={"lstm_units": [48, 24]},
        ),
        BatchCase(
            name="portable_no_identity_rank003",
            notes=(
                "Portable no-identity model with a very light cross-sectional pairwise rank sidecar. "
                "This is a conservative ranking smoke test against the all-VN portable base."
            ),
            feature_columns=anchor_features,
            disable_stock_identity=True,
            config_overrides={
                "signmag_rank_loss_weight": 0.03,
                "signmag_rank_temperature": 1.0,
                "signmag_rank_min_group_size": 8,
            },
        ),
        BatchCase(
            name="portable_no_identity_rank005",
            notes=(
                "Portable no-identity model with a light cross-sectional pairwise rank sidecar. "
                "Use this as the first active rank/portfolio objective challenger."
            ),
            feature_columns=anchor_features,
            disable_stock_identity=True,
            config_overrides={
                "signmag_rank_loss_weight": 0.05,
                "signmag_rank_temperature": 1.0,
                "signmag_rank_min_group_size": 8,
            },
        ),
        BatchCase(
            name="portable_no_identity_rank010",
            notes=(
                "Portable no-identity model with a stronger cross-sectional pairwise rank sidecar. "
                "This checks whether ranking pressure helps IC/equity or starts damaging rel_score."
            ),
            feature_columns=anchor_features,
            disable_stock_identity=True,
            config_overrides={
                "signmag_rank_loss_weight": 0.10,
                "signmag_rank_temperature": 1.0,
                "signmag_rank_min_group_size": 8,
            },
        ),
        BatchCase(
            name="portable_no_sector_context",
            notes="Keep stock identity, remove sector context to isolate the value of sector-relative signals.",
            feature_columns=no_sector_context,
            disable_stock_identity=False,
        ),
        BatchCase(
            name="portable_no_identity_no_sector_context",
            notes="Remove both stock identity and sector context for the cleanest transfer benchmark.",
            feature_columns=no_sector_context,
            disable_stock_identity=True,
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
) -> list[str]:
    train_end_date = args.train_end_date or str(source_config["train_end_date"])
    val_end_date = args.val_end_date or str(source_config["val_end_date"])
    effective_config = dict(source_config)
    effective_config.update(case.config_overrides or {})
    stocks_arg = None if args.stocks_mode == "all" else effective_config.get("stocks")
    epochs = int(args.override_epochs or effective_config["epochs"])
    patience = int(args.override_patience or effective_config["patience"])
    command = [
        str(python_bin),
        str(ROOT / "main.py"),
        "train",
        "--run-name",
        run_name,
        "--market",
        str(source_config.get("market", "VN")),
        "--target-mode",
        str(source_config["target_mode"]),
        "--train-end-date",
        train_end_date,
        "--val-end-date",
        val_end_date,
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
        str(epochs),
        "--patience",
        str(patience),
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
    if stocks_arg:
        command.extend(["--stocks", str(stocks_arg)])
    if args.allow_nonstandard_time:
        command.append("--allow-nonstandard-time")
    if effective_config.get("data_path"):
        command.extend(["--data-path", str(effective_config["data_path"])])
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
        "signmag_rank_loss_weight": "--signmag-rank-loss-weight",
        "signmag_rank_temperature": "--signmag-rank-temperature",
    }
    for config_key, cli_arg in optional_float_args.items():
        if effective_config.get(config_key) is not None:
            command.extend([cli_arg, str(effective_config[config_key])])
    if effective_config.get("signmag_rank_min_group_size") is not None:
        command.extend(
            [
                "--signmag-rank-min-group-size",
                str(effective_config["signmag_rank_min_group_size"]),
            ]
        )
    if not bool(effective_config.get("signmag_log_magnitude", True)):
        command.append("--no-signmag-log-magnitude")
    if case.disable_stock_identity:
        command.append("--disable-stock-identity")
    return command


def build_ood_command(
    python_bin: Path,
    run_dir: Path,
    market: str,
    output_name: str,
) -> list[str]:
    return [
        str(python_bin),
        str(ROOT / "experiments" / "analysis" / "evaluate_run_ood_readiness.py"),
        "--run-dir",
        str(run_dir),
        "--market",
        market,
        "--codes-path",
        str(MARKET_CODES_PATHS[market]),
        "--output-name",
        output_name,
    ]


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
    best_model = str((family.get("lstm_signmag") or {}).get("best_by_val", "lstm_signmag_best_by_val"))
    best_metrics = metrics[best_model]["val"]
    return {
        "batch_name": batch_name,
        "case_name": case.name,
        "run_name": run_dir.name,
        "feature_count": len(config["feature_columns"]),
        "disable_stock_identity": bool(config.get("use_stock_identity") is False or config.get("lstm_use_stock_identity") is False),
        "best_signmag_model": best_model,
        "val_rel_score": float(best_metrics.get("rel_score", float("nan"))),
        "val_directional_accuracy": float(best_metrics.get("directional_accuracy", float("nan"))),
        "val_mse": float(best_metrics.get("mse", float("nan"))),
        "window_size": int(config["window_size"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
        "signmag_rank_loss_weight": float(config.get("signmag_rank_loss_weight") or 0.0),
        "signmag_rank_temperature": float(config.get("signmag_rank_temperature") or 1.0),
        "signmag_rank_min_group_size": int(config.get("signmag_rank_min_group_size") or 5),
    }


def collect_ood_row(batch_name: str, case_name: str, market: str, output_name: str) -> dict[str, object]:
    summary_path = REPORT_ROOT.parent / "ood_readiness" / output_name / "summary.json"
    summary = load_json(summary_path)
    return {
        "batch_name": batch_name,
        "case_name": case_name,
        "market": market,
        "output_name": output_name,
        "run_name": summary["run_name"],
        "model_name": summary["model_name"],
        "stock_identity_enabled": bool(summary["stock_identity_enabled"]),
        "accepted_codes": int(summary["accepted_codes"]),
        "unknown_sector_share": float(summary["unknown_sector_share"]),
        "known_code_share": float(summary["known_code_share"]),
        "known_row_share": float(summary["known_row_share"]),
        "rel_score": float(summary["rel_score"]),
        "directional_accuracy": float(summary["directional_accuracy"]),
        "mean_spearman_ic": float(summary["mean_spearman_ic"]),
        "ic_t_stat": float(summary["ic_t_stat"]),
        "quartile_equity": float(summary["quartile_equity"]),
        "quartile_hit_rate": float(summary["quartile_hit_rate"]),
        "quartile_max_drawdown": float(summary["quartile_max_drawdown"]),
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


def write_summary_markdown(
    batch_dir: Path,
    batch_name: str,
    source_run_dir: Path,
    run_rows: list[dict[str, object]],
    ood_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Portability Ablation Batch",
        "",
        f"- Batch: `{batch_name}`",
        f"- Source anchor: `{source_run_dir.name}`",
        f"- Run summary CSV: `run_summary.csv`",
        f"- OOD summary CSV: `ood_summary.csv`",
        "",
        "## VN Validation",
    ]
    if run_rows:
        ranked_runs = sorted(run_rows, key=lambda row: float(row["val_rel_score"]), reverse=True)
        for row in ranked_runs:
            lines.append(
                f"- `{row['case_name']}`: val rel_score `{float(row['val_rel_score']):+.5f}`, "
                f"dir acc `{float(row['val_directional_accuracy']):.1%}`, "
                f"stock_identity_disabled `{bool(row['disable_stock_identity'])}`"
            )
    else:
        lines.append("- No VN runs were completed.")

    if ood_rows:
        lines.extend(["", "## OOD Readiness"])
        for market in DEFAULT_MARKETS:
            market_rows = [row for row in ood_rows if row["market"] == market]
            if not market_rows:
                continue
            best_row = max(market_rows, key=lambda row: float(row["rel_score"]))
            lines.append(
                f"- `{market}` best rel_score: `{best_row['case_name']}` at `{float(best_row['rel_score']):+.5f}`, "
                f"mean IC `{float(best_row['mean_spearman_ic']):+.5f}`, quartile equity `{float(best_row['quartile_equity']):.3f}`"
            )
    batch_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    batch_name: str,
    batch_dir: Path,
    source_config: dict,
    cases: list[BatchCase],
) -> dict[str, object]:
    universe_codes = None
    if args.stocks_mode == "all":
        data_path = Path(str(source_config["data_path"]))
        try:
            import pandas as pd

            universe_codes = int(pd.read_csv(data_path, usecols=["code"])["code"].astype(str).nunique())
        except Exception:
            universe_codes = None
    return {
        "batch_name": batch_name,
        "batch_dir": str(batch_dir),
        "source_run_dir": str(args.source_run_dir.resolve()),
        "source_run_name": args.source_run_dir.resolve().name,
        "source_config_path": str((args.source_run_dir.resolve() / "reports" / "core" / "config.json")),
        "python_bin": str(args.python_bin.resolve()),
        "markets": list(parse_csv_list(args.markets)),
        "stocks_mode": args.stocks_mode,
        "all_universe_code_count": universe_codes,
        "skip_ood": bool(args.skip_ood),
        "train_end_date": args.train_end_date or source_config["train_end_date"],
        "val_end_date": args.val_end_date or source_config["val_end_date"],
        "override_epochs": args.override_epochs,
        "override_patience": args.override_patience,
        "allow_nonstandard_time": bool(args.allow_nonstandard_time),
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
    markets = parse_csv_list(args.markets)
    cases = resolve_cases(source_config, parse_csv_list(args.case_filter))
    if not cases:
        raise ValueError("No cases selected for the portability batch.")

    batch_scope = "allvn" if args.stocks_mode == "all" else "portability"
    batch_name = f"broad_signmag_{batch_scope}_{args.stamp}"
    batch_dir = REPORT_ROOT / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    dump_json(batch_dir / "manifest.json", build_manifest(args, batch_name, batch_dir, source_config, cases))

    planned_commands: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    ood_rows: list[dict[str, object]] = []

    for case in cases:
        run_name = f"broad_signmag_{case.name}_{args.stamp}"
        command = build_train_command(args.python_bin, source_config, case, run_name, args)
        planned_commands.append({"stage": "train", "case_name": case.name, "command": command})
        if args.print_only:
            continue
        run_logged_command(command, batch_dir / "logs" / f"{run_name}.log", cwd=ROOT)
        run_dir = RUN_ROOT / run_name
        run_rows.append(collect_run_row(batch_name, case, run_dir))

    if args.print_only:
        dump_json(batch_dir / "planned_commands.json", planned_commands)
        print(json.dumps({"batch_dir": str(batch_dir), "planned_commands": planned_commands}, indent=2))
        return

    if not args.skip_ood:
        for case in cases:
            run_name = f"broad_signmag_{case.name}_{args.stamp}"
            run_dir = RUN_ROOT / run_name
            for market in markets:
                output_name = f"{run_name}__{market.lower()}__portability"
                command = build_ood_command(args.python_bin, run_dir, market, output_name)
                planned_commands.append(
                    {"stage": "ood", "case_name": case.name, "market": market, "command": command}
                )
                run_logged_command(
                    command,
                    batch_dir / "logs" / f"{output_name}.log",
                    cwd=ROOT,
                )
                ood_rows.append(collect_ood_row(batch_name, case.name, market, output_name))

    dump_json(batch_dir / "planned_commands.json", planned_commands)
    write_csv(batch_dir / "run_summary.csv", run_rows)
    write_csv(batch_dir / "ood_summary.csv", ood_rows)
    write_summary_markdown(batch_dir, batch_name, source_run_dir, run_rows, ood_rows)

    print(
        json.dumps(
            {
                "batch_name": batch_name,
                "batch_dir": str(batch_dir),
                "run_summary_rows": len(run_rows),
                "ood_summary_rows": len(ood_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
