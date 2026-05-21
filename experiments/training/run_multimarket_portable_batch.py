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

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metric import evaluate


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
VN_DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
JP_DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_jp" / "history" / "jp_gold_recommended.csv"
US_DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_us" / "history" / "us_gold_recommended.csv"
JP_CODES_PATH = ROOT / "market_lists" / "jp50.txt"
US_CODES_PATH = ROOT / "market_lists" / "us100.txt"
GLOBAL_HISTORY_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_global" / "history"
GLOBAL_RUN_ROOT = GLOBAL_HISTORY_ROOT / "training_runs"
GLOBAL_REPORT_ROOT = GLOBAL_RUN_ROOT / "reports" / "multimarket_portable"
KNOWN_MARKETS = ("VN", "JP", "US")
MULTIMARKET_BASE_COLUMNS = (
    "Date",
    "code",
    "native_code",
    "market",
    "sector",
    "open",
    "high",
    "low",
    "close",
    "adjust",
    "volume_match",
    "value_match",
    "target_next_price",
    "target_next_growth_pct",
    "target_next_return",
    "target_next_adjust_price",
    "target_next_adjust_growth_pct",
    "target_next_adjust_return",
    "target_next_3d_return",
    "target_next_5d_return",
    "market_is_vn",
    "market_is_jp",
    "market_is_us",
)

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
MARKET_FLAG_FEATURES = (
    "market_is_vn",
    "market_is_jp",
    "market_is_us",
)
MARKET_CONTEXT_FEATURES = (
    "market_return_5",
    "market_return_20",
    "market_return_60",
    "market_volatility_20",
    "market_ad_ratio_20",
    "breadth_20",
)
SIGNAL_FOCUS_FEATURES = (
    "sector_return",
    "ma_5_gap",
    "momentum_5",
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
        description="Train compact multi-market portable branches on combined VN/JP/US data."
    )
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_r%H%M"))
    parser.add_argument("--python-bin", type=Path, default=ROOT / "venv" / "bin" / "python")
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--val-end-date", default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-patience", type=int, default=None)
    parser.add_argument("--override-seeds", default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args(argv)


def parse_csv_list(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_codes(path: Path) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    return tuple(token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict | list[dict]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_remove(base_features: tuple[str, ...], removed_features: tuple[str, ...]) -> tuple[str, ...]:
    removed = set(removed_features)
    return tuple(feature for feature in base_features if feature not in removed)


def build_lstm_units_arg(value: int | list[int]) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def resolve_cases(source_config: dict, case_filter: tuple[str, ...]) -> list[BatchCase]:
    source_features = tuple(str(feature) for feature in source_config["feature_columns"])
    no_sector_features = ordered_remove(source_features, SECTOR_CONTEXT_FEATURES)
    core_features = tuple(dict.fromkeys([*no_sector_features, *MARKET_FLAG_FEATURES]))
    marketplus_features = tuple(dict.fromkeys([*core_features, *MARKET_CONTEXT_FEATURES]))
    marketplus_signal_focus_features = tuple(
        dict.fromkeys([*marketplus_features, *SIGNAL_FOCUS_FEATURES])
    )
    compact_signal_features = tuple(
        dict.fromkeys(
            [
                "volume_delta_1",
                "momentum_5",
                "momentum_20",
                "macd_hist",
                "ma_5_gap",
                "ma_20_gap",
                "volume_level_20",
                "volume_ratio_20",
                "close_position",
                "volatility_20",
                "day_of_week",
                *MARKET_FLAG_FEATURES,
                *MARKET_CONTEXT_FEATURES,
            ]
        )
    )
    cases = [
        BatchCase(
            name="multimarket_portable_core",
            notes="Portable shared model with no stock identity, no sector context, and market one-hot flags.",
            feature_columns=core_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="multimarket_portable_core_marketbalanced",
            notes="Portable core model with market-balanced sample weights so JP and VN are not dominated by US row count.",
            feature_columns=core_features,
            disable_stock_identity=True,
            config_overrides={"sample_weight_balance_mode": "market"},
        ),
        BatchCase(
            name="multimarket_portable_marketplus",
            notes="Portable shared model with extra per-market trend, volatility, and breadth context.",
            feature_columns=marketplus_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="multimarket_portable_marketplus_signal_focus",
            notes=(
                "Marketplus portable model plus train/validation IC-stable signals: "
                "sector_return, ma_5_gap, and momentum_5."
            ),
            feature_columns=marketplus_signal_focus_features,
            disable_stock_identity=True,
        ),
        BatchCase(
            name="multimarket_portable_compact_signal",
            notes="Compact portable signal set from train/validation IC audit: volume shock, short mean-reversion, MACD reversal, and market context.",
            feature_columns=compact_signal_features,
            disable_stock_identity=True,
            config_overrides={"sample_weight_balance_mode": "market"},
        ),
    ]
    if not case_filter:
        return cases
    allowed = set(case_filter)
    return [case for case in cases if case.name in allowed]


def build_multimarket_dataset(source_config: dict, stamp: str) -> tuple[Path, dict[str, object]]:
    GLOBAL_HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    vn_codes = {str(code).upper() for code in source_config.get("recipe_selected_stocks", [])}
    jp_codes = set(parse_codes(JP_CODES_PATH))
    us_codes = set(parse_codes(US_CODES_PATH))

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for market, path, allowed_codes in (
        ("VN", VN_DATA_PATH, vn_codes),
        ("JP", JP_DATA_PATH, jp_codes),
        ("US", US_DATA_PATH, us_codes),
    ):
        df = pd.read_csv(path)
        raw_code = df["code"].astype(str).str.upper()
        if allowed_codes:
            df = df.loc[raw_code.isin(allowed_codes)].copy()
            raw_code = df["code"].astype(str).str.upper()
        df["market"] = market
        df["native_code"] = raw_code
        df["code"] = market + ":" + raw_code
        if "sector" not in df.columns:
            df["sector"] = "Unknown"
        else:
            df["sector"] = df["sector"].fillna("Unknown")
        df["market_is_vn"] = 1.0 if market == "VN" else 0.0
        df["market_is_jp"] = 1.0 if market == "JP" else 0.0
        df["market_is_us"] = 1.0 if market == "US" else 0.0
        # Keep only raw/common fields so derived paper features are recomputed consistently
        # across VN, JP, and US instead of inheriting VN-only precomputed columns with NaN elsewhere.
        available_columns = [column for column in MULTIMARKET_BASE_COLUMNS if column in df.columns]
        df = df.loc[:, available_columns].copy()
        frames.append(df)
        summaries.append(
            {
                "market": market,
                "rows": int(len(df)),
                "codes": int(df["native_code"].nunique()),
                "start_date": str(pd.to_datetime(df["Date"]).min().date()) if not df.empty else None,
                "end_date": str(pd.to_datetime(df["Date"]).max().date()) if not df.empty else None,
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["market", "code", "Date"], kind="stable").reset_index(drop=True)
    output_path = GLOBAL_HISTORY_ROOT / f"multimarket_vn_jp_us_portable_{stamp}.csv"
    combined.to_csv(output_path, index=False)
    manifest = {
        "data_path": str(output_path),
        "rows": int(len(combined)),
        "panel_codes": int(combined["code"].nunique()),
        "native_codes": int(combined["native_code"].nunique()),
        "markets": summaries,
    }
    return output_path, manifest


def build_train_command(
    python_bin: Path,
    source_config: dict,
    case: BatchCase,
    run_name: str,
    args: argparse.Namespace,
    data_path: Path,
) -> list[str]:
    effective_config = dict(source_config)
    effective_config.update(case.config_overrides or {})
    seeds = parse_csv_list(args.override_seeds) or tuple(str(seed) for seed in effective_config["lstm_seeds"])
    command = [
        str(python_bin),
        str(ROOT / "main.py"),
        "train",
        "--run-name",
        run_name,
        "--market",
        "GLOBAL",
        "--allow-nonstandard-time",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(GLOBAL_RUN_ROOT),
        "--target-mode",
        str(effective_config["target_mode"]),
        "--train-end-date",
        str(args.train_end_date or effective_config["train_end_date"]),
        "--val-end-date",
        str(args.val_end_date or effective_config["val_end_date"]),
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
        ",".join(seeds),
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
    if effective_config.get("sample_weight_balance_mode") not in {None, "", "none"}:
        command.extend(["--sample-weight-balance-mode", str(effective_config["sample_weight_balance_mode"])])
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


def infer_market_from_code(code: object) -> str:
    text = str(code)
    market = text.split(":", 1)[0].upper() if ":" in text else "UNKNOWN"
    return market if market in KNOWN_MARKETS else "UNKNOWN"


def compute_prediction_summary(prediction_df: pd.DataFrame) -> dict[str, float | int] | None:
    work = prediction_df.dropna(subset=["prediction", "actual"]).copy()
    if work.empty:
        return None
    if {"code", "Date"}.issubset(work.columns):
        work["Date"] = pd.to_datetime(work["Date"])
        work = work.sort_values(["code", "Date"], kind="stable")
    group_ids = work["code"].astype(str).to_numpy() if "code" in work.columns else None
    try:
        eval_result = evaluate(
            work["prediction"].to_numpy(dtype=float),
            work["actual"].to_numpy(dtype=float),
            group_ids=group_ids,
        )
    except ValueError:
        return None
    error = work["actual"].to_numpy(dtype=float) - work["prediction"].to_numpy(dtype=float)
    return {
        "row_count": int(len(work)),
        "panel_count": int(work["code"].nunique()) if "code" in work.columns else 0,
        "rel_score": float(eval_result["rel_score"]),
        "directional_accuracy": float(eval_result["directional_accuracy"]),
        "rmse": float((error**2).mean() ** 0.5),
    }


def build_per_market_rows(run_name: str, case_name: str, prediction_df: pd.DataFrame) -> list[dict[str, object]]:
    if prediction_df.empty or "code" not in prediction_df.columns:
        return []
    work = prediction_df.copy()
    work["market"] = work["code"].map(infer_market_from_code)
    rows: list[dict[str, object]] = []
    for (model_name, split_name, market_name), group_df in work.groupby(["model", "split", "market"], sort=True):
        summary = compute_prediction_summary(group_df)
        if summary is None:
            continue
        rows.append(
            {
                "case_name": case_name,
                "run_name": run_name,
                "model": str(model_name),
                "split": str(split_name),
                "market": str(market_name),
                "row_count": int(summary["row_count"]),
                "panel_count": int(summary["panel_count"]),
                "rel_score": float(summary["rel_score"]),
                "directional_accuracy": float(summary["directional_accuracy"]),
                "rmse": float(summary["rmse"]),
            }
        )
    return rows


def write_per_market_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["split"]),
            str(row["market"]),
            -float(row["rel_score"]),
            str(row["model"]),
        ),
    )
    write_csv(path, ordered_rows)


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


def collect_run_artifacts(batch_name: str, case: BatchCase, run_dir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = load_json(run_dir / "reports" / "core" / "config.json")
    metrics = load_json(run_dir / "reports" / "core" / "metrics.json")
    prediction_df = pd.read_csv(run_dir / "reports" / "core" / "predictions.csv")
    train_panel_count = 0
    if not prediction_df.empty and {"code", "split"}.issubset(prediction_df.columns):
        train_panel_count = int(prediction_df.loc[prediction_df["split"] == "train", "code"].nunique())
    candidate_models = [
        name
        for name, payload in metrics.items()
        if isinstance(payload, dict) and "val" in payload and isinstance(payload["val"], dict)
    ]
    overall_best_model = max(
        candidate_models,
        key=lambda name: float(metrics[name]["val"].get("rel_score", float("-inf"))),
    )
    overall_best_metrics = metrics[overall_best_model]["val"]
    per_market_rows = build_per_market_rows(run_dir.name, case.name, prediction_df)
    write_per_market_summary(run_dir / "reports" / "core" / "per_market_summary.csv", per_market_rows)
    return {
        "batch_name": batch_name,
        "case_name": case.name,
        "run_name": run_dir.name,
        "feature_count": len(config["feature_columns"]),
        "trained_panel_count": train_panel_count,
        "disable_stock_identity": bool(config.get("lstm_use_stock_identity") is False),
        "overall_best_model": overall_best_model,
        "overall_val_rel_score": float(overall_best_metrics.get("rel_score", float("nan"))),
        "overall_val_directional_accuracy": float(overall_best_metrics.get("directional_accuracy", float("nan"))),
        "overall_val_mse": float(overall_best_metrics.get("mse", float("nan"))),
        "window_size": int(config["window_size"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["patience"]),
    }, per_market_rows


def write_summary_markdown(
    batch_dir: Path,
    batch_name: str,
    dataset_manifest: dict[str, object],
    run_rows: list[dict[str, object]],
    per_market_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Multi-Market Portable Batch",
        "",
        f"- Batch: `{batch_name}`",
        f"- Data path: `{dataset_manifest['data_path']}`",
        f"- Panel codes: `{dataset_manifest['panel_codes']}`",
        f"- Rows: `{dataset_manifest['rows']}`",
        f"- Run summary CSV: `run_summary.csv`",
        f"- Per-market summary CSV: `per_market_summary.csv`",
        "",
        "## Markets",
    ]
    for item in dataset_manifest["markets"]:
        lines.append(
            f"- `{item['market']}`: codes `{item['codes']}`, rows `{item['rows']}`, window `{item['start_date']}` -> `{item['end_date']}`"
        )
    lines.extend(["", "## Validation"])
    ranked_rows = sorted(run_rows, key=lambda row: float(row["overall_val_rel_score"]), reverse=True)
    for row in ranked_rows:
        lines.append(
            f"- `{row['case_name']}`: overall best `{row['overall_best_model']}` at rel_score `{float(row['overall_val_rel_score']):+.5f}`, "
            f"dir acc `{float(row['overall_val_directional_accuracy']):.2%}`, trained panels `{int(row['trained_panel_count'])}`"
        )
    if per_market_rows:
        lines.extend(["", "## Validation By Market"])
        val_rows = [row for row in per_market_rows if str(row["split"]) == "val"]
        for market_name in KNOWN_MARKETS:
            market_rows = [row for row in val_rows if str(row["market"]) == market_name]
            if not market_rows:
                continue
            best_row = max(market_rows, key=lambda row: float(row["rel_score"]))
            lines.append(
                f"- `{market_name}`: best `{best_row['model']}` from `{best_row['case_name']}` at rel_score `{float(best_row['rel_score']):+.5f}`, "
                f"dir acc `{float(best_row['directional_accuracy']):.2%}`, panels `{int(best_row['panel_count'])}`"
            )
    batch_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    batch_name: str,
    batch_dir: Path,
    source_config: dict,
    cases: list[BatchCase],
    dataset_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "batch_name": batch_name,
        "batch_dir": str(batch_dir),
        "source_run_dir": str(args.source_run_dir.resolve()),
        "source_run_name": args.source_run_dir.resolve().name,
        "source_config_path": str(args.source_run_dir.resolve() / "reports" / "core" / "config.json"),
        "python_bin": str(args.python_bin.resolve()),
        "market": "GLOBAL",
        "output_dir": str(GLOBAL_RUN_ROOT),
        "train_end_date": args.train_end_date or source_config["train_end_date"],
        "val_end_date": args.val_end_date or source_config["val_end_date"],
        "override_epochs": args.override_epochs,
        "override_patience": args.override_patience,
        "override_seeds": args.override_seeds,
        "dataset": dataset_manifest,
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
    cases = resolve_cases(source_config, parse_csv_list(args.case_filter))
    if not cases:
        raise ValueError("No multi-market cases selected.")

    batch_name = f"multimarket_portable_{args.stamp}"
    batch_dir = GLOBAL_REPORT_ROOT / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    data_path, dataset_manifest = build_multimarket_dataset(source_config, args.stamp)
    dump_json(
        batch_dir / "manifest.json",
        build_manifest(args, batch_name, batch_dir, source_config, cases, dataset_manifest),
    )

    planned_commands: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    per_market_rows: list[dict[str, object]] = []
    for case in cases:
        run_name = f"{case.name}_{args.stamp}"
        command = build_train_command(args.python_bin, source_config, case, run_name, args, data_path)
        planned_commands.append({"stage": "train", "case_name": case.name, "command": command})
        if args.print_only:
            continue
        run_logged_command(command, batch_dir / "logs" / f"{run_name}.log", cwd=ROOT)
        run_dir = GLOBAL_RUN_ROOT / run_name
        run_row, run_market_rows = collect_run_artifacts(batch_name, case, run_dir)
        run_rows.append(run_row)
        per_market_rows.extend(run_market_rows)

    dump_json(batch_dir / "planned_commands.json", planned_commands)
    if args.print_only:
        print(json.dumps({"batch_dir": str(batch_dir), "planned_commands": planned_commands, "dataset": dataset_manifest}, indent=2))
        return

    write_csv(batch_dir / "run_summary.csv", run_rows)
    write_per_market_summary(batch_dir / "per_market_summary.csv", per_market_rows)
    write_summary_markdown(batch_dir, batch_name, dataset_manifest, run_rows, per_market_rows)
    print(
        json.dumps(
            {
                "batch_name": batch_name,
                "batch_dir": str(batch_dir),
                "run_summary_rows": len(run_rows),
                "dataset_rows": dataset_manifest["rows"],
                "dataset_panels": dataset_manifest["panel_codes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
