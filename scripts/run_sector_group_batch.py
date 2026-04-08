from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.reporting import resolve_run_artifact
from src.models.training_recipe import DEFAULT_SEARCH_SUMMARY_PATH


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
DEFAULT_LOG_BASE = RUN_BASE / "sector_logs"
DEFAULT_CONTEXT_FEATURES = "alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"


@dataclass(frozen=True)
class SectorCandidate:
    sector: str
    eligible_stock_count: int
    supported_stock_count: int
    top_stocks: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sector-group training batches from stock search summary.")
    parser.add_argument("--search-summary", type=Path, default=DEFAULT_SEARCH_SUMMARY_PATH)
    parser.add_argument("--target-mode", choices=["return", "return_3d", "return_5d"], default="return")
    parser.add_argument("--min-stock-val-rel-score", type=float, default=0.03)
    parser.add_argument("--min-sector-stock-count", type=int, default=4)
    parser.add_argument("--max-sectors", type=int, default=4)
    parser.add_argument("--max-stocks-per-sector", type=int, default=10)
    parser.add_argument("--feature-top-k", type=int, default=10)
    parser.add_argument("--extra-context-features", default=DEFAULT_CONTEXT_FEATURES)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--lstm-units", default="48,24")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--loss", choices=["mse", "huber", "directional_huber", "rel_score"], default="rel_score")
    parser.add_argument("--huber-delta", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--lstm-seeds", default="42,52,62")
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default="none")
    parser.add_argument("--sample-weight-strength", type=float, default=0.8)
    parser.add_argument("--sample-weight-quantile", type=float, default=0.8)
    parser.add_argument("--sample-weight-clip", type=float, default=2.5)
    parser.add_argument("--signmag-signed-loss-weight", type=float, default=2.0)
    parser.add_argument("--signmag-sign-loss-weight", type=float, default=0.10)
    parser.add_argument("--signmag-magnitude-loss-weight", type=float, default=0.25)
    parser.add_argument("--enable-attention-family", action="store_true")
    parser.add_argument("--enable-quantile-family", action="store_true")
    parser.add_argument("--enable-fk-benchmark", action="store_true")
    parser.add_argument("--log-base", type=Path, default=DEFAULT_LOG_BASE)
    parser.add_argument("--run-name-suffix", default=None)
    parser.add_argument("--sector", action="append", default=None)
    return parser.parse_args()


def slugify_sector(sector: str) -> str:
    normalized = unicodedata.normalize("NFKD", sector)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = []
    for char in ascii_text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "sector"


def load_sector_candidates(
    search_summary_path: Path,
    *,
    min_stock_val_rel_score: float,
    min_sector_stock_count: int,
    max_sectors: int,
    selected_sectors: list[str] | None,
) -> list[SectorCandidate]:
    rows: list[dict[str, object]] = []
    with search_summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row["best_val_rel_score"] = float(row["best_val_rel_score"])
                row["best_test_rel_score"] = float(row["best_test_rel_score"])
                row["best_test_val_rel_score"] = float(row["best_test_val_rel_score"])
            except Exception:
                continue
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(f"No usable rows found in search summary: {search_summary_path}")

    if selected_sectors:
        summary = summary[summary["sector"].isin(selected_sectors)].copy()
        if summary.empty:
            raise ValueError("None of the requested sectors exist in the search summary.")

    candidates: list[SectorCandidate] = []
    for sector, sector_df in summary.groupby("sector"):
        eligible = sector_df[sector_df["best_val_rel_score"] >= min_stock_val_rel_score].copy()
        if len(eligible) < min_sector_stock_count:
            continue
        eligible = eligible.sort_values(
            ["best_val_rel_score", "best_test_rel_score", "stock"],
            ascending=[False, False, True],
            kind="stable",
        )
        supported_count = int(
            (
                (sector_df["best_test_rel_score"] >= 0.03)
                & (sector_df["best_test_val_rel_score"] >= 0.0)
            ).sum()
        )
        candidates.append(
            SectorCandidate(
                sector=str(sector),
                eligible_stock_count=int(len(eligible)),
                supported_stock_count=supported_count,
                top_stocks=tuple(eligible["stock"].astype(str).head(8).tolist()),
            )
        )

    candidates.sort(
        key=lambda item: (
            item.eligible_stock_count,
            item.supported_stock_count,
            item.sector,
        ),
        reverse=True,
    )
    return candidates[: max(1, max_sectors)]


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


def resolve_backtest_summary(run_dir: Path, target_mode: str) -> Path:
    if target_mode in {"return_3d", "return_5d"}:
        candidate = resolve_run_artifact(run_dir, "threshold_backtest_summary_non_overlap.json", "backtests")
        if candidate.exists():
            return candidate
    return resolve_run_artifact(run_dir, "threshold_backtest_summary.json", "backtests")


def collect_run_row(run_dir: Path) -> dict[str, object]:
    metrics_path = resolve_run_artifact(run_dir, "metrics.json", "core")
    config_path = resolve_run_artifact(run_dir, "config.json", "core")
    if not metrics_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Run artifacts missing in {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    backtest_path = resolve_backtest_summary(run_dir, str(config.get("target_mode")))
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}

    lstm_models = [name for name in metrics if name.startswith("lstm")]
    ranked = sorted(
        lstm_models,
        key=lambda name: (
            metrics.get(name, {}).get("test", {}).get("rel_score", float("-inf")),
            metrics.get(name, {}).get("val", {}).get("rel_score", float("-inf")),
        ),
        reverse=True,
    )
    best_model = ranked[0] if ranked else None
    best_test = metrics.get(best_model, {}).get("test", {}) if best_model else {}
    best_val = metrics.get(best_model, {}).get("val", {}) if best_model else {}
    lstm_test = metrics.get("lstm", {}).get("test", {})
    lstm_backtest = backtest.get("lstm", {})

    return {
        "run_name": run_dir.name,
        "sector": config.get("sector"),
        "stocks": config.get("stocks"),
        "feature_columns": ",".join(config.get("feature_columns", [])),
        "best_lstm_model": best_model,
        "best_lstm_val_rel_score": best_val.get("rel_score"),
        "best_lstm_test_rel_score": best_test.get("rel_score"),
        "lstm_test_rel_score": lstm_test.get("rel_score"),
        "lstm_test_directional_accuracy": lstm_test.get("directional_accuracy"),
        "lstm_backtest_final_equity": lstm_backtest.get("final_equity"),
        "lstm_backtest_trade_count": lstm_backtest.get("trade_count"),
        "lstm_backtest_threshold": lstm_backtest.get("threshold"),
        "run_dir": str(run_dir),
    }


def build_train_command(args: argparse.Namespace, run_name: str, sector: str) -> list[str]:
    cmd = [
        str(ROOT / "venv" / "bin" / "python"),
        "scripts/run_train.py",
        "--target-mode",
        args.target_mode,
        "--sector",
        sector,
        "--feature-selection-mode",
        "search_summary",
        "--stock-search-summary",
        str(args.search_summary),
        "--min-stock-val-rel-score",
        str(args.min_stock_val_rel_score),
        "--max-stocks",
        str(args.max_stocks_per_sector),
        "--feature-top-k",
        str(args.feature_top_k),
        "--extra-context-features",
        args.extra_context_features,
        "--window-size",
        str(args.window_size),
        "--lstm-units",
        args.lstm_units,
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--loss",
        args.loss,
        "--huber-delta",
        str(args.huber_delta),
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
        "--run-name",
        run_name,
    ]
    if args.sample_weight_mode != "none":
        cmd.extend(
            [
                "--sample-weight-mode",
                args.sample_weight_mode,
                "--sample-weight-strength",
                str(args.sample_weight_strength),
                "--sample-weight-quantile",
                str(args.sample_weight_quantile),
                "--sample-weight-clip",
                str(args.sample_weight_clip),
                "--signmag-signed-loss-weight",
                str(args.signmag_signed_loss_weight),
                "--signmag-sign-loss-weight",
                str(args.signmag_sign_loss_weight),
                "--signmag-magnitude-loss-weight",
                str(args.signmag_magnitude_loss_weight),
            ]
        )
    if args.enable_attention_family:
        cmd.append("--enable-attention-family")
    if args.enable_quantile_family:
        cmd.append("--enable-quantile-family")
    if args.enable_fk_benchmark:
        cmd.append("--enable-fk-benchmark")
    return cmd


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_base / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log dir: {log_dir}")

    candidates = load_sector_candidates(
        args.search_summary,
        min_stock_val_rel_score=args.min_stock_val_rel_score,
        min_sector_stock_count=args.min_sector_stock_count,
        max_sectors=args.max_sectors,
        selected_sectors=args.sector,
    )
    if not candidates:
        raise ValueError("No sector candidates met the selection thresholds.")
    print("Selected sectors:", ", ".join(candidate.sector for candidate in candidates))

    run_and_log(
        [str(ROOT / "venv" / "bin" / "python"), "scripts/run_build_dataset.py", "--market", "VN"],
        log_dir / "build_vn_dataset.log",
    )

    rows: list[dict[str, object]] = []
    run_name_suffix = f"_{slugify_sector(args.run_name_suffix)}" if args.run_name_suffix else ""
    for candidate in candidates:
        slug = slugify_sector(candidate.sector)
        run_name = f"sector_{slug}_{args.target_mode}_w{args.window_size}{run_name_suffix}"
        print(
            f"[sector] {candidate.sector} | eligible={candidate.eligible_stock_count} | "
            f"supported={candidate.supported_stock_count} | top={','.join(candidate.top_stocks)}"
        )

        run_and_log(
            build_train_command(args, run_name, candidate.sector),
            log_dir / f"{run_name}_train.log",
        )
        run_and_log(
            [
                str(ROOT / "venv" / "bin" / "python"),
                "src/backtesting/threshold_backtest.py",
                str(RUN_BASE / run_name),
                "--non-overlap",
            ],
            log_dir / f"{run_name}_backtest.log",
        )
        run_and_log(
            [
                str(ROOT / "venv" / "bin" / "python"),
                "src/reporting/update_run_reports.py",
                str(RUN_BASE / run_name),
            ],
            log_dir / f"{run_name}_report.log",
        )
        rows.append(collect_run_row(RUN_BASE / run_name))

    summary_df = pd.DataFrame(rows).sort_values(
        ["best_lstm_test_rel_score", "best_lstm_val_rel_score", "lstm_backtest_final_equity"],
        ascending=[False, False, False],
    )
    summary_path = log_dir / "sector_batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
