from __future__ import annotations

import argparse
import csv
import itertools
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

from src.models.report_layout import resolve_run_artifact
from src.models.training_recipe import DEFAULT_SEARCH_SUMMARY_PATH


RUN_BASE = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs"
DEFAULT_LOG_BASE = RUN_BASE / "mini_group_logs"
DEFAULT_CONTEXT_FEATURES = "alpha_sector,vingroup_momentum,vnindex_return,a_d_ratio,day_of_week,rsi_14"


@dataclass(frozen=True)
class MiniGroupCandidate:
    sector: str
    group_index: int
    stocks: tuple[str, ...]
    mean_best_val_rel_score: float
    mean_best_test_rel_score: float
    mean_pairwise_overlap: float
    support_count: int
    score: float


def parse_int_list(value: str) -> tuple[int, ...]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return tuple(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mini-group VN sector batches from stock search summary.")
    parser.add_argument("--search-summary", type=Path, default=DEFAULT_SEARCH_SUMMARY_PATH)
    parser.add_argument("--target-mode", choices=["return", "return_3d", "return_5d"], default="return")
    parser.add_argument("--min-stock-val-rel-score", type=float, default=0.03)
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--groups-per-sector", type=int, default=2)
    parser.add_argument("--group-sizes", type=parse_int_list, default=(3, 4))
    parser.add_argument("--max-sectors", type=int, default=3)
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
    parser.add_argument("--sample-weight-mode", choices=["none", "magnitude"], default="magnitude")
    parser.add_argument("--sample-weight-strength", type=float, default=0.8)
    parser.add_argument("--sample-weight-quantile", type=float, default=0.8)
    parser.add_argument("--sample-weight-clip", type=float, default=2.5)
    parser.add_argument("--signmag-signed-loss-weight", type=float, default=2.0)
    parser.add_argument("--signmag-sign-loss-weight", type=float, default=0.10)
    parser.add_argument("--signmag-magnitude-loss-weight", type=float, default=0.25)
    parser.add_argument("--enable-attention-family", action="store_true")
    parser.add_argument("--enable-quantile-family", action="store_true")
    parser.add_argument("--enable-fk-benchmark", action="store_true")
    parser.add_argument("--max-stock-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--log-base", type=Path, default=DEFAULT_LOG_BASE)
    parser.add_argument("--run-name-suffix", default=None)
    parser.add_argument("--sector", action="append", default=None)
    return parser.parse_args()


def slugify_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
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
    return slug.strip("_") or "group"


def parse_feature_combo(raw_value: object) -> tuple[str, ...]:
    if not isinstance(raw_value, str):
        return ()
    return tuple(dict.fromkeys(item.strip() for item in raw_value.split(",") if item.strip()))


def load_search_summary(search_summary_path: Path) -> pd.DataFrame:
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
            row["feature_set"] = parse_feature_combo(row.get("best_by_val"))
            rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError(f"No usable rows found in search summary: {search_summary_path}")
    return summary


def jaccard_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set.union(right_set)
    if not union:
        return 0.0
    return float(len(left_set.intersection(right_set)) / len(union))


def mean_pairwise_overlap(feature_sets: list[tuple[str, ...]]) -> float:
    if len(feature_sets) < 2:
        return 0.0
    overlaps = [
        jaccard_overlap(feature_sets[left_idx], feature_sets[right_idx])
        for left_idx in range(len(feature_sets))
        for right_idx in range(left_idx + 1, len(feature_sets))
    ]
    return float(sum(overlaps) / len(overlaps)) if overlaps else 0.0


def stock_overlap_ratio(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(set(left).intersection(right))
    return float(overlap / min(len(left), len(right)))


def rank_group_score(group_df: pd.DataFrame) -> tuple[float, float, float, int]:
    mean_best_val = float(group_df["best_val_rel_score"].mean())
    mean_best_test = float(group_df["best_test_rel_score"].mean())
    overlap = mean_pairwise_overlap(group_df["feature_set"].tolist())
    support_count = int((group_df["best_test_rel_score"] >= 0.03).sum())
    score = mean_best_val + 0.5 * mean_best_test + 0.03 * support_count + 0.02 * overlap
    return score, mean_best_val, mean_best_test, support_count


def build_group_candidates(
    summary: pd.DataFrame,
    *,
    min_stock_val_rel_score: float,
    candidate_pool: int,
    group_sizes: tuple[int, ...],
    groups_per_sector: int,
    max_sectors: int,
    selected_sectors: list[str] | None,
    max_stock_overlap_ratio: float,
) -> list[MiniGroupCandidate]:
    if selected_sectors:
        summary = summary[summary["sector"].isin(selected_sectors)].copy()
        if summary.empty:
            raise ValueError("None of the requested sectors exist in the search summary.")

    sector_rows: list[tuple[str, list[MiniGroupCandidate]]] = []
    for sector, sector_df in summary.groupby("sector"):
        eligible = sector_df[sector_df["best_val_rel_score"] >= min_stock_val_rel_score].copy()
        if len(eligible) < min(group_sizes):
            continue
        eligible = eligible.sort_values(
            ["best_val_rel_score", "best_test_rel_score", "stock"],
            ascending=[False, False, True],
            kind="stable",
        ).head(max(1, candidate_pool))

        raw_candidates: list[MiniGroupCandidate] = []
        for group_size in sorted(set(group_sizes)):
            if len(eligible) < group_size:
                continue
            for combo_idx, combo in enumerate(itertools.combinations(range(len(eligible)), group_size)):
                group_df = eligible.iloc[list(combo)].copy()
                score, mean_best_val, mean_best_test, support_count = rank_group_score(group_df)
                raw_candidates.append(
                    MiniGroupCandidate(
                        sector=str(sector),
                        group_index=combo_idx,
                        stocks=tuple(group_df["stock"].astype(str).tolist()),
                        mean_best_val_rel_score=mean_best_val,
                        mean_best_test_rel_score=mean_best_test,
                        mean_pairwise_overlap=mean_pairwise_overlap(group_df["feature_set"].tolist()),
                        support_count=support_count,
                        score=score,
                    )
                )

        raw_candidates.sort(
            key=lambda item: (
                item.score,
                item.mean_best_val_rel_score,
                item.mean_best_test_rel_score,
                item.mean_pairwise_overlap,
                item.stocks,
            ),
            reverse=True,
        )

        selected: list[MiniGroupCandidate] = []
        for candidate in raw_candidates:
            if any(stock_overlap_ratio(candidate.stocks, existing.stocks) > max_stock_overlap_ratio for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= groups_per_sector:
                break
        if selected:
            sector_rows.append((str(sector), selected))

    sector_rows.sort(
        key=lambda item: (
            max(candidate.score for candidate in item[1]),
            item[0],
        ),
        reverse=True,
    )

    flattened: list[MiniGroupCandidate] = []
    for _, candidates in sector_rows[: max(1, max_sectors)]:
        flattened.extend(candidates)
    return flattened


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


def rank_lstm_models(metrics: dict[str, object], split_primary: str, split_secondary: str) -> list[str]:
    lstm_models = [name for name in metrics if name.startswith("lstm")]
    return sorted(
        lstm_models,
        key=lambda name: (
            metrics.get(name, {}).get(split_primary, {}).get("rel_score", float("-inf")),
            metrics.get(name, {}).get(split_secondary, {}).get("rel_score", float("-inf")),
        ),
        reverse=True,
    )


def collect_run_row(run_dir: Path) -> dict[str, object]:
    metrics_path = resolve_run_artifact(run_dir, "metrics.json", "core")
    config_path = resolve_run_artifact(run_dir, "config.json", "core")
    if not metrics_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Run artifacts missing in {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    backtest_path = resolve_backtest_summary(run_dir, str(config.get("target_mode")))
    backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}

    ranked_by_val = rank_lstm_models(metrics, "val", "test")
    ranked_by_test = rank_lstm_models(metrics, "test", "val")
    best_by_val_model = ranked_by_val[0] if ranked_by_val else None
    best_by_test_model = ranked_by_test[0] if ranked_by_test else None

    best_by_val_test = metrics.get(best_by_val_model, {}).get("test", {}) if best_by_val_model else {}
    best_by_val_val = metrics.get(best_by_val_model, {}).get("val", {}) if best_by_val_model else {}
    best_by_test_test = metrics.get(best_by_test_model, {}).get("test", {}) if best_by_test_model else {}
    best_by_val_backtest = backtest.get(best_by_val_model, {}) if best_by_val_model else {}

    return {
        "run_name": run_dir.name,
        "sector": config.get("sector"),
        "stocks": config.get("stocks"),
        "feature_columns": ",".join(config.get("feature_columns", [])),
        "best_by_val_model": best_by_val_model,
        "best_by_val_val_rel_score": best_by_val_val.get("rel_score"),
        "best_by_val_test_rel_score": best_by_val_test.get("rel_score"),
        "best_by_val_test_directional_accuracy": best_by_val_test.get("directional_accuracy"),
        "best_by_val_backtest_final_equity": best_by_val_backtest.get("final_equity"),
        "best_by_val_backtest_trade_count": best_by_val_backtest.get("trade_count"),
        "best_by_val_backtest_threshold": best_by_val_backtest.get("threshold"),
        "best_by_test_model": best_by_test_model,
        "best_by_test_test_rel_score": best_by_test_test.get("rel_score"),
        "run_dir": str(run_dir),
    }


def build_train_command(args: argparse.Namespace, run_name: str, sector: str, stocks: tuple[str, ...]) -> list[str]:
    cmd = [
        str(ROOT / "venv" / "bin" / "python"),
        "scripts/run_train.py",
        "--target-mode",
        args.target_mode,
        "--sector",
        sector,
        "--stocks",
        ",".join(stocks),
        "--feature-selection-mode",
        "search_summary",
        "--stock-search-summary",
        str(args.search_summary),
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

    summary = load_search_summary(args.search_summary)
    candidates = build_group_candidates(
        summary,
        min_stock_val_rel_score=args.min_stock_val_rel_score,
        candidate_pool=args.candidate_pool,
        group_sizes=args.group_sizes,
        groups_per_sector=args.groups_per_sector,
        max_sectors=args.max_sectors,
        selected_sectors=args.sector,
        max_stock_overlap_ratio=args.max_stock_overlap_ratio,
    )
    if not candidates:
        raise ValueError("No mini-group candidates met the selection thresholds.")

    print("Selected mini-groups:")
    for candidate in candidates:
        print(
            f"  [{candidate.sector}] stocks={','.join(candidate.stocks)} | "
            f"score={candidate.score:.4f} | mean_val={candidate.mean_best_val_rel_score:.4f} | "
            f"mean_test={candidate.mean_best_test_rel_score:.4f} | overlap={candidate.mean_pairwise_overlap:.4f}"
        )

    run_and_log(
        [str(ROOT / "venv" / "bin" / "python"), "scripts/run_build_dataset.py", "--market", "VN"],
        log_dir / "build_vn_dataset.log",
    )

    rows: list[dict[str, object]] = []
    run_name_suffix = f"_{slugify_text(args.run_name_suffix)}" if args.run_name_suffix else ""
    for index, candidate in enumerate(candidates, start=1):
        sector_slug = slugify_text(candidate.sector)
        run_name = f"mini_{sector_slug}_g{index:02d}_{args.target_mode}_w{args.window_size}{run_name_suffix}"
        print(
            f"[mini-group {index}/{len(candidates)}] {candidate.sector} | "
            f"stocks={','.join(candidate.stocks)}"
        )
        run_and_log(
            build_train_command(args, run_name, candidate.sector, candidate.stocks),
            log_dir / f"{run_name}_train.log",
        )
        run_and_log(
            [
                str(ROOT / "venv" / "bin" / "python"),
                "src/models/backtest_threshold.py",
                str(RUN_BASE / run_name),
                "--non-overlap",
            ],
            log_dir / f"{run_name}_backtest.log",
        )
        run_and_log(
            [
                str(ROOT / "venv" / "bin" / "python"),
                "src/models/update_run_reports.py",
                str(RUN_BASE / run_name),
            ],
            log_dir / f"{run_name}_report.log",
        )
        row = collect_run_row(RUN_BASE / run_name)
        row.update(
            {
                "candidate_score": candidate.score,
                "candidate_mean_val_rel_score": candidate.mean_best_val_rel_score,
                "candidate_mean_test_rel_score": candidate.mean_best_test_rel_score,
                "candidate_feature_overlap": candidate.mean_pairwise_overlap,
                "candidate_support_count": candidate.support_count,
            }
        )
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        ["best_by_val_test_rel_score", "best_by_val_val_rel_score", "best_by_val_backtest_final_equity"],
        ascending=[False, False, False],
    )
    summary_path = log_dir / "mini_group_batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
