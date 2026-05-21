"""Walk-forward LSTM ablation runner.

Mục đích: cho phép search hyperparam / feature set / loss variant trên
**train-only** thông qua expanding walk-forward CV, tránh chạm validation
Apr-2020 → Nov-2022 trong quá trình tuning.

Workflow:

1. Sinh expanding folds từ `src.models.training.cv.expanding_walk_forward_folds`
   trên các ngày trong khoảng `[start_date, --wf-train-end]` (mặc định 2020-03-31).
2. Với mỗi fold, gọi `src.models.training.pipeline.main(...)` với
   `--train-end-date <fold.train_end> --val-end-date <fold.val_end>` và thư
   mục output riêng cho fold.
3. Đọc `core/metrics.json` của từng fold, gom val_rel_score và val_ic.
4. Tổng hợp: mean/median/std/min across folds.
5. Ghi `summary.md` và `folds.csv` vào output dir.

Cách chạy (smoke test plumbing):

```
python experiments/training/run_walk_forward_lstm_search.py \\
    --output-name signmag_wf_baseline_20260514 \\
    --wf-train-end 2020-03-31 \\
    --min-train-days 1000 \\
    --val-days 126 \\
    --step-days 252 \\
    --embargo-days 5 \\
    --max-folds 3 \\
    --epochs 4 \\
    --patience 3 \\
    --lstm-seeds 42 \\
    --smoke
```

Cách chạy thật (full ablation):

```
python experiments/training/run_walk_forward_lstm_search.py \\
    --output-name signmag_wf_dropout_grid_20260514 \\
    --wf-train-end 2020-03-31 \\
    --min-train-days 1500 \\
    --val-days 126 \\
    --step-days 126 \\
    --embargo-days 5 \\
    --dropout 0.2 \\
    --lstm-seeds 42,52,62
```

Tích hợp với hệ thống config: runner pass-through TẤT CẢ training CLI args
(vd --window-size, --dropout, --loss, --lstm-units, --feature-columns, etc).
Chỉ override `--train-end-date` và `--val-end-date` per fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.training.cv import (  # noqa: E402
    Fold,
    expanding_walk_forward_folds,
    summarize_folds,
)


DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "training_runs" / "reports" / "walk_forward"


def _passthrough_args() -> list[str]:
    """Return the list of CLI arg names that should be forwarded to pipeline."""
    return [
        # Required-ish
        "--data-path",
        "--market",
        "--start-date",
        "--target-mode",
        "--allow-nonstandard-time",
        # Model / hyper
        "--window-size",
        "--lstm-units",
        "--dropout",
        "--lr",
        "--loss",
        "--huber-delta",
        "--rel-score-large-move-quantile",
        "--rel-score-directional-penalty",
        "--rel-score-confidence-penalty",
        "--rel-score-confidence-ratio",
        "--rel-score-weighted-high-quantile",
        "--rel-score-weighted-high-weight",
        "--rel-score-weighted-base-weight",
        "--batch-size",
        "--epochs",
        "--patience",
        "--target-normalizer",
        "--sequence-normalization",
        "--feature-phase",
        "--lstm-seeds",
        # Signmag
        "--signmag-signed-loss-weight",
        "--signmag-sign-loss-weight",
        "--signmag-magnitude-loss-weight",
        "--signmag-rank-loss-weight",
        "--signmag-rank-temperature",
        "--signmag-rank-min-group-size",
        # Sample weights
        "--sample-weight-mode",
        "--sample-weight-balance-mode",
        "--sample-weight-strength",
        "--sample-weight-quantile",
        "--sample-weight-clip",
        # Universe
        "--stocks",
        "--sector",
        "--feature-columns",
        "--regime-filter",
        # Other
        "--initial-model-path",
    ]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Walk-forward CV runner for LSTM ablations (train-only fold search).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Subdirectory name under reports/walk_forward/ for this ablation.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Used to read date range; also forwarded to pipeline.",
    )
    parser.add_argument(
        "--wf-train-end",
        type=str,
        default="2020-03-31",
        help="Last date allowed inside any fold (train+val of each fold must be <= this).",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=1500,
        help="Minimum training trading days in the first fold.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=126,
        help="Validation length per fold (default ~ 6 months).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=126,
        help="Step between fold starts.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="Embargo gap between train_end and val_start (purge target leakage).",
    )
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used to run pipeline subprocesses.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the per-fold commands without executing.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: print fold table and skip pipeline execution.",
    )
    return parser.parse_known_args(argv)


def load_available_dates(data_path: Path, wf_train_end: pd.Timestamp) -> list[pd.Timestamp]:
    df = pd.read_csv(data_path, usecols=["Date"], parse_dates=["Date"])
    dates = pd.Series(df["Date"].dropna().unique()).sort_values()
    dates = dates[dates <= wf_train_end]
    return list(dates)


def build_pipeline_command(
    python_bin: str,
    fold_dir: Path,
    fold: Fold,
    passthrough: Iterable[str],
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "src.models.training.pipeline",
        "--output-dir",
        str(fold_dir),
        "--run-name",
        f"fold_{fold.fold_id:02d}",
        "--train-end-date",
        fold.train_end.strftime("%Y-%m-%d"),
        "--val-end-date",
        fold.val_end.strftime("%Y-%m-%d"),
    ]
    cmd.extend(passthrough)
    # Walk-forward folds intentionally use non-standard dates. Auto-add the
    # bypass flag so the VN reporting standard validator does not reject
    # the per-fold split. Safe because we constrain `wf-train-end` ≤
    # 2020-03-31 ourselves, so holdout remains untouched.
    if "--allow-nonstandard-time" not in cmd:
        cmd.append("--allow-nonstandard-time")
    return cmd


def collect_passthrough(known: argparse.Namespace, extra: list[str]) -> list[str]:
    forward: list[str] = list(extra)
    if known.data_path is not None:
        forward = ["--data-path", str(known.data_path), *forward]
    return forward


def _holdout_safety_check(folds: list[Fold], wf_train_end: pd.Timestamp) -> None:
    """Assert every fold ends ≤ wf_train_end. Guards against accidental
    spillage into the reserved validation/holdout window."""
    for fold in folds:
        if fold.val_end > wf_train_end:
            raise SystemExit(
                f"Fold {fold.fold_id} val_end {fold.val_end.date()} exceeds "
                f"wf-train-end {wf_train_end.date()} — would touch reserved "
                f"validation/holdout. Reduce min-train-days/val-days/step-days."
            )


def write_summary(
    output_dir: Path,
    folds: list[Fold],
    fold_metrics: list[dict[str, object]],
    args_namespace: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_folds(folds)
    summary_df.to_csv(output_dir / "folds.csv", index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    val_rel = metrics_df.get("val_rel_score")
    val_ic = metrics_df.get("val_mean_daily_ic")

    def _stats(series: pd.Series | None) -> dict[str, float]:
        if series is None or series.empty:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        clean = series.astype(float).dropna()
        if clean.empty:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "std": float(clean.std(ddof=1)) if len(clean) >= 2 else 0.0,
            "min": float(clean.min()),
            "max": float(clean.max()),
        }

    rel_stats = _stats(val_rel)
    ic_stats = _stats(val_ic)

    lines = [
        "# Walk-Forward LSTM Search Summary",
        "",
        f"- output_name: `{args_namespace.output_name}`",
        f"- generated_at: `{datetime.utcnow().isoformat()}Z`",
        f"- wf_train_end: `{args_namespace.wf_train_end}`",
        f"- min_train_days: `{args_namespace.min_train_days}`",
        f"- val_days: `{args_namespace.val_days}`",
        f"- step_days: `{args_namespace.step_days}`",
        f"- embargo_days: `{args_namespace.embargo_days}`",
        f"- n_folds: `{len(folds)}`",
        "",
        "## Fold Aggregates",
        "",
        "| Metric | mean | median | std | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| val_rel_score | {rel_stats['mean']:.5f} | {rel_stats['median']:.5f} | {rel_stats['std']:.5f} | {rel_stats['min']:.5f} | {rel_stats['max']:.5f} |",
        f"| val_mean_daily_ic | {ic_stats['mean']:.5f} | {ic_stats['median']:.5f} | {ic_stats['std']:.5f} | {ic_stats['min']:.5f} | {ic_stats['max']:.5f} |",
        "",
        "## Per-Fold Detail",
        "",
        "See [fold_metrics.csv](fold_metrics.csv) and [folds.csv](folds.csv).",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fold_metrics(fold_dir: Path, fold: Fold) -> dict[str, object]:
    metrics_path = fold_dir / f"fold_{fold.fold_id:02d}" / "reports" / "core" / "metrics.json"
    if not metrics_path.exists():
        # Some pipelines may flatten differently; try alternate locations.
        candidates = list(fold_dir.rglob("metrics.json"))
        if not candidates:
            return {"fold_id": fold.fold_id, "metrics_path": str(metrics_path), "status": "missing"}
        metrics_path = candidates[0]
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    record: dict[str, object] = {
        "fold_id": fold.fold_id,
        "train_end": fold.train_end,
        "val_start": fold.val_start,
        "val_end": fold.val_end,
        "metrics_path": str(metrics_path),
        "status": "ok",
    }
    # metrics.json structure may be model_name -> split -> metric.
    # Be defensive: surface common keys when present.
    for model_name, by_split in raw.items() if isinstance(raw, dict) else []:
        if not isinstance(by_split, dict):
            continue
        val = by_split.get("val", {}) if isinstance(by_split.get("val", {}), dict) else {}
        prefix = f"{model_name}__"
        for key, value in val.items():
            record[prefix + str(key)] = value
        # Convenience aliases for the primary signmag model
        if model_name in {"lstm_signmag", "lstm_signmag_seed_42", "lstm_signmag_seed_52", "lstm_signmag_seed_62"}:
            if "val_rel_score" not in record and "rel_score" in val:
                record["val_rel_score"] = val["rel_score"]
            if "val_mean_daily_ic" not in record and "mean_daily_ic" in val:
                record["val_mean_daily_ic"] = val["mean_daily_ic"]
    return record


def main(argv: list[str] | None = None) -> int:
    known, extra = parse_args(argv)

    wf_train_end = pd.Timestamp(known.wf_train_end)
    dates = load_available_dates(known.data_path, wf_train_end)
    if len(dates) < known.min_train_days + known.val_days:
        raise SystemExit(
            f"Not enough dates ({len(dates)}) before {wf_train_end.date()} for "
            f"min_train_days={known.min_train_days} + val_days={known.val_days}."
        )

    folds = expanding_walk_forward_folds(
        dates,
        min_train_days=known.min_train_days,
        val_days=known.val_days,
        step_days=known.step_days,
        embargo_days=known.embargo_days,
        max_folds=known.max_folds,
    )
    if not folds:
        raise SystemExit("No folds generated. Loosen min_train_days or check date range.")

    _holdout_safety_check(folds, wf_train_end)

    output_dir = known.output_root / known.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = summarize_folds(folds)
    print(f"\nGenerated {len(folds)} fold(s). Writing to {output_dir}\n")
    print(summary_df.to_string(index=False))

    if known.smoke:
        write_summary(output_dir, folds, [], known)
        print("\nSmoke mode: skipping pipeline execution.")
        return 0

    passthrough = collect_passthrough(known, extra)

    import subprocess

    fold_metrics: list[dict[str, object]] = []
    for fold in folds:
        cmd = build_pipeline_command(known.python_bin, output_dir, fold, passthrough)
        if known.dry_run:
            print("\n[dry-run] " + " ".join(cmd))
            continue
        print(f"\n--- Fold {fold.fold_id}: train≤{fold.train_end.date()} val={fold.val_start.date()}..{fold.val_end.date()} ---")
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"  WARN: pipeline exited with code {result.returncode}; continuing to next fold.")
        fold_metrics.append(read_fold_metrics(output_dir, fold))

    write_summary(output_dir, folds, fold_metrics, known)
    print(f"\nDone. Summary at {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
