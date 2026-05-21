from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_RUN = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "broad_signmag_portable_no_identity_20260428_allvn_r01"
)
BASE_CONFIG = BASE_RUN / "reports" / "core" / "config.json"
DATA_PATH = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
WORK_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "tail_confidence_lstm_ablation"
)
GOLD_ROOT = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_confidence_lstm_ablation"


@dataclass(frozen=True)
class Candidate:
    name: str
    loss: str
    epochs: int
    patience: int
    lr: float
    extra_args: tuple[str, ...]


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("base_loaded_no_finetune", "rel_score", 0, 1, 0.0001, ()),
    Candidate(
        "rel_sharp_finetune",
        "rel_score_sharp",
        8,
        3,
        0.0001,
        (
            "--rel-score-large-move-quantile",
            "0.75",
            "--rel-score-directional-penalty",
            "0.6",
            "--rel-score-confidence-penalty",
            "0.5",
            "--rel-score-confidence-ratio",
            "0.30",
        ),
    ),
    Candidate(
        "rel_weighted_finetune",
        "rel_score_weighted",
        8,
        3,
        0.0001,
        (
            "--rel-score-weighted-high-quantile",
            "0.8",
            "--rel-score-weighted-high-weight",
            "3.0",
            "--rel-score-weighted-base-weight",
            "1.0",
        ),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tail-confidence fine-tune seed sweep.")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--only-candidates", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_base_feature_columns() -> list[str]:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    columns = config.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Could not read feature_columns from {BASE_CONFIG}")
    return [str(column) for column in columns]


def run_name(candidate: Candidate, seed: int) -> str:
    if candidate.name == "base_loaded_no_finetune":
        return f"base{seed}_loaded_no_finetune_e0"
    suffix = "sharp" if candidate.name == "rel_sharp_finetune" else "weighted"
    return f"rel_{suffix}_finetune_base{seed}_lr1e4_e8"


def build_train_command(candidate: Candidate, seed: int, python_bin: str, feature_columns: list[str]) -> list[str]:
    initial_model = BASE_RUN / f"model_seed_{seed}.keras"
    if not initial_model.exists():
        raise FileNotFoundError(initial_model)
    cmd = [
        python_bin,
        "-m",
        "src.models.training.pipeline",
        "--output-dir",
        str(WORK_ROOT),
        "--run-name",
        run_name(candidate, seed),
        "--market",
        "VN",
        "--target-mode",
        "return",
        "--train-end-date",
        "2020-03-31",
        "--val-end-date",
        "2022-11-15",
        "--feature-columns",
        ",".join(feature_columns),
        "--window-size",
        "15",
        "--lstm-units",
        "64,32",
        "--dropout",
        "0.05",
        "--lr",
        str(candidate.lr),
        "--loss",
        candidate.loss,
        "--batch-size",
        "64",
        "--epochs",
        str(candidate.epochs),
        "--patience",
        str(candidate.patience),
        "--sequence-normalization",
        "none",
        "--lstm-seeds",
        str(seed),
        "--sample-weight-mode",
        "magnitude",
        "--sample-weight-strength",
        "1.5",
        "--sample-weight-quantile",
        "0.75",
        "--sample-weight-clip",
        "3.0",
        "--signmag-signed-loss-weight",
        "1.5",
        "--signmag-sign-loss-weight",
        "0.15",
        "--signmag-magnitude-loss-weight",
        "0.35",
        "--data-path",
        str(DATA_PATH),
        "--target-normalizer",
        "volatility_20",
        "--feature-phase",
        "paper_v1",
        "--huber-delta",
        "0.01",
        "--disable-stock-identity",
        "--initial-model-path",
        str(initial_model),
    ]
    cmd.extend(candidate.extra_args)
    return cmd


def run_if_needed(cmd: list[str], run_dir: Path, *, force: bool, dry_run: bool) -> None:
    prediction_path = run_dir / "reports" / "core" / "predictions.csv"
    if prediction_path.exists() and not force:
        print(f"Skip existing run: {run_dir.name}")
        return
    print(f"Run: {run_dir.name}")
    print(" ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Training failed for {run_dir.name} with exit code {result.returncode}")


def evaluate_if_needed(run_dir: Path, artifact_name: str, python_bin: str, *, force: bool, dry_run: bool) -> Path:
    output_dir = GOLD_ROOT / artifact_name
    summary_path = output_dir / "tail_error_summary.csv"
    if summary_path.exists() and not force:
        print(f"Skip existing evaluation: {artifact_name}")
        return summary_path
    cmd = [
        python_bin,
        str(ROOT / "experiments" / "analysis" / "evaluate_tail_error_predictions.py"),
        "--prediction-file",
        str(run_dir / "reports" / "core" / "predictions.csv"),
        "--prediction-format",
        "core_predictions",
        "--artifact-name",
        artifact_name,
        "--output-dir",
        str(output_dir),
    ]
    print(f"Evaluate: {artifact_name}")
    print(" ".join(cmd))
    if dry_run:
        return summary_path
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Evaluation failed for {artifact_name} with exit code {result.returncode}")
    return summary_path


def collect_row(candidate: Candidate, seed: int, run_dir: Path, tail_summary_path: Path) -> dict[str, object]:
    metrics = json.loads((run_dir / "reports" / "core" / "metrics.json").read_text(encoding="utf-8"))
    model_name = f"lstm_seed_{seed}"
    metric_model_name = model_name if model_name in metrics else "lstm"
    val_metrics = metrics[metric_model_name]["val"]
    tail = pd.read_csv(tail_summary_path)
    tail_model_name = model_name if tail["model"].eq(model_name).any() else "lstm"
    val = tail[
        tail["model"].eq(tail_model_name) & tail["split"].eq("val") & tail["scope"].eq("full_split")
    ].iloc[0]
    segment = tail[
        tail["model"].eq(tail_model_name) & tail["scope"].eq("segment_2017_d200_250")
    ].iloc[0]
    return {
        "candidate": candidate.name,
        "seed": seed,
        "run_name": run_dir.name,
        "official_val_rel_score": float(val_metrics["rel_score"]),
        "official_val_abs_loss": float(val_metrics["abs_loss"]),
        "official_val_directional_accuracy": float(val_metrics["directional_accuracy"]),
        "val_q90_abs_error": float(val["q90_abs_error"]),
        "val_daily_q90_median": float(val["daily_q90_abs_error_median"]),
        "val_daily_q90_q90": float(val["daily_q90_abs_error_q90"]),
        "val_spike_rate": float(val["spike_rate"]),
        "val_prediction_abs_q90": float(val["prediction_abs_q90"]),
        "val_pred_actual_abs_q90_ratio": float(val["prediction_actual_abs_q90_ratio"]),
        "segment_rel_score_eval": float(segment["rel_score"]),
        "segment_q90_abs_error": float(segment["q90_abs_error"]),
        "segment_daily_q90_median": float(segment["daily_q90_abs_error_median"]),
        "segment_spike_rate": float(segment["spike_rate"]),
        "segment_pred_actual_abs_q90_ratio": float(segment["prediction_actual_abs_q90_ratio"]),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in frame.columns]
        rows = [[str(value) for value in row] for row in frame.to_numpy()]
        widths = [
            max(len(column), *(len(row[idx]) for row in rows)) if rows else len(column)
            for idx, column in enumerate(columns)
        ]
        header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
        separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
        body = [
            "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
            for row in rows
        ]
        return "\n".join([header, separator, *body])


def write_summary(rows: list[dict[str, object]]) -> None:
    GOLD_ROOT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["candidate", "seed"], kind="stable")
    frame.to_csv(GOLD_ROOT / "tail_confidence_seed_sweep_comparison.csv", index=False)
    aggregate = (
        frame.groupby("candidate", sort=True)
        .agg(
            seeds=("seed", "count"),
            official_val_rel_score_mean=("official_val_rel_score", "mean"),
            official_val_rel_score_min=("official_val_rel_score", "min"),
            val_q90_abs_error_mean=("val_q90_abs_error", "mean"),
            val_spike_rate_mean=("val_spike_rate", "mean"),
            val_pred_actual_abs_q90_ratio_mean=("val_pred_actual_abs_q90_ratio", "mean"),
            official_val_directional_accuracy_mean=("official_val_directional_accuracy", "mean"),
            segment_rel_score_mean=("segment_rel_score_eval", "mean"),
            segment_q90_abs_error_mean=("segment_q90_abs_error", "mean"),
            segment_pred_actual_abs_q90_ratio_mean=("segment_pred_actual_abs_q90_ratio", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(GOLD_ROOT / "tail_confidence_seed_sweep_aggregate.csv", index=False)
    display_frame = frame.copy()
    display_agg = aggregate.copy()
    for table in (display_frame, display_agg):
        for column in table.columns:
            if column not in {"candidate", "run_name"}:
                table[column] = table[column].map(lambda value: f"{value:.5f}" if isinstance(value, float) else value)
    lines = [
        "# Tail-Confidence Seed Sweep",
        "",
        "Scope: VN train/validation development only. Holdout/test is not used.",
        "",
        "## Aggregate",
        "",
        markdown_table(display_agg),
        "",
        "## Per Seed",
        "",
        markdown_table(display_frame),
        "",
        "## Read",
        "",
        "- Promote only if the gain survives across seeds, not only seed 52.",
        "- `rel_weighted_finetune` is the rel_score challenger.",
        "- `rel_sharp_finetune` is the tail-confidence/uptrend challenger.",
    ]
    (GOLD_ROOT / "seed_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    only = {item.strip() for item in args.only_candidates.split(",") if item.strip()} if args.only_candidates else None
    candidates = [candidate for candidate in CANDIDATES if only is None or candidate.name in only]
    feature_columns = load_base_feature_columns()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for seed in seeds:
            name = run_name(candidate, seed)
            run_dir = WORK_ROOT / name
            cmd = build_train_command(candidate, seed, args.python_bin, feature_columns)
            run_if_needed(cmd, run_dir, force=args.force, dry_run=args.dry_run)
            if args.dry_run:
                continue
            tail_summary_path = evaluate_if_needed(run_dir, name, args.python_bin, force=args.force, dry_run=False)
            rows.append(collect_row(candidate, seed, run_dir, tail_summary_path))
    if rows:
        write_summary(rows)
    print(json.dumps({"runs": len(rows), "summary": str(GOLD_ROOT / "seed_sweep_summary.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
