from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.config import load_experiment_config
from fk_lstm_classifier.experiment import run_experiment
from fk_lstm_classifier.reporting import render_dashboard as render_fk_dashboard
from fk_lstm_classifier.reporting import summarize_fk_run
from render_run_index import render_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch of FK LSTM experiments from JSON configs.")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Path to a JSON experiment config. Repeat to run multiple configs.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs") / "fk_lstm",
        help="Directory containing JSON configs. Used when --config is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used when loading configs from --config-dir.",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("artifacts") / "run_index" / "fk_batch_dashboard.html",
        help="Output HTML path for the aggregate comparison page.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional override for config.output_dir, useful for Google Drive or shared scratch storage.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when its dashboard already exists under the resolved output directory.",
    )
    return parser.parse_args()


def _resolve_config_paths(args: argparse.Namespace) -> list[Path]:
    if args.config:
        return [Path(path).expanduser().resolve() for path in args.config]

    config_dir = args.config_dir.expanduser().resolve()
    paths = sorted(config_dir.glob(args.pattern))
    if not paths:
        raise ValueError(f"No config files matched {args.pattern} under {config_dir}")
    return paths


def main() -> None:
    args = parse_args()
    config_paths = _resolve_config_paths(args)

    entries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        if args.output_root is not None:
            config = replace(config, output_dir=args.output_root.expanduser().resolve())

        run_name = config.run_name or Path(config_path).stem
        run_dir = config.output_dir / run_name
        if args.skip_existing and (run_dir / "dashboard.html").exists():
            summary = summarize_fk_run(run_dir)
            entries.append(
                {
                    "label": run_name,
                    "run_dir": run_dir,
                    "dashboard_path": run_dir / "dashboard.html",
                    **summary,
                }
            )
            rows.append(
                {
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "dashboard_path": str(run_dir / "dashboard.html"),
                    "status": "skipped_existing",
                    **summary,
                }
            )
            continue

        result = run_experiment(config)
        render_fk_dashboard(result.run_dir)
        summary = summarize_fk_run(result.run_dir)
        entries.append(
            {
                "label": run_name,
                "run_dir": result.run_dir,
                "dashboard_path": result.dashboard_path,
                **summary,
            }
        )
        rows.append(
            {
                "config_path": str(config_path),
                "run_dir": str(result.run_dir),
                "dashboard_path": str(result.dashboard_path),
                "status": "completed",
                **summary,
            }
        )

    index_path = render_index(entries, args.index_output)
    summary_path = args.index_output.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nBatch index saved to: {index_path}")
    print(f"Batch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
