from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.config import load_experiment_config
from fk_lstm_classifier.experiment import run_experiment
from fk_lstm_classifier.reporting import summarize_fk_run
from render_run_index import render_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VN Fischer-Krauss pack and render one VN-only comparison report.")
    parser.add_argument(
        "--preset",
        choices=["fast", "full"],
        default="fast",
        help="Choose between the lighter fast walk-forward suite and the full VN walk-forward suite.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "fk_lstm_vn_pack",
        help="Root directory for VN run outputs.",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path("artifacts") / "run_index" / "vn_walkforward_report.html",
        help="Output HTML path for the VN comparison page.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose dashboards already exist under the output root.",
    )
    return parser.parse_args()


def _config_paths(preset: str) -> list[Path]:
    if preset == "full":
        return [
            Path("configs/fk_lstm/vn_baseline.json"),
            Path("configs/fk_lstm/vn_attention.json"),
        ]
    return [
        Path("configs/fk_lstm_fast/vn_baseline_fast.json"),
        Path("configs/fk_lstm_fast/vn_attention_fast.json"),
    ]


def main() -> None:
    args = parse_args()
    entries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for config_path in _config_paths(args.preset):
        config = replace(
            load_experiment_config(config_path),
            output_dir=args.output_root.expanduser().resolve(),
        )
        run_name = config.run_name or config_path.stem
        run_dir = config.output_dir / run_name

        if args.skip_existing and (run_dir / "dashboard.html").exists():
            summary = summarize_fk_run(run_dir)
            status = "skipped_existing"
            dashboard_path = run_dir / "dashboard.html"
        else:
            result = run_experiment(config)
            summary = summarize_fk_run(result.run_dir)
            status = "completed"
            dashboard_path = result.dashboard_path
            run_dir = result.run_dir

        entries.append(
            {
                "label": run_name,
                "run_dir": run_dir,
                "dashboard_path": dashboard_path,
                **summary,
            }
        )
        rows.append(
            {
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "dashboard_path": str(dashboard_path),
                "status": status,
                **summary,
            }
        )

    index_path = render_index(entries, args.index_output)
    summary_path = args.index_output.with_suffix(".csv")
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nVN walk-forward index saved to: {index_path}")
    print(f"VN walk-forward summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
