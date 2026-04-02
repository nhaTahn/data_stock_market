from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from model_benchmark.dashboard import render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard for a benchmark run directory.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing benchmark_summary.csv and related benchmark outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_path = render_dashboard(args.run_dir)
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
