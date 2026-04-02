from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from tf_lstm.dashboard import render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard for a tf_lstm run directory.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing baseline_metrics.csv, lstm_metrics.csv, and fit_history.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_path = render_dashboard(args.run_dir)
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
