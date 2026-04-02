from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.reporting import render_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard for an FK LSTM run directory.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing fit_history.csv, validation_predictions.csv, and validation_long_short_returns.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard_path = render_dashboard(args.run_dir)
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
