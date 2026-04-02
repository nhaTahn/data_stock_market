from __future__ import annotations

import pandas as pd

from _bootstrap import bootstrap_training_path

bootstrap_training_path()

from fk_lstm_classifier.config import namespace_to_config, parse_args
from fk_lstm_classifier.experiment import run_experiment


def main() -> None:
    args = parse_args()
    config = namespace_to_config(args)
    result = run_experiment(config)

    print("\nClassification metrics:")
    print(pd.DataFrame([result.metrics]).to_string(index=False))
    print("\nStrategy summary:")
    print(pd.DataFrame([result.strategy_summary]).to_string(index=False))
    print(f"\nPrediction rows: {result.prediction_count}")
    print(f"Folds: {result.fold_count}")
    print(f"\nArtifacts saved to: {result.run_dir}")
    print(f"Dashboard saved to: {result.dashboard_path}")


if __name__ == "__main__":
    main()
