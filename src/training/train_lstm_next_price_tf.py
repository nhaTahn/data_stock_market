from __future__ import annotations

import numpy as np
import pandas as pd

from tf_lstm.config import parse_args
from tf_lstm.data import (
    build_sequences,
    fit_feature_scaler,
    fit_target_scaler,
    load_dataset,
    scale_split,
    split_dataset,
)
from tf_lstm.experiment import run_experiment
from tf_lstm.model import set_seed


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    set_seed(args.seed)

    run_dir, baseline_metrics, lstm_metrics = run_experiment(args)

    print("\nBaseline 1 metrics:")
    print(pd.DataFrame([{"split": key, **value} for key, value in baseline_metrics.items()]).to_string(index=False))
    print("\nLSTM metrics:")
    print(pd.DataFrame([{"split": key, **value} for key, value in lstm_metrics.items()]).to_string(index=False))
    print("\nSaved training outputs to:", run_dir)


if __name__ == "__main__":
    main()
