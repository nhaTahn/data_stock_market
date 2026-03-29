from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT / "data" / "assests" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "assests" / "data_info_vn" / "history" / "training_runs"

FEATURE_COLUMNS = ["adjust", "volume_match"]
TARGET_COLUMN = "target_next_adjust_return"

# Shared time boundaries for all TensorFlow LSTM training scripts.
TRAIN_END_DATE = "2023-12-31"
VAL_END_DATE = "2024-12-31"
RETURN_DIRECTION_THRESHOLD = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a simple TensorFlow/Keras LSTM for next-day adjusted return prediction.\n"
            "Baseline 1: predict next-day return = 0, equivalent to reusing today's adjusted price."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to vn_gold_recommended.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for training outputs.")
    parser.add_argument("--train-end", default=TRAIN_END_DATE, help="Last date in the training split.")
    parser.add_argument("--val-end", default=VAL_END_DATE, help="Last date in the validation split.")
    parser.add_argument("--window-size", type=int, default=20, help="Number of historical days in each input sequence.")
    parser.add_argument("--lstm-units", type=int, default=32, help="Number of hidden units in the LSTM layer.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout after the LSTM layer.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()
