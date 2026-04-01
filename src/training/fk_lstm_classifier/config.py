from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fk_lstm_classifier"


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    markets: tuple[str, ...]
    lookback: int = 240
    train_ratio: float = 0.8
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 10
    lstm_units: int = 25
    dropout: float = 0.16
    learning_rate: float = 1e-3
    top_k: int = 10
    seed: int = 42
    model_type: str = "attention"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_cross_sectional_count: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Fischer-Krauss style binary LSTM classifier on daily stock returns "
            "with optional temporal attention."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing market folders such as data/VN and data/US.",
    )
    parser.add_argument(
        "--markets",
        default="VN,US",
        help="Comma-separated market folders to load, for example VN,US.",
    )
    parser.add_argument("--lookback", type=int, default=240, help="Sequence length in trading days.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological fraction of eligible dates reserved for the training split.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--max-epochs", type=int, default=1000, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--lstm-units", type=int, default=25, help="Number of LSTM hidden units.")
    parser.add_argument("--dropout", type=float, default=0.16, help="Input dropout inside the LSTM layer.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="RMSprop learning rate.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "attention"],
        default="attention",
        help="Baseline reproduces the paper-style LSTM, attention adds temporal attention on top.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top and bottom k names for long-short evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save predictions, backtest results, and fit history.",
    )
    parser.add_argument(
        "--min-cross-sectional-count",
        type=int,
        default=20,
        help="Minimum number of stocks required on a date to keep that cross-section.",
    )
    return parser.parse_args()


def namespace_to_config(args: argparse.Namespace) -> ExperimentConfig:
    markets = tuple(market.strip().upper() for market in args.markets.split(",") if market.strip())
    if not markets:
        raise ValueError("At least one market must be provided via --markets.")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    return ExperimentConfig(
        data_dir=args.data_dir,
        markets=markets,
        lookback=args.lookback,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        seed=args.seed,
        model_type=args.model_type,
        output_dir=args.output_dir,
        min_cross_sectional_count=args.min_cross_sectional_count,
    )
