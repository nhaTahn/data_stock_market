"""Fischer-Krauss style LSTM classification pipeline for stock returns."""

from fk_lstm_classifier.config import ExperimentConfig
from fk_lstm_classifier.data import (
    PreparedPanel,
    SequenceDataset,
    build_classification_panel,
    build_sequences,
    load_market_data,
    prepare_datasets,
)
from fk_lstm_classifier.evaluation import (
    build_prediction_frame,
    compute_classification_metrics,
    compute_long_short_returns,
)
from fk_lstm_classifier.model import build_lstm_classifier
from fk_lstm_classifier.training import train_classifier

__all__ = [
    "ExperimentConfig",
    "PreparedPanel",
    "SequenceDataset",
    "build_classification_panel",
    "build_sequences",
    "load_market_data",
    "prepare_datasets",
    "build_prediction_frame",
    "compute_classification_metrics",
    "compute_long_short_returns",
    "build_lstm_classifier",
    "train_classifier",
]
