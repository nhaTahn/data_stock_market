"""Fischer-Krauss style LSTM classification pipeline for stock returns."""

from fk_lstm_classifier.config import ExperimentConfig, load_experiment_config
from fk_lstm_classifier.data import (
    PreparedPanel,
    SequenceDataset,
    build_classification_panel,
    build_datasets_for_date_splits,
    build_sequences,
    load_market_data,
    prepare_datasets,
)
from fk_lstm_classifier.evaluation import (
    build_long_short_holdings,
    build_prediction_frame,
    compute_classification_metrics,
    compute_long_short_returns,
    compute_long_short_returns_from_holdings,
)
from fk_lstm_classifier.experiment import run_experiment
from fk_lstm_classifier.market_rules import MarketRuntimeSettings, resolve_market_runtime_settings
from fk_lstm_classifier.model import build_lstm_classifier
from fk_lstm_classifier.reporting import render_dashboard, summarize_fk_run
from fk_lstm_classifier.training import train_classifier

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "PreparedPanel",
    "SequenceDataset",
    "build_classification_panel",
    "build_datasets_for_date_splits",
    "build_sequences",
    "load_market_data",
    "prepare_datasets",
    "build_long_short_holdings",
    "build_prediction_frame",
    "compute_classification_metrics",
    "compute_long_short_returns",
    "compute_long_short_returns_from_holdings",
    "run_experiment",
    "MarketRuntimeSettings",
    "resolve_market_runtime_settings",
    "build_lstm_classifier",
    "render_dashboard",
    "summarize_fk_run",
    "train_classifier",
]
