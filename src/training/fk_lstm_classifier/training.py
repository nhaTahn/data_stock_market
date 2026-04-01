from __future__ import annotations

from dataclasses import dataclass

from fk_lstm_classifier.data import SequenceDataset
from fk_lstm_classifier.model import build_early_stopping

try:
    from tensorflow import keras  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is required for fk_lstm_classifier.\n"
        "Install it in your environment, for example: pip install tensorflow"
    ) from exc


@dataclass
class TrainResult:
    history: dict[str, list[float]]


def train_classifier(
    model: keras.Model,
    train_dataset: SequenceDataset,
    validation_dataset: SequenceDataset,
    batch_size: int = 128,
    max_epochs: int = 1000,
    patience: int = 10,
) -> TrainResult:
    history = model.fit(
        train_dataset.features,
        train_dataset.labels_one_hot,
        validation_data=(validation_dataset.features, validation_dataset.labels_one_hot),
        batch_size=batch_size,
        epochs=max_epochs,
        shuffle=True,
        verbose=2,
        callbacks=build_early_stopping(patience=patience),
    )
    return TrainResult(history=history.history)
