from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers


LSTM_L2_FACTOR = 1e-5


def _normalize_unit_stack(lstm_units: int | list[int]) -> list[int]:
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")
    return unit_stack


def build_lstm_backbone(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    dropout: float = 0.3,
) -> tuple[keras.layers.Input, keras.KerasTensor]:
    unit_stack = _normalize_unit_stack(lstm_units)

    inputs = layers.Input(shape=(window_size, num_features))
    x = inputs
    for layer_idx, units in enumerate(unit_stack):
        return_sequences = layer_idx < len(unit_stack) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            kernel_regularizer=regularizers.l2(LSTM_L2_FACTOR),
            recurrent_regularizer=regularizers.l2(LSTM_L2_FACTOR),
        )(x)
        if dropout > 0 and return_sequences:
            x = layers.Dropout(dropout)(x)
    return inputs, x


def build_lstm_sequence_backbone(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    dropout: float = 0.3,
) -> tuple[keras.layers.Input, keras.KerasTensor]:
    unit_stack = _normalize_unit_stack(lstm_units)

    inputs = layers.Input(shape=(window_size, num_features))
    x = inputs
    for units in unit_stack:
        x = layers.LSTM(
            units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(LSTM_L2_FACTOR),
            recurrent_regularizer=regularizers.l2(LSTM_L2_FACTOR),
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    return inputs, x
