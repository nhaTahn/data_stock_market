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
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> tuple[keras.layers.Input, keras.KerasTensor]:
    """Build a stacked LSTM backbone.

    Args:
        window_size, num_features: input shape.
        lstm_units: int or list of layer sizes.
        dropout: feature-dropout applied between LSTM layers (only when
            `return_sequences=True`) and replicated as the post-norm dropout
            on the final layer when `use_layer_norm` is set.
        recurrent_dropout: dropout applied INSIDE each LSTM cell on the
            recurrent connections. L1 of plan — defaults to 0.0 for backward
            compat with the canonical signmag run.
        use_layer_norm: when True, insert `LayerNormalization` after every
            LSTM layer (before any feature dropout). L1 of plan.
    """
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
            recurrent_dropout=recurrent_dropout,
        )(x)
        if use_layer_norm:
            x = layers.LayerNormalization()(x)
        if dropout > 0 and return_sequences:
            x = layers.Dropout(dropout)(x)
    return inputs, x


def build_lstm_sequence_backbone(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
) -> tuple[keras.layers.Input, keras.KerasTensor]:
    """Same as `build_lstm_backbone` but keeps `return_sequences=True` at every
    layer (used by attention/event heads that pool the full sequence).
    """
    unit_stack = _normalize_unit_stack(lstm_units)

    inputs = layers.Input(shape=(window_size, num_features))
    x = inputs
    for units in unit_stack:
        x = layers.LSTM(
            units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(LSTM_L2_FACTOR),
            recurrent_regularizer=regularizers.l2(LSTM_L2_FACTOR),
            recurrent_dropout=recurrent_dropout,
        )(x)
        if use_layer_norm:
            x = layers.LayerNormalization()(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    return inputs, x
