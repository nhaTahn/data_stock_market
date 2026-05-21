from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_backbone
from src.models.components.losses import QuantilePinballLoss


def build_quantile_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    q50 = layers.Dense(1, name="q50")(encoded)
    q90_delta = layers.Dense(1, activation="softplus", name="q90_delta")(encoded)
    q90 = layers.Add(name="q90")([q50, q90_delta])
    outputs = layers.Concatenate(name="quantile_prediction")([q50, q90])
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=QuantilePinballLoss(),
    )
    return model
