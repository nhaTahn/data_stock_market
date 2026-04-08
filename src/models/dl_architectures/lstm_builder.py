from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.models.components.losses import QuantilePinballLoss, resolve_loss
from src.models.sequence_utils import LocalTargetNormalizer, TargetScaler

LSTM_L2_FACTOR = 1e-5


def build_lstm_backbone(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    dropout: float = 0.3,
) -> tuple[keras.layers.Input, keras.KerasTensor]:
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")

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
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")

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


def build_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "mse",
    huber_delta: float = 0.01,
    target_scaler: TargetScaler | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    outputs = layers.Dense(1)(encoded)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=resolve_loss(
            loss,
            huber_delta,
            target_mean=target_scaler.mean if target_scaler is not None else 0.0,
            target_std=target_scaler.std if target_scaler is not None else 1.0,
            use_target_scaler=target_scaler is not None,
            local_scale_floor=local_target_normalizer.floor if local_target_normalizer is not None else 0.0,
        ),
    )
    return model


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


def build_sign_magnitude_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    sign_loss_weight: float = 0.15,
    magnitude_loss_weight: float = 0.35,
    signed_loss_weight: float = 1.5,
    use_log_magnitude: bool = True,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    sign_prob = layers.Dense(1, activation="sigmoid", name="sign_prob")(encoded)
    sign_centered = layers.Rescaling(scale=2.0, offset=-1.0, name="sign_centered")(sign_prob)
    magnitude_raw = layers.Dense(1, activation="softplus", name="magnitude_raw")(encoded)
    if use_log_magnitude:
        magnitude = layers.Lambda(tf.math.expm1, name="magnitude")(magnitude_raw)
    else:
        magnitude = layers.Activation("linear", name="magnitude")(magnitude_raw)
    signed_prediction = layers.Multiply(name="signed_prediction")([sign_centered, magnitude])

    model = keras.Model(
        inputs=inputs,
        outputs={
            "signed_prediction": signed_prediction,
            "sign_prob": sign_prob,
            "magnitude": magnitude,
        },
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            "signed_prediction": resolve_loss(
                loss,
                huber_delta,
                local_scale_floor=local_target_normalizer.floor if local_target_normalizer is not None else 0.0,
            ),
            "sign_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.02),
            "magnitude": keras.losses.Huber(delta=huber_delta),
        },
        loss_weights={
            "signed_prediction": signed_loss_weight,
            "sign_prob": sign_loss_weight,
            "magnitude": magnitude_loss_weight,
        },
    )
    return model


def build_attention_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    target_scaler: TargetScaler | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, sequence_encoded = build_lstm_sequence_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    attended = layers.MultiHeadAttention(
        num_heads=max(1, attention_heads),
        key_dim=max(1, attention_key_dim),
        dropout=max(0.0, min(0.5, dropout)),
        name="self_attention",
    )(sequence_encoded, sequence_encoded)
    x = layers.Add(name="attn_residual")([sequence_encoded, attended])
    x = layers.LayerNormalization(name="attn_norm")(x)
    ff = layers.Dense(int(x.shape[-1]), activation="swish", name="attn_ff")(x)
    x = layers.Add(name="attn_ff_residual")([x, ff])
    x = layers.LayerNormalization(name="attn_ff_norm")(x)
    x = layers.GlobalAveragePooling1D(name="attn_pool")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, name="prediction")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=resolve_loss(
            loss,
            huber_delta,
            target_mean=target_scaler.mean if target_scaler is not None else 0.0,
            target_std=target_scaler.std if target_scaler is not None else 1.0,
            use_target_scaler=target_scaler is not None,
            local_scale_floor=local_target_normalizer.floor if local_target_normalizer is not None else 0.0,
        ),
    )
    return model


def build_event_gated_attention_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    event_loss_weight: float = 0.4,
    sign_loss_weight: float = 0.1,
    magnitude_loss_weight: float = 0.3,
    signed_loss_weight: float = 2.0,
    use_log_magnitude: bool = True,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, sequence_encoded = build_lstm_sequence_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    attended = layers.MultiHeadAttention(
        num_heads=max(1, attention_heads),
        key_dim=max(1, attention_key_dim),
        dropout=max(0.0, min(0.5, dropout)),
        name="event_self_attention",
    )(sequence_encoded, sequence_encoded)
    x = layers.Add(name="event_attn_residual")([sequence_encoded, attended])
    x = layers.LayerNormalization(name="event_attn_norm")(x)
    ff = layers.Dense(int(x.shape[-1]), activation="swish", name="event_attn_ff")(x)
    x = layers.Add(name="event_ff_residual")([x, ff])
    x = layers.LayerNormalization(name="event_ff_norm")(x)
    x = layers.GlobalAveragePooling1D(name="event_pool")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    event_prob = layers.Dense(1, activation="sigmoid", name="event_prob")(x)
    sign_prob = layers.Dense(1, activation="sigmoid", name="sign_prob")(x)
    sign_centered = layers.Rescaling(scale=2.0, offset=-1.0, name="sign_centered")(sign_prob)
    magnitude_raw = layers.Dense(1, activation="softplus", name="magnitude_raw")(x)
    if use_log_magnitude:
        magnitude = layers.Lambda(tf.math.expm1, name="magnitude")(magnitude_raw)
    else:
        magnitude = layers.Activation("linear", name="magnitude")(magnitude_raw)
    gated_sign = layers.Multiply(name="gated_sign")([event_prob, sign_centered])
    signed_prediction = layers.Multiply(name="signed_prediction")([gated_sign, magnitude])

    model = keras.Model(
        inputs=inputs,
        outputs={
            "signed_prediction": signed_prediction,
            "event_prob": event_prob,
            "sign_prob": sign_prob,
            "magnitude": magnitude,
        },
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss={
            "signed_prediction": resolve_loss(
                loss,
                huber_delta,
                local_scale_floor=local_target_normalizer.floor if local_target_normalizer is not None else 0.0,
            ),
            "event_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "sign_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "magnitude": keras.losses.Huber(delta=huber_delta),
        },
        loss_weights={
            "signed_prediction": signed_loss_weight,
            "event_prob": event_loss_weight,
            "sign_prob": sign_loss_weight,
            "magnitude": magnitude_loss_weight,
        },
    )
    return model
