from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_sequence_backbone
from src.models.components.losses import resolve_loss
from src.models.training.scalers import LocalTargetNormalizer


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
    gated_sign = layers.Multiply(name="gated_sign")([sign_centered, event_prob])
    magnitude_raw = layers.Dense(1, activation="softplus", name="magnitude_raw")(x)
    if use_log_magnitude:
        magnitude = layers.Lambda(tf.math.expm1, name="magnitude")(magnitude_raw)
    else:
        magnitude = layers.Activation("linear", name="magnitude")(magnitude_raw)
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
