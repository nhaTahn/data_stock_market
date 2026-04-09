from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_backbone
from src.models.components.losses import resolve_loss
from src.models.training.scalers import LocalTargetNormalizer


def build_sign_magnitude_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
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
                large_move_quantile=rel_score_large_move_quantile,
                directional_penalty_weight=rel_score_directional_penalty,
                confidence_penalty_weight=rel_score_confidence_penalty,
                confidence_ratio=rel_score_confidence_ratio,
                weighted_high_quantile=rel_score_weighted_high_quantile,
                weighted_high_weight=rel_score_weighted_high_weight,
                weighted_base_weight=rel_score_weighted_base_weight,
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
