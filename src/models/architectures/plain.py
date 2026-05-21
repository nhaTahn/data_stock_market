from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_backbone
from src.models.components.losses import resolve_loss
from src.models.training.scalers import LocalTargetNormalizer, TargetScaler


def build_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
    loss: str = "mse",
    huber_delta: float = 0.01,
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
    target_scaler: TargetScaler | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        use_layer_norm=use_layer_norm,
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
            large_move_quantile=rel_score_large_move_quantile,
            directional_penalty_weight=rel_score_directional_penalty,
            confidence_penalty_weight=rel_score_confidence_penalty,
            confidence_ratio=rel_score_confidence_ratio,
            weighted_high_quantile=rel_score_weighted_high_quantile,
            weighted_high_weight=rel_score_weighted_high_weight,
            weighted_base_weight=rel_score_weighted_base_weight,
        ),
    )
    return model
