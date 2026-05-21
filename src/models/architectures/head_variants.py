"""Head-variant LSTMs to beat plain LSTM.

Three architectures sharing the same backbone but with different head designs:

- **hetero** (F2): heteroscedastic regression with learned per-sample variance.
  `pred = Dense(1)(encoded)`, `log_var = Dense(1)(encoded)`. Inference uses `pred`.
  Loss: Gaussian NLL `0.5 * exp(-log_var) * (pred - y)^2 + 0.5 * log_var`.
  Rationale: rel_score's q90(|err|) term implicitly weights tails. Heteroscedastic
  model down-weights inherently noisy days, focuses gradient on stable ones.

- **skip** (F5): skip connection from last raw input timestep to head.
  `pred = Dense(1)(concat([encoded, features[:, -1, :]]))`.
  Rationale: LSTM compresses 15×26 into 32 hidden units. Direct access to the
  last raw timestep gives the head the strongest predictor (autocorrelation)
  without information loss.

- **deep_head** (F6): two-layer MLP head instead of single Dense.
  `pred = Dense(1)(relu(Dense(32)(encoded)))`.
  Rationale: extra nonlinear capacity at the OUTPUT, not the backbone. May
  capture nonlinear mapping from hidden state to return that linear Dense misses.

All three use plain regression target (no sign/magnitude decomposition). The
loss configured via `loss` argument is applied to `pred` directly.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_backbone
from src.models.components.losses import resolve_loss
from src.models.training.scalers import LocalTargetNormalizer, TargetScaler


# ----------------------------------------------------------------------
# F2 — Heteroscedastic regression
# ----------------------------------------------------------------------


@keras.utils.register_keras_serializable(package="custom")
class GaussianNLLLoss(keras.losses.Loss):
    """Gaussian Negative Log-Likelihood.

    Expects y_pred to be concatenated [mu, log_var] along the last axis.
    Includes a stability term to keep log_var in a sane range.
    """

    def __init__(self, log_var_clip: tuple[float, float] = (-8.0, 4.0), name: str = "gaussian_nll"):
        super().__init__(name=name)
        self.log_var_clip = (float(log_var_clip[0]), float(log_var_clip[1]))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        # y_true may be 1-D (batch,) or 2-D (batch, K) when the pipeline
        # appends auxiliary scale columns (rel_score loss convention).
        # Only the first column is the actual target.
        if y_true.shape.rank is not None and y_true.shape.rank >= 2:
            y_true = y_true[:, 0]
        y_true = tf.reshape(y_true, [-1])

        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank is None or y_pred.shape[-1] is None or y_pred.shape[-1] < 2:
            raise ValueError("GaussianNLLLoss expects y_pred with last-dim>=2 (mu, log_var).")
        mu = tf.reshape(y_pred[..., 0], [-1])
        log_var = tf.reshape(y_pred[..., 1], [-1])
        log_var = tf.clip_by_value(log_var, self.log_var_clip[0], self.log_var_clip[1])
        inv_var = tf.exp(-log_var)
        loss = 0.5 * inv_var * tf.square(y_true - mu) + 0.5 * log_var
        return tf.reduce_mean(loss)

    def get_config(self) -> dict[str, object]:
        return {"name": self.name, "log_var_clip": list(self.log_var_clip)}


def build_hetero_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
    log_var_clip: tuple[float, float] = (-8.0, 4.0),
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        use_layer_norm=use_layer_norm,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_var = layers.Dense(1, name="log_var")(encoded)
    # Concatenate into single output for Gaussian NLL loss.
    output = layers.Concatenate(axis=-1, name="mu_logvar")([mu, log_var])
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=GaussianNLLLoss(log_var_clip=log_var_clip),
    )
    return model


def hetero_predict(model: keras.Model, x) -> "tf.Tensor":
    """Inference: return mu only (drop log_var)."""
    import numpy as np

    full = model.predict(x, verbose=0)
    if full.ndim == 1:
        return full
    return full[..., 0]


# ----------------------------------------------------------------------
# F5 — Skip connection from raw last-step features
# ----------------------------------------------------------------------


def build_skip_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
    loss: str = "rel_score",
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
    # Pull last-step raw features as a skip connection.
    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(inputs)
    combined = layers.Concatenate(axis=-1, name="encoded_with_skip")([encoded, last_step])
    pred = layers.Dense(1, name="pred")(combined)
    model = keras.Model(inputs=inputs, outputs=pred)
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


# ----------------------------------------------------------------------
# F6 — Deep head (two-layer MLP at output)
# ----------------------------------------------------------------------


def build_deep_head_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
    head_hidden_units: int = 32,
    head_dropout: float = 0.0,
    loss: str = "rel_score",
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
    h = layers.Dense(head_hidden_units, activation="relu", name="head_hidden")(encoded)
    if head_dropout > 0.0:
        h = layers.Dropout(head_dropout, name="head_dropout")(h)
    pred = layers.Dense(1, name="pred")(h)
    model = keras.Model(inputs=inputs, outputs=pred)
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
