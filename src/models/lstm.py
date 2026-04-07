from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.evaluation.metric import evaluate

LSTM_L2_FACTOR = 1e-5


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_columns: tuple[str, ...]


@dataclass
class TargetScaler:
    mean: float
    std: float


@dataclass
class LocalTargetNormalizer:
    column: str
    floor: float


def build_magnitude_sample_weights(
    y: np.ndarray,
    strength: float = 1.5,
    reference_quantile: float = 0.75,
    clip_multiple: float = 3.0,
) -> np.ndarray:
    abs_y = np.abs(np.asarray(y, dtype=np.float32).reshape(-1))
    if len(abs_y) == 0:
        return np.ones(0, dtype=np.float32)

    valid = abs_y[np.isfinite(abs_y)]
    if len(valid) == 0:
        return np.ones_like(abs_y, dtype=np.float32)

    reference = float(np.quantile(valid, reference_quantile))
    reference = max(reference, 1e-4)
    normalized = np.clip(abs_y / reference, 0.0, clip_multiple)
    weights = 1.0 + strength * np.tanh(normalized)
    return weights.astype(np.float32)


def build_sign_magnitude_sample_weights(sample_weight: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    return {
        "signed_prediction": sample_weight,
        "magnitude": sample_weight,
        "sign_prob": np.sqrt(sample_weight).astype(np.float32),
    }


def build_event_gated_sample_weights(
    sample_weight: np.ndarray | None,
    event_target: np.ndarray,
) -> dict[str, np.ndarray]:
    event_target = np.asarray(event_target, dtype=np.float32).reshape(-1)
    if sample_weight is None:
        base = np.ones_like(event_target, dtype=np.float32)
    else:
        base = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    event_focus = 0.25 + 0.75 * event_target
    magnitude_focus = 0.1 + 0.9 * event_target
    return {
        "signed_prediction": base,
        "event_prob": base,
        "sign_prob": (base * event_focus).astype(np.float32),
        "magnitude": (base * magnitude_focus).astype(np.float32),
    }


def _extract_prediction_array(raw_prediction, prediction_key: str | int | None = None) -> np.ndarray:
    if isinstance(raw_prediction, dict):
        if prediction_key is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output dict predictions.")
            raw_prediction = next(iter(raw_prediction.values()))
        else:
            raw_prediction = raw_prediction[prediction_key]
    elif isinstance(raw_prediction, (list, tuple)):
        if prediction_key is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output list predictions.")
            raw_prediction = raw_prediction[0]
        else:
            raw_prediction = raw_prediction[int(prediction_key)]
    return np.asarray(raw_prediction, dtype=np.float32).reshape(-1)


class ValidationRelScoreCallback(keras.callbacks.Callback):
    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        group_ids: np.ndarray | None = None,
        target_scaler: TargetScaler | None = None,
        metric_y_val: np.ndarray | None = None,
        local_target_normalizer: LocalTargetNormalizer | None = None,
        local_target_scale_values: np.ndarray | None = None,
        prediction_key: str | int | None = None,
    ):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.group_ids = group_ids
        self.target_scaler = target_scaler
        self.metric_y_val = metric_y_val
        self.local_target_normalizer = local_target_normalizer
        self.local_target_scale_values = local_target_scale_values
        self.prediction_key = prediction_key

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        if len(self.x_val) < 3 or len(self.y_val) < 3:
            logs["val_rel_score"] = np.nan
            return
        raw_prediction = self.model.predict(self.x_val, verbose=0)
        prediction = _extract_prediction_array(raw_prediction, self.prediction_key)
        try:
            metric_prediction = inverse_target_scaler_values(prediction, self.target_scaler)
            metric_prediction = inverse_local_target_normalizer(
                metric_prediction,
                self.local_target_scale_values,
                self.local_target_normalizer,
            )
            if self.metric_y_val is not None:
                metric_target = np.asarray(self.metric_y_val, dtype=np.float32)
            else:
                metric_target = inverse_target_scaler_values(self.y_val, self.target_scaler)
            logs["val_rel_score"] = float(
                evaluate(metric_prediction, metric_target, group_ids=self.group_ids)["rel_score"]
            )
        except Exception:
            logs["val_rel_score"] = np.nan


class DirectionalHuberLoss(keras.losses.Loss):
    def __init__(self, delta: float = 0.01, penalty_weight: float = 20.0, name: str = "directional_huber_loss"):
        super().__init__(name=name)
        self.delta = delta
        self.penalty_weight = penalty_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return directional_huber_loss(
            y_true,
            y_pred,
            delta=self.delta,
            penalty_weight=self.penalty_weight,
        )

    def get_config(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "delta": self.delta,
            "penalty_weight": self.penalty_weight,
        }


def directional_huber_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    delta: float = 0.01,
    penalty_weight: float = 20.0,
) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * tf.square(quadratic) + delta * linear
    sign_penalty = tf.maximum(0.0, -(y_true * y_pred))
    return tf.reduce_mean(huber + penalty_weight * sign_penalty)


def resolve_loss(loss: str, huber_delta: float):
    if loss == "mse":
        return "mse"
    if loss == "huber":
        return keras.losses.Huber(delta=huber_delta)
    if loss == "directional_huber":
        return DirectionalHuberLoss(delta=huber_delta)
    raise ValueError("loss must be one of: mse, huber, directional_huber")


def build_training_callbacks(
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_group_ids: np.ndarray | None,
    patience: int,
    monitor_metric: str,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    prediction_key: str | int | None = None,
) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = []
    if monitor_metric == "val_rel_score":
        callbacks.append(
            ValidationRelScoreCallback(
                x_val,
                y_val,
                group_ids=val_group_ids,
                target_scaler=target_scaler,
                metric_y_val=metric_y_val,
                local_target_normalizer=local_target_normalizer,
                local_target_scale_values=local_target_scale_values,
                prediction_key=prediction_key,
            )
        )
        mode = "max"
    else:
        mode = "min"
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
        )
    )
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode=mode,
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-5,
            verbose=0,
        )
    )
    return callbacks


def build_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    loss: str = "mse",
    huber_delta: float = 0.01,
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
        loss=resolve_loss(loss, huber_delta),
    )
    return model


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
            "signed_prediction": resolve_loss(loss, huber_delta),
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
        loss=resolve_loss(loss, huber_delta),
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
            "signed_prediction": resolve_loss(loss, huber_delta),
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


def fit_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    loss: str = "mse",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    model = build_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
        loss=loss,
        huber_delta=huber_delta,
    )
    callbacks = build_training_callbacks(
        x_val,
        y_val,
        val_group_ids,
        patience,
        monitor_metric,
        target_scaler=target_scaler,
        metric_y_val=metric_y_val,
        local_target_normalizer=local_target_normalizer,
        local_target_scale_values=local_target_scale_values,
    )
    history = model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(x_val, y_val, val_sample_weight) if val_sample_weight is not None else (x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def build_sign_magnitude_targets(y: np.ndarray, use_log_magnitude: bool = True) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    magnitude_target = np.abs(y)
    if use_log_magnitude:
        magnitude_target = np.log1p(magnitude_target)
    return {
        "signed_prediction": y,
        "sign_prob": (y >= 0).astype(np.float32),
        "magnitude": magnitude_target.astype(np.float32),
    }


def fit_sign_magnitude_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    sign_loss_weight: float = 0.15,
    magnitude_loss_weight: float = 0.35,
    signed_loss_weight: float = 1.5,
    use_log_magnitude: bool = True,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    model = build_sign_magnitude_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
        loss=loss,
        huber_delta=huber_delta,
        sign_loss_weight=sign_loss_weight,
        magnitude_loss_weight=magnitude_loss_weight,
        signed_loss_weight=signed_loss_weight,
        use_log_magnitude=use_log_magnitude,
    )
    callbacks = build_training_callbacks(
        x_val,
        y_val,
        val_group_ids,
        patience,
        monitor_metric,
        metric_y_val=metric_y_val,
        local_target_normalizer=local_target_normalizer,
        local_target_scale_values=local_target_scale_values,
        prediction_key="signed_prediction",
    )
    history = model.fit(
        x_train,
        build_sign_magnitude_targets(y_train, use_log_magnitude=use_log_magnitude),
        sample_weight=build_sign_magnitude_sample_weights(sample_weight),
        validation_data=(
            x_val,
            build_sign_magnitude_targets(y_val, use_log_magnitude=use_log_magnitude),
            build_sign_magnitude_sample_weights(val_sample_weight),
        )
        if val_sample_weight is not None
        else (x_val, build_sign_magnitude_targets(y_val, use_log_magnitude=use_log_magnitude)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def fit_attention_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    model = build_attention_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
        loss=loss,
        huber_delta=huber_delta,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
    )
    callbacks = build_training_callbacks(
        x_val,
        y_val,
        val_group_ids,
        patience,
        monitor_metric,
        target_scaler=target_scaler,
        metric_y_val=metric_y_val,
        local_target_normalizer=local_target_normalizer,
        local_target_scale_values=local_target_scale_values,
    )
    history = model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(x_val, y_val, val_sample_weight) if val_sample_weight is not None else (x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def build_event_gated_targets(
    y: np.ndarray,
    event_threshold: float = 0.75,
    use_log_magnitude: bool = True,
) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    abs_y = np.abs(y)
    magnitude_target = np.log1p(abs_y) if use_log_magnitude else abs_y
    event_target = (abs_y >= event_threshold).astype(np.float32)
    return {
        "signed_prediction": y.astype(np.float32),
        "event_prob": event_target.astype(np.float32),
        "sign_prob": (y >= 0).astype(np.float32),
        "magnitude": magnitude_target.astype(np.float32),
    }


def fit_event_gated_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    loss: str = "huber",
    huber_delta: float = 0.01,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    event_threshold: float = 0.75,
    event_loss_weight: float = 0.4,
    sign_loss_weight: float = 0.1,
    magnitude_loss_weight: float = 0.3,
    signed_loss_weight: float = 2.0,
    use_log_magnitude: bool = True,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    model = build_event_gated_attention_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
        loss=loss,
        huber_delta=huber_delta,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        event_loss_weight=event_loss_weight,
        sign_loss_weight=sign_loss_weight,
        magnitude_loss_weight=magnitude_loss_weight,
        signed_loss_weight=signed_loss_weight,
        use_log_magnitude=use_log_magnitude,
    )
    callbacks = build_training_callbacks(
        x_val,
        y_val,
        val_group_ids,
        patience,
        monitor_metric,
        metric_y_val=metric_y_val,
        local_target_normalizer=local_target_normalizer,
        local_target_scale_values=local_target_scale_values,
        prediction_key="signed_prediction",
    )
    train_targets = build_event_gated_targets(
        y_train,
        event_threshold=event_threshold,
        use_log_magnitude=use_log_magnitude,
    )
    val_targets = build_event_gated_targets(
        y_val,
        event_threshold=event_threshold,
        use_log_magnitude=use_log_magnitude,
    )
    history = model.fit(
        x_train,
        train_targets,
        sample_weight=build_event_gated_sample_weights(sample_weight, train_targets["event_prob"]),
        validation_data=(
            x_val,
            val_targets,
            build_event_gated_sample_weights(val_sample_weight, val_targets["event_prob"]),
        )
        if val_sample_weight is not None
        else (x_val, val_targets),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def set_global_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def predict(model: keras.Model, x, prediction_key: str | int | None = None):
    raw_prediction = model.predict(x, verbose=0)
    return _extract_prediction_array(raw_prediction, prediction_key)


def split_frame_by_date(df: pd.DataFrame, train_end_date: str, val_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"])
    train_df = work[work["Date"] <= pd.Timestamp(train_end_date)].copy()
    val_df = work[(work["Date"] > pd.Timestamp(train_end_date)) & (work["Date"] <= pd.Timestamp(val_end_date))].copy()
    test_df = work[work["Date"] > pd.Timestamp(val_end_date)].copy()
    return train_df, val_df, test_df


def fit_feature_scaler(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> FeatureScaler:
    features = df.loc[:, feature_columns].astype(float)
    mean = features.mean(axis=0).to_numpy()
    std = features.std(axis=0).replace(0, 1).fillna(1).to_numpy()
    return FeatureScaler(mean=mean, std=std, feature_columns=feature_columns)


def fit_target_scaler(y: np.ndarray) -> TargetScaler:
    y = np.asarray(y, dtype=np.float32)
    std = float(np.std(y))
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    return TargetScaler(mean=float(np.mean(y)), std=std)


def fit_local_target_normalizer(
    scale_values: np.ndarray,
    column: str,
) -> LocalTargetNormalizer:
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.abs(scale_values)
    valid = scale_values[np.isfinite(scale_values) & (scale_values > 0)]
    if len(valid) == 0:
        floor = 1.0
    else:
        floor = max(float(np.quantile(valid, 0.25)), 1e-4)
    return LocalTargetNormalizer(column=column, floor=floor)


def apply_target_scaler(y: np.ndarray, scaler: TargetScaler | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if scaler is None:
        return y
    return ((y - scaler.mean) / scaler.std).astype(np.float32)


def inverse_target_scaler_values(y: np.ndarray, scaler: TargetScaler | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if scaler is None:
        return y
    return (y * scaler.std + scaler.mean).astype(np.float32)


def apply_local_target_normalizer(
    y: np.ndarray,
    scale_values: np.ndarray | None,
    normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if normalizer is None or scale_values is None:
        return y
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.where(np.isfinite(scale_values), np.abs(scale_values), normalizer.floor)
    denom = np.maximum(scale_values, normalizer.floor)
    return (y / denom).astype(np.float32)


def inverse_local_target_normalizer(
    y: np.ndarray,
    scale_values: np.ndarray | None,
    normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if normalizer is None or scale_values is None:
        return y
    scale_values = np.asarray(scale_values, dtype=np.float32)
    scale_values = np.where(np.isfinite(scale_values), np.abs(scale_values), normalizer.floor)
    denom = np.maximum(scale_values, normalizer.floor)
    return (y * denom).astype(np.float32)


def apply_feature_scaler(df: pd.DataFrame, scaler: FeatureScaler) -> pd.DataFrame:
    work = df.copy()
    work = work.astype({col: float for col in scaler.feature_columns})
    values = work.loc[:, scaler.feature_columns].to_numpy()
    work.loc[:, scaler.feature_columns] = (values - scaler.mean) / scaler.std
    return work


def build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    window_size: int,
    extra_meta_columns: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list = []
    y_list = []
    meta_rows = []
    required_cols = list(feature_columns) + [target_column, "Date", "code", *extra_meta_columns]

    for code, group in df.sort_values(["code", "Date"]).groupby("code"):
        group = group.dropna(subset=required_cols).reset_index(drop=True)
        if len(group) < window_size:
            continue
        feature_values = group.loc[:, feature_columns].to_numpy(dtype=float)
        target_values = group.loc[:, target_column].to_numpy(dtype=float)
        dates = pd.to_datetime(group["Date"])

        for end_idx in range(window_size - 1, len(group)):
            start_idx = end_idx - window_size + 1
            x_list.append(feature_values[start_idx : end_idx + 1])
            y_list.append(target_values[end_idx])
            meta_rows.append(
                {
                    "code": code,
                    "Date": dates.iloc[end_idx],
                    "target": target_values[end_idx],
                    **{
                        col: group.iloc[end_idx][col]
                        for col in extra_meta_columns
                    },
                }
            )

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return x, y, meta


def split_sequence_dataset(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    dates = pd.to_datetime(meta["Date"])
    train_mask = dates <= pd.Timestamp(train_end_date)
    val_mask = (dates > pd.Timestamp(train_end_date)) & (dates <= pd.Timestamp(val_end_date))
    test_mask = dates > pd.Timestamp(val_end_date)

    return {
        "train": (x[train_mask], y[train_mask], meta.loc[train_mask].reset_index(drop=True)),
        "val": (x[val_mask], y[val_mask], meta.loc[val_mask].reset_index(drop=True)),
        "test": (x[test_mask], y[test_mask], meta.loc[test_mask].reset_index(drop=True)),
    }
