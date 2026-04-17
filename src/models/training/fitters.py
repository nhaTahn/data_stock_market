from __future__ import annotations

import numpy as np
from tensorflow import keras

from src.models.architectures.attention import build_attention_model
from src.models.architectures.event import build_event_gated_attention_model
from src.models.architectures.pcie_lite import build_pcie_lite_model
from src.models.architectures.plain import build_model
from src.models.architectures.quantile import build_quantile_model
from src.models.architectures.signal import build_signal_attention_lstm_model
from src.models.architectures.signmag import build_sign_magnitude_model
from src.models.components.callbacks import build_training_callbacks
from src.models.training.sample_weights import (
    build_event_gated_sample_weights,
    build_sign_magnitude_sample_weights,
)
from src.models.training.scalers import LocalTargetNormalizer, TargetScaler
from src.models.training.targets import (
    build_event_gated_targets,
    build_sign_magnitude_targets,
    primary_target_array,
)


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
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
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
    initial_model_path: str | None = None,
):
    if initial_model_path:
        model = keras.models.load_model(initial_model_path)
        model.optimizer.learning_rate = lr
    else:
        model = build_model(
            window_size=window_size,
            num_features=num_features,
            lstm_units=lstm_units,
            lr=lr,
            dropout=dropout,
            loss=loss,
            huber_delta=huber_delta,
            rel_score_large_move_quantile=rel_score_large_move_quantile,
            rel_score_directional_penalty=rel_score_directional_penalty,
            rel_score_confidence_penalty=rel_score_confidence_penalty,
            rel_score_confidence_ratio=rel_score_confidence_ratio,
            rel_score_weighted_high_quantile=rel_score_weighted_high_quantile,
            rel_score_weighted_high_weight=rel_score_weighted_high_weight,
            rel_score_weighted_base_weight=rel_score_weighted_base_weight,
            target_scaler=target_scaler,
            local_target_normalizer=local_target_normalizer,
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
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
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
        rel_score_large_move_quantile=rel_score_large_move_quantile,
        rel_score_directional_penalty=rel_score_directional_penalty,
        rel_score_confidence_penalty=rel_score_confidence_penalty,
        rel_score_confidence_ratio=rel_score_confidence_ratio,
        rel_score_weighted_high_quantile=rel_score_weighted_high_quantile,
        rel_score_weighted_high_weight=rel_score_weighted_high_weight,
        rel_score_weighted_base_weight=rel_score_weighted_base_weight,
        sign_loss_weight=sign_loss_weight,
        magnitude_loss_weight=magnitude_loss_weight,
        signed_loss_weight=signed_loss_weight,
        use_log_magnitude=use_log_magnitude,
        local_target_normalizer=local_target_normalizer,
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
        build_sign_magnitude_targets(
            y_train,
            auxiliary_target=primary_target_array(y_train),
            use_log_magnitude=use_log_magnitude,
        ),
        sample_weight=build_sign_magnitude_sample_weights(sample_weight),
        validation_data=(
            x_val,
            build_sign_magnitude_targets(
                y_val,
                auxiliary_target=primary_target_array(y_val),
                use_log_magnitude=use_log_magnitude,
            ),
            build_sign_magnitude_sample_weights(val_sample_weight),
        )
        if val_sample_weight is not None
        else (
            x_val,
            build_sign_magnitude_targets(
                y_val,
                auxiliary_target=primary_target_array(y_val),
                use_log_magnitude=use_log_magnitude,
            ),
        ),
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
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
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
        rel_score_large_move_quantile=rel_score_large_move_quantile,
        rel_score_directional_penalty=rel_score_directional_penalty,
        rel_score_confidence_penalty=rel_score_confidence_penalty,
        rel_score_confidence_ratio=rel_score_confidence_ratio,
        rel_score_weighted_high_quantile=rel_score_weighted_high_quantile,
        rel_score_weighted_high_weight=rel_score_weighted_high_weight,
        rel_score_weighted_base_weight=rel_score_weighted_base_weight,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        target_scaler=target_scaler,
        local_target_normalizer=local_target_normalizer,
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


def fit_quantile_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
):
    model = build_quantile_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        dropout=dropout,
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
        prediction_key=0,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def fit_signal_attention_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    patch_length: int = 5,
    patch_stride: int = 3,
    d_patch: int = 16,
    future_steps: int = 1,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    attention_ff_dim: int | None = None,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train[:, None, None]
    elif y_train.ndim == 2:
        y_train = y_train[..., None]
    if y_val.ndim == 1:
        y_val = y_val[:, None, None]
    elif y_val.ndim == 2:
        y_val = y_val[..., None]

    model = build_signal_attention_lstm_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        patch_length=patch_length,
        patch_stride=patch_stride,
        d_patch=d_patch,
        future_steps=future_steps,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        attention_ff_dim=attention_ff_dim,
        dropout=dropout,
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
        prediction_key=(0, 0),
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


def fit_pcie_lite_model(
    x_train,
    y_train,
    x_val,
    y_val,
    window_size: int,
    num_features: int,
    lstm_units: int | list[int] = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 10,
    patience: int = 3,
    monitor_metric: str = "val_loss",
    val_group_ids: np.ndarray | None = None,
    target_scaler: TargetScaler | None = None,
    metric_y_val: np.ndarray | None = None,
    local_target_normalizer: LocalTargetNormalizer | None = None,
    local_target_scale_values: np.ndarray | None = None,
    patch_length: int = 5,
    patch_stride: int = 5,
    d_patch: int = 16,
    future_steps: int = 3,
    sample_weight: np.ndarray | None = None,
    val_sample_weight: np.ndarray | None = None,
):
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_val.ndim == 1:
        y_val = y_val[:, None]

    model = build_pcie_lite_model(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        lr=lr,
        patch_length=patch_length,
        patch_stride=patch_stride,
        d_patch=d_patch,
        future_steps=future_steps,
        dropout=dropout,
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
        prediction_key=0,
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
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
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
        rel_score_large_move_quantile=rel_score_large_move_quantile,
        rel_score_directional_penalty=rel_score_directional_penalty,
        rel_score_confidence_penalty=rel_score_confidence_penalty,
        rel_score_confidence_ratio=rel_score_confidence_ratio,
        rel_score_weighted_high_quantile=rel_score_weighted_high_quantile,
        rel_score_weighted_high_weight=rel_score_weighted_high_weight,
        rel_score_weighted_base_weight=rel_score_weighted_base_weight,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        event_loss_weight=event_loss_weight,
        sign_loss_weight=sign_loss_weight,
        magnitude_loss_weight=magnitude_loss_weight,
        signed_loss_weight=signed_loss_weight,
        use_log_magnitude=use_log_magnitude,
        local_target_normalizer=local_target_normalizer,
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
        auxiliary_target=primary_target_array(y_train),
        event_threshold=event_threshold,
        use_log_magnitude=use_log_magnitude,
    )
    val_targets = build_event_gated_targets(
        y_val,
        auxiliary_target=primary_target_array(y_val),
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
