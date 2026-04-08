from __future__ import annotations

import numpy as np
from tensorflow import keras

from src.evaluation.metric import evaluate
from src.models.training.prediction import extract_prediction_array
from src.models.training.scalers import (
    LocalTargetNormalizer,
    TargetScaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)


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
        prediction = extract_prediction_array(raw_prediction, self.prediction_key)
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
