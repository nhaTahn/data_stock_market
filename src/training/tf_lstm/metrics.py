from __future__ import annotations

import numpy as np
import pandas as pd

from tf_lstm.config import RETURN_DIRECTION_THRESHOLD


def baseline_predict(targets_scaled: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    del target_mean, target_std
    return np.zeros_like(targets_scaled, dtype=np.float32)


def invert_target_scale(values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    return values * target_std + target_mean


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    directional_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    active_mask = np.abs(y_true) > RETURN_DIRECTION_THRESHOLD
    if np.any(active_mask):
        thresholded_directional_accuracy = float(
            np.mean(np.sign(y_true[active_mask]) == np.sign(y_pred[active_mask]))
        )
    else:
        thresholded_directional_accuracy = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "thresholded_directional_accuracy": thresholded_directional_accuracy,
    }
