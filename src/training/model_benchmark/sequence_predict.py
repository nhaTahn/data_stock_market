from __future__ import annotations

import numpy as np

from tf_lstm.metrics import compute_metrics, invert_target_scale

try:
    from tensorflow import keras  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is not installed in the current Python environment.\n"
        "Use a local virtualenv with tensorflow, then rerun this script.\n"
        "Example: pip install tensorflow"
    ) from exc


def load_keras_model(model_path):
    return keras.models.load_model(model_path)


def predict_returns(model, batch, target_mean: float, target_std: float) -> np.ndarray:
    pred_scaled = model.predict(batch.features, verbose=0).reshape(-1)
    return invert_target_scale(pred_scaled, target_mean, target_std)


def evaluate_return_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return compute_metrics(y_true, y_pred)

