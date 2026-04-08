from __future__ import annotations

import numpy as np
from tensorflow import keras


def extract_prediction_array(raw_prediction, prediction_key: str | int | None = None) -> np.ndarray:
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
    array = np.asarray(raw_prediction, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(-1)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.shape[-1] == 1:
        return array.reshape(-1)
    if prediction_key is None:
        raise ValueError("prediction_key is required for multi-column predictions.")
    return np.asarray(array[:, int(prediction_key)], dtype=np.float32).reshape(-1)


def predict(model: keras.Model, x, prediction_key: str | int | None = None):
    raw_prediction = model.predict(x, verbose=0)
    return extract_prediction_array(raw_prediction, prediction_key)
