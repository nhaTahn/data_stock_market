from __future__ import annotations

import numpy as np
from tensorflow import keras


def _split_prediction_key(prediction_key):
    if isinstance(prediction_key, (tuple, list)):
        if len(prediction_key) == 0:
            return None, None
        head = prediction_key[0]
        tail = tuple(prediction_key[1:]) if len(prediction_key) > 1 else None
        return head, tail
    return prediction_key, None


def _slice_prediction_array(array: np.ndarray, prediction_key: str | int | tuple[int, ...] | list[int] | None) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(-1)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.shape[-1] == 1:
        return array.reshape(-1)
    if prediction_key is None:
        raise ValueError("prediction_key is required for multi-column predictions.")
    if isinstance(prediction_key, (tuple, list)):
        sliced = array
        for key in prediction_key:
            if sliced.ndim <= 1:
                break
            sliced = np.take(sliced, int(key), axis=1)
        return np.asarray(sliced, dtype=np.float32).reshape(-1)
    return np.asarray(array[:, int(prediction_key)], dtype=np.float32).reshape(-1)


def extract_prediction_array(raw_prediction, prediction_key: str | int | None = None) -> np.ndarray:
    if isinstance(raw_prediction, dict):
        selector, remainder = _split_prediction_key(prediction_key)
        if selector is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output dict predictions.")
            raw_prediction = next(iter(raw_prediction.values()))
        else:
            raw_prediction = raw_prediction[selector]
            prediction_key = remainder
    elif isinstance(raw_prediction, (list, tuple)):
        selector, remainder = _split_prediction_key(prediction_key)
        if selector is None:
            if len(raw_prediction) != 1:
                raise ValueError("prediction_key is required for multi-output list predictions.")
            raw_prediction = raw_prediction[0]
        else:
            raw_prediction = raw_prediction[int(selector)]
            prediction_key = remainder
    array = np.asarray(raw_prediction, dtype=np.float32)
    return _slice_prediction_array(array, prediction_key)


def predict(model: keras.Model, x, prediction_key: str | int | None = None):
    raw_prediction = model.predict(x, verbose=0)
    return extract_prediction_array(raw_prediction, prediction_key)
