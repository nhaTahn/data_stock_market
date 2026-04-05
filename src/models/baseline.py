from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(flatten_sequences(x_train), y_train)
    return model


def predict_linear_regression(model: LinearRegression, x: np.ndarray) -> np.ndarray:
    return model.predict(flatten_sequences(x)).reshape(-1)
