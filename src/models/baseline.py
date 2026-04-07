from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    ARIMA = None


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(flatten_sequences(x_train), y_train)
    return model


def predict_linear_regression(model: LinearRegression, x: np.ndarray) -> np.ndarray:
    return model.predict(flatten_sequences(x)).reshape(-1)


@dataclass
class ArimaSequenceWrapper:
    proxy_feature_index: int
    ar1_intercept: float
    ar1_coef: float
    calibration_intercept: float
    calibration_coef: float
    use_fast: bool = True
    arima_order: tuple[int, int, int] = (1, 0, 0)

    def _extract_proxy_series(self, sequence: np.ndarray) -> np.ndarray:
        proxy_series = np.asarray(sequence[:, self.proxy_feature_index], dtype=float)
        return np.nan_to_num(proxy_series, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_calibration(self, raw_forecast: float) -> float:
        forecast = self.calibration_intercept + self.calibration_coef * raw_forecast
        if not np.isfinite(forecast):
            return 0.0
        return float(forecast)

    def _predict_single_fast(self, sequence: np.ndarray) -> float:
        proxy_series = self._extract_proxy_series(sequence)
        raw_forecast = self.ar1_intercept + self.ar1_coef * proxy_series[-1]
        return self._apply_calibration(raw_forecast)

    def _predict_single_arima(self, sequence: np.ndarray) -> float:
        proxy_series = self._extract_proxy_series(sequence)
        if ARIMA is None or len(proxy_series) < 3:
            return self._predict_single_fast(sequence)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted = ARIMA(proxy_series, order=self.arima_order).fit()
                raw_forecast = float(fitted.forecast(steps=1)[0])
            except Exception:
                return self._predict_single_fast(sequence)
        return self._apply_calibration(raw_forecast)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictor = self._predict_single_fast if self.use_fast else self._predict_single_arima
        return np.asarray([predictor(sequence) for sequence in x], dtype=np.float32)


def _select_proxy_feature_index(x_train: np.ndarray, y_train: np.ndarray) -> int:
    last_step = np.asarray(x_train[:, -1, :], dtype=float)
    scores: list[float] = []
    for feature_idx in range(last_step.shape[1]):
        values = last_step[:, feature_idx]
        if np.std(values) == 0 or np.std(y_train) == 0:
            scores.append(0.0)
            continue
        corr = np.corrcoef(values, y_train)[0, 1]
        scores.append(float(abs(corr)) if np.isfinite(corr) else 0.0)
    return int(np.argmax(scores))


def _fit_ar1_coefficients(proxy_windows: np.ndarray) -> tuple[float, float]:
    lag_values = proxy_windows[:, :-1].reshape(-1)
    next_values = proxy_windows[:, 1:].reshape(-1)
    mask = np.isfinite(lag_values) & np.isfinite(next_values)
    if mask.sum() < 2:
        return 0.0, 1.0

    design = np.column_stack([np.ones(mask.sum()), lag_values[mask]])
    target = next_values[mask]
    try:
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    return float(coef[0]), float(coef[1])


def _fit_calibration(raw_prediction: np.ndarray, y_train: np.ndarray) -> tuple[float, float]:
    raw_prediction = np.asarray(raw_prediction, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    mask = np.isfinite(raw_prediction) & np.isfinite(y_train)
    if mask.sum() < 2:
        return 0.0, 1.0

    design = np.column_stack([np.ones(mask.sum()), raw_prediction[mask]])
    target = y_train[mask]
    try:
        coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0
    return float(coef[0]), float(coef[1])


def predict_arima_fast(x: np.ndarray, model: ArimaSequenceWrapper | None = None) -> np.ndarray:
    if model is None:
        return np.asarray([sequence[-1, -1] for sequence in x], dtype=np.float32)
    return np.asarray([model._predict_single_fast(sequence) for sequence in x], dtype=np.float32)


def fit_arima(x_train: np.ndarray, y_train: np.ndarray) -> ArimaSequenceWrapper:
    proxy_feature_index = _select_proxy_feature_index(x_train, y_train)
    proxy_windows = np.asarray(x_train[:, :, proxy_feature_index], dtype=float)
    ar1_intercept, ar1_coef = _fit_ar1_coefficients(proxy_windows)
    raw_prediction = ar1_intercept + ar1_coef * proxy_windows[:, -1]
    calibration_intercept, calibration_coef = _fit_calibration(raw_prediction, y_train)
    return ArimaSequenceWrapper(
        proxy_feature_index=proxy_feature_index,
        ar1_intercept=ar1_intercept,
        ar1_coef=ar1_coef,
        calibration_intercept=calibration_intercept,
        calibration_coef=calibration_coef,
        use_fast=True,
    )


def predict_arima(model: ArimaSequenceWrapper, x: np.ndarray) -> np.ndarray:
    return model.predict(x)
