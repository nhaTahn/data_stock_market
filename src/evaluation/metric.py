import numpy as np

# * Metric Fomular
# Error = actual_returns - prediction_returns 
# Base = actual_returns
# loss = np.quantile(Error, 0.5) + 0.5 np.quantile(Error, 0.9)
# base = loss(Base), abs = loss(Error), rel = 1 - abs / base

def loss_fn(values):
    """
    Compute the custom loss:
    loss = q50(values) + 0.5 * q90(values)
    """
    values = np.asarray(values, dtype=float)
    values = np.abs(values)
    return np.quantile(values, 0.5) + 0.5 * np.quantile(values, 0.9)


def align_prediction_actual(predict, actual):
    predict = np.asarray(predict, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(predict) != len(actual):
        raise ValueError("`predict` and `actual` must have the same length.")
    if len(actual) < 3:
        raise ValueError("Input arrays must contain at least 3 elements.")

    predict, actual = predict[1:], actual[1:]
    n = len(actual)
    aligned_predict, aligned_actual = [], []

    for i in range(1, n):
        aligned_predict.append(predict[i - 1])
        aligned_actual.append(actual[i])

    return np.asarray(aligned_predict, dtype=float), np.asarray(aligned_actual, dtype=float)


def directional_accuracy(predict, actual):
    aligned_predict, aligned_actual = align_prediction_actual(predict, actual)
    return float(np.mean(np.sign(aligned_predict) == np.sign(aligned_actual)))


def evaluate(predict, actual, dspl=False):
    """
    Evaluate prediction quality based on the custom metric.

    Metric definition:
        error = actual_returns - prediction_returns
        base  = actual_returns
        loss  = q50(x) + 0.5 * q90(x)
        rel   = 1 - loss(error) / loss(base)

    Args:
        predict: Sequence of predicted values.
        actual: Sequence of actual values.
        dspl: If True, print intermediate metric values.

    Returns:
        A dictionary containing error array, base array,
        base loss, absolute loss, and relative score.

    Raises:
        ValueError: If input lengths do not match.
        ValueError: If input length is too short.
        ZeroDivisionError: If base loss is zero.
    """
    aligned_predict, base = align_prediction_actual(predict, actual)
    error = base - aligned_predict

    base_loss = loss_fn(base)
    abs_loss = loss_fn(error)

    if base_loss == 0:
        raise ZeroDivisionError("Base loss is zero, relative score cannot be computed.")

    rel_score = 1 - abs_loss / base_loss

    if dspl:
        print(f"base_loss = {base_loss:.6f}")
        print(f"abs_loss  = {abs_loss:.6f}")
        print(f"rel_score = {rel_score:.6f}")

    return {
        "error": error,
        "base": base,
        "base_loss": base_loss,
        "abs_loss": abs_loss,
        "rel_score": rel_score,
        "directional_accuracy": directional_accuracy(predict, actual),
    }
