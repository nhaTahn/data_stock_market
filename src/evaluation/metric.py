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


def _iter_group_slices(group_ids):
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("`group_ids` must be a 1D array.")
    if len(group_ids) == 0:
        return

    start = 0
    for idx in range(1, len(group_ids)):
        if group_ids[idx] != group_ids[idx - 1]:
            yield start, idx
            start = idx
    yield start, len(group_ids)


def _align_single_group(predict, actual):
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


def align_prediction_actual(predict, actual, group_ids=None):
    if group_ids is None:
        return _align_single_group(predict, actual)

    predict = np.asarray(predict, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(predict) != len(actual):
        raise ValueError("`predict` and `actual` must have the same length.")
    if len(group_ids) != len(actual):
        raise ValueError("`group_ids` and `actual` must have the same length.")

    aligned_predict_parts = []
    aligned_actual_parts = []
    for start, end in _iter_group_slices(group_ids):
        if end - start < 3:
            continue
        group_predict, group_actual = _align_single_group(predict[start:end], actual[start:end])
        aligned_predict_parts.append(group_predict)
        aligned_actual_parts.append(group_actual)

    if not aligned_predict_parts:
        raise ValueError("Grouped inputs must contain at least one group with 3 elements.")

    return np.concatenate(aligned_predict_parts), np.concatenate(aligned_actual_parts)


def directional_accuracy(predict, actual, group_ids=None):
    aligned_predict, aligned_actual = align_prediction_actual(predict, actual, group_ids=group_ids)
    return float(np.mean(np.sign(aligned_predict) == np.sign(aligned_actual)))


def evaluate(predict, actual, dspl=False, group_ids=None):
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
    aligned_predict, base = align_prediction_actual(predict, actual, group_ids=group_ids)
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
        "directional_accuracy": directional_accuracy(predict, actual, group_ids=group_ids),
    }
