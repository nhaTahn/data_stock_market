from __future__ import annotations

import numpy as np


def primary_target_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y[:, :1].reshape(-1, 1)


def build_sign_magnitude_targets(
    signed_target: np.ndarray,
    auxiliary_target: np.ndarray | None = None,
    use_log_magnitude: bool = True,
) -> dict[str, np.ndarray]:
    signed_target = np.asarray(signed_target, dtype=np.float32)
    primary_target = primary_target_array(auxiliary_target if auxiliary_target is not None else signed_target)
    magnitude_target = np.abs(primary_target)
    if use_log_magnitude:
        magnitude_target = np.log1p(magnitude_target)
    return {
        "signed_prediction": signed_target,
        "sign_prob": (primary_target >= 0).astype(np.float32),
        "magnitude": magnitude_target.astype(np.float32),
    }


def build_event_gated_targets(
    signed_target: np.ndarray,
    auxiliary_target: np.ndarray | None = None,
    event_threshold: float = 0.75,
    use_log_magnitude: bool = True,
) -> dict[str, np.ndarray]:
    signed_target = np.asarray(signed_target, dtype=np.float32)
    primary_target = primary_target_array(auxiliary_target if auxiliary_target is not None else signed_target)
    abs_y = np.abs(primary_target)
    magnitude_target = np.log1p(abs_y) if use_log_magnitude else abs_y
    event_target = (abs_y >= event_threshold).astype(np.float32)
    return {
        "signed_prediction": signed_target.astype(np.float32),
        "event_prob": event_target.astype(np.float32),
        "sign_prob": (primary_target >= 0).astype(np.float32),
        "magnitude": magnitude_target.astype(np.float32),
    }
