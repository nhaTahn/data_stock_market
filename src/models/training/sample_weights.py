from __future__ import annotations

import numpy as np


def build_magnitude_sample_weights(
    y: np.ndarray,
    strength: float = 1.5,
    reference_quantile: float = 0.75,
    clip_multiple: float = 3.0,
) -> np.ndarray:
    abs_y = np.abs(np.asarray(y, dtype=np.float32).reshape(-1))
    if len(abs_y) == 0:
        return np.ones(0, dtype=np.float32)

    valid = abs_y[np.isfinite(abs_y)]
    if len(valid) == 0:
        return np.ones_like(abs_y, dtype=np.float32)

    reference = float(np.quantile(valid, reference_quantile))
    reference = max(reference, 1e-4)
    normalized = np.clip(abs_y / reference, 0.0, clip_multiple)
    weights = 1.0 + strength * np.tanh(normalized)
    return weights.astype(np.float32)


def build_sign_magnitude_sample_weights(sample_weight: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if sample_weight is None:
        return None
    sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    return {
        "signed_prediction": sample_weight,
        "magnitude": sample_weight,
        "sign_prob": np.sqrt(sample_weight).astype(np.float32),
    }


def build_event_gated_sample_weights(
    sample_weight: np.ndarray | None,
    event_target: np.ndarray,
) -> dict[str, np.ndarray]:
    event_target = np.asarray(event_target, dtype=np.float32).reshape(-1)
    if sample_weight is None:
        base = np.ones_like(event_target, dtype=np.float32)
    else:
        base = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    event_focus = 0.25 + 0.75 * event_target
    magnitude_focus = 0.1 + 0.9 * event_target
    return {
        "signed_prediction": base,
        "event_prob": base,
        "sign_prob": (base * event_focus).astype(np.float32),
        "magnitude": (base * magnitude_focus).astype(np.float32),
    }
