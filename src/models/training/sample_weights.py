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


def build_inv_volatility_sample_weights(
    scale_values: np.ndarray,
    strength: float = 1.0,
    reference_quantile: float = 0.5,
    clip_multiple: float = 3.0,
) -> np.ndarray:
    """Downweight high-volatility days.

    L4 of plan: dùng rolling volatility (vd `volatility_20`) đã có trong
    feature/target normalizer để tính weight ngược với vol:

        w_i = 1 / (1 + strength * tanh(|vol_i| / ref - 1))   when vol > ref
        w_i ≈ 1                                              when vol <= ref

    Motivation: rel_score loss là quantile-based (q50 + 0.5·q90). Ngày bão
    (vol cao) thường có target |y| lớn → magnitude weighting nâng weight →
    model dồn capacity vào outlier. Inv-volatility làm ngược lại: giảm
    weight ngày bão để model học median behavior tốt hơn.

    Args:
        scale_values: per-sample volatility proxy (same shape as y).
            Typically the same column used for `local_target_normalizer`
            (e.g. `volatility_20`).
        strength: how aggressively to downweight high-vol days.
        reference_quantile: pivot quantile of scale_values for normalization.
        clip_multiple: cap on normalized ratio.
    """
    abs_scale = np.abs(np.asarray(scale_values, dtype=np.float32).reshape(-1))
    if len(abs_scale) == 0:
        return np.ones(0, dtype=np.float32)
    valid = abs_scale[np.isfinite(abs_scale) & (abs_scale > 0)]
    if len(valid) == 0:
        return np.ones_like(abs_scale, dtype=np.float32)
    reference = max(float(np.quantile(valid, reference_quantile)), 1e-4)
    normalized = np.clip(abs_scale / reference, 0.0, clip_multiple)
    # weight is 1.0 at reference vol, decreasing as vol grows.
    weights = 1.0 / (1.0 + strength * np.tanh(np.maximum(normalized - 1.0, 0.0)))
    # NaN fallback to 1.0 — never zero out a sample because of missing vol.
    weights = np.where(np.isfinite(weights), weights, 1.0)
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


def balance_sample_weights_by_group(
    sample_weight: np.ndarray | None,
    group_labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(group_labels).reshape(-1)
    if len(labels) == 0:
        return np.ones(0, dtype=np.float32)

    if sample_weight is None:
        base = np.ones(len(labels), dtype=np.float32)
    else:
        base = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
        if len(base) != len(labels):
            raise ValueError("sample_weight and group_labels must have the same length.")

    unique_labels, counts = np.unique(labels.astype(str), return_counts=True)
    if len(unique_labels) <= 1:
        return base.astype(np.float32)

    total_rows = float(len(labels))
    group_count = float(len(unique_labels))
    balance_map = {
        label: total_rows / (group_count * float(count))
        for label, count in zip(unique_labels.tolist(), counts.tolist())
    }
    balance = np.asarray([balance_map[str(label)] for label in labels], dtype=np.float32)
    weights = base * balance
    weight_mean = float(np.mean(weights))
    if np.isfinite(weight_mean) and weight_mean > 0.0:
        weights = weights / weight_mean
    return weights.astype(np.float32)


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
