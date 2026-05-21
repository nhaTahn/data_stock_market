from __future__ import annotations

import numpy as np


def primary_target_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y[:, :1].reshape(-1, 1)


def build_rank_targets(
    rank_target: np.ndarray,
    rank_group_ids: np.ndarray,
) -> np.ndarray:
    target = primary_target_array(rank_target)
    group_ids = np.asarray(rank_group_ids, dtype=np.float32).reshape(-1, 1)
    if len(target) != len(group_ids):
        raise ValueError("rank_target and rank_group_ids must have the same length.")
    return np.concatenate([target, group_ids], axis=1).astype(np.float32)


def build_aux_plain_targets(
    signed_target: np.ndarray,
    auxiliary_target: np.ndarray | None = None,
    use_log_magnitude: bool = True,
) -> dict[str, np.ndarray]:
    """Targets for the aux_plain architecture.

    - `pred`: passed signed_target unchanged (rel_score loss handles 1D/2D).
    - `sign_prob`: binary indicator of (y > 0).
    - `magnitude_aux`: |y| optionally log1p-transformed (matches signmag default).
    """
    signed_target = np.asarray(signed_target, dtype=np.float32)
    primary_target = primary_target_array(auxiliary_target if auxiliary_target is not None else signed_target)
    magnitude_target = np.abs(primary_target)
    if use_log_magnitude:
        magnitude_target = np.log1p(magnitude_target)
    return {
        "pred": signed_target,
        "sign_prob": (primary_target >= 0).astype(np.float32),
        "magnitude_aux": magnitude_target.astype(np.float32),
    }


def build_sign_magnitude_targets(
    signed_target: np.ndarray,
    auxiliary_target: np.ndarray | None = None,
    use_log_magnitude: bool = True,
    rank_target: np.ndarray | None = None,
    rank_group_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    signed_target = np.asarray(signed_target, dtype=np.float32)
    primary_target = primary_target_array(auxiliary_target if auxiliary_target is not None else signed_target)
    magnitude_target = np.abs(primary_target)
    if use_log_magnitude:
        magnitude_target = np.log1p(magnitude_target)
    targets = {
        "signed_prediction": signed_target,
        "sign_prob": (primary_target >= 0).astype(np.float32),
        "magnitude": magnitude_target.astype(np.float32),
    }
    if rank_target is not None or rank_group_ids is not None:
        if rank_target is None or rank_group_ids is None:
            raise ValueError("rank_target and rank_group_ids must both be provided for rank sidecar targets.")
        targets["rank_score"] = build_rank_targets(rank_target, rank_group_ids)
    return targets


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
