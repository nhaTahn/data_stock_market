from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Union

import numpy as np
from tensorflow import keras


Payload = Union[np.ndarray, Dict[str, np.ndarray]]


def _validate_payload_length(name: str, payload: Payload | None, expected_length: int) -> None:
    if payload is None:
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if len(value) != expected_length:
                raise ValueError(f"{name}[{key!r}] has length {len(value)} but expected {expected_length}.")
        return
    if len(payload) != expected_length:
        raise ValueError(f"{name} has length {len(payload)} but expected {expected_length}.")


def _slice_payload(payload: Payload, indices: np.ndarray) -> Payload:
    if isinstance(payload, dict):
        return {key: np.asarray(value)[indices] for key, value in payload.items()}
    return np.asarray(payload)[indices]


def build_group_ids(values: Sequence[object]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Group values must be a 1D array.")
    if len(array) == 0:
        return np.zeros(0, dtype=np.int32)

    _, inverse = np.unique(array, return_inverse=True)
    return inverse.astype(np.int32)


class DateGroupedBatchSequence(keras.utils.Sequence):
    def __init__(
        self,
        x: np.ndarray,
        y: Payload,
        batch_group_ids: np.ndarray,
        batch_size: int,
        sample_weight: Payload | None = None,
        shuffle_dates: bool = True,
        min_groups_per_batch: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        x = np.asarray(x)
        batch_group_ids = np.asarray(batch_group_ids).reshape(-1)
        if x.shape[0] != len(batch_group_ids):
            raise ValueError("`x` and `batch_group_ids` must have the same number of rows.")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if min_groups_per_batch <= 0:
            raise ValueError("`min_groups_per_batch` must be positive.")

        _validate_payload_length("y", y, x.shape[0])
        _validate_payload_length("sample_weight", sample_weight, x.shape[0])

        self.x = x
        self.y = y
        self.sample_weight = sample_weight
        self.batch_size = int(batch_size)
        self.shuffle_dates = bool(shuffle_dates)
        self.min_groups_per_batch = int(min_groups_per_batch)
        self._group_indices = self._build_group_indices(batch_group_ids)
        self._batches: list[np.ndarray] = []
        self.on_epoch_end()

    @staticmethod
    def _build_group_indices(batch_group_ids: np.ndarray) -> list[np.ndarray]:
        groups: dict[int, list[int]] = {}
        for row_idx, group_id in enumerate(batch_group_ids.astype(np.int32).tolist()):
            groups.setdefault(int(group_id), []).append(row_idx)
        return [np.asarray(indices, dtype=np.int32) for _, indices in sorted(groups.items(), key=lambda item: item[0])]

    def _pack_batches(self, group_order: np.ndarray) -> list[np.ndarray]:
        batches: list[np.ndarray] = []
        current_groups: list[np.ndarray] = []
        current_count = 0

        for group_idx in group_order.tolist():
            group_indices = self._group_indices[int(group_idx)]
            proposed_count = current_count + len(group_indices)
            should_flush = (
                len(current_groups) >= self.min_groups_per_batch
                and current_count > 0
                and proposed_count > self.batch_size
            )
            if should_flush:
                batches.append(np.concatenate(current_groups))
                current_groups = []
                current_count = 0

            current_groups.append(group_indices)
            current_count += len(group_indices)

        if current_groups:
            batches.append(np.concatenate(current_groups))
        return batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int):
        indices = self._batches[index]
        x_batch = self.x[indices]
        y_batch = _slice_payload(self.y, indices)
        if self.sample_weight is None:
            return x_batch, y_batch
        sample_weight_batch = _slice_payload(self.sample_weight, indices)
        return x_batch, y_batch, sample_weight_batch

    def on_epoch_end(self) -> None:
        group_order = np.arange(len(self._group_indices), dtype=np.int32)
        if self.shuffle_dates and len(group_order) > 1:
            group_order = np.random.permutation(group_order)
        self._batches = self._pack_batches(group_order)
