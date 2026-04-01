from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tf_lstm.data import (
    SequenceBatch,
    build_sequences,
    fit_feature_scaler,
    fit_target_scaler,
    load_dataset,
    scale_split,
    split_dataset,
)
from tf_lstm.metrics import baseline_predict, compute_metrics, invert_target_scale


@dataclass
class SequenceDataBundle:
    feature_columns: list[str]
    feature_groups: list[str]
    target_mean: float
    target_std: float
    train_seq: SequenceBatch
    val_seq: SequenceBatch
    test_seq: SequenceBatch
    train_targets: np.ndarray
    val_targets: np.ndarray
    test_targets: np.ndarray
    baseline_metrics: dict[str, dict[str, float]]


def prepare_sequence_data(
    data_path: Path,
    train_end: str,
    val_end: str,
    window_size: int,
    feature_groups: list[str] | tuple[str, ...] | None,
    max_tickers: int | None = None,
) -> SequenceDataBundle:
    df, feature_columns, normalized_groups = load_dataset(data_path, feature_groups=feature_groups)
    if max_tickers is not None:
        selected_codes = sorted(df["code"].unique())[:max_tickers]
        df = df[df["code"].isin(selected_codes)].copy()
    train_df, val_df, test_df = split_dataset(df, train_end=train_end, val_end=val_end)
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the splits is empty. Adjust split dates.")

    feature_mean, feature_std = fit_feature_scaler(train_df, feature_columns)
    target_mean, target_std = fit_target_scaler(train_df)

    scaled_train = scale_split(train_df, feature_mean, feature_std, target_mean, target_std, feature_columns)
    scaled_val = scale_split(val_df, feature_mean, feature_std, target_mean, target_std, feature_columns)
    scaled_test = scale_split(test_df, feature_mean, feature_std, target_mean, target_std, feature_columns)

    train_seq = build_sequences(scaled_train, window_size=window_size, feature_columns=feature_columns)
    val_seq = build_sequences(scaled_val, window_size=window_size, feature_columns=feature_columns)
    test_seq = build_sequences(scaled_test, window_size=window_size, feature_columns=feature_columns)
    if len(train_seq.targets) == 0 or len(val_seq.targets) == 0 or len(test_seq.targets) == 0:
        raise ValueError("Sequence generation returned an empty split.")

    train_targets = invert_target_scale(train_seq.targets, target_mean, target_std)
    val_targets = invert_target_scale(val_seq.targets, target_mean, target_std)
    test_targets = invert_target_scale(test_seq.targets, target_mean, target_std)

    baseline_metrics = {
        "train": compute_metrics(train_targets, baseline_predict(train_seq.targets, target_mean, target_std)),
        "val": compute_metrics(val_targets, baseline_predict(val_seq.targets, target_mean, target_std)),
        "test": compute_metrics(test_targets, baseline_predict(test_seq.targets, target_mean, target_std)),
    }

    return SequenceDataBundle(
        feature_columns=feature_columns,
        feature_groups=normalized_groups,
        target_mean=target_mean,
        target_std=target_std,
        train_seq=train_seq,
        val_seq=val_seq,
        test_seq=test_seq,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        baseline_metrics=baseline_metrics,
    )


def flatten_sequences(batch: SequenceBatch) -> np.ndarray:
    return batch.features.reshape(len(batch.features), -1)
