from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _normalize_unit_stack(lstm_units: int | list[int]) -> list[int]:
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")
    if len(unit_stack) > 2:
        raise ValueError("Shallow PCIE-LSTM supports at most 2 LSTM layers.")
    return [int(units) for units in unit_stack]


@dataclass(frozen=True)
class DataPreprocessor:
    base_columns: tuple[str, ...]
    context_columns: tuple[str, ...] = ()
    ma_windows: tuple[int, ...] = (10, 20)
    column_aliases: dict[str, tuple[str, ...]] | None = None

    def output_feature_columns(self, df: pd.DataFrame) -> tuple[str, ...]:
        self._validate_columns(df)
        columns: list[str] = []
        for column in self.base_columns:
            columns.append(f"pcie_level_{column}")
            columns.append(f"pcie_delta_{column}")
        for window in self.ma_windows:
            columns.append(f"pcie_ma{window}")
        for column in self.context_columns:
            if column in df.columns:
                columns.append(f"pcie_ctx_{column}")
        return tuple(columns)

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(df)
        work = df.sort_values(["code", "Date"], kind="stable").copy()
        source_map = self._resolve_source_columns(df)
        close_series = work[source_map["close"]].astype(float)

        for column in self.base_columns:
            source_column = source_map[column]
            series = work[source_column].astype(float)
            work[f"pcie_level_{column}"] = series
            delta = work.groupby("code", sort=False)[source_column].pct_change()
            delta = delta.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            work[f"pcie_delta_{column}"] = delta.astype(np.float32)

        for window in self.ma_windows:
            ma = close_series.groupby(work["code"], sort=False).transform(
                lambda s, w=window: s.rolling(w, min_periods=w).mean()
            )
            work[f"pcie_ma{window}"] = ma.astype(np.float32)

        for column in self.context_columns:
            if column in work.columns:
                work[f"pcie_ctx_{column}"] = work[column].astype(float)
        return work

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = {"Date", "code", "close"}
        missing = [column for column in required if column not in df.columns]
        missing.extend(
            [
                column
                for column in self.base_columns
                if self._resolve_column_name(column, df.columns) is None
            ]
        )
        if missing:
            raise ValueError(f"Missing columns for PCIE-lite preprocessing: {missing}")

    def _resolve_source_columns(self, df: pd.DataFrame) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for column in self.base_columns:
            source_column = self._resolve_column_name(column, df.columns)
            if source_column is None:
                raise ValueError(f"Missing source column for PCIE-lite base feature '{column}'.")
            resolved[column] = source_column
        return resolved

    def _resolve_column_name(self, column: str, available_columns) -> str | None:
        if column in available_columns:
            return column
        alias_map = self.column_aliases or {
            "volume": ("volume_match",),
            "close": ("adjust",),
        }
        for candidate in alias_map.get(column, ()):
            if candidate in available_columns:
                return candidate
        return None


@keras.utils.register_keras_serializable(package="custom")
class PatchingLayer(layers.Layer):
    def __init__(self, patch_length: int, stride: int, **kwargs):
        super().__init__(**kwargs)
        if patch_length <= 0:
            raise ValueError("patch_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.patch_length = int(patch_length)
        self.stride = int(stride)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        sequence = tf.transpose(tf.cast(inputs, tf.float32), [0, 2, 1])
        seq_len = tf.shape(sequence)[-1]
        patch_length = tf.constant(self.patch_length, dtype=tf.int32)
        stride = tf.constant(self.stride, dtype=tf.int32)

        rem = tf.math.mod(tf.maximum(seq_len - patch_length, 0), stride)
        extra_pad = tf.where(tf.equal(rem, 0), 0, stride - rem)
        pad_len = tf.where(seq_len < patch_length, patch_length - seq_len, extra_pad)
        paddings = tf.concat(
            [
                tf.constant([[0, 0], [0, 0]], dtype=tf.int32),
                tf.reshape(tf.stack([0, pad_len]), [1, 2]),
            ],
            axis=0,
        )
        padded = tf.pad(sequence, paddings)
        patches = tf.signal.frame(padded, frame_length=self.patch_length, frame_step=self.stride, axis=-1)
        return tf.transpose(patches, [0, 2, 1, 3])

    def get_config(self) -> dict[str, int | str]:
        config = super().get_config()
        config.update({"patch_length": self.patch_length, "stride": self.stride})
        return config


@keras.utils.register_keras_serializable(package="custom")
class SharedLinearATL(layers.Layer):
    def __init__(self, d_patch: int, **kwargs):
        super().__init__(**kwargs)
        if d_patch <= 0:
            raise ValueError("d_patch must be positive.")
        self.d_patch = int(d_patch)
        self.shared_projection: layers.Dense | None = None

    def build(self, input_shape) -> None:
        self.shared_projection = layers.Dense(self.d_patch, name=f"{self.name}_shared_linear")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.shared_projection is None:
            raise RuntimeError("SharedLinearATL must be built before calling.")
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        patch_count = input_shape[1]
        channel_count = input_shape[2]
        patch_length = input_shape[3]

        flat = tf.reshape(inputs, [-1, patch_length])
        projected = self.shared_projection(flat)
        projected = tf.reshape(projected, [batch_size, patch_count, channel_count, self.d_patch])
        return tf.reshape(projected, [batch_size, patch_count, channel_count * self.d_patch])

    def get_config(self) -> dict[str, int | str]:
        config = super().get_config()
        config.update({"d_patch": self.d_patch})
        return config


@keras.utils.register_keras_serializable(package="custom")
class ShallowPCIELSTM(layers.Layer):
    def __init__(
        self,
        lstm_units: int | list[int],
        future_steps: int,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lstm_units = _normalize_unit_stack(lstm_units)
        self.future_steps = int(future_steps)
        self.dropout = float(dropout)
        self.lstm_stack = [
            layers.LSTM(units, return_sequences=True, name=f"{self.name}_lstm_{idx}")
            for idx, units in enumerate(self.lstm_units)
        ]
        self.dropout_layers = [
            layers.Dropout(self.dropout, name=f"{self.name}_dropout_{idx}")
            for idx in range(len(self.lstm_units))
        ]
        self.flatten = layers.Flatten(name=f"{self.name}_flatten")
        self.output_head = layers.Dense(self.future_steps, name=f"{self.name}_forecast_head")

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = inputs
        for lstm_layer, dropout_layer in zip(self.lstm_stack, self.dropout_layers):
            x = lstm_layer(x, training=training)
            if self.dropout > 0:
                x = dropout_layer(x, training=training)
        x = self.flatten(x)
        return self.output_head(x)

    def get_config(self) -> dict[str, int | float | str | list[int]]:
        config = super().get_config()
        config.update(
            {
                "lstm_units": self.lstm_units,
                "future_steps": self.future_steps,
                "dropout": self.dropout,
            }
        )
        return config


def build_pcie_lite_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    patch_length: int,
    patch_stride: int,
    d_patch: int,
    future_steps: int = 3,
    dropout: float = 0.2,
) -> keras.Model:
    inputs = layers.Input(shape=(window_size, num_features), name="pcie_lite_input")
    patched = PatchingLayer(
        patch_length=patch_length,
        stride=patch_stride,
        name="pcie_lite_patching",
    )(inputs)
    mixed = SharedLinearATL(
        d_patch=d_patch,
        name="pcie_lite_shared_linear_atl",
    )(patched)
    outputs = ShallowPCIELSTM(
        lstm_units=lstm_units,
        future_steps=future_steps,
        dropout=dropout,
        name="pcie_lite_shallow_lstm",
    )(mixed)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pcie_lite_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
    )
    return model
