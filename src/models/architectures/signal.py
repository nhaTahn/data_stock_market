from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.models.components.losses import CustomPinballLoss


LSTM_L2_FACTOR = 1e-5


def _normalize_unit_stack(lstm_units: int | list[int]) -> list[int]:
    unit_stack = [lstm_units] if isinstance(lstm_units, int) else list(lstm_units)
    if not unit_stack:
        raise ValueError("lstm_units must contain at least one layer size.")
    return [int(units) for units in unit_stack]


@keras.utils.register_keras_serializable(package="custom")
class DualDataPreprocessor(layers.Layer):
    def __init__(self, epsilon: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        values = tf.cast(inputs, tf.float32)
        prev = tf.concat([values[:, :1, :], values[:, :-1, :]], axis=1)
        denom = tf.maximum(tf.abs(prev), tf.cast(self.epsilon, tf.float32))
        pct_change = (values - prev) / denom
        pct_change = tf.where(tf.math.is_finite(pct_change), pct_change, tf.zeros_like(pct_change))
        return tf.concat([values, pct_change], axis=-1)

    def get_config(self) -> dict[str, float | str]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@keras.utils.register_keras_serializable(package="custom")
class Patching(layers.Layer):
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
class ATLChannelMixing(layers.Layer):
    def __init__(
        self,
        d_patch: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if d_patch <= 0:
            raise ValueError("d_patch must be positive.")
        self.d_patch = int(d_patch)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.dropout = float(dropout)
        self.activation = activation
        self.patch_projection: keras.Sequential | None = None
        self.mixing_dropout = layers.Dropout(self.dropout) if self.dropout > 0 else None

    def build(self, input_shape) -> None:
        patch_length = int(input_shape[-1])
        hidden_dim = self.hidden_dim or max(self.d_patch * 2, patch_length * 2)
        self.patch_projection = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation=self.activation, name=f"{self.name}_dense_1"),
                layers.Dense(self.d_patch, name=f"{self.name}_dense_2"),
            ],
            name=f"{self.name}_mlp",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if self.patch_projection is None:
            raise RuntimeError("ATLChannelMixing must be built before calling.")
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        patch_count = input_shape[1]
        channel_count = input_shape[2]
        patch_length = input_shape[3]

        flat = tf.reshape(inputs, [-1, patch_length])
        projected = self.patch_projection(flat, training=training)
        projected = tf.reshape(projected, [batch_size, patch_count, channel_count, self.d_patch])
        mixed = tf.reshape(projected, [batch_size, patch_count, channel_count * self.d_patch])
        if self.mixing_dropout is not None:
            mixed = self.mixing_dropout(mixed, training=training)
        return mixed

    def get_config(self) -> dict[str, int | float | str | None]:
        config = super().get_config()
        config.update(
            {
                "d_patch": self.d_patch,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="custom")
class AttentionLSTM(layers.Layer):
    def __init__(
        self,
        lstm_units: int | list[int],
        future_steps: int,
        attention_heads: int = 2,
        attention_key_dim: int = 16,
        attention_ff_dim: int | None = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lstm_units = _normalize_unit_stack(lstm_units)
        self.future_steps = int(future_steps)
        self.attention_heads = int(attention_heads)
        self.attention_key_dim = int(attention_key_dim)
        self.attention_ff_dim = None if attention_ff_dim is None else int(attention_ff_dim)
        self.dropout = float(dropout)

        self.lstm_stack = [
            layers.LSTM(
                units,
                return_sequences=True,
                kernel_regularizer=regularizers.l2(LSTM_L2_FACTOR),
                recurrent_regularizer=regularizers.l2(LSTM_L2_FACTOR),
                name=f"{self.name}_lstm_{idx}",
            )
            for idx, units in enumerate(self.lstm_units)
        ]
        self.lstm_dropouts = [
            layers.Dropout(self.dropout, name=f"{self.name}_lstm_dropout_{idx}")
            for idx in range(len(self.lstm_units))
        ]
        self.temporal_attention = layers.MultiHeadAttention(
            num_heads=max(1, self.attention_heads),
            key_dim=max(1, self.attention_key_dim),
            dropout=max(0.0, min(0.5, self.dropout)),
            name=f"{self.name}_mha",
        )
        self.attn_batch_norm = layers.BatchNormalization(name=f"{self.name}_attn_bn")
        self.ffn_batch_norm = layers.BatchNormalization(name=f"{self.name}_ffn_bn")
        self.sequence_flatten = layers.Flatten(name=f"{self.name}_flatten")
        self.output_dropout = layers.Dropout(self.dropout, name=f"{self.name}_output_dropout")
        self.ffn_dense_1: layers.Dense | None = None
        self.ffn_dense_2: layers.Dense | None = None
        self.ffn_dropout: layers.Dropout | None = None
        self.q50_head: layers.Dense | None = None
        self.q90_delta_head: layers.Dense | None = None

    def build(self, input_shape) -> None:
        sequence_dim = int(self.lstm_units[-1])
        ff_dim = self.attention_ff_dim or max(sequence_dim * 2, 32)
        self.ffn_dense_1 = layers.Dense(ff_dim, activation="gelu", name=f"{self.name}_ffn_dense_1")
        self.ffn_dropout = layers.Dropout(self.dropout, name=f"{self.name}_ffn_dropout")
        self.ffn_dense_2 = layers.Dense(sequence_dim, name=f"{self.name}_ffn_dense_2")
        self.q50_head = layers.Dense(self.future_steps, name=f"{self.name}_q50")
        self.q90_delta_head = layers.Dense(self.future_steps, activation="softplus", name=f"{self.name}_q90_delta")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if any(layer is None for layer in (self.ffn_dense_1, self.ffn_dense_2, self.ffn_dropout, self.q50_head, self.q90_delta_head)):
            raise RuntimeError("AttentionLSTM must be built before calling.")
        x = inputs
        for lstm_layer, dropout_layer in zip(self.lstm_stack, self.lstm_dropouts):
            x = lstm_layer(x, training=training)
            if self.dropout > 0:
                x = dropout_layer(x, training=training)

        attended = self.temporal_attention(x, x, training=training)
        x = self.attn_batch_norm(x + attended, training=training)

        ff = self.ffn_dense_1(x)
        ff = self.ffn_dropout(ff, training=training)
        ff = self.ffn_dense_2(ff)
        x = self.ffn_batch_norm(x + ff, training=training)

        flat = self.sequence_flatten(x)
        if self.dropout > 0:
            flat = self.output_dropout(flat, training=training)

        q50 = self.q50_head(flat)
        q90_delta = self.q90_delta_head(flat)
        q90 = q50 + q90_delta
        return tf.stack([q50, q90], axis=-1)

    def get_config(self) -> dict[str, int | float | str | list[int] | None]:
        config = super().get_config()
        config.update(
            {
                "lstm_units": self.lstm_units,
                "future_steps": self.future_steps,
                "attention_heads": self.attention_heads,
                "attention_key_dim": self.attention_key_dim,
                "attention_ff_dim": self.attention_ff_dim,
                "dropout": self.dropout,
            }
        )
        return config


def build_signal_attention_lstm_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    patch_length: int,
    patch_stride: int,
    d_patch: int,
    future_steps: int = 1,
    attention_heads: int = 2,
    attention_key_dim: int = 16,
    attention_ff_dim: int | None = None,
    dropout: float = 0.3,
) -> keras.Model:
    inputs = layers.Input(shape=(window_size, num_features), name="signal_input")
    preprocessed = DualDataPreprocessor(name="signal_dual_preprocess")(inputs)
    patches = Patching(
        patch_length=patch_length,
        stride=patch_stride,
        name="signal_patching",
    )(preprocessed)
    mixed = ATLChannelMixing(
        d_patch=d_patch,
        dropout=dropout,
        name="signal_atl_channel_mixing",
    )(patches)
    outputs = AttentionLSTM(
        lstm_units=lstm_units,
        future_steps=future_steps,
        attention_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        attention_ff_dim=attention_ff_dim,
        dropout=dropout,
        name="signal_attention_lstm",
    )(mixed)

    model = keras.Model(inputs=inputs, outputs=outputs, name="signal_attention_quantile")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=CustomPinballLoss(),
    )
    return model
