from __future__ import annotations

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is required for fk_lstm_classifier.\n"
        "Install it in your environment, for example: pip install tensorflow"
    ) from exc


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


class TemporalAttention(keras.layers.Layer):
    """Simple additive attention over the temporal axis."""

    def __init__(self, attention_units: int, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = keras.layers.Dense(attention_units, activation="tanh")
        self.weight_dense = keras.layers.Dense(1, activation=None)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        scores = self.weight_dense(self.score_dense(inputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context


def build_lstm_classifier(
    lookback: int = 240,
    feature_dim: int = 1,
    lstm_units: int = 25,
    dropout: float = 0.16,
    learning_rate: float = 1e-3,
    use_attention: bool = False,
) -> keras.Model:
    inputs = keras.layers.Input(shape=(lookback, feature_dim), name="returns_sequence")
    lstm_output = keras.layers.LSTM(
        units=lstm_units,
        dropout=dropout,
        recurrent_dropout=0.0,
        return_sequences=use_attention,
        name="lstm",
    )(inputs)

    if use_attention:
        hidden = TemporalAttention(attention_units=lstm_units, name="temporal_attention")(lstm_output)
    else:
        hidden = lstm_output

    outputs = keras.layers.Dense(2, activation="softmax", name="class_probs")(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name="fk_lstm_classifier")
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_early_stopping(patience: int = 10) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
