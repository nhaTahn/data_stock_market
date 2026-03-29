from __future__ import annotations

try:
    import tensorflow as tf # type: ignore
    from tensorflow import keras # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is not installed in the current Python environment.\n"
        "Use a local virtualenv with tensorflow, then rerun this script.\n"
        "Example: pip install tensorflow"
    ) from exc


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def build_model(window_size: int, num_features: int, lstm_units: int, dropout: float, lr: float) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size, num_features)),
            keras.layers.LSTM(lstm_units),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def build_callbacks() -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
