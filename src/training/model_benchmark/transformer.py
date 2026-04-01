from __future__ import annotations

import numpy as np
import pandas as pd

from tf_lstm.metrics import compute_metrics, invert_target_scale
from tf_lstm.model import build_callbacks, set_seed

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "TensorFlow is not installed in the current Python environment.\n"
        "Use a local virtualenv with tensorflow, then rerun this script.\n"
        "Example: pip install tensorflow"
    ) from exc


def build_transformer_model(
    window_size: int,
    num_features: int,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    lr: float,
) -> keras.Model:
    inputs = keras.layers.Input(shape=(window_size, num_features))
    x = inputs

    attention_output = keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout,
    )(x, x)
    x = keras.layers.Add()([x, attention_output])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ff = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = keras.layers.Dropout(dropout)(ff)
    ff = keras.layers.Dense(num_features)(ff)
    x = keras.layers.Add()([x, ff])
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(ff_dim, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def run_transformer(bundle, args) -> tuple[dict[str, dict[str, float]], pd.DataFrame, keras.Model]:
    np.random.seed(args.seed)
    set_seed(args.seed)

    model = build_transformer_model(
        window_size=args.window_size,
        num_features=len(bundle.feature_columns),
        head_size=args.transformer_head_size,
        num_heads=args.transformer_num_heads,
        ff_dim=args.transformer_ff_dim,
        dropout=args.dropout,
        lr=args.lr,
    )

    history = model.fit(
        bundle.train_seq.features,
        bundle.train_seq.targets,
        validation_data=(bundle.val_seq.features, bundle.val_seq.targets),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=build_callbacks(),
    )

    train_pred = model.predict(bundle.train_seq.features, verbose=0).reshape(-1)
    val_pred = model.predict(bundle.val_seq.features, verbose=0).reshape(-1)
    test_pred = model.predict(bundle.test_seq.features, verbose=0).reshape(-1)

    metrics = {
        "train": compute_metrics(
            bundle.train_targets,
            invert_target_scale(train_pred, bundle.target_mean, bundle.target_std),
        ),
        "val": compute_metrics(
            bundle.val_targets,
            invert_target_scale(val_pred, bundle.target_mean, bundle.target_std),
        ),
        "test": compute_metrics(
            bundle.test_targets,
            invert_target_scale(test_pred, bundle.target_mean, bundle.target_std),
        ),
    }
    return metrics, pd.DataFrame(history.history), model

