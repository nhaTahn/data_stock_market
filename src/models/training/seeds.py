from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def set_global_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
