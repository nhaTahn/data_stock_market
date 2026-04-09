from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def _linear_quantile(values: tf.Tensor, q: float) -> tf.Tensor:
    values = tf.reshape(tf.cast(values, tf.float32), [-1])
    values = tf.sort(values)
    count = tf.shape(values)[0]

    def compute() -> tf.Tensor:
        max_index = tf.cast(count - 1, tf.float32)
        rank = tf.cast(q, tf.float32) * max_index
        lower = tf.cast(tf.math.floor(rank), tf.int32)
        upper = tf.cast(tf.math.ceil(rank), tf.int32)
        lower_value = tf.gather(values, lower)
        upper_value = tf.gather(values, upper)
        mix = rank - tf.math.floor(rank)
        return lower_value * (1.0 - mix) + upper_value * mix

    return tf.cond(count > 0, compute, lambda: tf.constant(0.0, dtype=tf.float32))


def _weighted_linear_quantile(values: tf.Tensor, weights: tf.Tensor, q: float) -> tf.Tensor:
    values = tf.reshape(tf.cast(values, tf.float32), [-1])
    weights = tf.reshape(tf.cast(weights, tf.float32), [-1])
    count = tf.shape(values)[0]

    def compute() -> tf.Tensor:
        safe_weights = tf.maximum(weights, tf.constant(1e-6, dtype=tf.float32))
        order = tf.argsort(values, axis=0, direction="ASCENDING")
        sorted_values = tf.gather(values, order)
        sorted_weights = tf.gather(safe_weights, order)
        cumulative = tf.cumsum(sorted_weights)
        total = cumulative[-1]
        target = tf.cast(q, tf.float32) * total
        index = tf.cast(tf.argmax(tf.cast(cumulative >= target, tf.int32)), tf.int32)
        return tf.gather(sorted_values, index)

    return tf.cond(count > 0, compute, lambda: tf.constant(0.0, dtype=tf.float32))


def _rel_score_quantile_loss(values: tf.Tensor) -> tf.Tensor:
    abs_values = tf.abs(tf.reshape(tf.cast(values, tf.float32), [-1]))
    return _linear_quantile(abs_values, 0.5) + 0.5 * _linear_quantile(abs_values, 0.9)


def _weighted_rel_score_quantile_loss(values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    abs_values = tf.abs(tf.reshape(tf.cast(values, tf.float32), [-1]))
    weights = tf.reshape(tf.cast(weights, tf.float32), [-1])
    return _weighted_linear_quantile(abs_values, weights, 0.5) + 0.5 * _weighted_linear_quantile(abs_values, weights, 0.9)


def _split_rel_score_target(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor | None]:
    y_true = tf.cast(y_true, tf.float32)
    if y_true.shape.rank is None:
        y_true = tf.reshape(y_true, [-1, 1])
    elif y_true.shape.rank == 1:
        y_true = tf.reshape(y_true, [-1, 1])
    else:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])

    primary = y_true[:, :1]
    aux_scale = y_true[:, 1:2]
    return primary, aux_scale


def _restore_rel_score_scale(
    values: tf.Tensor,
    target_mean: float,
    target_std: float,
    use_target_scaler: bool,
    local_scale: tf.Tensor | None,
    local_scale_floor: float,
) -> tf.Tensor:
    restored = tf.cast(values, tf.float32)
    if use_target_scaler:
        restored = restored * tf.cast(target_std, tf.float32) + tf.cast(target_mean, tf.float32)
    if local_scale is not None:
        local_scale = tf.cast(local_scale, tf.float32)

        def apply_local_scale() -> tf.Tensor:
            denom = tf.maximum(tf.abs(local_scale), tf.cast(local_scale_floor, tf.float32))
            return restored * denom

        restored = tf.cond(
            tf.shape(local_scale)[1] > 0,
            apply_local_scale,
            lambda: restored,
        )
    return restored


def uses_rel_score_target(loss_name: str) -> bool:
    return loss_name in {"rel_score", "rel_score_sharp", "rel_score_weighted"}


@keras.utils.register_keras_serializable(package="custom")
class DirectionalHuberLoss(keras.losses.Loss):
    def __init__(self, delta: float = 0.01, penalty_weight: float = 20.0, name: str = "directional_huber_loss"):
        super().__init__(name=name)
        self.delta = delta
        self.penalty_weight = penalty_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return directional_huber_loss(
            y_true,
            y_pred,
            delta=self.delta,
            penalty_weight=self.penalty_weight,
        )

    def get_config(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "delta": self.delta,
            "penalty_weight": self.penalty_weight,
        }


@keras.utils.register_keras_serializable(package="custom")
class RelScoreLoss(keras.losses.Loss):
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        use_target_scaler: bool = False,
        local_scale_floor: float = 0.0,
        epsilon: float = 1e-6,
        name: str = "rel_score_loss",
    ):
        super().__init__(name=name)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.use_target_scaler = bool(use_target_scaler)
        self.local_scale_floor = float(local_scale_floor)
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        primary_true, local_scale = _split_rel_score_target(y_true)
        restored_true = _restore_rel_score_scale(
            primary_true,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        restored_pred = _restore_rel_score_scale(
            y_pred,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        abs_loss = _rel_score_quantile_loss(restored_true - restored_pred)
        base_loss = _rel_score_quantile_loss(restored_true)
        return abs_loss / tf.maximum(base_loss, tf.cast(self.epsilon, tf.float32))

    def get_config(self) -> dict[str, float | bool | str]:
        return {
            "name": self.name,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "use_target_scaler": self.use_target_scaler,
            "local_scale_floor": self.local_scale_floor,
            "epsilon": self.epsilon,
        }


@keras.utils.register_keras_serializable(package="custom")
class RelScoreSharpLoss(keras.losses.Loss):
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        use_target_scaler: bool = False,
        local_scale_floor: float = 0.0,
        large_move_quantile: float = 0.8,
        directional_penalty_weight: float = 0.6,
        confidence_penalty_weight: float = 0.35,
        confidence_ratio: float = 0.25,
        epsilon: float = 1e-6,
        name: str = "rel_score_sharp_loss",
    ):
        super().__init__(name=name)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.use_target_scaler = bool(use_target_scaler)
        self.local_scale_floor = float(local_scale_floor)
        self.large_move_quantile = float(large_move_quantile)
        self.directional_penalty_weight = float(directional_penalty_weight)
        self.confidence_penalty_weight = float(confidence_penalty_weight)
        self.confidence_ratio = float(confidence_ratio)
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        primary_true, local_scale = _split_rel_score_target(y_true)
        restored_true = _restore_rel_score_scale(
            primary_true,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        restored_pred = _restore_rel_score_scale(
            y_pred,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        abs_loss = _rel_score_quantile_loss(restored_true - restored_pred)
        base_loss = _rel_score_quantile_loss(restored_true)
        rel_component = abs_loss / tf.maximum(base_loss, tf.cast(self.epsilon, tf.float32))

        abs_true = tf.abs(tf.reshape(restored_true, [-1]))
        abs_pred = tf.abs(tf.reshape(restored_pred, [-1]))
        flat_true = tf.reshape(restored_true, [-1])
        flat_pred = tf.reshape(restored_pred, [-1])
        cutoff = _linear_quantile(abs_true, self.large_move_quantile)
        large_mask = tf.cast(abs_true >= cutoff, tf.float32)
        active_count = tf.maximum(tf.reduce_sum(large_mask), 1.0)

        directional_gap = tf.nn.relu(-(flat_true * flat_pred))
        directional_penalty = tf.reduce_sum(large_mask * directional_gap) / active_count

        min_required_abs_pred = self.confidence_ratio * abs_true
        confidence_gap = tf.nn.relu(min_required_abs_pred - abs_pred)
        confidence_penalty = tf.reduce_sum(large_mask * confidence_gap) / active_count

        avg_large_move = tf.maximum(
            tf.reduce_sum(large_mask * abs_true) / active_count,
            tf.cast(self.epsilon, tf.float32),
        )
        directional_penalty = directional_penalty / avg_large_move
        confidence_penalty = confidence_penalty / avg_large_move

        return (
            rel_component
            + tf.cast(self.directional_penalty_weight, tf.float32) * directional_penalty
            + tf.cast(self.confidence_penalty_weight, tf.float32) * confidence_penalty
        )

    def get_config(self) -> dict[str, float | bool | str]:
        return {
            "name": self.name,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "use_target_scaler": self.use_target_scaler,
            "local_scale_floor": self.local_scale_floor,
            "large_move_quantile": self.large_move_quantile,
            "directional_penalty_weight": self.directional_penalty_weight,
            "confidence_penalty_weight": self.confidence_penalty_weight,
            "confidence_ratio": self.confidence_ratio,
            "epsilon": self.epsilon,
        }


@keras.utils.register_keras_serializable(package="custom")
class RelScoreWeightedLoss(keras.losses.Loss):
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        use_target_scaler: bool = False,
        local_scale_floor: float = 0.0,
        high_quantile: float = 0.8,
        high_weight: float = 3.0,
        base_weight: float = 1.0,
        epsilon: float = 1e-6,
        name: str = "rel_score_weighted_loss",
    ):
        super().__init__(name=name)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.use_target_scaler = bool(use_target_scaler)
        self.local_scale_floor = float(local_scale_floor)
        self.high_quantile = float(high_quantile)
        self.high_weight = float(high_weight)
        self.base_weight = float(base_weight)
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        primary_true, local_scale = _split_rel_score_target(y_true)
        restored_true = _restore_rel_score_scale(
            primary_true,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        restored_pred = _restore_rel_score_scale(
            y_pred,
            target_mean=self.target_mean,
            target_std=self.target_std,
            use_target_scaler=self.use_target_scaler,
            local_scale=local_scale,
            local_scale_floor=self.local_scale_floor,
        )
        abs_true = tf.abs(tf.reshape(restored_true, [-1]))
        cutoff = _linear_quantile(abs_true, self.high_quantile)
        weights = tf.where(
            abs_true >= cutoff,
            tf.cast(self.high_weight, tf.float32),
            tf.cast(self.base_weight, tf.float32),
        )
        abs_loss = _weighted_rel_score_quantile_loss(restored_true - restored_pred, weights)
        base_loss = _weighted_rel_score_quantile_loss(restored_true, weights)
        return abs_loss / tf.maximum(base_loss, tf.cast(self.epsilon, tf.float32))

    def get_config(self) -> dict[str, float | bool | str]:
        return {
            "name": self.name,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "use_target_scaler": self.use_target_scaler,
            "local_scale_floor": self.local_scale_floor,
            "high_quantile": self.high_quantile,
            "high_weight": self.high_weight,
            "base_weight": self.base_weight,
            "epsilon": self.epsilon,
        }


@keras.utils.register_keras_serializable(package="custom")
class QuantilePinballLoss(keras.losses.Loss):
    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.5, 0.9),
        weights: tuple[float, ...] = (1.0, 0.5),
        name: str = "quantile_pinball_loss",
    ):
        super().__init__(name=name)
        if len(quantiles) == 0:
            raise ValueError("quantiles must not be empty.")
        if len(quantiles) != len(weights):
            raise ValueError("quantiles and weights must have the same length.")
        self.quantiles = tuple(float(q) for q in quantiles)
        self.weights = tuple(float(weight) for weight in weights)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape.rank is None:
            y_true = tf.reshape(y_true, [-1, 1])
        elif y_true.shape.rank == 1:
            y_true = tf.reshape(y_true, [-1, 1])
        else:
            y_true = tf.reshape(y_true[:, :1], [-1, 1])

        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank is None:
            y_pred = tf.reshape(y_pred, [-1, len(self.quantiles)])

        total = tf.constant(0.0, dtype=tf.float32)
        for index, (quantile, weight) in enumerate(zip(self.quantiles, self.weights)):
            pred = y_pred[:, index : index + 1]
            error = y_true - pred
            pinball = tf.maximum(quantile * error, (quantile - 1.0) * error)
            total = total + tf.cast(weight, tf.float32) * tf.reduce_mean(pinball)
        return total

    def get_config(self) -> dict[str, tuple[float, ...] | str]:
        return {
            "name": self.name,
            "quantiles": self.quantiles,
            "weights": self.weights,
        }


def directional_huber_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    delta: float = 0.01,
    penalty_weight: float = 20.0,
) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * tf.square(quadratic) + delta * linear
    sign_penalty = tf.maximum(0.0, -(y_true * y_pred))
    return tf.reduce_mean(huber + penalty_weight * sign_penalty)


def resolve_loss(
    loss: str,
    huber_delta: float,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_target_scaler: bool = False,
    local_scale_floor: float = 0.0,
    large_move_quantile: float = 0.8,
    directional_penalty_weight: float = 0.6,
    confidence_penalty_weight: float = 0.35,
    confidence_ratio: float = 0.25,
    weighted_high_quantile: float = 0.8,
    weighted_high_weight: float = 3.0,
    weighted_base_weight: float = 1.0,
):
    if loss == "mse":
        return "mse"
    if loss == "huber":
        return keras.losses.Huber(delta=huber_delta)
    if loss == "directional_huber":
        return DirectionalHuberLoss(delta=huber_delta)
    if loss == "rel_score":
        return RelScoreLoss(
            target_mean=target_mean,
            target_std=target_std,
            use_target_scaler=use_target_scaler,
            local_scale_floor=local_scale_floor,
        )
    if loss == "rel_score_sharp":
        return RelScoreSharpLoss(
            target_mean=target_mean,
            target_std=target_std,
            use_target_scaler=use_target_scaler,
            local_scale_floor=local_scale_floor,
            large_move_quantile=large_move_quantile,
            directional_penalty_weight=directional_penalty_weight,
            confidence_penalty_weight=confidence_penalty_weight,
            confidence_ratio=confidence_ratio,
        )
    if loss == "rel_score_weighted":
        return RelScoreWeightedLoss(
            target_mean=target_mean,
            target_std=target_std,
            use_target_scaler=use_target_scaler,
            local_scale_floor=local_scale_floor,
            high_quantile=weighted_high_quantile,
            high_weight=weighted_high_weight,
            base_weight=weighted_base_weight,
        )
    raise ValueError("loss must be one of: mse, huber, directional_huber, rel_score, rel_score_sharp, rel_score_weighted")
