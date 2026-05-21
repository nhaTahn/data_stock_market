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


def _split_rank_target(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    if y_true.shape.rank is None:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    elif y_true.shape.rank == 1:
        y_true = tf.reshape(y_true, [-1, 1])
    else:
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])

    if y_true.shape.rank is not None and y_true.shape[-1] is not None and y_true.shape[-1] < 2:
        raise ValueError("Rank targets require at least 2 columns: target value and group id.")

    return y_true[:, :1], y_true[:, 1:2]


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
class CustomPinballLoss(keras.losses.Loss):
    def __init__(
        self,
        q50: float = 0.5,
        q90: float = 0.9,
        q90_weight: float = 0.5,
        name: str = "custom_pinball_loss",
    ):
        super().__init__(name=name)
        self.q50 = float(q50)
        self.q90 = float(q90)
        self.q90_weight = float(q90_weight)

    @staticmethod
    def _ensure_target_rank(y_true: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        if y_true.shape.rank is None:
            y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1, 1])
        elif y_true.shape.rank == 1:
            y_true = tf.reshape(y_true, [-1, 1, 1])
        elif y_true.shape.rank == 2:
            y_true = y_true[..., tf.newaxis]
        elif y_true.shape.rank >= 3:
            y_true = y_true[..., :1]
        return y_true

    @staticmethod
    def _ensure_prediction_rank(y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.cast(y_pred, tf.float32)
        if y_pred.shape.rank is None:
            y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, 2])
        elif y_pred.shape.rank == 1:
            y_pred = tf.reshape(y_pred, [-1, 1, 1])
        elif y_pred.shape.rank == 2:
            y_pred = y_pred[:, tf.newaxis, :]
        return y_pred

    @staticmethod
    def _pinball(target: tf.Tensor, prediction: tf.Tensor, tau: float) -> tf.Tensor:
        error = target - prediction
        return tf.maximum(tf.cast(tau, tf.float32) * error, (tf.cast(tau, tf.float32) - 1.0) * error)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target = self._ensure_target_rank(y_true)
        prediction = self._ensure_prediction_rank(y_pred)

        q50_pred = prediction[..., 0:1]
        q90_pred = prediction[..., 1:2]
        q50_loss = tf.reduce_mean(self._pinball(target, q50_pred, self.q50))
        q90_loss = tf.reduce_mean(self._pinball(target, q90_pred, self.q90))
        return q50_loss + tf.cast(self.q90_weight, tf.float32) * q90_loss

    def get_config(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "q50": self.q50,
            "q90": self.q90,
            "q90_weight": self.q90_weight,
        }


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
class RelScoreWeightedTailLoss(keras.losses.Loss):
    def __init__(
        self,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        use_target_scaler: bool = False,
        local_scale_floor: float = 0.0,
        high_quantile: float = 0.8,
        high_weight: float = 3.0,
        base_weight: float = 1.0,
        tail_error_threshold: float = 0.05,
        tail_penalty_weight: float = 0.10,
        large_move_quantile: float = 0.80,
        directional_penalty_weight: float = 0.0,
        epsilon: float = 1e-6,
        name: str = "rel_score_weighted_tail_loss",
    ):
        super().__init__(name=name)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.use_target_scaler = bool(use_target_scaler)
        self.local_scale_floor = float(local_scale_floor)
        self.high_quantile = float(high_quantile)
        self.high_weight = float(high_weight)
        self.base_weight = float(base_weight)
        self.tail_error_threshold = float(tail_error_threshold)
        self.tail_penalty_weight = float(tail_penalty_weight)
        self.large_move_quantile = float(large_move_quantile)
        self.directional_penalty_weight = float(directional_penalty_weight)
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
        error = restored_true - restored_pred
        abs_loss = _weighted_rel_score_quantile_loss(error, weights)
        base_loss = _weighted_rel_score_quantile_loss(restored_true, weights)
        rel_component = abs_loss / tf.maximum(base_loss, tf.cast(self.epsilon, tf.float32))

        abs_error = tf.abs(tf.reshape(error, [-1]))
        threshold = tf.cast(self.tail_error_threshold, tf.float32)
        excess = tf.nn.relu(abs_error - threshold)
        normalizer = tf.maximum(_linear_quantile(abs_true, 0.9), threshold)
        tail_penalty = tf.reduce_mean(tf.square(excess / tf.maximum(normalizer, tf.cast(self.epsilon, tf.float32))))

        flat_true = tf.reshape(restored_true, [-1])
        flat_pred = tf.reshape(restored_pred, [-1])
        large_cutoff = _linear_quantile(abs_true, self.large_move_quantile)
        large_mask = tf.cast(abs_true >= large_cutoff, tf.float32)
        active_count = tf.maximum(tf.reduce_sum(large_mask), 1.0)
        directional_gap = tf.nn.relu(-(flat_true * flat_pred))
        avg_large_move = tf.maximum(
            tf.reduce_sum(large_mask * abs_true) / active_count,
            tf.cast(self.epsilon, tf.float32),
        )
        directional_penalty = (
            tf.reduce_sum(large_mask * directional_gap) / active_count
        ) / avg_large_move
        return (
            rel_component
            + tf.cast(self.tail_penalty_weight, tf.float32) * tail_penalty
            + tf.cast(self.directional_penalty_weight, tf.float32) * directional_penalty
        )

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
            "tail_error_threshold": self.tail_error_threshold,
            "tail_penalty_weight": self.tail_penalty_weight,
            "large_move_quantile": self.large_move_quantile,
            "directional_penalty_weight": self.directional_penalty_weight,
            "epsilon": self.epsilon,
        }


@keras.utils.register_keras_serializable(package="custom")
class CrossSectionalPairwiseRankLoss(keras.losses.Loss):
    def __init__(
        self,
        temperature: float = 1.0,
        min_group_size: int = 5,
        epsilon: float = 1e-6,
        name: str = "cross_sectional_pairwise_rank_loss",
    ):
        super().__init__(name=name)
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if min_group_size <= 1:
            raise ValueError("min_group_size must be greater than 1.")
        self.temperature = float(temperature)
        self.min_group_size = int(min_group_size)
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target, group_id = _split_rank_target(y_true)
        target = tf.reshape(tf.cast(target, tf.float32), [-1])
        prediction = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        group_id = tf.reshape(tf.cast(tf.round(group_id), tf.int32), [-1])
        unique_groups, _ = tf.unique(group_id)

        def compute_group_loss(active_group_id: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            mask = tf.equal(group_id, active_group_id)
            group_target = tf.boolean_mask(target, mask)
            group_prediction = tf.boolean_mask(prediction, mask)
            group_size = tf.shape(group_target)[0]

            def compute_valid_group() -> tuple[tf.Tensor, tf.Tensor]:
                centered_prediction = group_prediction - tf.reduce_mean(group_prediction)
                prediction_scale = tf.maximum(
                    tf.math.reduce_std(centered_prediction),
                    tf.cast(self.epsilon, tf.float32),
                )
                normalized_prediction = centered_prediction / prediction_scale
                target_diff = group_target[:, tf.newaxis] - group_target[tf.newaxis, :]
                prediction_diff = (
                    normalized_prediction[:, tf.newaxis] - normalized_prediction[tf.newaxis, :]
                ) / tf.cast(self.temperature, tf.float32)
                pair_mask = target_diff > 0.0
                pair_count = tf.reduce_sum(tf.cast(pair_mask, tf.float32))
                pairwise_loss = tf.nn.softplus(-prediction_diff)
                masked_loss = tf.where(pair_mask, pairwise_loss, tf.zeros_like(pairwise_loss))
                return (
                    tf.reduce_sum(masked_loss) / tf.maximum(pair_count, 1.0),
                    tf.cast(pair_count > 0.0, tf.float32),
                )

            return tf.cond(
                group_size >= tf.cast(self.min_group_size, tf.int32),
                compute_valid_group,
                lambda: (tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)),
            )

        losses, active_counts = tf.map_fn(
            compute_group_loss,
            unique_groups,
            fn_output_signature=(tf.TensorSpec([], dtype=tf.float32), tf.TensorSpec([], dtype=tf.float32)),
        )
        return tf.reduce_sum(losses) / tf.maximum(tf.reduce_sum(active_counts), 1.0)

    def get_config(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "min_group_size": self.min_group_size,
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
    if loss == "rel_score_weighted_tail":
        return RelScoreWeightedTailLoss(
            target_mean=target_mean,
            target_std=target_std,
            use_target_scaler=use_target_scaler,
            local_scale_floor=local_scale_floor,
            high_quantile=weighted_high_quantile,
            high_weight=weighted_high_weight,
            base_weight=weighted_base_weight,
        )
    raise ValueError(
        "loss must be one of: mse, huber, directional_huber, rel_score, "
        "rel_score_sharp, rel_score_weighted, rel_score_weighted_tail"
    )
