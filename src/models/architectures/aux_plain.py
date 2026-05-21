"""Aux-loss Plain LSTM (Option A from sign×magnitude alternatives).

Mục đích: thay thế multiplicative composition `sign × magnitude` của
signmag bằng **direct regression head** + **auxiliary sign/magnitude heads**.

Architecture:
```
encoded = LSTM_backbone(...)
pred           = Dense(1, linear)(encoded)      # main, used for inference
sign_prob      = Dense(1, sigmoid)(encoded)     # aux: shape backbone with direction info
magnitude_aux  = Dense(1, softplus)(encoded)    # aux: shape backbone with scale info
```

Loss:
```
total_loss = w_pred * rel_score(pred, signed_target)
           + w_sign * BCE(sign_prob, [y > 0])
           + w_mag  * Huber(magnitude_aux, |y|)
```

Trade-off so với signmag:

| Property | signmag (`sign × mag`) | aux_plain (direct + aux) |
| --- | --- | --- |
| Inference path | `(2σ-1) × softplus(...)` multiplicative | `Dense(1, linear)` additive |
| Gradient to magnitude when sign≈0.5 | ≈ 0 (vanishing) | Full, unaffected |
| Asymmetric uptick/downtick magnitude | No (sym) | No (sym, but main head doesn't bottleneck) |
| Aux supervision shapes backbone | Yes (joint training) | Yes (joint training) |

Lý do để thử: plain LSTM (direct regression) đã beat signmag on val
(+0.0070 vs +0.0027 mean_ensemble). Aux heads giữ signmag's
representation-learning advantage mà không phá main inference path.

See docs/lstm_isolated_ablation_findings_20260515.md for the empirical
motivation.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from src.models.architectures.backbone import build_lstm_backbone
from src.models.components.losses import resolve_loss
from src.models.training.scalers import LocalTargetNormalizer


def build_aux_plain_model(
    window_size: int,
    num_features: int,
    lstm_units: int | list[int],
    lr: float,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.0,
    use_layer_norm: bool = False,
    loss: str = "rel_score",
    huber_delta: float = 0.01,
    rel_score_large_move_quantile: float = 0.8,
    rel_score_directional_penalty: float = 0.6,
    rel_score_confidence_penalty: float = 0.35,
    rel_score_confidence_ratio: float = 0.25,
    rel_score_weighted_high_quantile: float = 0.8,
    rel_score_weighted_high_weight: float = 3.0,
    rel_score_weighted_base_weight: float = 1.0,
    pred_loss_weight: float = 1.0,
    sign_loss_weight: float = 0.25,
    magnitude_loss_weight: float = 0.25,
    local_target_normalizer: LocalTargetNormalizer | None = None,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=window_size,
        num_features=num_features,
        lstm_units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        use_layer_norm=use_layer_norm,
    )

    pred = layers.Dense(1, name="pred")(encoded)
    sign_prob = layers.Dense(1, activation="sigmoid", name="sign_prob")(encoded)
    magnitude_aux = layers.Dense(1, activation="softplus", name="magnitude_aux")(encoded)

    outputs = {
        "pred": pred,
        "sign_prob": sign_prob,
        "magnitude_aux": magnitude_aux,
    }

    loss_map = {
        "pred": resolve_loss(
            loss,
            huber_delta,
            local_scale_floor=local_target_normalizer.floor if local_target_normalizer is not None else 0.0,
            large_move_quantile=rel_score_large_move_quantile,
            directional_penalty_weight=rel_score_directional_penalty,
            confidence_penalty_weight=rel_score_confidence_penalty,
            confidence_ratio=rel_score_confidence_ratio,
            weighted_high_quantile=rel_score_weighted_high_quantile,
            weighted_high_weight=rel_score_weighted_high_weight,
            weighted_base_weight=rel_score_weighted_base_weight,
        ),
        "sign_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        "magnitude_aux": keras.losses.Huber(delta=huber_delta),
    }
    loss_weights = {
        "pred": pred_loss_weight,
        "sign_prob": sign_loss_weight,
        "magnitude_aux": magnitude_loss_weight,
    }

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_map,
        loss_weights=loss_weights,
    )
    return model
