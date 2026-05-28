"""Static-error risk auxiliary probe on the stronger hetero anchor.

Stage 1 trains the heteroscedastic anchor return model. Stage 2 computes
static train labels |y - p_stage1| >= threshold, then trains a shared-backbone
model with two heads:
  - mu_logsigma return head with CombinedRelScoreNLLLoss
  - risk_aux classification head with BCE

Validation is used only for readout/early stopping. Holdout/test is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training.run_hetero_nll_probe import CombinedRelScoreNLLLoss, evaluate_predictions, predict_raw, train_model  # noqa: E402
from experiments.training.run_tail_aware_lstm_probe import balanced_binary_weights  # noqa: E402
from experiments.training.run_vn_improvement_smoke import build_hetero_model, build_variants, load_variant, parse_seeds  # noqa: E402
from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.seeds import set_global_seed  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/static_error_hetero_anchor_probe_20260528"
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/static_error_hetero_anchor_probe_20260528"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--w-rel", type=float, default=0.7, dest="w_rel")
    parser.add_argument("--w-nll", type=float, default=0.3, dest="w_nll")
    parser.add_argument("--risk-weight", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--variant", default="baseline", help="run_vn_improvement_smoke variant, e.g. baseline/rich_feat")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv")
    return parser.parse_args()


def build_static_error_model(data, args: argparse.Namespace, cfg: dict) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=data.x_train.shape[1],
        num_features=data.x_train.shape[2],
        lstm_units=cfg["lstm_units"],
        dropout=args.dropout,
        use_layer_norm=cfg["use_layer_norm"],
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    mu_logsigma = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    risk = layers.Dense(1, activation="sigmoid", name="risk_aux")(encoded)
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=data.target_scaler.mean,
        target_std=data.target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=data.local_normalizer.floor,
        high_quantile=0.85,
        high_weight=1.75,
        base_weight=1.0,
        tail_error_threshold=0.035,
        tail_penalty_weight=0.05,
    )
    model = keras.Model(inputs=inputs, outputs={"mu_logsigma": mu_logsigma, "risk_aux": risk})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "mu_logsigma": CombinedRelScoreNLLLoss(rel_loss, w_rel=args.w_rel, w_nll=args.w_nll),
            "risk_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"mu_logsigma": 1.0, "risk_aux": args.risk_weight},
    )
    return model


def predict_raw_multi(model: keras.Model, data, x: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = model.predict(x, verbose=0)
    if isinstance(output, dict):
        raw_output = output["mu_logsigma"]
        risk = np.asarray(output.get("risk_aux"), dtype=np.float32).reshape(-1)
    else:
        output_map = {name: value for name, value in zip(model.output_names, output)}
        raw_output = output_map["mu_logsigma"]
        risk = np.asarray(output_map["risk_aux"], dtype=np.float32).reshape(-1)
    class Proxy:
        def predict(self, _x, verbose=0):
            return raw_output
    mu, sigma = predict_raw(Proxy(), data, x, scale)
    return mu, sigma, risk


def rel_score_value(actual: np.ndarray, pred: np.ndarray) -> float:
    err = actual - pred
    base = float(np.quantile(np.abs(actual), 0.5) + 0.5 * np.quantile(np.abs(actual), 0.9))
    loss = float(np.quantile(np.abs(err), 0.5) + 0.5 * np.quantile(np.abs(err), 0.9))
    return 1.0 - loss / base


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    all_variants = build_variants(tuple(DEFAULT_FEATURE_COLUMNS))
    if args.variant not in all_variants:
        raise ValueError(f"Unknown variant {args.variant}; available={sorted(all_variants)}")
    cfg = all_variants[args.variant]
    data = load_variant(args, cfg["feature_columns"], cfg["window_size"])
    rows: list[dict] = []

    pred_dir = args.output_dir / "predictions"
    gold_pred_dir = args.gold_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
        gold_pred_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"Seed {seed}: Stage 1 hetero anchor")
        set_global_seed(seed)
        stage1_args = argparse.Namespace(**vars(args))
        stage1_args.epochs = args.epochs_stage1
        model1 = build_hetero_model(data, stage1_args, cfg["lstm_units"], cfg["use_layer_norm"], w_rel=args.w_rel, w_nll=args.w_nll)
        train_model(model1, data, stage1_args, seed)
        mu_train_1, sigma_train_1 = predict_raw(model1, data, data.x_train, data.train_scale)
        mu_val_1, sigma_val_1 = predict_raw(model1, data, data.x_val, data.val_scale)
        train_label = (np.abs(data.y_train_raw - mu_train_1).reshape(-1, 1) >= args.threshold).astype(np.float32)
        val_label = (np.abs(data.y_val_raw - mu_val_1).reshape(-1, 1) >= args.threshold).astype(np.float32)
        for split, y, mu, sigma, meta in [
            ("train", data.y_train_raw, mu_train_1, sigma_train_1, data.meta_train),
            ("val", data.y_val_raw, mu_val_1, sigma_val_1, data.meta_val),
        ]:
            metric = evaluate_predictions(y, mu, sigma, meta)
            rows.append({"variant": "stage1_hetero", "seed": seed, "split": split, **metric})

        print(f"Seed {seed}: Stage 2 static-error riskaux")
        set_global_seed(seed)
        model2 = build_static_error_model(data, args, cfg)
        y_train = {"mu_logsigma": data.y_train_model, "risk_aux": train_label}
        y_val = {"mu_logsigma": data.y_val_model, "risk_aux": val_label}
        sample_weight = {
            "mu_logsigma": np.ones(len(data.y_train_raw), dtype=np.float32),
            "risk_aux": balanced_binary_weights(train_label, max_weight=6.0),
        }
        model2.fit(
            data.x_train,
            y_train,
            sample_weight=sample_weight,
            validation_data=(data.x_val, y_val),
            epochs=args.epochs_stage2,
            batch_size=args.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_mu_logsigma_loss",
                    mode="min",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        mu_train_2, sigma_train_2, risk_train = predict_raw_multi(model2, data, data.x_train, data.train_scale)
        mu_val_2, sigma_val_2, risk_val = predict_raw_multi(model2, data, data.x_val, data.val_scale)
        for split, y, mu, sigma, risk, meta in [
            ("train", data.y_train_raw, mu_train_2, sigma_train_2, risk_train, data.meta_train),
            ("val", data.y_val_raw, mu_val_2, sigma_val_2, risk_val, data.meta_val),
        ]:
            metric = evaluate_predictions(y, mu, sigma, meta)
            metric["mean_risk"] = float(np.mean(risk))
            metric["static_error_label_rate_train"] = float(train_label.mean())
            metric["static_error_label_rate_val"] = float(val_label.mean())
            rows.append({"variant": "stage2_static_error_riskaux", "seed": seed, "split": split, **metric})

        if args.save_predictions:
            payload = {
                "mu_train": mu_train_2.astype(np.float32),
                "sigma_train": sigma_train_2.astype(np.float32),
                "risk_train": risk_train.astype(np.float32),
                "y_train": data.y_train_raw.astype(np.float32),
                "mu_val": mu_val_2.astype(np.float32),
                "sigma_val": sigma_val_2.astype(np.float32),
                "risk_val": risk_val.astype(np.float32),
                "y_val": data.y_val_raw.astype(np.float32),
            }
            np.savez_compressed(pred_dir / f"predictions_seed_{seed}.npz", **payload)
            np.savez_compressed(gold_pred_dir / f"predictions_seed_{seed}.npz", **payload)

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    result.to_csv(args.gold_dir / "results_per_seed.csv", index=False)
    val = result[result["split"] == "val"].copy()
    agg = val.groupby("variant").agg(
        rel_score_mean=("rel_score", "mean"),
        rel_score_std=("rel_score", "std"),
        q90_abs_error_mean=("q90_abs_error", "mean"),
        directional_accuracy_mean=("directional_accuracy", "mean"),
        pred_actual_q90_ratio_mean=("pred_actual_q90_ratio", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index().sort_values("rel_score_mean", ascending=False)
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    agg.to_csv(args.gold_dir / "aggregate.csv", index=False)
    text = "\n".join([
        "# Static-Error Hetero Anchor Probe",
        "",
        "Protocol: Stage-1 hetero anchor; Stage-2 attached static-error risk head. Holdout/test not used.",
        "",
        "## Aggregate Validation",
        "",
        agg.round(6).to_markdown(index=False),
        "",
        "## Per Seed",
        "",
        result.round(6).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2),
    ])
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
