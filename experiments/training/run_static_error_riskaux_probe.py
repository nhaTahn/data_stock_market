"""Two-stage static-error risk auxiliary probe.

Stage 1 trains a return-only LSTM on the train split. Stage 2 freezes the
tail-risk labels as |y - p_stage1| >= threshold, then trains a detached
two-head LSTM (return + risk head). Validation is used only for readout and
early stopping; holdout/test is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.training import run_tail_aware_lstm_probe as probe  # noqa: E402

DEFAULT_OUTPUT = (
    ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports/static_error_riskaux_probe_20260528"
)
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/static_error_riskaux_probe_20260528"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--epochs-stage1", type=int, default=8)
    parser.add_argument("--epochs-stage2", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--risk-weight", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--base-variant", default="plain_global_weighted_mild_tail35_p05")
    parser.add_argument("--attached", action="store_true", help="Let risk loss update shared backbone instead of detached risk head.")
    return parser.parse_args()


def make_probe_args(args: argparse.Namespace, epochs: int) -> argparse.Namespace:
    parsed = probe.parse_args([])
    parsed.output_dir = args.output_dir
    parsed.gold_dir = args.gold_dir
    parsed.seed = args.seed
    parsed.epochs = epochs
    parsed.patience = args.patience
    return parsed


def fit_return_stage(data: probe.PreparedData, variant: probe.Variant, args: argparse.Namespace) -> keras.Model:
    model = probe.build_plain_probe_model(data, args, variant)
    x_train = probe.apply_training_feature_dropout(data.x_train, data.feature_columns, variant, args.seed)
    model.fit(
        x_train,
        data.y_train_model,
        validation_data=(data.x_val, data.y_val_model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    return model


def fit_static_error_risk_stage(
    data: probe.PreparedData,
    variant: probe.Variant,
    args: argparse.Namespace,
    train_label: np.ndarray,
    val_label: np.ndarray,
) -> keras.Model:
    if variant.model_type == "riskaux":
        model = probe.build_riskaux_probe_model(data, args, variant)
    else:
        model = probe.build_riskaux_detached_probe_model(data, args, variant)
    x_train = probe.apply_training_feature_dropout(data.x_train, data.feature_columns, variant, args.seed)
    y_train = {"pred": data.y_train_model, "risk_aux": train_label}
    y_val = {"pred": data.y_val_model, "risk_aux": val_label}
    sample_weight = {
        "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
        "risk_aux": probe.balanced_binary_weights(train_label, max_weight=6.0),
    }
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(data.x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_pred_loss",
                mode="min",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=0,
    )
    return model


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    probe_args_stage1 = make_probe_args(args, args.epochs_stage1)
    probe_args_stage2 = make_probe_args(args, args.epochs_stage2)
    variants = probe.all_variants()
    base_variant = variants[args.base_variant]
    risk_variant = replace(
        base_variant,
        name=f"{base_variant.name}_static_error_riskaux_{'attached' if args.attached else 'detached'}_w{int(args.risk_weight * 100)}",
        model_type="riskaux" if args.attached else "riskaux_detached",
        risk_aux_weight=args.risk_weight,
        risk_aux_threshold=args.threshold,
    )
    feature_columns = probe.load_base_feature_columns()
    raw = probe.load_frame(
        probe_args_stage1.data,
        feature_columns,
        probe_args_stage1.target_column,
        probe_args_stage1.target_normalizer,
    )
    feature_columns = tuple(raw.attrs.get("feature_columns", feature_columns))
    data = probe.prepare_data(raw, feature_columns, base_variant, probe_args_stage1)

    probe.set_global_seed(args.seed)
    print("Stage 1: return-only model")
    stage1 = fit_return_stage(data, base_variant, probe_args_stage1)
    p_train, _, _, _ = probe.predict_raw_return(stage1, data, data.x_train, data.train_scale_values)
    p_val, _, _, _ = probe.predict_raw_return(stage1, data, data.x_val, data.val_scale_values)
    train_label = (np.abs(data.y_train_raw - p_train).reshape(-1, 1) >= args.threshold).astype(np.float32)
    val_label = (np.abs(data.y_val_raw - p_val).reshape(-1, 1) >= args.threshold).astype(np.float32)
    print(
        json.dumps(
            {
                "train_static_error_label_rate": float(train_label.mean()),
                "val_static_error_label_rate": float(val_label.mean()),
            },
            indent=2,
        )
    )

    print(f"Stage 2: {'attached' if args.attached else 'detached'} static-error riskaux model")
    probe.set_global_seed(args.seed)
    stage2 = fit_static_error_risk_stage(data, risk_variant, probe_args_stage2, train_label, val_label)

    pred_stage1 = probe.prediction_frame(base_variant, stage1, data)
    pred_stage1["variant"] = f"{base_variant.name}_stage1"
    pred_stage2 = probe.prediction_frame(risk_variant, stage2, data)
    predictions = pd.concat([pred_stage1, pred_stage2], ignore_index=True)
    summary = probe.summarize_predictions(predictions, tuple(float(x) for x in probe.parse_csv(probe_args_stage1.spike_thresholds)))
    predictions.to_csv(args.output_dir / "predictions_all_variants.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    summary.to_csv(args.gold_dir / "summary.csv", index=False)

    text = "\n".join(
        [
            "# Static-Error RiskAux Probe",
            "",
            "Protocol: Stage-1 train return head; Stage-2 train risk head using static |y-p_stage1| labels. Holdout/test not used.",
            "",
            f"- seed: {args.seed}",
            f"- base_variant: `{base_variant.name}`",
            f"- risk_variant: `{risk_variant.name}`",
            f"- train static error label rate: {float(train_label.mean()):.4f}",
            f"- val static error label rate: {float(val_label.mean()):.4f}",
            "",
            "## Summary",
            "",
            summary.round(6).to_markdown(index=False),
            "",
            json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2),
        ]
    )
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
