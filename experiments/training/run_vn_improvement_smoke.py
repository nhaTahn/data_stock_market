"""VN improvement smoke: test architectural and feature variants against baseline.

Variants vs. baseline (DEFAULT_FEATURE_COLUMNS + [64,32] + window=15):
  deeper_96:  [96,64,32] + LayerNorm
  rich_feat:  + vwap_gap_20, above_ma_200, alpha_sector + cs-rank features
  window20:   window_size=20
  rich_deeper: rich_feat + deeper_96

Scope: VN train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    apply_feature_scaler, apply_local_target_normalizer, apply_target_scaler,
    fit_feature_scaler, fit_local_target_normalizer, fit_target_scaler,
    inverse_local_target_normalizer, inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402
from experiments.training.run_hetero_nll_probe import (  # noqa: E402
    CombinedRelScoreNLLLoss, PreparedData, evaluate_predictions,
    predict_raw, train_model,
)

DEFAULT_DATA = ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT / "data/processed/assets/data_info_vn/history/training_runs/reports"
    / "vn_improvement_smoke_20260526"
)
DEFAULT_GOLD = ROOT / "gold/vn_transition_pressure_20260512/plots/vn_improvement_smoke_20260526"

RICH_EXTRA: tuple[str, ...] = ("vwap_gap_20", "above_ma_200", "alpha_sector")
CS_RANK_EXTRAS: tuple[str, ...] = (
    "cs_rank_momentum_20", "cs_rank_volume_ratio_20", "cs_rank_wyckoff_phase_60d"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--train-end-date", default="2020-03-31")
    p.add_argument("--val-end-date", default="2022-11-15")
    p.add_argument("--seeds", default="43,52,62")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-column", default="target_next_return")
    p.add_argument("--target-normalizer", default="volatility_20")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument(
        "--variants",
        default="baseline,rich_feat",
        help="Comma-separated variants to run.",
    )
    p.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-seed train/validation predictions for downstream ensemble calibration.",
    )
    return p.parse_args(argv)


def parse_seeds(value: str) -> list[int]:
    return [int(s.strip()) for s in value.split(",") if s.strip()]


def add_cs_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add daily cross-sectional percentile ranks of key signals."""
    frame = frame.copy()
    for src_col, dst_col in [
        ("momentum_20", "cs_rank_momentum_20"),
        ("volume_ratio_20", "cs_rank_volume_ratio_20"),
        ("wyckoff_phase_60d", "cs_rank_wyckoff_phase_60d"),
    ]:
        if src_col in frame.columns and dst_col not in frame.columns:
            frame[dst_col] = frame.groupby("Date")[src_col].rank(pct=True, na_option="keep")
    return frame


def load_variant(
    args: argparse.Namespace,
    feature_columns: tuple[str, ...],
    window_size: int,
) -> PreparedData:
    frame = load_training_frame(str(args.data), stocks=None)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = add_cs_rank_features(frame)

    available = tuple(c for c in feature_columns if c in frame.columns)
    dropped = sorted(set(feature_columns) - set(available))
    if dropped:
        print(f"  [WARN] Dropping missing columns: {dropped}")

    required = {"Date", "code", args.target_column, args.target_normalizer, *available}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    target_alias = args.target_normalizer
    train_df = frame.loc[frame["Date"] <= args.train_end_date].copy()
    feat_scaler = fit_feature_scaler(train_df.dropna(subset=available), available)
    scaled = apply_feature_scaler(frame, feat_scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled, available, args.target_column, window_size,
        extra_meta_columns=(target_alias,), sequence_normalization="none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, args.train_end_date, args.val_end_date)
    x_train, y_train_raw, meta_train = splits["train"]
    x_val, y_val_raw, meta_val = splits["val"]
    train_scale = meta_train[target_alias].to_numpy(dtype=np.float32)
    val_scale = meta_val[target_alias].to_numpy(dtype=np.float32)
    local_norm = fit_local_target_normalizer(train_scale, target_alias)
    y_train_local = apply_local_target_normalizer(y_train_raw, train_scale, local_norm)
    ts = fit_target_scaler(y_train_local)
    y_train_scaled = apply_target_scaler(y_train_local, ts).reshape(-1, 1)
    y_train_model = np.concatenate([y_train_scaled, train_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    y_val_local = apply_local_target_normalizer(y_val_raw, val_scale, local_norm)
    y_val_scaled = apply_target_scaler(y_val_local, ts).reshape(-1, 1)
    y_val_model = np.concatenate([y_val_scaled, val_scale.reshape(-1, 1)], axis=1).astype(np.float32)
    return PreparedData(
        feature_columns=available,
        x_train=x_train, x_val=x_val,
        y_train_raw=y_train_raw.astype(np.float32), y_val_raw=y_val_raw.astype(np.float32),
        y_train_model=y_train_model, y_val_model=y_val_model,
        meta_train=meta_train.reset_index(drop=True), meta_val=meta_val.reset_index(drop=True),
        train_scale=train_scale, val_scale=val_scale,
        target_scaler=ts, local_normalizer=local_norm,
    )


def build_hetero_model(
    data: PreparedData,
    args: argparse.Namespace,
    lstm_units: list[int],
    use_layer_norm: bool = False,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=data.x_train.shape[1],
        num_features=data.x_train.shape[2],
        lstm_units=lstm_units,
        dropout=args.dropout,
        use_layer_norm=use_layer_norm,
    )
    mu = layers.Dense(1, name="mu")(encoded)
    log_sigma = layers.Dense(1, name="log_sigma")(encoded)
    output = layers.Concatenate(name="mu_logsigma")([mu, log_sigma])
    rel_loss = RelScoreWeightedTailLoss(
        target_mean=data.target_scaler.mean,
        target_std=data.target_scaler.std,
        use_target_scaler=True,
        local_scale_floor=data.local_normalizer.floor,
        high_quantile=0.85, high_weight=1.75,
        base_weight=1.0, tail_error_threshold=0.035, tail_penalty_weight=0.05,
    )
    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=CombinedRelScoreNLLLoss(rel_loss, w_rel=0.7, w_nll=0.3),
    )
    return model


def build_variants(default_feats: tuple[str, ...]) -> dict[str, dict]:
    rich = default_feats + RICH_EXTRA + CS_RANK_EXTRAS
    return {
        "baseline": {
            "feature_columns": default_feats,
            "lstm_units": [64, 32],
            "window_size": 15,
            "use_layer_norm": False,
            "description": "DEFAULT 26 feats, [64,32], w=15",
        },
        "deeper_96": {
            "feature_columns": default_feats,
            "lstm_units": [96, 64, 32],
            "window_size": 15,
            "use_layer_norm": True,
            "description": "DEFAULT 26 feats, [96,64,32]+LN, w=15",
        },
        "rich_feat": {
            "feature_columns": rich,
            "lstm_units": [64, 32],
            "window_size": 15,
            "use_layer_norm": False,
            "description": "DEFAULT + vwap_gap_20/above_ma_200/alpha_sector/cs_ranks, [64,32], w=15",
        },
        "window20": {
            "feature_columns": default_feats,
            "lstm_units": [64, 32],
            "window_size": 20,
            "use_layer_norm": False,
            "description": "DEFAULT 26 feats, [64,32], w=20",
        },
        "rich_deeper": {
            "feature_columns": rich,
            "lstm_units": [96, 64, 32],
            "window_size": 15,
            "use_layer_norm": True,
            "description": "DEFAULT + rich feats, [96,64,32]+LN, w=15",
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.gold_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    all_variants = build_variants(tuple(DEFAULT_FEATURE_COLUMNS))
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(all_variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(all_variants)}")
    variants = {name: all_variants[name] for name in requested}

    all_rows: list[dict] = []
    for variant_name, cfg in variants.items():
        print(f"\n=== Variant: {variant_name} ===  {cfg['description']}")
        data = load_variant(args, cfg["feature_columns"], cfg["window_size"])
        print(f"  train={data.x_train.shape}, val={data.x_val.shape}")
        pred_dir = args.output_dir / "predictions" / variant_name
        gold_pred_dir = args.gold_dir / "predictions" / variant_name
        if args.save_predictions:
            pred_dir.mkdir(parents=True, exist_ok=True)
            gold_pred_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            model = build_hetero_model(data, args, cfg["lstm_units"], cfg["use_layer_norm"])
            train_model(model, data, args, seed)
            mu_val, sigma_val = predict_raw(model, data, data.x_val, data.val_scale)
            result = evaluate_predictions(data.y_val_raw, mu_val, sigma_val, data.meta_val)
            if args.save_predictions:
                mu_train, sigma_train = predict_raw(model, data, data.x_train, data.train_scale)
                payload = {
                    "mu_train": mu_train,
                    "sigma_train": sigma_train,
                    "y_train": data.y_train_raw,
                    "meta_train_date": pd.to_datetime(data.meta_train["Date"]).astype("datetime64[ns]").to_numpy(),
                    "meta_train_code": data.meta_train["code"].astype(str).to_numpy(),
                    "mu_val": mu_val,
                    "sigma_val": sigma_val,
                    "y_val": data.y_val_raw,
                    "meta_val_date": pd.to_datetime(data.meta_val["Date"]).astype("datetime64[ns]").to_numpy(),
                    "meta_val_code": data.meta_val["code"].astype(str).to_numpy(),
                }
                np.savez_compressed(pred_dir / f"predictions_seed_{seed}.npz", **payload)
                np.savez_compressed(gold_pred_dir / f"predictions_seed_{seed}.npz", **payload)
            row = {
                "variant": variant_name,
                "seed": seed,
                "rel_score": result["rel_score"],
                "directional_accuracy": result["directional_accuracy"],
                "pred_actual_q90_ratio": result["pred_actual_q90_ratio"],
                "description": cfg["description"],
            }
            all_rows.append(row)
            print(f"  seed={seed}: rel_score={result['rel_score']:.5f}  DA={result['directional_accuracy']:.4f}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(args.output_dir / "results_per_seed.csv", index=False)
    result_df.to_csv(args.gold_dir / "results_per_seed.csv", index=False)

    agg = (
        result_df.groupby("variant")
        .agg(
            rel_score_mean=("rel_score", "mean"),
            rel_score_std=("rel_score", "std"),
            DA_mean=("directional_accuracy", "mean"),
            n_seeds=("seed", "count"),
            description=("description", "first"),
        )
        .sort_values("rel_score_mean", ascending=False)
        .reset_index()
    )
    agg.to_csv(args.output_dir / "aggregate.csv", index=False)
    agg.to_csv(args.gold_dir / "aggregate.csv", index=False)

    text = "\n".join([
        "# VN Improvement Smoke",
        "",
        "Protocol: train <= 2020-03-31, validation 2020-04-01..2022-11-15. Holdout/test not used.",
        "Reference: hetero_combined_full5 raw ensemble = 0.0339 (pre-calibration)",
        "",
        "## Aggregate (sorted by rel_score_mean)",
        "",
        agg.round(6).to_markdown(index=False),
        "",
        "## Per Seed",
        "",
        result_df.round(6).to_markdown(index=False),
        "",
        json.dumps({"output_dir": str(args.output_dir), "gold_dir": str(args.gold_dir)}, indent=2),
    ])
    (args.output_dir / "summary.md").write_text(text, encoding="utf-8")
    (args.gold_dir / "summary.md").write_text(text, encoding="utf-8")
    print("\n\n" + text)


if __name__ == "__main__":
    main()
