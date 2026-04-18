from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.run_train import augment_sequence_with_stock_identity, load_frame, validate_columns
from src.evaluation.metric import evaluate
from src.models.architectures.attention import build_attention_model
from src.models.architectures.signmag import build_sign_magnitude_model
from src.models.training.datasets import build_sequence_dataset, split_sequence_dataset
from src.models.training.prediction import predict
from src.models.training.scalers import (
    FeatureScaler,
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    fit_local_target_normalizer,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.utils.features import ensure_paper_features


@dataclass
class ModelSpec:
    family: str
    checkpoint_name: str
    prediction_key: str | int | tuple[int, ...] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD inference for a saved model on a target universe and package plots.")
    parser.add_argument("--package-dir", type=Path, required=True, help="Gold package directory that contains model/core artifacts.")
    parser.add_argument("--universe-path", type=Path, required=True, help="Text file with target codes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output package directory.")
    parser.add_argument("--label", required=True, help="Human-readable title for plots/readme.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data/processed/assets/data_info_vn/history/vn_gold_recommended.csv",
    )
    return parser.parse_args()


def parse_code_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token.strip().upper() for token in re.split(r"[\s,]+", text) if token.strip()]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_scaler(path: Path) -> FeatureScaler:
    payload = np.load(path, allow_pickle=True)
    feature_columns = tuple(str(item) for item in payload["feature_columns"].tolist())
    return FeatureScaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        std=np.asarray(payload["std"], dtype=np.float32),
        feature_columns=feature_columns,
    )


def load_target_scaler(path: Path | None) -> TargetScaler | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path)
    return TargetScaler(
        mean=float(np.asarray(payload["mean"]).reshape(-1)[0]),
        std=float(np.asarray(payload["std"]).reshape(-1)[0]),
    )


def resolve_model_spec(package_dir: Path) -> ModelSpec:
    model_dir = package_dir / "model"
    if (model_dir / "model_signmag_seed_52.keras").exists():
        return ModelSpec(
            family="signmag",
            checkpoint_name="model_signmag_seed_52.keras",
            prediction_key="signed_prediction",
        )
    if (model_dir / "model_attention.keras").exists():
        return ModelSpec(
            family="attention",
            checkpoint_name="model_attention.keras",
            prediction_key=None,
        )
    raise FileNotFoundError(f"Unable to resolve supported checkpoint in {model_dir}")


def build_model_for_inference(
    spec: ModelSpec,
    config: dict,
    num_features: int,
    target_scaler: TargetScaler | None,
    local_target_normalizer: LocalTargetNormalizer | None,
):
    common_kwargs = dict(
        window_size=int(config["window_size"]),
        num_features=num_features,
        lstm_units=config["lstm_units"],
        lr=float(config["lr"]),
        dropout=float(config["dropout"]),
        loss=config["loss"],
        huber_delta=float(config.get("huber_delta", 0.01)),
        rel_score_large_move_quantile=float(config.get("rel_score_large_move_quantile", 0.8)),
        rel_score_directional_penalty=float(config.get("rel_score_directional_penalty", 0.6)),
        rel_score_confidence_penalty=float(config.get("rel_score_confidence_penalty", 0.35)),
        rel_score_confidence_ratio=float(config.get("rel_score_confidence_ratio", 0.25)),
        rel_score_weighted_high_quantile=float(config.get("rel_score_weighted_high_quantile", 0.8)),
        rel_score_weighted_high_weight=float(config.get("rel_score_weighted_high_weight", 3.0)),
        rel_score_weighted_base_weight=float(config.get("rel_score_weighted_base_weight", 1.0)),
    )
    if spec.family == "attention":
        return build_attention_model(
            attention_heads=int(config.get("attention_heads", 2)),
            attention_key_dim=int(config.get("attention_key_dim", 16)),
            target_scaler=target_scaler,
            local_target_normalizer=local_target_normalizer,
            **common_kwargs,
        )
    if spec.family == "signmag":
        return build_sign_magnitude_model(
            sign_loss_weight=float(config.get("signmag_sign_loss_weight", 0.15)),
            magnitude_loss_weight=float(config.get("signmag_magnitude_loss_weight", 0.35)),
            signed_loss_weight=float(config.get("signmag_signed_loss_weight", 1.5)),
            use_log_magnitude=bool(config.get("signmag_log_magnitude", True)),
            local_target_normalizer=local_target_normalizer,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported family: {spec.family}")


def build_local_target_normalizer(
    config: dict,
    data_path: Path,
    feature_columns: tuple[str, ...],
) -> LocalTargetNormalizer | None:
    target_normalizer = config.get("target_normalizer")
    if not target_normalizer:
        return None

    train_stocks = config.get("stocks")
    if not train_stocks:
        return None

    train_df = load_frame(data_path, train_stocks)
    if config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        train_df = ensure_paper_features(train_df)
    validate_columns(train_df, feature_columns, config["target_column"], target_normalizer)

    alias = f"__target_normalizer__{target_normalizer}"
    train_df[alias] = train_df[target_normalizer].astype(float)
    x_all, y_all, meta_all = build_sequence_dataset(
        train_df,
        feature_columns,
        config["target_column"],
        int(config["window_size"]),
        extra_meta_columns=(alias,),
        sequence_normalization=config.get("sequence_normalization", "none"),
    )
    if len(x_all) == 0:
        return None
    splits = split_sequence_dataset(
        x_all,
        y_all,
        meta_all,
        config["train_end_date"],
        config["val_end_date"],
    )
    meta_train = splits["train"][2]
    if meta_train.empty:
        return None
    scale_values = meta_train[alias].to_numpy(dtype=np.float32)
    return fit_local_target_normalizer(scale_values, target_normalizer)


def build_stock_identity_map(config: dict) -> dict[str, int]:
    codes = [str(code) for code in config.get("lstm_stock_identity_codes", [])]
    return {code: idx for idx, code in enumerate(codes)}


def prepare_universe_sequences(
    config: dict,
    data_path: Path,
    target_codes: list[str],
    feature_scaler: FeatureScaler,
    stock_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
    requested = set(target_codes)
    df = load_frame(data_path, ",".join(target_codes))
    if config.get("feature_phase") in {"paper_v1", "paper_denoise_v1"}:
        df = ensure_paper_features(df)
    validate_columns(df, feature_scaler.feature_columns, config["target_column"], config.get("target_normalizer"))

    present_codes = sorted(df["code"].astype(str).unique().tolist())
    missing_codes = sorted(requested - set(present_codes))

    alias = None
    extra_meta_columns: tuple[str, ...] = ()
    if config.get("target_normalizer"):
        alias = f"__target_normalizer__{config['target_normalizer']}"
        df[alias] = df[config["target_normalizer"]].astype(float)
        extra_meta_columns = (alias,)

    scaled_df = apply_feature_scaler(df, feature_scaler)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled_df,
        feature_scaler.feature_columns,
        config["target_column"],
        int(config["window_size"]),
        extra_meta_columns=extra_meta_columns,
        sequence_normalization=config.get("sequence_normalization", "none"),
    )
    splits = split_sequence_dataset(
        x_all,
        y_all,
        meta_all,
        config["train_end_date"],
        config["val_end_date"],
    )
    x_test, y_test, meta_test = splits["test"]
    if stock_to_idx:
        x_test = augment_sequence_with_stock_identity(x_test, meta_test, stock_to_idx)
    available_test_codes = sorted(meta_test["code"].astype(str).unique().tolist()) if not meta_test.empty else []
    return x_test, y_test, meta_test, missing_codes, available_test_codes


def postprocess_prediction(
    prediction: np.ndarray,
    meta_test: pd.DataFrame,
    target_scaler: TargetScaler | None,
    local_target_normalizer: LocalTargetNormalizer | None,
) -> np.ndarray:
    work = np.asarray(prediction, dtype=np.float32).reshape(-1)
    work = inverse_target_scaler_values(work, target_scaler)
    if local_target_normalizer is not None:
        alias = f"__target_normalizer__{local_target_normalizer.column}"
        scale_values = meta_test[alias].to_numpy(dtype=np.float32) if alias in meta_test.columns else None
        work = inverse_local_target_normalizer(work, scale_values, local_target_normalizer)
    return np.asarray(work, dtype=np.float32).reshape(-1)


def build_price_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    prices: list[pd.DataFrame] = []
    for code in sorted(pred_df["code"].astype(str).unique().tolist()):
        raw_path = ROOT / "data/raw/VN" / f"{code}.csv"
        if not raw_path.exists():
            continue
        raw_df = pd.read_csv(raw_path, usecols=["Date", "code", "adjust", "close"])
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        price_col = "adjust" if "adjust" in raw_df.columns else "close"
        raw_df = raw_df[["Date", "code", price_col]].rename(columns={price_col: "base_price"})
        prices.append(raw_df)
    if not prices:
        return pred_df.copy()
    price_df = pd.concat(prices, ignore_index=True)
    merged = pred_df.merge(price_df, on=["Date", "code"], how="left")
    merged["actual_next_price"] = merged["base_price"] * (1.0 + merged["actual"])
    merged["predicted_next_price"] = merged["base_price"] * (1.0 + merged["prediction"])
    return merged


def save_plot_by_code(
    df: pd.DataFrame,
    code_metrics: pd.DataFrame,
    y_actual: str,
    y_pred: str,
    output_path: Path,
    title: str,
    actual_label: str,
    pred_label: str,
) -> None:
    codes = sorted(df["code"].astype(str).unique().tolist())
    metric_map = code_metrics.set_index("code").to_dict(orient="index") if not code_metrics.empty else {}
    ncols = 4 if len(codes) > 40 else (3 if len(codes) > 12 else 2)
    nrows = ceil(len(codes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, code in zip(axes_flat, codes):
        part = df[df["code"] == code].sort_values("Date")
        metrics = metric_map.get(code, {})
        ax.plot(part["Date"], part[y_actual], color="#1f77b4", linewidth=1.1, label=actual_label)
        ax.plot(part["Date"], part[y_pred], color="#d62728", linewidth=0.95, alpha=0.9, label=pred_label)
        if y_actual == "actual":
            ax.axhline(0.0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_title(
            (
                f"{code} | "
                f"rel={metrics.get('rel_score', float('nan')):.4f} | "
                f"abs={metrics.get('abs_score', float('nan')):.4f} | "
                f"err={metrics.get('error_mean', float('nan')):.4f}"
            ),
            fontsize=8.2,
        )
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(fontsize=6, loc="best")
        if metrics:
            stats_text = (
                f"base={metrics['base_score']:.4f}\n"
                f"abs={metrics['abs_score']:.4f}\n"
                f"err={metrics['error_mean']:.4f}"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
            )
    for ax in axes_flat[len(codes):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_error_hist_by_code(
    df: pd.DataFrame,
    code_metrics: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    codes = sorted(df["code"].astype(str).unique().tolist())
    metric_map = code_metrics.set_index("code").to_dict(orient="index") if not code_metrics.empty else {}
    ncols = 4 if len(codes) > 40 else (3 if len(codes) > 12 else 2)
    nrows = ceil(len(codes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, code in zip(axes_flat, codes):
        part = df[df["code"] == code].sort_values("Date")
        metrics = metric_map.get(code, {})
        error = part["prediction"].to_numpy(dtype=np.float32) - part["actual"].to_numpy(dtype=np.float32)
        ax.hist(error, bins=30, color="#d62728", alpha=0.78, edgecolor="white")
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        if metrics:
            ax.axvline(metrics["error_q25"], color="#1f77b4", linestyle=":", linewidth=0.9)
            ax.axvline(metrics["error_mean"], color="#2ca02c", linestyle="-", linewidth=0.9)
            ax.axvline(metrics["error_q75"], color="#ff7f0e", linestyle=":", linewidth=0.9)
        ax.set_title(
            (
                f"{code} | "
                f"rel={metrics.get('rel_score', float('nan')):.4f} | "
                f"abs={metrics.get('abs_score', float('nan')):.4f} | "
                f"err={metrics.get('error_mean', float('nan')):.4f}"
            ),
            fontsize=8.2,
        )
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if metrics:
            stats_text = (
                f"min={error.min():.4f}\n"
                f"q25={metrics['error_q25']:.4f}\n"
                f"mean={metrics['error_mean']:.4f}\n"
                f"q75={metrics['error_q75']:.4f}\n"
                f"max={error.max():.4f}"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#cccccc"},
            )
    for ax in axes_flat[len(codes):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_code_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for code, part in df.groupby("code", sort=True):
        metric = evaluate(part["prediction"].to_numpy(), part["actual"].to_numpy())
        err = part["prediction"].to_numpy(dtype=np.float32) - part["actual"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "code": str(code),
                "rows": int(len(part)),
                "rel_score": float(metric["rel_score"]),
                "base_score": float(metric["base_loss"]),
                "abs_score": float(metric["abs_loss"]),
                "error_mean": float(np.mean(err)),
                "error_abs_mean": float(np.mean(np.abs(err))),
                "error_q25": float(np.quantile(err, 0.25)),
                "error_q75": float(np.quantile(err, 0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values(["rel_score", "code"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = load_json(args.package_dir / "core" / "source_config.json")
    spec = resolve_model_spec(args.package_dir)
    feature_scaler = load_feature_scaler(args.package_dir / "model" / "feature_scaler.npz")
    target_scaler = load_target_scaler(args.package_dir / "model" / "target_scaler.npz")
    local_target_normalizer = build_local_target_normalizer(config, args.data_path, feature_scaler.feature_columns)
    stock_to_idx = build_stock_identity_map(config)
    target_codes = parse_code_list(args.universe_path)

    x_test, y_test, meta_test, missing_codes, available_test_codes = prepare_universe_sequences(
        config=config,
        data_path=args.data_path,
        target_codes=target_codes,
        feature_scaler=feature_scaler,
        stock_to_idx=stock_to_idx,
    )
    if len(x_test) == 0:
        raise ValueError("No test sequences available for target universe.")

    model = build_model_for_inference(
        spec=spec,
        config=config,
        num_features=int(x_test.shape[2]),
        target_scaler=target_scaler,
        local_target_normalizer=local_target_normalizer,
    )
    model.load_weights(args.package_dir / "model" / spec.checkpoint_name)
    raw_pred = predict(model, x_test, prediction_key=spec.prediction_key)
    pred = postprocess_prediction(raw_pred, meta_test, target_scaler, local_target_normalizer)

    pred_df = meta_test.copy()
    pred_df["prediction"] = pred
    pred_df["actual"] = y_test.astype(np.float32)
    pred_df["split"] = "test"
    pred_df["model"] = spec.family
    pred_df = pred_df.sort_values(["code", "Date"]).reset_index(drop=True)
    pred_df = build_price_frame(pred_df)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    (args.output_dir / "core").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(parents=True, exist_ok=True)

    metric = evaluate(
        pred_df["prediction"].to_numpy(),
        pred_df["actual"].to_numpy(),
        group_ids=pred_df["code"].to_numpy(),
    )
    code_metrics = build_code_metrics(pred_df)
    err = pred_df["prediction"].to_numpy(dtype=np.float32) - pred_df["actual"].to_numpy(dtype=np.float32)
    summary = {
        "label": args.label,
        "source_package_dir": str(args.package_dir.resolve()),
        "checkpoint_name": spec.checkpoint_name,
        "model_family": spec.family,
        "prediction_key": spec.prediction_key if spec.prediction_key is None else str(spec.prediction_key),
        "requested_universe_count": len(target_codes),
        "missing_from_dataset_count": len(missing_codes),
        "missing_from_dataset_codes": missing_codes,
        "available_test_code_count": len(available_test_codes),
        "available_test_codes": available_test_codes,
        "rows": int(len(pred_df)),
        "rel_score": float(metric["rel_score"]),
        "base_score": float(metric["base_loss"]),
        "abs_score": float(metric["abs_loss"]),
        "error_mean": float(np.mean(err)),
        "error_q25": float(np.quantile(err, 0.25)),
        "error_q75": float(np.quantile(err, 0.75)),
        "error_min": float(np.min(err)),
        "error_max": float(np.max(err)),
        "ood_note": "Models were trained with stock-identity one-hot. Codes outside the original training universe are inferred with all-zero identity, so this package is an OOD stress test, not a clean validation run.",
    }

    pred_df.to_csv(args.output_dir / "core" / "predictions_test.csv", index=False)
    code_metrics.to_csv(args.output_dir / "core" / "code_metrics.csv", index=False)
    (args.output_dir / "core" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    shutil.copy2(args.package_dir / "core" / "source_config.json", args.output_dir / "core" / "source_config.json")

    save_plot_by_code(
        pred_df,
        code_metrics=code_metrics,
        y_actual="actual",
        y_pred="prediction",
        output_path=args.output_dir / "plots" / "test_actual_vs_predicted_return_by_code.png",
        title=f"{args.label}\nTest Actual Return vs Predicted Return by Code",
        actual_label="actual_return",
        pred_label="predicted_return",
    )
    if "actual_next_price" in pred_df.columns and "predicted_next_price" in pred_df.columns:
        save_plot_by_code(
            pred_df.dropna(subset=["actual_next_price", "predicted_next_price"]),
            code_metrics=code_metrics,
            y_actual="actual_next_price",
            y_pred="predicted_next_price",
            output_path=args.output_dir / "plots" / "test_actual_vs_predicted_price_from_return_by_code.png",
            title=f"{args.label}\nTest Actual Next Close vs Predicted Next Close by Code",
            actual_label="actual_next_close",
            pred_label="predicted_next_close",
        )
    save_error_hist_by_code(
        pred_df,
        code_metrics=code_metrics,
        output_path=args.output_dir / "plots" / "test_return_error_hist_by_code.png",
        title=f"{args.label}\nHistogram of Prediction Return - Actual Return by Code",
    )

    readme = (
        f"# {args.label}\n\n"
        f"- source_package: `{args.package_dir}`\n"
        f"- checkpoint: `{spec.checkpoint_name}`\n"
        f"- rel_score: `{summary['rel_score']:.6f}`\n"
        f"- base_score: `{summary['base_score']:.6f}`\n"
        f"- abs_score: `{summary['abs_score']:.6f}`\n"
        f"- requested_universe_count: `{summary['requested_universe_count']}`\n"
        f"- available_test_code_count: `{summary['available_test_code_count']}`\n"
        f"- missing_from_dataset_count: `{summary['missing_from_dataset_count']}`\n\n"
        f"{summary['ood_note']}\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
