from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.architectures.backbone import build_lstm_backbone  # noqa: E402
from src.models.components.losses import RelScoreLoss, RelScoreWeightedLoss, RelScoreWeightedTailLoss  # noqa: E402
from src.models.config import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from src.models.training.datasets import build_sequence_dataset, split_frame_by_date, split_sequence_dataset  # noqa: E402
from src.models.training.feature_normalization import add_multimarket_feature_normalization  # noqa: E402
from src.models.training.scalers import (  # noqa: E402
    LocalTargetNormalizer,
    TargetScaler,
    apply_feature_scaler,
    apply_local_target_normalizer,
    apply_target_scaler,
    fit_feature_scaler,
    fit_local_target_normalizer,
    fit_target_scaler,
    inverse_local_target_normalizer,
    inverse_target_scaler_values,
)
from src.models.training.seeds import set_global_seed  # noqa: E402
from src.models.training.pipeline import load_frame as load_training_frame  # noqa: E402


BASE_RUN_CONFIG = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "broad_signmag_portable_no_identity_20260428_allvn_r01"
    / "reports"
    / "core"
    / "config.json"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "assets"
    / "data_info_vn"
    / "history"
    / "training_runs"
    / "reports"
    / "tail_aware_lstm_probe_20260519"
)
DEFAULT_GOLD = ROOT / "gold" / "vn_transition_pressure_20260512" / "plots" / "tail_aware_lstm_probe_20260519"


@dataclass(frozen=True)
class Variant:
    name: str
    normalization: str
    model_type: str
    loss: str
    use_instance_zscore: bool = False
    weighted_high_quantile: float = 0.80
    weighted_high_weight: float = 4.0
    tail_error_threshold: float = 0.05
    tail_penalty_weight: float = 0.10
    extra_feature_columns: tuple[str, ...] = ()
    directional_penalty_weight: float = 0.0
    stress_aux_column: str = "market_negative_ratio"
    stress_aux_weight: float = 0.10
    stress_aux_quantile: float = 0.75
    market_aux_return_column: str = "future_market_return_mean"
    market_aux_abs_column: str = "future_market_abs_return_q90"
    market_aux_weight: float = 0.10
    risk_aux_weight: float = 0.10
    risk_aux_threshold: float = 0.035
    feature_clip_abs: float | None = None
    target_scale_mode: str = "stock_vol"
    target_scale_floor_quantile: float = 0.25
    input_feature_dropout_rate: float = 0.0
    input_feature_dropout_columns: tuple[str, ...] = ()
    context_gate_columns: tuple[str, ...] = ()
    context_gate_units: int = 16


@dataclass(frozen=True)
class PreparedData:
    feature_columns: tuple[str, ...]
    x_train: np.ndarray
    y_train_raw: np.ndarray
    y_train_model: np.ndarray
    meta_train: pd.DataFrame
    train_scale_values: np.ndarray
    x_val: np.ndarray
    y_val_raw: np.ndarray
    y_val_model: np.ndarray
    meta_val: pd.DataFrame
    val_scale_values: np.ndarray
    target_scaler: TargetScaler
    local_target_normalizer: LocalTargetNormalizer


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe tail-aware LSTM variants for return forecasting.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--train-end-date", default="2020-03-31")
    parser.add_argument("--val-end-date", default="2022-11-15")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--target-column", default="target_next_return")
    parser.add_argument("--target-normalizer", default="volatility_20")
    parser.add_argument("--feature-columns", default=None)
    parser.add_argument(
        "--variants",
        default="plain_global_rel,plain_global_instance_rel,plain_multimarket_rel,tailaware_multimarket_weighted",
    )
    parser.add_argument("--tail5-threshold", type=float, default=0.05)
    parser.add_argument("--tail7-threshold", type=float, default=0.07)
    parser.add_argument("--spike-thresholds", default="0.05,0.07,0.08")
    return parser.parse_args(argv)


def load_base_feature_columns() -> tuple[str, ...]:
    config = json.loads(BASE_RUN_CONFIG.read_text(encoding="utf-8"))
    columns = config.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Could not read feature_columns from {BASE_RUN_CONFIG}")
    return tuple(str(column) for column in columns)


def parse_lstm_units(value: str) -> list[int]:
    units = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not units:
        raise ValueError("lstm_units must not be empty.")
    return units


def all_variants() -> dict[str, Variant]:
    tailstress_features = (
        "market_return_q10",
        "market_return_q25",
        "market_negative_ratio",
        "market_left_tail_4pct_ratio",
        "market_left_tail_6pct_ratio",
        "market_abs_return_q90",
    )
    tailstress_past_features = (
        "market_return_q10_lag1",
        "market_return_q25_lag1",
        "market_negative_ratio_lag1",
        "market_left_tail_4pct_ratio_lag1",
        "market_left_tail_6pct_ratio_lag1",
        "market_abs_return_q90_lag1",
        "market_negative_ratio_ewm5_lag1",
        "market_abs_return_q90_ewm5_lag1",
    )
    return {
        "plain_global_rel": Variant("plain_global_rel", "global", "plain", "rel_score"),
        "plain_global_weighted": Variant("plain_global_weighted", "global", "plain", "rel_score_weighted"),
        "plain_global_weighted_mild": Variant(
            "plain_global_weighted_mild",
            "global",
            "plain",
            "rel_score_weighted",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
        ),
        "plain_global_weighted_mild_tail35_p05": Variant(
            "plain_global_weighted_mild_tail35_p05",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
        ),
        "plain_global_weighted_mild_tail35_stressaux": Variant(
            "plain_global_weighted_mild_tail35_stressaux",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.10,
            stress_aux_quantile=0.75,
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_clip3": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_clip3",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            feature_clip_abs=3.0,
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_clip5": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_clip5",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            feature_clip_abs=5.0,
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_floor50": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_floor50",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            target_scale_floor_quantile=0.50,
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_mktmaxscale": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_mktmaxscale",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            target_scale_mode="stock_market_max_ewm5_lag1",
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_mktblendscale": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_mktblendscale",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            target_scale_mode="stock_market_blend_ewm5_lag1",
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop10": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop10",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            input_feature_dropout_rate=0.10,
            input_feature_dropout_columns=(
                "vnindex_return",
                "market_leader_return",
                "a_d_ratio",
                "sector_momentum_rank",
                "is_top_2_sector",
            ),
        ),
        "plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop20": Variant(
            "plain_global_weighted_mild_tail35_stressaux_w20_ctxdrop20",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            input_feature_dropout_rate=0.20,
            input_feature_dropout_columns=(
                "vnindex_return",
                "market_leader_return",
                "a_d_ratio",
                "sector_momentum_rank",
                "is_top_2_sector",
            ),
        ),
        "plain_global_weighted_mild_tail35_contextgate_w20": Variant(
            "plain_global_weighted_mild_tail35_contextgate_w20",
            "global",
            "contextgate_stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            context_gate_columns=(
                "vnindex_return",
                "market_leader_return",
                "a_d_ratio",
                "sector_momentum_rank",
                "is_top_2_sector",
            ),
            context_gate_units=16,
        ),
        "plain_global_weighted_mild_tail35_contextresid_w20": Variant(
            "plain_global_weighted_mild_tail35_contextresid_w20",
            "global",
            "contextresid_stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
            context_gate_columns=(
                "vnindex_return",
                "market_leader_return",
                "a_d_ratio",
                "sector_momentum_rank",
                "is_top_2_sector",
            ),
            context_gate_units=12,
        ),
        "plain_global_weighted_mild_tail35_stressaux_past": Variant(
            "plain_global_weighted_mild_tail35_stressaux_past",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="market_negative_ratio_lag1",
            stress_aux_weight=0.10,
            stress_aux_quantile=0.75,
        ),
        "plain_global_weighted_mild_tail35_futurestress_w10": Variant(
            "plain_global_weighted_mild_tail35_futurestress_w10",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="future_market_abs_return_q90",
            stress_aux_weight=0.10,
            stress_aux_quantile=0.75,
        ),
        "plain_global_weighted_mild_tail35_futurestress_w20": Variant(
            "plain_global_weighted_mild_tail35_futurestress_w20",
            "global",
            "stressaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            stress_aux_column="future_market_abs_return_q90",
            stress_aux_weight=0.20,
            stress_aux_quantile=0.75,
        ),
        "plain_global_weighted_mild_tail35_marketaux_w10": Variant(
            "plain_global_weighted_mild_tail35_marketaux_w10",
            "global",
            "marketaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            market_aux_weight=0.10,
        ),
        "plain_global_weighted_mild_tail35_marketaux_w20": Variant(
            "plain_global_weighted_mild_tail35_marketaux_w20",
            "global",
            "marketaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            market_aux_weight=0.20,
        ),
        "plain_global_weighted_mild_tail35_riskaux_w10": Variant(
            "plain_global_weighted_mild_tail35_riskaux_w10",
            "global",
            "riskaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            risk_aux_weight=0.10,
            risk_aux_threshold=0.035,
        ),
        "plain_global_weighted_mild_tail35_riskaux_w20": Variant(
            "plain_global_weighted_mild_tail35_riskaux_w20",
            "global",
            "riskaux",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            risk_aux_weight=0.20,
            risk_aux_threshold=0.035,
        ),
        "plain_global_weighted_mild_tail35_riskaux_detached_w10": Variant(
            "plain_global_weighted_mild_tail35_riskaux_detached_w10",
            "global",
            "riskaux_detached",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            risk_aux_weight=0.10,
            risk_aux_threshold=0.035,
        ),
        "plain_global_weighted_mild_tail35_riskaux_detached_w20": Variant(
            "plain_global_weighted_mild_tail35_riskaux_detached_w20",
            "global",
            "riskaux_detached",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            risk_aux_weight=0.20,
            risk_aux_threshold=0.035,
        ),
        "plain_global_weighted_mild_tail35_p05_tailstress": Variant(
            "plain_global_weighted_mild_tail35_p05_tailstress",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            extra_feature_columns=tailstress_features,
        ),
        "plain_global_weighted_mild_tail35_p05_tailstress_past": Variant(
            "plain_global_weighted_mild_tail35_p05_tailstress_past",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            extra_feature_columns=tailstress_past_features,
        ),
        "plain_global_weighted_mild_tail50_p10": Variant(
            "plain_global_weighted_mild_tail50_p10",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.05,
            tail_penalty_weight=0.10,
        ),
        "plain_global_weighted_mild_tail50_p20": Variant(
            "plain_global_weighted_mild_tail50_p20",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.05,
            tail_penalty_weight=0.20,
        ),
        "plain_global_weighted_mild_tail35_sign_p05": Variant(
            "plain_global_weighted_mild_tail35_sign_p05",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            directional_penalty_weight=0.05,
        ),
        "plain_global_weighted_mild_tail35_sign_p10": Variant(
            "plain_global_weighted_mild_tail35_sign_p10",
            "global",
            "plain",
            "rel_score_weighted_tail",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
            tail_error_threshold=0.035,
            tail_penalty_weight=0.05,
            directional_penalty_weight=0.10,
        ),
        "plain_global_instance_rel": Variant(
            "plain_global_instance_rel",
            "global",
            "plain",
            "rel_score",
            use_instance_zscore=True,
        ),
        "plain_global_instance_weighted_mild": Variant(
            "plain_global_instance_weighted_mild",
            "global",
            "plain",
            "rel_score_weighted",
            use_instance_zscore=True,
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
        ),
        "plain_multimarket_rel": Variant("plain_multimarket_rel", "multimarket_v1", "plain", "rel_score"),
        "plain_multimarket_weighted_mild": Variant(
            "plain_multimarket_weighted_mild",
            "multimarket_v1",
            "plain",
            "rel_score_weighted",
            weighted_high_quantile=0.85,
            weighted_high_weight=1.75,
        ),
        "plain_multimarket_weighted": Variant(
            "plain_multimarket_weighted",
            "multimarket_v1",
            "plain",
            "rel_score_weighted",
        ),
        "tailaware_multimarket_weighted": Variant(
            "tailaware_multimarket_weighted",
            "multimarket_v1",
            "tailaware",
            "rel_score_weighted",
        ),
    }


def robust_loss(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan")
    abs_values = np.abs(clean)
    return float(np.quantile(abs_values, 0.50) + 0.5 * np.quantile(abs_values, 0.90))


def rel_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    base = robust_loss(actual)
    if not np.isfinite(base) or base <= 0.0:
        return float("nan")
    return float(1.0 - robust_loss(actual - predicted) / base)


def add_tail_stress_features(df: pd.DataFrame, target_column: str = "target_next_return") -> pd.DataFrame:
    work = df.sort_values(["code", "Date"], kind="stable").copy()
    work["return_1"] = work.groupby("code", sort=False)["adjust"].pct_change()
    daily_agg: dict[str, tuple[str, object]] = {
        "market_return_q10": ("return_1", lambda values: float(np.nanquantile(values, 0.10))),
        "market_return_q25": ("return_1", lambda values: float(np.nanquantile(values, 0.25))),
        "market_negative_ratio": ("return_1", lambda values: float(np.nanmean(values < 0.0))),
        "market_left_tail_4pct_ratio": ("return_1", lambda values: float(np.nanmean(values <= -0.04))),
        "market_left_tail_6pct_ratio": ("return_1", lambda values: float(np.nanmean(values <= -0.06))),
        "market_abs_return_q90": ("return_1", lambda values: float(np.nanquantile(np.abs(values), 0.90))),
    }
    if target_column in work.columns:
        daily_agg.update(
            {
                "future_market_return_q10": (target_column, lambda values: float(np.nanquantile(values, 0.10))),
                "future_market_return_q25": (target_column, lambda values: float(np.nanquantile(values, 0.25))),
                "future_market_return_mean": (target_column, lambda values: float(np.nanmean(values))),
                "future_market_negative_ratio": (target_column, lambda values: float(np.nanmean(values < 0.0))),
                "future_market_left_tail_4pct_ratio": (target_column, lambda values: float(np.nanmean(values <= -0.04))),
                "future_market_left_tail_6pct_ratio": (target_column, lambda values: float(np.nanmean(values <= -0.06))),
                "future_market_abs_return_q90": (
                    target_column,
                    lambda values: float(np.nanquantile(np.abs(values), 0.90)),
                ),
            }
        )
    daily = work.groupby("Date", sort=True).agg(**daily_agg).reset_index()
    daily = daily.sort_values("Date", kind="stable")
    tail_columns = [
        "market_return_q10",
        "market_return_q25",
        "market_negative_ratio",
        "market_left_tail_4pct_ratio",
        "market_left_tail_6pct_ratio",
        "market_abs_return_q90",
    ]
    for column in tail_columns:
        daily[f"{column}_lag1"] = daily[column].shift(1)
    daily["market_negative_ratio_ewm5_lag1"] = daily["market_negative_ratio"].shift(1).ewm(span=5, adjust=False).mean()
    daily["market_abs_return_q90_ewm5_lag1"] = daily["market_abs_return_q90"].shift(1).ewm(span=5, adjust=False).mean()
    return work.merge(daily, on="Date", how="left")


def load_frame(data_path: Path, feature_columns: tuple[str, ...], target_column: str, target_normalizer: str) -> pd.DataFrame:
    frame = load_training_frame(data_path, stocks=None)
    required = {"Date", "code", target_column, target_normalizer, "adjust", *feature_columns}
    available = set(frame.columns)
    missing = sorted(required.difference(available))
    if missing:
        fallback = tuple(column for column in DEFAULT_FEATURE_COLUMNS if column in available)
        if not fallback:
            raise ValueError(f"Missing required columns after pipeline feature prep: {missing}")
        print(
            "Info: requested feature set is not fully available after current feature prep; "
            f"falling back to current config features ({len(fallback)} columns). Missing: {missing}"
        )
        feature_columns = fallback
        required = {"Date", "code", target_column, target_normalizer, "adjust", *feature_columns}
        missing = sorted(required.difference(set(frame.columns)))
        if missing:
            raise ValueError(f"Missing fallback columns after pipeline feature prep: {missing}")
    frame = frame.loc[:, sorted(required)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["code"] = frame["code"].astype(str).str.upper()
    frame = add_tail_stress_features(frame, target_column=target_column)
    frame.attrs["feature_columns"] = feature_columns
    return frame


def apply_input_normalization(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    variant: Variant,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, object]]:
    if variant.normalization == "global":
        return df.copy(), feature_columns, {"mode": "global_train_zscore"}
    if variant.normalization == "multimarket_v1":
        result = add_multimarket_feature_normalization(
            df,
            feature_columns,
            rolling_window=60,
            min_periods=20,
            include_cross_sectional_z=True,
            include_cross_sectional_rank=True,
            strict_past=True,
        )
        return result.frame, result.feature_columns, result.metadata
    raise ValueError(f"Unsupported normalization: {variant.normalization}")


def build_model_target(
    y_raw: np.ndarray,
    scale_values: np.ndarray,
    local_target_normalizer: LocalTargetNormalizer,
    target_scaler: TargetScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, TargetScaler]:
    y_local = apply_local_target_normalizer(y_raw, scale_values, local_target_normalizer)
    fitted_scaler = target_scaler or fit_target_scaler(y_local)
    y_scaled = apply_target_scaler(y_local, fitted_scaler).reshape(-1, 1)
    return (
        np.concatenate([y_scaled, np.asarray(scale_values, dtype=np.float32).reshape(-1, 1)], axis=1).astype(np.float32),
        y_local.reshape(-1),
        fitted_scaler,
    )


def fit_variant_local_target_normalizer(
    scale_values: np.ndarray,
    column: str,
    floor_quantile: float,
) -> LocalTargetNormalizer:
    values = np.abs(np.asarray(scale_values, dtype=np.float32))
    valid = values[np.isfinite(values) & (values > 0)]
    if len(valid) == 0:
        floor = 1.0
    else:
        clipped_quantile = min(max(float(floor_quantile), 0.0), 0.95)
        floor = max(float(np.quantile(valid, clipped_quantile)), 1e-4)
    return LocalTargetNormalizer(column=column, floor=floor)


def build_target_scale_values(normalized: pd.DataFrame, variant: Variant, args: argparse.Namespace) -> pd.Series:
    stock_scale = normalized[args.target_normalizer].astype(float).abs()
    if variant.target_scale_mode == "stock_vol":
        return stock_scale
    if "market_abs_return_q90_ewm5_lag1" not in normalized.columns:
        raise ValueError(f"Missing market tail scale column for {variant.name}: market_abs_return_q90_ewm5_lag1")
    market_scale = normalized["market_abs_return_q90_ewm5_lag1"].astype(float).abs()
    market_scale = market_scale.replace([np.inf, -np.inf], np.nan).fillna(float(np.nanmedian(stock_scale)))
    if variant.target_scale_mode == "stock_market_max_ewm5_lag1":
        return pd.Series(np.maximum(stock_scale.to_numpy(dtype=float), market_scale.to_numpy(dtype=float)), index=normalized.index)
    if variant.target_scale_mode == "stock_market_blend_ewm5_lag1":
        return 0.70 * stock_scale + 0.30 * market_scale
    raise ValueError(f"Unsupported target_scale_mode for {variant.name}: {variant.target_scale_mode}")


def prepare_data(
    raw: pd.DataFrame,
    feature_columns: tuple[str, ...],
    variant: Variant,
    args: argparse.Namespace,
) -> PreparedData:
    if variant.extra_feature_columns:
        missing = sorted(set(variant.extra_feature_columns).difference(raw.columns))
        if missing:
            raise ValueError(f"Missing extra feature columns for {variant.name}: {missing}")
        feature_columns = tuple(dict.fromkeys([*feature_columns, *variant.extra_feature_columns]))
    normalized, input_features, _ = apply_input_normalization(raw, feature_columns, variant)
    target_alias = f"__target_normalizer__{variant.target_scale_mode}"
    normalized[target_alias] = build_target_scale_values(normalized, variant, args)
    extra_meta_columns = [target_alias]
    if variant.model_type in {"stressaux", "contextgate_stressaux", "contextresid_stressaux"}:
        if variant.stress_aux_column not in normalized.columns:
            raise ValueError(f"Missing stress aux column for {variant.name}: {variant.stress_aux_column}")
        extra_meta_columns.append(variant.stress_aux_column)
    if variant.model_type == "marketaux":
        missing_aux = [
            column
            for column in (variant.market_aux_return_column, variant.market_aux_abs_column)
            if column not in normalized.columns
        ]
        if missing_aux:
            raise ValueError(f"Missing market aux columns for {variant.name}: {missing_aux}")
        extra_meta_columns.extend([variant.market_aux_return_column, variant.market_aux_abs_column])
    train_df, _, _ = split_frame_by_date(normalized, args.train_end_date, args.val_end_date)
    scaler = fit_feature_scaler(train_df.dropna(subset=input_features), input_features)
    scaled = apply_feature_scaler(normalized, scaler)
    if variant.feature_clip_abs is not None:
        clip_abs = float(variant.feature_clip_abs)
        if clip_abs <= 0.0:
            raise ValueError(f"feature_clip_abs must be positive for {variant.name}.")
        scaled.loc[:, input_features] = scaled.loc[:, input_features].clip(lower=-clip_abs, upper=clip_abs)
    x_all, y_all, meta_all = build_sequence_dataset(
        scaled,
        input_features,
        args.target_column,
        args.window_size,
        extra_meta_columns=tuple(extra_meta_columns),
        sequence_normalization="instance_zscore" if variant.use_instance_zscore else "none",
    )
    splits = split_sequence_dataset(x_all, y_all, meta_all, args.train_end_date, args.val_end_date)
    x_train, y_train_raw, meta_train = splits["train"]
    x_val, y_val_raw, meta_val = splits["val"]
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError(f"Not enough train/val sequences for {variant.name}.")
    train_scale_values = meta_train[target_alias].to_numpy(dtype=np.float32)
    val_scale_values = meta_val[target_alias].to_numpy(dtype=np.float32)
    local_normalizer = fit_variant_local_target_normalizer(
        train_scale_values,
        f"{args.target_normalizer}:{variant.target_scale_mode}",
        variant.target_scale_floor_quantile,
    )
    y_train_model, _, target_scaler = build_model_target(
        y_train_raw,
        train_scale_values,
        local_normalizer,
    )
    y_val_model, _, _ = build_model_target(
        y_val_raw,
        val_scale_values,
        local_normalizer,
        target_scaler,
    )
    return PreparedData(
        feature_columns=input_features,
        x_train=x_train,
        y_train_raw=y_train_raw.astype(np.float32),
        y_train_model=y_train_model,
        meta_train=meta_train,
        train_scale_values=train_scale_values,
        x_val=x_val,
        y_val_raw=y_val_raw.astype(np.float32),
        y_val_model=y_val_model,
        meta_val=meta_val,
        val_scale_values=val_scale_values,
        target_scaler=target_scaler,
        local_target_normalizer=local_normalizer,
    )


def build_pred_loss(
    variant: Variant,
    target_scaler: TargetScaler,
    local_target_normalizer: LocalTargetNormalizer,
) -> keras.losses.Loss:
    kwargs = {
        "target_mean": target_scaler.mean,
        "target_std": target_scaler.std,
        "use_target_scaler": True,
        "local_scale_floor": local_target_normalizer.floor,
    }
    if variant.loss == "rel_score":
        return RelScoreLoss(**kwargs)
    if variant.loss == "rel_score_weighted":
        return RelScoreWeightedLoss(
            **kwargs,
            high_quantile=variant.weighted_high_quantile,
            high_weight=variant.weighted_high_weight,
            base_weight=1.0,
        )
    if variant.loss == "rel_score_weighted_tail":
        return RelScoreWeightedTailLoss(
            **kwargs,
            high_quantile=variant.weighted_high_quantile,
            high_weight=variant.weighted_high_weight,
            base_weight=1.0,
            tail_error_threshold=variant.tail_error_threshold,
            tail_penalty_weight=variant.tail_penalty_weight,
            directional_penalty_weight=variant.directional_penalty_weight,
        )
    raise ValueError(f"Unsupported loss: {variant.loss}")


def build_plain_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    model = keras.Model(inputs=inputs, outputs=pred)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
    )
    return model


def build_stressaux_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    stress = layers.Dense(1, activation="sigmoid", name="stress_aux")(encoded)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "stress_aux": stress})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "stress_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"pred": 1.0, "stress_aux": variant.stress_aux_weight},
    )
    return model


def _stack_lstm_on_tensor(
    x: keras.KerasTensor,
    unit_stack: list[int],
    dropout: float,
    name_prefix: str,
) -> keras.KerasTensor:
    for layer_idx, units in enumerate(unit_stack):
        return_sequences = layer_idx < len(unit_stack) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            kernel_regularizer=keras.regularizers.l2(1e-5),
            recurrent_regularizer=keras.regularizers.l2(1e-5),
            name=f"{name_prefix}_lstm_{layer_idx + 1}",
        )(x)
        if dropout > 0.0 and return_sequences:
            x = layers.Dropout(dropout, name=f"{name_prefix}_dropout_{layer_idx + 1}")(x)
    return x


def build_contextgate_stressaux_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    feature_index = {feature: idx for idx, feature in enumerate(data.feature_columns)}
    context_indices = [feature_index[feature] for feature in variant.context_gate_columns if feature in feature_index]
    if not context_indices:
        raise ValueError(f"No context gate columns found in feature set for {variant.name}.")
    stock_indices = [idx for idx in range(len(data.feature_columns)) if idx not in set(context_indices)]
    if not stock_indices:
        raise ValueError(f"No stock feature columns left after context split for {variant.name}.")

    inputs = layers.Input(shape=(args.window_size, data.x_train.shape[2]), name="features")
    stock_x = layers.Lambda(
        lambda value: tf.gather(value, stock_indices, axis=-1),
        name="stock_feature_slice",
    )(inputs)
    context_x = layers.Lambda(
        lambda value: tf.gather(value, context_indices, axis=-1),
        name="context_feature_slice",
    )(inputs)

    unit_stack = parse_lstm_units(args.lstm_units)
    stock_encoded = _stack_lstm_on_tensor(stock_x, unit_stack, args.dropout, "stock")
    context_units = max(4, int(variant.context_gate_units))
    context_encoded = layers.LSTM(
        context_units,
        kernel_regularizer=keras.regularizers.l2(1e-5),
        recurrent_regularizer=keras.regularizers.l2(1e-5),
        name="context_lstm",
    )(context_x)
    context_projected = layers.Dense(unit_stack[-1], activation="tanh", name="context_project")(context_encoded)
    context_gate = layers.Dense(unit_stack[-1], activation="sigmoid", name="context_gate")(context_encoded)
    gated_context = layers.Multiply(name="gated_context")([context_projected, context_gate])
    combined = layers.Concatenate(name="stock_context_concat")([stock_encoded, gated_context])
    encoded = layers.Dense(unit_stack[-1], activation="tanh", name="gated_fusion")(combined)
    if args.dropout > 0.0:
        encoded = layers.Dropout(args.dropout, name="gated_fusion_dropout")(encoded)

    pred = layers.Dense(1, name="pred")(encoded)
    stress = layers.Dense(1, activation="sigmoid", name="stress_aux")(encoded)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "stress_aux": stress})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "stress_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"pred": 1.0, "stress_aux": variant.stress_aux_weight},
    )
    return model


def build_contextresid_stressaux_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    feature_index = {feature: idx for idx, feature in enumerate(data.feature_columns)}
    context_indices = [feature_index[feature] for feature in variant.context_gate_columns if feature in feature_index]
    if not context_indices:
        raise ValueError(f"No context gate columns found in feature set for {variant.name}.")

    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    context_x = layers.Lambda(
        lambda value: tf.gather(value, context_indices, axis=-1),
        name="context_feature_slice",
    )(inputs)
    context_state = layers.LSTM(
        max(4, int(variant.context_gate_units)),
        kernel_regularizer=keras.regularizers.l2(1e-5),
        recurrent_regularizer=keras.regularizers.l2(1e-5),
        name="context_resid_lstm",
    )(context_x)
    context_gate = layers.Dense(
        encoded.shape[-1],
        activation="sigmoid",
        bias_initializer=keras.initializers.Constant(2.0),
        name="context_resid_gate",
    )(context_state)
    context_delta = layers.Dense(
        encoded.shape[-1],
        activation="tanh",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="context_resid_delta",
    )(context_state)
    gated_delta = layers.Multiply(name="context_resid_gated_delta")([context_gate, context_delta])
    encoded = layers.Add(name="context_resid_add")([encoded, gated_delta])

    pred = layers.Dense(1, name="pred")(encoded)
    stress = layers.Dense(1, activation="sigmoid", name="stress_aux")(encoded)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "stress_aux": stress})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "stress_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"pred": 1.0, "stress_aux": variant.stress_aux_weight},
    )
    return model


def build_marketaux_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    market_return = layers.Dense(1, name="market_return_aux")(encoded)
    market_abs = layers.Dense(1, activation="softplus", name="market_abs_aux")(encoded)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "market_return_aux": market_return, "market_abs_aux": market_abs})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "market_return_aux": keras.losses.Huber(delta=0.01),
            "market_abs_aux": keras.losses.Huber(delta=0.01),
        },
        loss_weights={
            "pred": 1.0,
            "market_return_aux": variant.market_aux_weight,
            "market_abs_aux": variant.market_aux_weight,
        },
    )
    return model


def build_riskaux_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    risk = layers.Dense(1, activation="sigmoid", name="risk_aux")(encoded)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "risk_aux": risk})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "risk_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"pred": 1.0, "risk_aux": variant.risk_aux_weight},
    )
    return model


def build_riskaux_detached_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    detached = layers.Lambda(lambda value: tf.stop_gradient(value), name="risk_stop_gradient")(encoded)
    risk = layers.Dense(1, activation="sigmoid", name="risk_aux")(detached)
    model = keras.Model(inputs=inputs, outputs={"pred": pred, "risk_aux": risk})
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "risk_aux": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        },
        loss_weights={"pred": 1.0, "risk_aux": variant.risk_aux_weight},
    )
    return model


def build_tailaware_probe_model(
    data: PreparedData,
    args: argparse.Namespace,
    variant: Variant,
) -> keras.Model:
    inputs, encoded = build_lstm_backbone(
        window_size=args.window_size,
        num_features=data.x_train.shape[2],
        lstm_units=parse_lstm_units(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=0.0,
        use_layer_norm=False,
    )
    pred = layers.Dense(1, name="pred")(encoded)
    tail5 = layers.Dense(1, activation="sigmoid", name="tail5_prob")(encoded)
    tail7 = layers.Dense(1, activation="sigmoid", name="tail7_prob")(encoded)
    magnitude = layers.Dense(1, activation="softplus", name="magnitude_aux")(encoded)
    model = keras.Model(
        inputs=inputs,
        outputs={
            "pred": pred,
            "tail5_prob": tail5,
            "tail7_prob": tail7,
            "magnitude_aux": magnitude,
        },
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "pred": build_pred_loss(variant, data.target_scaler, data.local_target_normalizer),
            "tail5_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "tail7_prob": keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            "magnitude_aux": keras.losses.Huber(delta=0.01),
        },
        loss_weights={
            "pred": 1.0,
            "tail5_prob": 0.12,
            "tail7_prob": 0.18,
            "magnitude_aux": 0.20,
        },
    )
    return model


def balanced_binary_weights(labels: np.ndarray, max_weight: float = 8.0) -> np.ndarray:
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    positive_rate = float(np.mean(y)) if len(y) else 0.0
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return np.ones_like(y, dtype=np.float32)
    pos_weight = min(max_weight, 0.5 / positive_rate)
    neg_weight = min(max_weight, 0.5 / (1.0 - positive_rate))
    weights = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)
    return (weights / np.mean(weights)).astype(np.float32)


def tail_targets(y_raw: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    y = np.asarray(y_raw, dtype=np.float32).reshape(-1, 1)
    return {
        "tail5_prob": (np.abs(y) >= args.tail5_threshold).astype(np.float32),
        "tail7_prob": (np.abs(y) >= args.tail7_threshold).astype(np.float32),
        "magnitude_aux": np.log1p(np.abs(y)).astype(np.float32),
    }


def stress_aux_targets(data: PreparedData, variant: Variant) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_values = data.meta_train[variant.stress_aux_column].to_numpy(dtype=np.float32)
    val_values = data.meta_val[variant.stress_aux_column].to_numpy(dtype=np.float32)
    threshold = float(np.nanquantile(train_values, variant.stress_aux_quantile))
    train_label = (train_values >= threshold).astype(np.float32).reshape(-1, 1)
    val_label = (val_values >= threshold).astype(np.float32).reshape(-1, 1)
    weights = balanced_binary_weights(train_label, max_weight=6.0)
    return (
        {"stress_aux": train_label},
        {"stress_aux": val_label},
        {"stress_aux": weights},
    )


def market_aux_targets(data: PreparedData, variant: Variant) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_return = data.meta_train[variant.market_aux_return_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    val_return = data.meta_val[variant.market_aux_return_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    train_abs = np.log1p(data.meta_train[variant.market_aux_abs_column].to_numpy(dtype=np.float32)).reshape(-1, 1)
    val_abs = np.log1p(data.meta_val[variant.market_aux_abs_column].to_numpy(dtype=np.float32)).reshape(-1, 1)
    return (
        {
            "market_return_aux": train_return,
            "market_abs_aux": train_abs,
        },
        {
            "market_return_aux": val_return,
            "market_abs_aux": val_abs,
        },
    )


def risk_aux_targets(data: PreparedData, variant: Variant) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_label = (np.abs(data.y_train_raw).reshape(-1, 1) >= variant.risk_aux_threshold).astype(np.float32)
    val_label = (np.abs(data.y_val_raw).reshape(-1, 1) >= variant.risk_aux_threshold).astype(np.float32)
    return (
        {"risk_aux": train_label},
        {"risk_aux": val_label},
        {"risk_aux": balanced_binary_weights(train_label, max_weight=6.0)},
    )


def apply_training_feature_dropout(x: np.ndarray, feature_columns: tuple[str, ...], variant: Variant, seed: int) -> np.ndarray:
    if variant.input_feature_dropout_rate <= 0.0 or not variant.input_feature_dropout_columns:
        return x
    feature_index = {feature: idx for idx, feature in enumerate(feature_columns)}
    indices = [feature_index[feature] for feature in variant.input_feature_dropout_columns if feature in feature_index]
    if not indices:
        return x
    rate = min(max(float(variant.input_feature_dropout_rate), 0.0), 0.95)
    rng = np.random.default_rng(seed + 7307)
    x_aug = np.array(x, copy=True)
    mask = rng.random((x_aug.shape[0], x_aug.shape[1], len(indices))) < rate
    x_aug[:, :, indices] = np.where(mask, 0.0, x_aug[:, :, indices])
    return x_aug.astype(np.float32)


def fit_variant(
    variant: Variant,
    data: PreparedData,
    args: argparse.Namespace,
) -> tuple[keras.Model, keras.callbacks.History]:
    set_global_seed(args.seed)
    x_train = apply_training_feature_dropout(data.x_train, data.feature_columns, variant, args.seed)
    if variant.model_type == "plain":
        model = build_plain_probe_model(data, args, variant)
        y_train: np.ndarray | dict[str, np.ndarray] = data.y_train_model
        y_val: np.ndarray | dict[str, np.ndarray] = data.y_val_model
        sample_weight = None
    elif variant.model_type == "stressaux":
        model = build_stressaux_probe_model(data, args, variant)
        train_stress, val_stress, stress_weights = stress_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_stress}
        y_val = {"pred": data.y_val_model, **val_stress}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            **stress_weights,
        }
    elif variant.model_type == "contextgate_stressaux":
        model = build_contextgate_stressaux_probe_model(data, args, variant)
        train_stress, val_stress, stress_weights = stress_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_stress}
        y_val = {"pred": data.y_val_model, **val_stress}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            **stress_weights,
        }
    elif variant.model_type == "contextresid_stressaux":
        model = build_contextresid_stressaux_probe_model(data, args, variant)
        train_stress, val_stress, stress_weights = stress_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_stress}
        y_val = {"pred": data.y_val_model, **val_stress}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            **stress_weights,
        }
    elif variant.model_type == "marketaux":
        model = build_marketaux_probe_model(data, args, variant)
        train_market, val_market = market_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_market}
        y_val = {"pred": data.y_val_model, **val_market}
        sample_weight = None
    elif variant.model_type == "riskaux":
        model = build_riskaux_probe_model(data, args, variant)
        train_risk, val_risk, risk_weights = risk_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_risk}
        y_val = {"pred": data.y_val_model, **val_risk}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            **risk_weights,
        }
    elif variant.model_type == "riskaux_detached":
        model = build_riskaux_detached_probe_model(data, args, variant)
        train_risk, val_risk, risk_weights = risk_aux_targets(data, variant)
        y_train = {"pred": data.y_train_model, **train_risk}
        y_val = {"pred": data.y_val_model, **val_risk}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            **risk_weights,
        }
    elif variant.model_type == "tailaware":
        model = build_tailaware_probe_model(data, args, variant)
        train_tail = tail_targets(data.y_train_raw, args)
        val_tail = tail_targets(data.y_val_raw, args)
        y_train = {"pred": data.y_train_model, **train_tail}
        y_val = {"pred": data.y_val_model, **val_tail}
        sample_weight = {
            "pred": np.ones(len(data.y_train_raw), dtype=np.float32),
            "tail5_prob": balanced_binary_weights(train_tail["tail5_prob"]),
            "tail7_prob": balanced_binary_weights(train_tail["tail7_prob"]),
            "magnitude_aux": np.ones(len(data.y_train_raw), dtype=np.float32),
        }
    else:
        raise ValueError(f"Unsupported model type: {variant.model_type}")
    monitor = "val_pred_loss" if variant.model_type == "riskaux_detached" else "val_loss"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        validation_data=(data.x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history


def predict_raw_return(
    model: keras.Model,
    data: PreparedData,
    x: np.ndarray,
    scale_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output = model.predict(x, verbose=0)
    if isinstance(output, dict):
        pred_scaled = np.asarray(output["pred"], dtype=np.float32).reshape(-1)
        tail5 = np.asarray(output.get("tail5_prob", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
        tail7 = np.asarray(output.get("tail7_prob", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
        risk = np.asarray(output.get("risk_aux", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
    elif isinstance(output, list):
        output_map = {name: value for name, value in zip(model.output_names, output)}
        pred_source = output_map.get("pred", output[0])
        pred_scaled = np.asarray(pred_source, dtype=np.float32).reshape(-1)
        tail5 = np.asarray(output_map.get("tail5_prob", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
        tail7 = np.asarray(output_map.get("tail7_prob", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
        risk = np.asarray(output_map.get("risk_aux", np.full((len(pred_scaled), 1), np.nan)), dtype=np.float32).reshape(-1)
    else:
        pred_scaled = np.asarray(output, dtype=np.float32).reshape(-1)
        tail5 = np.full(len(pred_scaled), np.nan)
        tail7 = np.full(len(pred_scaled), np.nan)
        risk = np.full(len(pred_scaled), np.nan)
    pred_local = inverse_target_scaler_values(pred_scaled, data.target_scaler)
    pred_raw = inverse_local_target_normalizer(
        pred_local,
        scale_values,
        data.local_target_normalizer,
    ).reshape(-1)
    return pred_raw.astype(np.float32), tail5.astype(np.float32), tail7.astype(np.float32), risk.astype(np.float32)


def prediction_frame(
    variant: Variant,
    model: keras.Model,
    data: PreparedData,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split, x, y, meta, scale_values in [
        ("train", data.x_train, data.y_train_raw, data.meta_train, data.train_scale_values),
        ("val", data.x_val, data.y_val_raw, data.meta_val, data.val_scale_values),
    ]:
        pred, tail5, tail7, risk = predict_raw_return(model, data, x, scale_values)
        part = meta.loc[:, ["code", "Date"]].copy()
        part["split"] = split
        part["variant"] = variant.name
        part["prediction"] = pred
        part["actual"] = y
        part["tail5_probability"] = tail5
        part["tail7_probability"] = tail7
        part["risk_probability"] = risk
        part["error"] = part["actual"] - part["prediction"]
        part["abs_error"] = part["error"].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def summarize_predictions(frame: pd.DataFrame, spike_thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, split), group in frame.groupby(["variant", "split"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        abs_error = np.abs(actual - pred)
        daily = (
            group.groupby("Date", sort=True)
            .agg(daily_q90_abs_error=("abs_error", lambda values: float(np.quantile(values, 0.90))))
            .reset_index()
        )
        row: dict[str, object] = {
            "variant": variant,
            "split": split,
            "n_obs": int(len(group)),
            "n_days": int(daily.shape[0]),
            "rel_score": rel_score(actual, pred),
            "median_abs_error": float(np.quantile(abs_error, 0.50)),
            "q90_abs_error": float(np.quantile(abs_error, 0.90)),
            "daily_q90_abs_error_median": float(daily["daily_q90_abs_error"].median()),
            "daily_q90_abs_error_q90": float(daily["daily_q90_abs_error"].quantile(0.90)),
            "daily_q90_abs_error_max": float(daily["daily_q90_abs_error"].max()),
            "actual_abs_q90": float(np.quantile(np.abs(actual), 0.90)),
            "prediction_abs_q90": float(np.quantile(np.abs(pred), 0.90)),
            "prediction_actual_abs_q90_ratio": float(
                np.quantile(np.abs(pred), 0.90) / max(np.quantile(np.abs(actual), 0.90), 1e-8)
            ),
            "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
        }
        for threshold in spike_thresholds:
            key = int(round(threshold * 100))
            row[f"spike_days_ge_{key}pct"] = int(daily["daily_q90_abs_error"].ge(threshold).sum())
            row[f"spike_rate_ge_{key}pct"] = float(daily["daily_q90_abs_error"].ge(threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    display = summary[summary["split"].eq("val")].copy()
    metric_cols = [
        "rel_score",
        "median_abs_error",
        "q90_abs_error",
        "daily_q90_abs_error_median",
        "daily_q90_abs_error_q90",
        "daily_q90_abs_error_max",
        "prediction_actual_abs_q90_ratio",
        "directional_accuracy",
        "spike_rate_ge_5pct",
        "spike_rate_ge_7pct",
        "spike_rate_ge_8pct",
    ]
    for column in metric_cols:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.5f}" if np.isfinite(float(value)) else "n/a")
    lines = [
        "# Tail-Aware LSTM Probe",
        "",
        "Scope: VN train/validation only. Holdout/test is not used.",
        "",
        "Goal: test whether input normalization and a tail-aware auxiliary head can improve all-days forecasting metrics, not trading filters.",
        "",
        f"- seed: `{args.seed}`",
        f"- window_size: `{args.window_size}`",
        f"- lstm_units: `{args.lstm_units}`",
        f"- train_end_date: `{args.train_end_date}`",
        f"- val_end_date: `{args.val_end_date}`",
        "",
        "## Validation Results",
        "",
        display.to_markdown(index=False),
        "",
        "## Read",
        "",
        "- `plain_global_rel` mirrors the current broad portable setup most closely.",
        "- `plain_global_instance_rel` tests per-window instance normalization.",
        "- `plain_multimarket_rel` tests strict-past rolling/cross-sectional input normalization.",
        "- `tailaware_multimarket_weighted` keeps LSTM as the backbone and adds tail/magnitude auxiliary supervision.",
        "",
        "A variant is useful only if it improves `rel_score` while also reducing daily q90 error p90/max and spike counts.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    gold_dir = args.gold_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = parse_csv(args.feature_columns) if args.feature_columns else load_base_feature_columns()
    selected = parse_csv(args.variants)
    variant_map = all_variants()
    unknown = sorted(set(selected).difference(variant_map))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    variants = [variant_map[name] for name in selected]
    spike_thresholds = tuple(float(item) for item in parse_csv(args.spike_thresholds))

    raw = load_frame(args.data, feature_columns, args.target_column, args.target_normalizer)
    feature_columns = tuple(raw.attrs.get("feature_columns", feature_columns))
    all_predictions: list[pd.DataFrame] = []
    manifest: dict[str, object] = {
        "data": str(args.data),
        "feature_columns": list(feature_columns),
        "variants": [variant.name for variant in variants],
        "holdout_test_used": False,
        "references": [
            "Deep Imbalanced Regression / LDS-FDS",
            "DeepAR probabilistic RNN forecasting",
            "Tail-aware auxiliary supervision",
            "Target scale normalization with lagged market tail volatility",
        ],
    }
    for variant in variants:
        print(f"Run variant: {variant.name}")
        data = prepare_data(raw, feature_columns, variant, args)
        model, history = fit_variant(variant, data, args)
        model.save(output_dir / f"{variant.name}.keras")
        pd.DataFrame(history.history).to_csv(output_dir / f"history_{variant.name}.csv", index=False)
        pred = prediction_frame(variant, model, data)
        pred.to_csv(output_dir / f"predictions_{variant.name}.csv", index=False)
        all_predictions.append(pred)
        manifest[f"{variant.name}_input_features"] = len(data.feature_columns)
        manifest[f"{variant.name}_target_scale_mode"] = variant.target_scale_mode
        manifest[f"{variant.name}_target_scale_floor_quantile"] = variant.target_scale_floor_quantile
        manifest[f"{variant.name}_input_feature_dropout_rate"] = variant.input_feature_dropout_rate
        manifest[f"{variant.name}_input_feature_dropout_columns"] = list(variant.input_feature_dropout_columns)
        manifest[f"{variant.name}_context_gate_columns"] = list(variant.context_gate_columns)
        manifest[f"{variant.name}_context_gate_units"] = variant.context_gate_units

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = summarize_predictions(predictions, spike_thresholds)
    predictions.to_csv(output_dir / "predictions_all_variants.csv", index=False)
    summary.to_csv(output_dir / "tail_aware_lstm_probe_summary.csv", index=False)
    write_summary(output_dir, summary, args)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Mirror compact report artifacts to gold.
    summary.to_csv(gold_dir / "tail_aware_lstm_probe_summary.csv", index=False)
    (gold_dir / "summary.md").write_text((output_dir / "summary.md").read_text(encoding="utf-8"), encoding="utf-8")
    (gold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "gold_dir": str(gold_dir), "variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
