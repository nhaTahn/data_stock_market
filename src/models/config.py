from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "lstm_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = json.load(f)

DEFAULT_DATA_PATH = ROOT / _cfg["paths"]["default_data_path_rel"]
DEFAULT_OUTPUT_DIR = ROOT / _cfg["paths"]["default_output_dir_rel"]
DEFAULT_PLOT_DIR = ROOT / _cfg["paths"]["default_plot_dir_rel"]

TRAIN_END_DATE = _cfg["dates"]["train_end_date"]
VAL_END_DATE = _cfg["dates"]["val_end_date"]

TARGET_COLUMNS = _cfg["targets"]
MACRO_FEATURE_COLUMNS = ("vingroup_momentum", "vnindex_return", "a_d_ratio")
DEFAULT_FEATURE_COLUMNS = tuple(dict.fromkeys([*_cfg["default_features"], *MACRO_FEATURE_COLUMNS]))
SECTOR_FEATURES_MAP = {
    key: tuple(dict.fromkeys([*values, *MACRO_FEATURE_COLUMNS]))
    for key, values in _cfg.get("sector_features", {}).items()
}
ALL_FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        [
            *DEFAULT_FEATURE_COLUMNS,
            *(feature for values in SECTOR_FEATURES_MAP.values() for feature in values),
        ]
    )
)


def resolve_target_column(target_mode: str) -> str:
    if target_mode not in TARGET_COLUMNS:
        raise ValueError(f"target_mode must be one of {list(TARGET_COLUMNS)}")
    return TARGET_COLUMNS[target_mode]


@dataclass
class LSTMConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    plot_dir: Path = DEFAULT_PLOT_DIR
    train_end_date: str = TRAIN_END_DATE
    val_end_date: str = VAL_END_DATE
    target_mode: str = "return"
    target_column: str | None = None
    feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    sector_features_map: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(SECTOR_FEATURES_MAP))
    window_size: int = _cfg["hyperparameters"]["window_size"]
    lstm_units: int | list[int] = _cfg["hyperparameters"]["lstm_units"]
    dropout: float = _cfg["hyperparameters"]["dropout"]
    lr: float = _cfg["hyperparameters"]["lr"]
    loss: str = _cfg["hyperparameters"]["loss"]
    huber_delta: float = _cfg["hyperparameters"]["huber_delta"]
    rel_score_large_move_quantile: float = _cfg["hyperparameters"].get("rel_score_large_move_quantile", 0.8)
    rel_score_directional_penalty: float = _cfg["hyperparameters"].get("rel_score_directional_penalty", 0.6)
    rel_score_confidence_penalty: float = _cfg["hyperparameters"].get("rel_score_confidence_penalty", 0.35)
    rel_score_confidence_ratio: float = _cfg["hyperparameters"].get("rel_score_confidence_ratio", 0.25)
    rel_score_weighted_high_quantile: float = _cfg["hyperparameters"].get("rel_score_weighted_high_quantile", 0.8)
    rel_score_weighted_high_weight: float = _cfg["hyperparameters"].get("rel_score_weighted_high_weight", 3.0)
    rel_score_weighted_base_weight: float = _cfg["hyperparameters"].get("rel_score_weighted_base_weight", 1.0)
    batch_size: int = _cfg["hyperparameters"]["batch_size"]
    epochs: int = _cfg["hyperparameters"]["epochs"]
    patience: int = _cfg["hyperparameters"]["patience"]
    target_normalizer: str | None = _cfg["hyperparameters"].get("target_normalizer")
    sequence_normalization: str = _cfg["hyperparameters"].get("sequence_normalization", "none")
    feature_phase: str = _cfg["hyperparameters"].get("feature_phase", "none")
    lstm_seeds: list[int] = field(default_factory=lambda: list(_cfg["hyperparameters"].get("lstm_seeds", [42])))
    signmag_signed_loss_weight: float = _cfg["hyperparameters"].get("signmag_signed_loss_weight", 1.5)
    signmag_sign_loss_weight: float = _cfg["hyperparameters"].get("signmag_sign_loss_weight", 0.15)
    signmag_magnitude_loss_weight: float = _cfg["hyperparameters"].get("signmag_magnitude_loss_weight", 0.35)
    signmag_log_magnitude: bool = _cfg["hyperparameters"].get("signmag_log_magnitude", True)
    sample_weight_mode: str = _cfg["hyperparameters"].get("sample_weight_mode", "none")
    sample_weight_strength: float = _cfg["hyperparameters"].get("sample_weight_strength", 1.5)
    sample_weight_quantile: float = _cfg["hyperparameters"].get("sample_weight_quantile", 0.75)
    sample_weight_clip: float = _cfg["hyperparameters"].get("sample_weight_clip", 3.0)
    attention_enabled: bool = _cfg["hyperparameters"].get("attention_enabled", False)
    attention_heads: int = _cfg["hyperparameters"].get("attention_heads", 2)
    attention_key_dim: int = _cfg["hyperparameters"].get("attention_key_dim", 16)
    signal_enabled: bool = _cfg["hyperparameters"].get("signal_enabled", False)
    signal_patch_length: int = _cfg["hyperparameters"].get("signal_patch_length", 5)
    signal_patch_stride: int = _cfg["hyperparameters"].get("signal_patch_stride", 3)
    signal_patch_dim: int = _cfg["hyperparameters"].get("signal_patch_dim", 16)
    signal_future_steps: int = _cfg["hyperparameters"].get("signal_future_steps", 1)
    signal_attention_heads: int = _cfg["hyperparameters"].get("signal_attention_heads", 2)
    signal_attention_key_dim: int = _cfg["hyperparameters"].get("signal_attention_key_dim", 16)
    signal_attention_ff_dim: int | None = _cfg["hyperparameters"].get("signal_attention_ff_dim")
    pcie_lite_enabled: bool = _cfg["hyperparameters"].get("pcie_lite_enabled", False)
    pcie_lite_base_columns: tuple[str, ...] = tuple(_cfg["hyperparameters"].get("pcie_lite_base_columns", ["open", "high", "low", "close", "volume"]))
    pcie_lite_context_columns: tuple[str, ...] = tuple(
        _cfg["hyperparameters"].get("pcie_lite_context_columns", ["vnindex_return", "vingroup_momentum", "a_d_ratio", "day_of_week"])
    )
    pcie_lite_patch_length: int = _cfg["hyperparameters"].get("pcie_lite_patch_length", 5)
    pcie_lite_patch_stride: int = _cfg["hyperparameters"].get("pcie_lite_patch_stride", 5)
    pcie_lite_patch_dim: int = _cfg["hyperparameters"].get("pcie_lite_patch_dim", 16)
    pcie_lite_future_steps: int = _cfg["hyperparameters"].get("pcie_lite_future_steps", 3)
    quantile_enabled: bool = _cfg["hyperparameters"].get("quantile_enabled", False)
    event_enabled: bool = _cfg["hyperparameters"].get("event_enabled", False)
    event_threshold: float = _cfg["hyperparameters"].get("event_threshold", 0.75)
    event_signed_loss_weight: float = _cfg["hyperparameters"].get("event_signed_loss_weight", 2.0)
    event_prob_loss_weight: float = _cfg["hyperparameters"].get("event_prob_loss_weight", 0.4)
    event_sign_loss_weight: float = _cfg["hyperparameters"].get("event_sign_loss_weight", 0.1)
    event_magnitude_loss_weight: float = _cfg["hyperparameters"].get("event_magnitude_loss_weight", 0.3)
    event_log_magnitude: bool = _cfg["hyperparameters"].get("event_log_magnitude", True)
    fk_benchmark_enabled: bool = _cfg["hyperparameters"].get("fk_benchmark_enabled", False)
    fk_window_size: int = _cfg["hyperparameters"].get("fk_window_size", 240)
    fk_hidden_units: int = _cfg["hyperparameters"].get("fk_hidden_units", 25)
    fk_dropout: float = _cfg["hyperparameters"].get("fk_dropout", 0.16)
    fk_learning_rate: float = _cfg["hyperparameters"].get("fk_learning_rate", 1e-3)
    fk_batch_size: int = _cfg["hyperparameters"].get("fk_batch_size", 32)
    fk_epochs: int = _cfg["hyperparameters"].get("fk_epochs", 1000)
    fk_patience: int = _cfg["hyperparameters"].get("fk_patience", 10)
    fk_train_fraction: float = _cfg["hyperparameters"].get("fk_train_fraction", 0.8)
    fk_top_k: int = _cfg["hyperparameters"].get("fk_top_k", 10)

    def __post_init__(self) -> None:
        if self.target_column is None:
            self.target_column = resolve_target_column(self.target_mode)
        if isinstance(self.lstm_units, (list, tuple)):
            self.lstm_units = [int(item) for item in self.lstm_units]
        self.lstm_seeds = [int(seed) for seed in self.lstm_seeds]


def get_config(target_mode: str = "return") -> LSTMConfig:
    config = LSTMConfig(target_mode=target_mode)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    return config
