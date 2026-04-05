from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "assets" / "data_info_vn" / "history" / "vn_gold_recommended.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "assets" / "data_info_vn" / "history" / "training_runs"
DEFAULT_PLOT_DIR = ROOT / "artifacts" / "plots"

TRAIN_END_DATE = "2023-12-31"
VAL_END_DATE = "2024-12-31"

TARGET_COLUMNS = {
    "price": "target_next_price",
    "growth": "target_next_growth_pct",
    "return": "target_next_return",
    "return_3d": "target_next_3d_return",
    "return_5d": "target_next_5d_return",
}

FEATURE_COLUMNS = (
    "volume_ratio_20",
    "volume_zscore_20",
    "intraday_return",
    "gap_open",
    "close_position",
    "upper_shadow",
    "lower_shadow",
    "momentum_5",
    "momentum_20",
    "atr_gap",
    "volatility_20",
    "ma_20_gap",
    "ma_200_gap",
    "above_ma_200",
    "rolling_max_20_gap",
    "bb_width",
    "bb_position",
    "bb_zscore",
    "vwap_gap",
    "obv_change",
    "macd",
    "macd_signal",
    "macd_hist",
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
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    window_size: int = 20
    lstm_units: int = 64
    dropout: float = 0.2
    lr: float = 1e-3
    loss: str = "mse"
    huber_delta: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    patience: int = 3

    def __post_init__(self) -> None:
        if self.target_column is None:
            self.target_column = resolve_target_column(self.target_mode)


def get_config(target_mode: str = "return") -> LSTMConfig:
    config = LSTMConfig(target_mode=target_mode)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    return config
