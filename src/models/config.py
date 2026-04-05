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
FEATURE_COLUMNS = tuple(_cfg["features"])


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
    window_size: int = _cfg["hyperparameters"]["window_size"]
    lstm_units: int = _cfg["hyperparameters"]["lstm_units"]
    dropout: float = _cfg["hyperparameters"]["dropout"]
    lr: float = _cfg["hyperparameters"]["lr"]
    loss: str = _cfg["hyperparameters"]["loss"]
    huber_delta: float = _cfg["hyperparameters"]["huber_delta"]
    batch_size: int = _cfg["hyperparameters"]["batch_size"]
    epochs: int = _cfg["hyperparameters"]["epochs"]
    patience: int = _cfg["hyperparameters"]["patience"]

    def __post_init__(self) -> None:
        if self.target_column is None:
            self.target_column = resolve_target_column(self.target_mode)


def get_config(target_mode: str = "return") -> LSTMConfig:
    config = LSTMConfig(target_mode=target_mode)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    return config
