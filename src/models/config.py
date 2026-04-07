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
MACRO_FEATURE_COLUMNS = ("vingroup_momentum", "vnindex_return")
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
    batch_size: int = _cfg["hyperparameters"]["batch_size"]
    epochs: int = _cfg["hyperparameters"]["epochs"]
    patience: int = _cfg["hyperparameters"]["patience"]

    def __post_init__(self) -> None:
        if self.target_column is None:
            self.target_column = resolve_target_column(self.target_mode)
        if isinstance(self.lstm_units, (list, tuple)):
            self.lstm_units = [int(item) for item in self.lstm_units]


def get_config(target_mode: str = "return") -> LSTMConfig:
    config = LSTMConfig(target_mode=target_mode)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plot_dir.mkdir(parents=True, exist_ok=True)
    return config
