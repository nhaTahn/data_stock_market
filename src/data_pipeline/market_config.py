from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_START_DATE = "2020-01-01"


@dataclass(frozen=True)
class CleanConfig:
    market: str
    data_dir: Path
    output_dir: Path
    output_prefix: str
    train_start_date: str = TRAIN_START_DATE
    min_coverage: float = 0.95
    recent_active_tolerance_days: int = 30
    max_close_return_abs: float = 0.10
    max_adjust_return_abs: float = 0.16
    drop_imputed_value_match: bool = True
    drop_neighbors_around_events: bool = True


def get_market_config(market: str, train_start_date: str = TRAIN_START_DATE) -> CleanConfig:
    market = market.upper()
    if market == "VN":
        return CleanConfig(
            market="VN",
            data_dir=ROOT / "data" / "VN",
            output_dir=ROOT / "data" / "assets" / "data_info_vn" / "history",
            output_prefix="vn",
            train_start_date=train_start_date,
            min_coverage=0.95,
            recent_active_tolerance_days=30,
            max_close_return_abs=0.075,
            max_adjust_return_abs=0.155,
            drop_imputed_value_match=True,
            drop_neighbors_around_events=True,
        )
    if market == "US":
        return CleanConfig(
            market="US",
            data_dir=ROOT / "data" / "US",
            output_dir=ROOT / "data" / "assets" / "data_info_us" / "history",
            output_prefix="us",
            train_start_date=train_start_date,
            min_coverage=0.97,
            recent_active_tolerance_days=10,
            max_close_return_abs=0.20,
            max_adjust_return_abs=0.20,
            drop_imputed_value_match=False,
            drop_neighbors_around_events=False,
        )
    if market == "JP":
        return CleanConfig(
            market="JP",
            data_dir=ROOT / "data" / "JP",
            output_dir=ROOT / "data" / "assets" / "data_info_jp" / "history",
            output_prefix="jp",
            train_start_date=train_start_date,
            min_coverage=0.97,
            recent_active_tolerance_days=15,
            max_close_return_abs=0.15,
            max_adjust_return_abs=0.15,
            drop_imputed_value_match=False,
            drop_neighbors_around_events=False,
        )
    raise ValueError(f"Unsupported market: {market}")
