from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "markets.json"


@dataclass(frozen=True)
class CleanConfig:
    market: str
    data_dir: Path
    output_dir: Path
    output_prefix: str
    train_start_date: str
    min_coverage: float
    recent_active_tolerance_days: int
    drop_imputed_value_match: bool
    drop_neighbors_around_events: bool
    max_close_return_abs: float | None = None
    max_adjust_return_abs: float | None = None
    exchange_limits: dict | None = None


def get_market_config(market: str, train_start_date: str | None = None) -> CleanConfig:
    market = market.upper()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if market not in config_data:
        raise ValueError(f"Unsupported market: {market}")
    
    m_cfg = config_data[market]
    start_date = train_start_date if train_start_date else m_cfg.get("train_start_date", "2020-01-01")
    
    return CleanConfig(
        market=market,
        data_dir=ROOT / m_cfg["data_dir_rel"],
        output_dir=ROOT / m_cfg["output_dir_rel"],
        output_prefix=m_cfg["output_prefix"],
        train_start_date=start_date,
        min_coverage=m_cfg["min_coverage"],
        recent_active_tolerance_days=m_cfg["recent_active_tolerance_days"],
        drop_imputed_value_match=m_cfg["drop_imputed_value_match"],
        drop_neighbors_around_events=m_cfg["drop_neighbors_around_events"],
        max_close_return_abs=m_cfg.get("max_close_return_abs"),
        max_adjust_return_abs=m_cfg.get("max_adjust_return_abs"),
        exchange_limits=m_cfg.get("exchange_limits"),
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = json.load(f)
TRAIN_START_DATE = _cfg.get("VN", {}).get("train_start_date", "2020-01-01")
