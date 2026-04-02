from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fk_lstm_classifier"


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    markets: tuple[str, ...] = ("VN",)
    vn_data_profile: str | None = None
    lookback: int = 240
    train_ratio: float = 0.8
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 10
    lstm_units: int = 25
    dropout: float = 0.16
    learning_rate: float = 1e-3
    top_k: int = 10
    seed: int = 42
    model_type: str = "attention"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_cross_sectional_count: int = 20
    run_name: str | None = None
    evaluation_mode: str = "holdout"
    min_train_days: int = 756
    validation_days: int = 126
    test_days: int = 63
    step_days: int = 63
    window_scheme: str = "expanding"
    transaction_cost_bps: float = 0.0
    forward_horizon_days: int | None = None
    allow_short: bool | None = None
    respect_market_rules: bool = True
    buy_cost_bps: float | None = None
    sell_cost_bps: float | None = None
    sell_tax_bps: float | None = None
    min_daily_value_traded: float | None = None
    min_adv20_value_traded: float | None = None
    max_position_adv_fraction: float | None = None
    portfolio_notional: float | None = None
    block_limit_up_entry: bool | None = None
    exclude_hard_issues: bool | None = None


def _coerce_path(value: Any, *, base_dir: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _coerce_markets(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    markets = tuple(str(item).strip().upper() for item in items if str(item).strip())
    if not markets:
        raise ValueError("At least one market must be configured.")
    return markets


def _normalize_config_values(raw: dict[str, Any], *, base_dir: Path = ROOT) -> dict[str, Any]:
    normalized = dict(raw)
    if "data_dir" in normalized:
        normalized["data_dir"] = _coerce_path(normalized["data_dir"], base_dir=base_dir)
    if "output_dir" in normalized:
        normalized["output_dir"] = _coerce_path(normalized["output_dir"], base_dir=base_dir)
    if "markets" in normalized:
        normalized["markets"] = _coerce_markets(normalized["markets"])
    return normalized


def load_config_file(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    return _normalize_config_values(raw, base_dir=ROOT)


def _validate_payload(payload: dict[str, Any]) -> None:
    if not 0.0 < float(payload["train_ratio"]) < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if payload["evaluation_mode"] not in {"holdout", "walk_forward"}:
        raise ValueError("evaluation_mode must be 'holdout' or 'walk_forward'.")
    if payload["window_scheme"] not in {"expanding", "rolling"}:
        raise ValueError("window_scheme must be 'expanding' or 'rolling'.")
    if int(payload["lookback"]) <= 1:
        raise ValueError("lookback must be greater than 1.")
    if int(payload["top_k"]) <= 0:
        raise ValueError("top_k must be positive.")
    if float(payload["transaction_cost_bps"]) < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if payload["forward_horizon_days"] is not None and int(payload["forward_horizon_days"]) <= 0:
        raise ValueError("forward_horizon_days must be positive when provided.")
    for key in ("buy_cost_bps", "sell_cost_bps", "sell_tax_bps"):
        if payload[key] is not None and float(payload[key]) < 0:
            raise ValueError(f"{key} cannot be negative.")
    for key in ("min_daily_value_traded", "min_adv20_value_traded", "portfolio_notional"):
        if payload[key] is not None and float(payload[key]) <= 0:
            raise ValueError(f"{key} must be positive when provided.")
    if payload["max_position_adv_fraction"] is not None:
        max_participation = float(payload["max_position_adv_fraction"])
        if not 0.0 < max_participation <= 1.0:
            raise ValueError("max_position_adv_fraction must be between 0 and 1.")


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    payload = config_to_dict(ExperimentConfig())
    payload.update(load_config_file(config_path))
    payload["markets"] = _coerce_markets(payload["markets"])
    _validate_payload(payload)
    return ExperimentConfig(**payload)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["markets"] = list(config.markets)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Fischer-Krauss style binary LSTM classifier on daily stock returns "
            "with optional temporal attention."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file under configs/fk_lstm or any absolute path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing market folders such as data/VN and data/US.",
    )
    parser.add_argument(
        "--vn-data-profile",
        default=None,
        help="Optional VN curated profile to load instead of raw per-ticker files, for example balanced/model_strict/trust_max.",
    )
    parser.add_argument(
        "--markets",
        default=None,
        help="Comma-separated market folders to load, for example VN,US.",
    )
    parser.add_argument("--lookback", type=int, default=None, help="Sequence length in trading days.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Chronological fraction of eligible dates reserved for the training split.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--lstm-units", type=int, default=None, help="Number of LSTM hidden units.")
    parser.add_argument("--dropout", type=float, default=None, help="Input dropout inside the LSTM layer.")
    parser.add_argument("--learning-rate", type=float, default=None, help="RMSprop learning rate.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "attention"],
        default=None,
        help="Baseline reproduces the paper-style LSTM, attention adds temporal attention on top.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Top and bottom k names for long-short evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save predictions, backtest results, and fit history.",
    )
    parser.add_argument(
        "--min-cross-sectional-count",
        type=int,
        default=None,
        help="Minimum number of stocks required on a date to keep that cross-section.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Defaults to <markets>_<model_type>.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["holdout", "walk_forward"],
        default=None,
        help="Holdout reproduces the earlier single split, walk_forward evaluates rolling test windows.",
    )
    parser.add_argument("--min-train-days", type=int, default=None, help="Minimum train window for walk-forward runs.")
    parser.add_argument("--validation-days", type=int, default=None, help="Validation window per walk-forward fold.")
    parser.add_argument("--test-days", type=int, default=None, help="Test window per walk-forward fold.")
    parser.add_argument("--step-days", type=int, default=None, help="Fold advance in trading days.")
    parser.add_argument(
        "--window-scheme",
        choices=["expanding", "rolling"],
        default=None,
        help="Walk-forward train window scheme.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=None,
        help="One-way transaction cost in basis points applied to portfolio turnover.",
    )
    parser.add_argument("--buy-cost-bps", type=float, default=None, help="Buy-side cost in basis points.")
    parser.add_argument("--sell-cost-bps", type=float, default=None, help="Sell-side commission cost in basis points.")
    parser.add_argument("--sell-tax-bps", type=float, default=None, help="Sell-side transfer tax in basis points.")
    parser.add_argument(
        "--min-daily-value-traded",
        type=float,
        default=None,
        help="Minimum same-day traded value required for a name to be eligible.",
    )
    parser.add_argument(
        "--min-adv20-value-traded",
        type=float,
        default=None,
        help="Minimum 20-day average traded value required for a name to be eligible.",
    )
    parser.add_argument(
        "--max-position-adv-fraction",
        type=float,
        default=None,
        help="Maximum fraction of ADV20 that a single position may represent.",
    )
    parser.add_argument(
        "--portfolio-notional",
        type=float,
        default=None,
        help="Portfolio capital used when translating weights into ADV-based position caps.",
    )
    parser.add_argument(
        "--forward-horizon-days",
        type=int,
        default=None,
        help="Optional forward return horizon in trading days. If omitted, market rules infer it.",
    )
    parser.set_defaults(allow_short=None, respect_market_rules=None, block_limit_up_entry=None, exclude_hard_issues=None)
    parser.add_argument(
        "--allow-short",
        dest="allow_short",
        action="store_true",
        help="Allow a short leg in strategy evaluation.",
    )
    parser.add_argument(
        "--long-only",
        dest="allow_short",
        action="store_false",
        help="Disable the short leg and evaluate only the long portfolio.",
    )
    parser.add_argument(
        "--disable-market-rules",
        dest="respect_market_rules",
        action="store_false",
        help="Disable market-specific defaults such as VN settlement and long-only handling.",
    )
    parser.add_argument(
        "--block-limit-up-entry",
        dest="block_limit_up_entry",
        action="store_true",
        help="Exclude names that appear limit-up at the entry date.",
    )
    parser.add_argument(
        "--allow-limit-up-entry",
        dest="block_limit_up_entry",
        action="store_false",
        help="Allow entries even when the name appears limit-up on daily bars.",
    )
    parser.add_argument(
        "--exclude-hard-issues",
        dest="exclude_hard_issues",
        action="store_true",
        help="Exclude rows tagged with hard quality issues when that metadata is available.",
    )
    parser.add_argument(
        "--include-hard-issues",
        dest="exclude_hard_issues",
        action="store_false",
        help="Keep rows even when they carry hard issue flags.",
    )
    return parser.parse_args()


def namespace_to_config(args: argparse.Namespace) -> ExperimentConfig:
    base_payload = config_to_dict(ExperimentConfig())
    if args.config is not None:
        base_payload.update(load_config_file(args.config))

    overrides: dict[str, Any] = {}
    for field in fields(ExperimentConfig):
        if not hasattr(args, field.name):
            continue
        value = getattr(args, field.name)
        if value is not None:
            overrides[field.name] = value

    normalized_overrides = _normalize_config_values(overrides, base_dir=ROOT)
    payload = {**base_payload, **normalized_overrides}
    payload["markets"] = _coerce_markets(payload["markets"])
    _validate_payload(payload)
    return ExperimentConfig(**payload)
