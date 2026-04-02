from __future__ import annotations

from dataclasses import asdict, dataclass

from fk_lstm_classifier.config import ExperimentConfig


@dataclass(frozen=True)
class MarketRuntimeSettings:
    market_scope: str
    forward_horizon_days: int
    allow_short: bool
    strategy_mode: str
    vn_data_profile: str | None
    settlement_cycle: str
    description: str
    buy_cost_bps: float
    sell_cost_bps: float
    sell_tax_bps: float
    min_daily_value_traded: float | None
    min_adv20_value_traded: float | None
    max_position_adv_fraction: float | None
    portfolio_notional: float | None
    block_limit_up_entry: bool
    exclude_hard_issues: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_DEFAULT_RULES = MarketRuntimeSettings(
    market_scope="generic",
    forward_horizon_days=1,
    allow_short=True,
    strategy_mode="long_short",
    vn_data_profile=None,
    settlement_cycle="T+1 stylized close-to-close approximation",
    description="Generic research default matching the original next-day ranking setup.",
    buy_cost_bps=0.0,
    sell_cost_bps=0.0,
    sell_tax_bps=0.0,
    min_daily_value_traded=None,
    min_adv20_value_traded=None,
    max_position_adv_fraction=None,
    portfolio_notional=None,
    block_limit_up_entry=False,
    exclude_hard_issues=False,
)

_VN_RULES = MarketRuntimeSettings(
    market_scope="VN",
    forward_horizon_days=2,
    allow_short=False,
    strategy_mode="long_only",
    vn_data_profile="balanced",
    settlement_cycle="T+2.5, approximated with sellability at T+2 close on daily bars",
    description=(
        "Vietnam cash equities are modeled as long-only by default. The return horizon is extended "
        "to two trading days to approximate T+2.5 settlement using daily close data."
    ),
    buy_cost_bps=0.0,
    sell_cost_bps=0.0,
    sell_tax_bps=10.0,
    min_daily_value_traded=1_000_000_000.0,
    min_adv20_value_traded=2_000_000_000.0,
    max_position_adv_fraction=0.05,
    portfolio_notional=1_000_000_000.0,
    block_limit_up_entry=True,
    exclude_hard_issues=True,
)


def resolve_market_runtime_settings(config: ExperimentConfig) -> MarketRuntimeSettings:
    if not config.respect_market_rules:
        base = _DEFAULT_RULES
    elif len(config.markets) == 1 and config.markets[0] == "VN":
        base = _VN_RULES
    else:
        base = _DEFAULT_RULES

    return MarketRuntimeSettings(
        market_scope=base.market_scope,
        forward_horizon_days=(
            config.forward_horizon_days if config.forward_horizon_days is not None else base.forward_horizon_days
        ),
        allow_short=config.allow_short if config.allow_short is not None else base.allow_short,
        strategy_mode=(
            "long_short"
            if (config.allow_short if config.allow_short is not None else base.allow_short)
            else "long_only"
        ),
        vn_data_profile=config.vn_data_profile if config.vn_data_profile is not None else base.vn_data_profile,
        settlement_cycle=base.settlement_cycle,
        description=base.description,
        buy_cost_bps=config.buy_cost_bps if config.buy_cost_bps is not None else base.buy_cost_bps,
        sell_cost_bps=config.sell_cost_bps if config.sell_cost_bps is not None else base.sell_cost_bps,
        sell_tax_bps=config.sell_tax_bps if config.sell_tax_bps is not None else base.sell_tax_bps,
        min_daily_value_traded=(
            config.min_daily_value_traded if config.min_daily_value_traded is not None else base.min_daily_value_traded
        ),
        min_adv20_value_traded=(
            config.min_adv20_value_traded if config.min_adv20_value_traded is not None else base.min_adv20_value_traded
        ),
        max_position_adv_fraction=(
            config.max_position_adv_fraction
            if config.max_position_adv_fraction is not None
            else base.max_position_adv_fraction
        ),
        portfolio_notional=config.portfolio_notional if config.portfolio_notional is not None else base.portfolio_notional,
        block_limit_up_entry=(
            config.block_limit_up_entry if config.block_limit_up_entry is not None else base.block_limit_up_entry
        ),
        exclude_hard_issues=(
            config.exclude_hard_issues if config.exclude_hard_issues is not None else base.exclude_hard_issues
        ),
    )
