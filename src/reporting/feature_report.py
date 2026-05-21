from __future__ import annotations

from pathlib import Path

from src.models.config import ALL_FEATURE_COLUMNS


FEATURE_FORMULAS: dict[str, str] = {
    "volume_ratio_20": "volume_match / MA20(volume_match)",
    "intraday_return": "close / open - 1",
    "gap_open": "open / close_prev - 1",
    "close_position": "(close - low) / (high - low)",
    "upper_shadow": "(high - max(open, close)) / close",
    "lower_shadow": "(min(open, close) - low) / close",
    "momentum_5": "adjust / adjust[t-5] - 1",
    "momentum_20": "adjust / adjust[t-20] - 1",
    "volatility_20": "Std20(adjust_return)",
    "ma_200_gap": "adjust / MA200(adjust) - 1",
    "rolling_max_20_gap": "adjust / RollingMax20(adjust) - 1",
    "bb_width": "(BB_upper_20 - BB_lower_20) / BB_mid_20",
    "vwap_gap": "close / vwap_proxy - 1, with vwap_proxy = value_match / volume_match",
    "obv_change": "pct_change(OBV), with OBV = cumulative(sign(close_return) * volume_match)",
    "macd_hist": "MACD - MACD_signal, MACD = EMA12(adjust) - EMA26(adjust), MACD_signal = EMA9(MACD)",
    "effort_result_ratio": "volume_norm / (high - low), volume_norm = volume / RollingMax20(volume)",
    "buying_pressure": "((close - low) / (high - low)) * volume_norm",
    "selling_pressure": "((high - close) / (high - low)) * volume_norm",
    "wyckoff_phase_60d": "(close - RollingMin60(low)) / (RollingMax60(high) - RollingMin60(low))",
    "a_d_ratio": "advancing_count / (declining_count + 1)",
    "market_leader_return": "liquidity-weighted return of the top-K market leaders selected by lagged rolling traded value",
    "vingroup_momentum": "deprecated alias for market_leader_return kept for old configs",
    "vnindex_return": "daily VNIndex return built in training pipeline",
    "market_return_5": "RollingMean5(vnindex_return), min_periods=3",
    "market_return_20": "RollingMean20(vnindex_return), min_periods=5",
    "market_volatility_20": "RollingStd20(vnindex_return), min_periods=5",
    "market_ad_ratio_20": "RollingMean20(a_d_ratio), min_periods=5",
    "rsi_14": "100 - 100 / (1 + AvgGain14 / AvgLoss14)",
    "day_of_week": "calendar day index from Date.dt.dayofweek",
    "sector_momentum_rank": "daily rank of each sector by yesterday's sector mean momentum_20; rank 1 is strongest",
    "sector_momentum_rank_pct": "normalized sector_momentum_rank by date; 0 is strongest and 1 is weakest",
    "sector_momentum_20": "yesterday's sector mean momentum_20",
    "relative_sector_momentum_20": "stock momentum_20 - yesterday's sector mean momentum_20",
    "is_top_2_sector": "1 if sector_momentum_rank <= 2, else 0",
    "sector_return": "leave-one-out sector mean adjust_return by date",
    "sector_positive_ratio": "leave-one-out ratio of positive-return stocks in the same sector by date",
    "sector_ad_ratio": "leave-one-out sector advancing count / (declining count + 1) by date",
    "vwap_gap_20": "close / VWAP20 - 1, with VWAP20 = RollingSum20(value_match) / RollingSum20(volume_match)",
    "above_ma_200": "1 if adjust > MA200(adjust), else 0",
    "alpha_sector": "adjust_return - sector_return, where sector_return is leave-one-out sector mean return by date",
}


def _build_lines(feature_columns: tuple[str, ...]) -> list[str]:
    lines = [
        "# Feature Formula Report",
        "",
        "This document lists the active engineered features and the formula used to build each signal.",
        "",
        "| Feature | Formula |",
        "| --- | --- |",
    ]
    for feature in feature_columns:
        formula = FEATURE_FORMULAS.get(feature, "Formula not documented yet.")
        lines.append(f"| `{feature}` | {formula} |")
    return lines


def render_feature_formula_report(feature_columns: tuple[str, ...] | None = None) -> str:
    columns = feature_columns or tuple(ALL_FEATURE_COLUMNS)
    return "\n".join(_build_lines(columns)) + "\n"


def write_feature_formula_report(output_path: Path, feature_columns: tuple[str, ...] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_feature_formula_report(feature_columns), encoding="utf-8")
    return output_path
