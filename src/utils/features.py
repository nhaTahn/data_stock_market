from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import norm


def _market_group_columns(df: pd.DataFrame) -> list[str]:
    return ["market"] if "market" in df.columns else []


def _market_date_group_columns(df: pd.DataFrame) -> list[str]:
    return [*_market_group_columns(df), "Date"]


def _market_date_sector_group_columns(df: pd.DataFrame) -> list[str]:
    return [*_market_group_columns(df), "Date", "sector"]


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    if "close_prev" not in df.columns and "close" in df.columns:
        df["close_prev"] = by_code["close"].shift(1)
    if "adjust_prev" not in df.columns and "adjust" in df.columns:
        df["adjust_prev"] = by_code["adjust"].shift(1)
    if "adjust_return" not in df.columns and "adjust" in df.columns:
        df["adjust_return"] = by_code["adjust"].pct_change()
    if "close_return" not in df.columns and "close" in df.columns:
        df["close_return"] = by_code["close"].pct_change()
    if "intraday_return" not in df.columns and {"close", "open"}.issubset(df.columns):
        df["intraday_return"] = df["close"] / df["open"] - 1
    if "overnight_return" not in df.columns and {"open", "close_prev"}.issubset(df.columns):
        df["overnight_return"] = df["open"] / df["close_prev"] - 1
    if "gap_open" not in df.columns and {"open", "close_prev"}.issubset(df.columns):
        df["gap_open"] = df["open"] / df["close_prev"] - 1
    return df


def add_price_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "range_pct" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    if "body_pct" not in df.columns and {"close", "open"}.issubset(df.columns):
        df["body_pct"] = (df["close"] - df["open"]) / df["open"]
    if "high_close_gap" not in df.columns and {"high", "close"}.issubset(df.columns):
        df["high_close_gap"] = df["high"] / df["close"] - 1
    if "close_low_gap" not in df.columns and {"close", "low"}.issubset(df.columns):
        df["close_low_gap"] = df["close"] / df["low"] - 1
    if "close_position" not in df.columns and {"high", "low", "close"}.issubset(df.columns):
        spread = (df["high"] - df["low"]).replace(0, np.nan)
        df["close_position"] = (df["close"] - df["low"]) / spread
    if "upper_shadow" not in df.columns and {"high", "open", "close"}.issubset(df.columns):
        upper_ref = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
        df["upper_shadow"] = (df["high"] - upper_ref) / df["close"]
    if "lower_shadow" not in df.columns and {"low", "open", "close"}.issubset(df.columns):
        lower_ref = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)
        df["lower_shadow"] = (lower_ref - df["low"]) / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    if "volume_match" in df.columns:
        if "volume_ma_5" not in df.columns:
            df["volume_ma_5"] = by_code["volume_match"].rolling(5).mean().reset_index(level=0, drop=True)
        if "volume_ma_20" not in df.columns:
            df["volume_ma_20"] = by_code["volume_match"].rolling(20).mean().reset_index(level=0, drop=True)
    if "volume_change" not in df.columns and "volume_match" in df.columns:
        df["volume_change"] = by_code["volume_match"].pct_change()
    # FIX: thêm .replace(0, np.nan) để tránh chia cho 0 khi MA = 0
    if "volume_ratio_5" not in df.columns and {"volume_match", "volume_ma_5"}.issubset(df.columns):
        df["volume_ratio_5"] = df["volume_match"] / df["volume_ma_5"].replace(0, np.nan)
    if "volume_ratio_20" not in df.columns and {"volume_match", "volume_ma_20"}.issubset(df.columns):
        df["volume_ratio_20"] = df["volume_match"] / df["volume_ma_20"].replace(0, np.nan)
    if "volume_zscore_20" not in df.columns and "volume_match" in df.columns:
        vol_mean = by_code["volume_match"].rolling(20).mean().reset_index(level=0, drop=True)
        vol_std = by_code["volume_match"].rolling(20).std().reset_index(level=0, drop=True).replace(0, np.nan)
        df["volume_zscore_20"] = (df["volume_match"] - vol_mean) / vol_std
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    if "return_3" not in df.columns and "adjust" in df.columns:
        df["return_3"] = by_code["adjust"].pct_change(3)
    if "momentum_5" not in df.columns and "adjust" in df.columns:
        df["momentum_5"] = by_code["adjust"].pct_change(5)
    if "momentum_20" not in df.columns and "adjust" in df.columns:
        df["momentum_20"] = by_code["adjust"].pct_change(20)
    if "return_10" not in df.columns and "adjust" in df.columns:
        df["return_10"] = by_code["adjust"].pct_change(10)
    if "price_acceleration" not in df.columns and {"momentum_5", "momentum_20"}.issubset(df.columns):
        df["price_acceleration"] = df["momentum_5"] - df["momentum_20"]
    return df


def add_oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_column = "adjust" if "adjust" in df.columns else "close" if "close" in df.columns else None
    if price_column is None:
        return df

    by_code = df.groupby("code", group_keys=False)
    if "rsi_14" not in df.columns:
        price_delta = by_code[price_column].diff()
        gain = price_delta.clip(lower=0.0)
        loss = (-price_delta).clip(lower=0.0)
        avg_gain = gain.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
        avg_loss = loss.groupby(df["code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns:
        return df
    date_series = pd.to_datetime(df["Date"])
    if "day_of_week" not in df.columns:
        df["day_of_week"] = date_series.dt.dayofweek.astype(float)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    if "true_range" not in df.columns and {"high", "low", "close_prev"}.issubset(df.columns):
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close_prev"]).abs()
        tr3 = (df["low"] - df["close_prev"]).abs()
        df["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if "atr_14" not in df.columns and "true_range" in df.columns:
        df["atr_14"] = by_code["true_range"].rolling(14).mean().reset_index(level=0, drop=True)
    if "atr_gap" not in df.columns and {"atr_14", "close"}.issubset(df.columns):
        df["atr_gap"] = df["atr_14"] / df["close"]
    if "volatility_5" not in df.columns and "adjust_return" in df.columns:
        df["volatility_5"] = by_code["adjust_return"].rolling(5).std().reset_index(level=0, drop=True)
    if "volatility_20" not in df.columns and "adjust_return" in df.columns:
        df["volatility_20"] = by_code["adjust_return"].rolling(20).std().reset_index(level=0, drop=True)
    if "volatility_10" not in df.columns and "adjust_return" in df.columns:
        df["volatility_10"] = by_code["adjust_return"].rolling(10).std().reset_index(level=0, drop=True)
    if "volatility_ratio" not in df.columns and {"volatility_5", "volatility_20"}.issubset(df.columns):
        df["volatility_ratio"] = df["volatility_5"] / df["volatility_20"]
    return df


def add_moving_average_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    ma_5 = None
    ma_10 = None
    ma_20 = None
    ma_50 = None
    ma_200 = None
    if "adjust" in df.columns:
        if "ma_5" not in df.columns:
            ma_5 = by_code["adjust"].rolling(5).mean().reset_index(level=0, drop=True)
            df["ma_5"] = ma_5
        else:
            ma_5 = df["ma_5"]
        if "ma_10" not in df.columns:
            ma_10 = by_code["adjust"].rolling(10).mean().reset_index(level=0, drop=True)
            df["ma_10"] = ma_10
        else:
            ma_10 = df["ma_10"]
        if "ma_20" not in df.columns:
            ma_20 = by_code["adjust"].rolling(20).mean().reset_index(level=0, drop=True)
            df["ma_20"] = ma_20
        else:
            ma_20 = df["ma_20"]
        if "ma_50" not in df.columns:
            ma_50 = by_code["adjust"].rolling(50).mean().reset_index(level=0, drop=True)
            df["ma_50"] = ma_50
        else:
            ma_50 = df["ma_50"]
        if "ma_200" not in df.columns:
            ma_200 = by_code["adjust"].rolling(200).mean().reset_index(level=0, drop=True)
            df["ma_200"] = ma_200
        else:
            ma_200 = df["ma_200"]
    if "ma_5_gap" not in df.columns and "adjust" in df.columns:
        if ma_5 is None:
            ma_5 = by_code["adjust"].rolling(5).mean().reset_index(level=0, drop=True)
        df["ma_5_gap"] = df["adjust"] / ma_5 - 1
    if "ma_10_gap" not in df.columns and "adjust" in df.columns:
        if ma_10 is None:
            ma_10 = by_code["adjust"].rolling(10).mean().reset_index(level=0, drop=True)
        df["ma_10_gap"] = df["adjust"] / ma_10 - 1
    if "ma_20_gap" not in df.columns and "adjust" in df.columns:
        if ma_20 is None:
            ma_20 = by_code["adjust"].rolling(20).mean().reset_index(level=0, drop=True)
        df["ma_20_gap"] = df["adjust"] / ma_20 - 1
    if "ma_50_gap" not in df.columns and "adjust" in df.columns:
        if ma_50 is None:
            ma_50 = by_code["adjust"].rolling(50).mean().reset_index(level=0, drop=True)
        df["ma_50_gap"] = df["adjust"] / ma_50 - 1
    if "ma_cross_5_20" not in df.columns and {"ma_5", "ma_20"}.issubset(df.columns):
        df["ma_cross_5_20"] = df["ma_5"] / df["ma_20"] - 1
    if "ma_200_gap" not in df.columns and "adjust" in df.columns:
        if ma_200 is None:
            ma_200 = by_code["adjust"].rolling(200).mean().reset_index(level=0, drop=True)
        df["ma_200_gap"] = df["adjust"] / ma_200 - 1
    if "above_ma_200" not in df.columns and {"adjust", "ma_200"}.issubset(df.columns):
        df["above_ma_200"] = (df["adjust"] > df["ma_200"]).astype(float)
    if "ma_20_ma_200_gap" not in df.columns and {"ma_20", "ma_200"}.issubset(df.columns):
        df["ma_20_ma_200_gap"] = df["ma_20"] / df["ma_200"] - 1
    if "rolling_max_20_gap" not in df.columns and "adjust" in df.columns:
        rolling_max_20 = by_code["adjust"].rolling(20).max().reset_index(level=0, drop=True)
        df["rolling_max_20_gap"] = df["adjust"] / rolling_max_20 - 1
    if "rolling_min_20_gap" not in df.columns and "adjust" in df.columns:
        rolling_min_20 = by_code["adjust"].rolling(20).min().reset_index(level=0, drop=True)
        df["rolling_min_20_gap"] = df["adjust"] / rolling_min_20 - 1
    return df


def _rolling_midpoint(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    high_roll = high.rolling(window, min_periods=window).max()
    low_roll = low.rolling(window, min_periods=window).min()
    return (high_roll + low_roll) / 2.0


def _grouped_rolling_midpoint(
    high: pd.Series,
    low: pd.Series,
    groups: pd.Series,
    window: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=high.index, dtype=float)
    for _, index in groups.groupby(groups, sort=False).groups.items():
        out.loc[index] = _rolling_midpoint(high.loc[index], low.loc[index], window)
    return out


def add_ichimoku_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not {"high", "low", "close", "code"}.issubset(df.columns):
        return df

    feature_specs = {
        "ichi_8_21_42_tenkan_kijun_gap": (8, 21),
        "ichi_8_22_44_tenkan_kijun_gap": (8, 22),
    }
    missing_features = [feature for feature in feature_specs if feature not in df.columns]
    if not missing_features:
        return df

    close = df["close"].replace(0.0, np.nan)
    adjustment_factor = df["adjust"] / close if "adjust" in df.columns else pd.Series(1.0, index=df.index)
    adjusted_high = df["high"] * adjustment_factor
    adjusted_low = df["low"] * adjustment_factor
    adjusted_close = df["adjust"] if "adjust" in df.columns else df["close"]
    close_safe = adjusted_close.replace(0.0, np.nan)

    for feature, (conversion_window, base_window) in feature_specs.items():
        if feature not in missing_features:
            continue
        tenkan = _grouped_rolling_midpoint(adjusted_high, adjusted_low, df["code"], conversion_window)
        kijun = _grouped_rolling_midpoint(adjusted_high, adjusted_low, df["code"], base_window)
        df[feature] = (tenkan - kijun) / close_safe
    return df


def add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adjust" not in df.columns:
        return df
    by_code = df.groupby("code", group_keys=False)
    if "bb_mid_20" not in df.columns:
        df["bb_mid_20"] = by_code["adjust"].rolling(20).mean().reset_index(level=0, drop=True)
    if "bb_std_20" not in df.columns:
        df["bb_std_20"] = by_code["adjust"].rolling(20).std().reset_index(level=0, drop=True)
    if "bb_upper_20" not in df.columns and {"bb_mid_20", "bb_std_20"}.issubset(df.columns):
        df["bb_upper_20"] = df["bb_mid_20"] + 2 * df["bb_std_20"]
    if "bb_lower_20" not in df.columns and {"bb_mid_20", "bb_std_20"}.issubset(df.columns):
        df["bb_lower_20"] = df["bb_mid_20"] - 2 * df["bb_std_20"]
    if "bb_width" not in df.columns and {"bb_upper_20", "bb_lower_20", "bb_mid_20"}.issubset(df.columns):
        mid = df["bb_mid_20"].replace(0, np.nan)
        df["bb_width"] = (df["bb_upper_20"] - df["bb_lower_20"]) / mid
    if "bb_position" not in df.columns and {"adjust", "bb_upper_20", "bb_lower_20"}.issubset(df.columns):
        spread = (df["bb_upper_20"] - df["bb_lower_20"]).replace(0, np.nan)
        df["bb_position"] = (df["adjust"] - df["bb_lower_20"]) / spread
    if "bb_zscore" not in df.columns and {"adjust", "bb_mid_20", "bb_std_20"}.issubset(df.columns):
        std = df["bb_std_20"].replace(0, np.nan)
        df["bb_zscore"] = (df["adjust"] - df["bb_mid_20"]) / std
    return df


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adjust" not in df.columns:
        return df
    if "macd" not in df.columns:
        ema_12 = df.groupby("code")["adjust"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
        ema_26 = df.groupby("code")["adjust"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
        df["macd"] = ema_12 - ema_26
    if "macd_signal" not in df.columns and "macd" in df.columns:
        df["macd_signal"] = df.groupby("code")["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    if "macd_hist" not in df.columns and {"macd", "macd_signal"}.issubset(df.columns):
        df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_price_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    if "vwap_proxy" not in df.columns and {"value_match", "volume_match"}.issubset(df.columns):
        volume = df["volume_match"].replace(0, np.nan)
        df["vwap_proxy"] = df["value_match"] / volume
    if "vwap_gap" not in df.columns and {"close", "vwap_proxy"}.issubset(df.columns):
        df["vwap_gap"] = df["close"] / df["vwap_proxy"] - 1
    if "signed_volume" not in df.columns and {"close_return", "volume_match"}.issubset(df.columns):
        df["signed_volume"] = np.sign(df["close_return"]).fillna(0) * df["volume_match"]
    if "obv" not in df.columns and {"close_return", "volume_match"}.issubset(df.columns):
        delta = np.sign(df["close_return"]).fillna(0) * df["volume_match"]
        df["obv"] = delta.groupby(df["code"]).cumsum()
    if "obv_change" not in df.columns and "obv" in df.columns:
        df["obv_change"] = by_code["obv"].pct_change()
        
    if "vwap_20" not in df.columns and {"value_match", "volume_match"}.issubset(df.columns):
        roll_vol = by_code["volume_match"].rolling(20).sum().reset_index(level=0, drop=True)
        roll_val = by_code["value_match"].rolling(20).sum().reset_index(level=0, drop=True)
        df["vwap_20"] = roll_val / roll_vol.replace(0, np.nan)
        
    if "vwap_gap_20" not in df.columns and {"close", "vwap_20"}.issubset(df.columns):
        df["vwap_gap_20"] = df["close"] / df["vwap_20"] - 1
        
    return df


def add_wyckoff_vsa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wyckoff / VSA features.

    Fixes applied vs original:
    - spread=0 (Doji candles) now produces NaN instead of near-inf via epsilon trick.
    - Volume is normalised by 20-day rolling max for cross-sectional comparability.
    - wyckoff_phase_60d requires min_periods=20 (was 1) for statistical validity.
    """
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    volume_column = "volume" if "volume" in df.columns else "volume_match" if "volume_match" in df.columns else None
    if volume_column is None:
        return df

    spread = (df["high"] - df["low"]).astype(float)
    # FIX: replace(0, np.nan) preserves Doji signal; epsilon trick created extreme |inf| values
    spread_safe = spread.replace(0, np.nan)

    by_code = df.groupby("code", group_keys=False)
    # FIX: normalise volume by rolling-20 max so features are comparable across stocks
    vol_max_20 = (
        by_code[volume_column]
        .rolling(20, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )
    vol_norm = df[volume_column] / vol_max_20.replace(0, np.nan)

    if "effort_result_ratio" not in df.columns:
        df["effort_result_ratio"] = vol_norm / spread_safe
    if "buying_pressure" not in df.columns:
        df["buying_pressure"] = ((df["close"] - df["low"]) / spread_safe) * vol_norm
    if "selling_pressure" not in df.columns:
        df["selling_pressure"] = ((df["high"] - df["close"]) / spread_safe) * vol_norm
    if "wyckoff_phase_60d" not in df.columns:
        # FIX: min_periods=20 (was 1) — phase has no meaning with fewer than ~20 days
        min_60d = by_code["low"].rolling(window=60, min_periods=20).min().reset_index(level=0, drop=True)
        max_60d = by_code["high"].rolling(window=60, min_periods=20).max().reset_index(level=0, drop=True)
        range_safe = (max_60d - min_60d).replace(0, np.nan)
        df["wyckoff_phase_60d"] = (df["close"] - min_60d) / range_safe
    return df



# https://github.com/romanmichaelpaolucci/Quant-Guild-Library/blob/655c00f733382e177b0d7fe8f0db80f244f5e3ed/2025%20Video%20Lectures/6.%20How%20to%20Trade%20with%20the%20Black-Scholes%20Model/Black-ScholesTrading.ipynb
def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional alpha features.

    FIX: Uses leave-one-out sector/market mean to prevent circular data leakage.
    Original code included the stock's own return in the sector mean, causing
    systematic alpha underestimation (especially severe for small sectors <=5 stocks).
    """
    df = df.copy()
    if "adjust_return" not in df.columns:
        return df
    market_date_groups = _market_date_group_columns(df)

    # Leave-one-out market mean: exclude self from the mean
    if "market_return" not in df.columns:
        date_sum = df.groupby(market_date_groups)["adjust_return"].transform("sum")
        date_count = df.groupby(market_date_groups)["adjust_return"].transform("count")
        # Subtract self, divide by (N-1); clip(1) avoids div/0 for 1-stock dates
        df["market_return"] = (date_sum - df["adjust_return"]) / (date_count - 1).clip(lower=1)

    if "sector" in df.columns and "sector_return" not in df.columns:
        sector_groups = _market_date_sector_group_columns(df)
        sec_sum = df.groupby(sector_groups)["adjust_return"].transform("sum")
        sec_count = df.groupby(sector_groups)["adjust_return"].transform("count")
        df["sector_return"] = (sec_sum - df["adjust_return"]) / (sec_count - 1).clip(lower=1)

    if "alpha_market" not in df.columns:
        df["alpha_market"] = df["adjust_return"] - df["market_return"]

    if "sector" in df.columns and "alpha_sector" not in df.columns:
        df["alpha_sector"] = df["adjust_return"] - df["sector_return"]

    if "sector" in df.columns and "sector_positive_ratio" not in df.columns:
        positive = (df["adjust_return"] > 0).astype(float)
        sector_series_groups = [df[column] for column in _market_date_sector_group_columns(df)]
        sec_positive = positive.groupby(sector_series_groups).transform("sum")
        sec_count = df.groupby(_market_date_sector_group_columns(df))["code"].transform("count")
        denom = (sec_count - 1).clip(lower=1)
        df["sector_positive_ratio"] = (sec_positive - positive) / denom

    if "sector" in df.columns and "sector_ad_ratio" not in df.columns:
        positive = (df["adjust_return"] > 0).astype(float)
        negative = (df["adjust_return"] < 0).astype(float)
        sector_series_groups = [df[column] for column in _market_date_sector_group_columns(df)]
        sec_positive = positive.groupby(sector_series_groups).transform("sum")
        sec_negative = negative.groupby(sector_series_groups).transform("sum")
        advancing_ex_self = sec_positive - positive
        declining_ex_self = sec_negative - negative
        df["sector_ad_ratio"] = advancing_ex_self / (declining_ex_self + 1.0)

    return df


def _forward_fill_array(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.ffill().bfill()
    return series.to_numpy(dtype=float)


def _causal_fft_last(values: np.ndarray, keep_ratio: float = 0.25, window: int = 32) -> np.ndarray:
    clean = _forward_fill_array(values)
    out = np.empty_like(clean, dtype=float)
    for idx in range(len(clean)):
        start_idx = max(0, idx - window + 1)
        segment = clean[start_idx : idx + 1]
        if len(segment) < 4:
            out[idx] = segment[-1]
            continue
        coeffs = np.fft.rfft(segment)
        keep_n = max(2, int(np.ceil(len(coeffs) * keep_ratio)))
        coeffs[keep_n:] = 0.0
        recon = np.fft.irfft(coeffs, n=len(segment))
        out[idx] = float(recon[-1])
    return out


def _haar_dwt(signal: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    current = signal.astype(float)
    details: list[np.ndarray] = []
    while len(current) > 1:
        if len(current) % 2 == 1:
            current = np.append(current, current[-1])
        avg = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        diff = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        details.append(diff)
        current = avg
    return current, details


def _haar_idwt(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    current = approx.astype(float)
    for diff in reversed(details):
        restored = np.empty(diff.size * 2, dtype=float)
        restored[0::2] = (current + diff) / np.sqrt(2.0)
        restored[1::2] = (current - diff) / np.sqrt(2.0)
        current = restored
    return current


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _causal_haar_last(values: np.ndarray, threshold_scale: float = 0.8, window: int = 32) -> np.ndarray:
    clean = _forward_fill_array(values)
    out = np.empty_like(clean, dtype=float)
    for idx in range(len(clean)):
        start_idx = max(0, idx - window + 1)
        segment = clean[start_idx : idx + 1]
        if len(segment) < 8:
            out[idx] = segment[-1]
            continue
        target_len = 1 << int(np.ceil(np.log2(len(segment))))
        padded = np.pad(segment, (0, target_len - len(segment)), mode="edge")
        approx, details = _haar_dwt(padded)
        finest = details[0]
        sigma = np.median(np.abs(finest)) / 0.6745 if len(finest) else 0.0
        threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(len(padded), 2)))
        shrinked = [_soft_threshold(detail, threshold) for detail in details]
        recon = _haar_idwt(approx, shrinked)[: len(segment)]
        out[idx] = float(recon[-1])
    return out


def _causal_savgol_last(values: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    clean = _forward_fill_array(values)
    out = np.empty_like(clean, dtype=float)
    for idx in range(len(clean)):
        start_idx = max(0, idx - window + 1)
        segment = clean[start_idx : idx + 1]
        if len(segment) < polyorder + 2:
            out[idx] = segment[-1]
            continue
        local_window = min(window, len(segment))
        if local_window % 2 == 0:
            local_window -= 1
        if local_window < polyorder + 2:
            deg = min(polyorder, len(segment) - 1)
            xs = np.arange(len(segment), dtype=float)
            coeffs = np.polyfit(xs, segment, deg)
            out[idx] = float(np.polyval(coeffs, xs[-1]))
            continue
        fitted = savgol_filter(segment, window_length=local_window, polyorder=min(polyorder, local_window - 1), mode="interp")
        out[idx] = float(fitted[-1])
    return out


def _causal_kalman(values: np.ndarray, process_scale: float = 1e-3, measurement_scale: float = 5e-3) -> np.ndarray:
    clean = _forward_fill_array(values)
    out = np.empty_like(clean, dtype=float)
    out[0] = clean[0]
    diff_var = np.nanvar(np.diff(clean)) if len(clean) > 1 else 0.0
    level_var = np.nanvar(clean)
    q = max(diff_var * process_scale, 1e-8)
    r = max(level_var * measurement_scale, q * 10.0, 1e-8)
    state_cov = 1.0
    for idx in range(1, len(clean)):
        pred = out[idx - 1]
        pred_cov = state_cov + q
        gain = pred_cov / (pred_cov + r)
        out[idx] = pred + gain * (clean[idx] - pred)
        state_cov = (1.0 - gain) * pred_cov
    return out


def add_paper_price_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_code = df.groupby("code", group_keys=False)
    price_columns = [col for col in ("open", "high", "low", "close") if col in df.columns]
    for column in price_columns:
        level_name = f"{column}_level_20"
        delta_name = f"{column}_delta_1"
        if level_name not in df.columns:
            rolling_mean = by_code[column].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
            df[level_name] = df[column] / rolling_mean.replace(0, np.nan) - 1.0
        if delta_name not in df.columns:
            df[delta_name] = by_code[column].pct_change()
    volume_column = "volume_match" if "volume_match" in df.columns else "volume" if "volume" in df.columns else None
    if volume_column is not None:
        if "volume_level_20" not in df.columns:
            volume_mean = by_code[volume_column].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
            df["volume_level_20"] = df[volume_column] / volume_mean.replace(0, np.nan) - 1.0
        if "volume_delta_1" not in df.columns:
            df["volume_delta_1"] = by_code[volume_column].pct_change()
    return df


def add_causal_denoise_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_column = "adjust" if "adjust" in df.columns else "close" if "close" in df.columns else None
    if price_column is None:
        return df

    work_frames: list[pd.DataFrame] = []
    for _, group in df.groupby("code", sort=False):
        part = group.sort_values("Date", kind="stable").copy()
        price = part[price_column].to_numpy(dtype=float)
        fft_trend = _causal_fft_last(price)
        wavelet_trend = _causal_haar_last(price)
        savgol_trend = _causal_savgol_last(price)
        kalman_trend = _causal_kalman(price)
        denom = np.where(np.abs(price) > 1e-8, price, np.nan)

        part["fft_trend_gap_32"] = fft_trend / denom - 1.0
        part["wavelet_trend_gap_32"] = wavelet_trend / denom - 1.0
        part["savgol_trend_gap_11"] = savgol_trend / denom - 1.0
        part["kalman_trend_gap"] = kalman_trend / denom - 1.0

        part["fft_noise_ratio_32"] = (price - fft_trend) / denom
        part["wavelet_noise_ratio_32"] = (price - wavelet_trend) / denom
        part["savgol_noise_ratio_11"] = (price - savgol_trend) / denom
        part["kalman_noise_ratio"] = (price - kalman_trend) / denom

        consensus_trend = (fft_trend + wavelet_trend + savgol_trend + kalman_trend) / 4.0
        part["denoise_consensus_gap"] = consensus_trend / denom - 1.0
        part["denoise_method_dispersion"] = np.nanstd(
            np.vstack([fft_trend, wavelet_trend, savgol_trend, kalman_trend]),
            axis=0,
        ) / denom
        work_frames.append(part)

    out = pd.concat(work_frames, ignore_index=True)
    return out.sort_values(["code", "Date"], kind="stable").reset_index(drop=True)


def add_sector_leadership_features(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Add causal sector-leadership features based on 20-day momentum."""
    df = df.copy()
    if "Date" not in df.columns or "sector" not in df.columns or "momentum_20" not in df.columns:
        return df

    leadership_columns = {
        "sector_momentum_20",
        "sector_momentum_rank",
        "sector_momentum_rank_pct",
        f"is_top_{k}_sector",
    }
    if not leadership_columns.issubset(df.columns):
        df = df.drop(columns=[column for column in leadership_columns if column in df.columns])
        market_groups = _market_group_columns(df)
        market_date_sector_groups = _market_date_sector_group_columns(df)
        rank_groups = [*market_groups, "Date"]
        sector_shift_groups = [*market_groups, "sector"]
        sector_daily = df.groupby(market_date_sector_groups)["momentum_20"].mean().reset_index()
        sector_daily["sector_momentum_20"] = sector_daily.groupby(sector_shift_groups)["momentum_20"].shift(1)
        sector_daily["sector_momentum_rank"] = sector_daily.groupby(rank_groups)["sector_momentum_20"].rank(
            ascending=False,
            method="min",
        )
        sector_count = sector_daily.groupby(rank_groups)["sector_momentum_20"].transform("count")
        sector_daily["sector_momentum_rank_pct"] = (
            (sector_daily["sector_momentum_rank"] - 1.0) / (sector_count - 1.0).replace(0.0, np.nan)
        )
        sector_daily[f"is_top_{k}_sector"] = (sector_daily["sector_momentum_rank"] <= k).astype(float)
        sector_daily = sector_daily.drop(columns=["momentum_20"])
        df = df.merge(sector_daily, on=market_date_sector_groups, how="left")

    if "relative_sector_momentum_20" not in df.columns and "sector_momentum_20" in df.columns:
        df["relative_sector_momentum_20"] = df["momentum_20"] - df["sector_momentum_20"]

    return df


def ensure_paper_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)
    df = add_paper_price_delta_features(df)
    df = add_causal_denoise_features(df)
    return df.replace([np.inf, -np.inf], np.nan)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_price_shape_features(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    df = add_oscillator_features(df)
    df = add_volatility_features(df)
    df = add_moving_average_gap_features(df)
    df = add_ichimoku_cycle_features(df)
    df = add_bollinger_features(df)
    df = add_macd_features(df)
    df = add_price_volume_features(df)
    df = add_wyckoff_vsa_features(df)
    df = add_cross_sectional_features(df)
    df = add_calendar_features(df)
    df = add_sector_leadership_features(df, k=2)

    return df.replace([np.inf, -np.inf], np.nan)
