from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


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
    if "volume_ratio_5" not in df.columns and {"volume_match", "volume_ma_5"}.issubset(df.columns):
        df["volume_ratio_5"] = df["volume_match"] / df["volume_ma_5"]
    if "volume_ratio_20" not in df.columns and {"volume_match", "volume_ma_20"}.issubset(df.columns):
        df["volume_ratio_20"] = df["volume_match"] / df["volume_ma_20"]
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
    df = df.copy()
    if not {"high", "low", "close"}.issubset(df.columns):
        return df

    volume_column = "volume" if "volume" in df.columns else "volume_match" if "volume_match" in df.columns else None
    if volume_column is None:
        return df

    spread = (df["high"] - df["low"]).astype(float) + 1e-8
    if "effort_result_ratio" not in df.columns:
        df["effort_result_ratio"] = df[volume_column] / spread
    if "buying_pressure" not in df.columns:
        df["buying_pressure"] = ((df["close"] - df["low"]) / spread) * df[volume_column]
    if "selling_pressure" not in df.columns:
        df["selling_pressure"] = ((df["high"] - df["close"]) / spread) * df[volume_column]
    if "wyckoff_phase_60d" not in df.columns:
        by_code = df.groupby("code", group_keys=False)
        min_60d = by_code["low"].rolling(window=60, min_periods=1).min().reset_index(level=0, drop=True)
        max_60d = by_code["high"].rolling(window=60, min_periods=1).max().reset_index(level=0, drop=True)
        df["wyckoff_phase_60d"] = (df["close"] - min_60d) / (max_60d - min_60d + 1e-8)
    return df


# https://github.com/romanmichaelpaolucci/Quant-Guild-Library/blob/655c00f733382e177b0d7fe8f0db80f244f5e3ed/2025%20Video%20Lectures/6.%20How%20to%20Trade%20with%20the%20Black-Scholes%20Model/Black-ScholesTrading.ipynb
def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C


def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adjust_return" not in df.columns:
        return df
        
    # Group by Date to get cross-sectional means
    if "market_return" not in df.columns:
        df["market_return"] = df.groupby("Date")["adjust_return"].transform(lambda x: x.mean())
        
    if "sector" in df.columns and "sector_return" not in df.columns:
        df["sector_return"] = df.groupby(["Date", "sector"])["adjust_return"].transform(lambda x: x.mean())
        
    if "alpha_market" not in df.columns:
        df["alpha_market"] = df["adjust_return"] - df["market_return"]
        
    if "sector" in df.columns and "alpha_sector" not in df.columns:
        df["alpha_sector"] = df["adjust_return"] - df["sector_return"]
        
    return df




def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = add_return_features(df)
    df = add_price_shape_features(df)
    df = add_volume_features(df)
    df = add_momentum_features(df)
    df = add_oscillator_features(df)
    df = add_volatility_features(df)
    df = add_moving_average_gap_features(df)
    df = add_bollinger_features(df)
    df = add_macd_features(df)
    df = add_price_volume_features(df)
    df = add_wyckoff_vsa_features(df)
    df = add_cross_sectional_features(df)
    df = add_calendar_features(df)

    return df.replace([np.inf, -np.inf], np.nan)
