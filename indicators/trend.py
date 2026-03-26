"""
Trend-following indicators.
"""

import pandas as pd
from indicators.base import Indicator


def create_sma_trend_indicator(period: int = 200) -> Indicator:
    """
    Returns True when the current price is above the N-day simple moving average.

    Args:
        period: SMA lookback window in trading days
    """
    def sma_trend(df: pd.DataFrame) -> bool:
        if len(df) < period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        sma = float(prices.rolling(window=period).mean().iloc[-1])
        current = float(prices.iloc[-1])
        if pd.isna(sma):
            return False
        return current > sma

    sma_trend.__name__ = f'SMA_Trend({period})'
    return sma_trend


def create_ema_crossover_indicator(short_period: int = 12, long_period: int = 26) -> Indicator:
    """
    Returns True when the short-period EMA is above the long-period EMA.

    Args:
        short_period: Fast EMA window
        long_period: Slow EMA window
    """
    def ema_crossover(df: pd.DataFrame) -> bool:
        if len(df) < long_period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        short_ema = float(prices.ewm(span=short_period, adjust=False).mean().iloc[-1])
        long_ema = float(prices.ewm(span=long_period, adjust=False).mean().iloc[-1])
        if pd.isna(short_ema) or pd.isna(long_ema):
            return False
        return short_ema > long_ema

    ema_crossover.__name__ = f'EMA_Cross({short_period},{long_period})'
    return ema_crossover
