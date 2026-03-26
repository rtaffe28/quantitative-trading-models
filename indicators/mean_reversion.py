"""
Mean-reversion indicators.
"""

import pandas as pd
from indicators.base import Indicator


def create_mean_reversion_indicator(
    period: int = 200,
    deviation_pct: float = 0.10,
) -> Indicator:
    """
    Returns True when the current price is at least deviation_pct below
    the N-day simple moving average (buy-the-dip signal).

    Args:
        period: SMA lookback window
        deviation_pct: Required deviation below SMA (0.10 = 10%)
    """
    def mean_reversion(df: pd.DataFrame) -> bool:
        if len(df) < period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        sma = float(prices.rolling(window=period).mean().iloc[-1])
        current = float(prices.iloc[-1])

        if pd.isna(sma) or sma == 0:
            return False
        return current < sma * (1 - deviation_pct)

    mean_reversion.__name__ = f'MeanReversion({period},{deviation_pct:.0%})'
    return mean_reversion


def create_zscore_indicator(
    period: int = 50,
    threshold: float = -2.0,
) -> Indicator:
    """
    Returns True when the z-score of the current price relative to the
    N-day rolling mean/std is below the threshold (oversold signal).

    Args:
        period: Rolling window for mean and std
        threshold: Z-score threshold (negative = below mean)
    """
    def zscore(df: pd.DataFrame) -> bool:
        if len(df) < period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        mean = float(prices.rolling(window=period).mean().iloc[-1])
        std = float(prices.rolling(window=period).std().iloc[-1])
        current = float(prices.iloc[-1])

        if pd.isna(mean) or pd.isna(std) or std == 0:
            return False
        z = (current - mean) / std
        return z < threshold

    zscore.__name__ = f'ZScore({period},<{threshold})'
    return zscore
