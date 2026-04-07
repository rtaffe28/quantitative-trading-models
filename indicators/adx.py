"""
ADX (Average Directional Index) indicator.

Measures trend strength regardless of direction.
ADX > 25 typically indicates a strong trend; < 20 indicates no trend.
"""

import numpy as np
import pandas as pd
from indicators.base import Indicator


def create_adx_indicator(
    period: int = 14,
    threshold: float = 25.0,
) -> Indicator:
    """
    Returns True when the ADX is above the threshold, indicating a strong trend.

    Args:
        period: Lookback window for ADX calculation
        threshold: ADX value above which the trend is considered strong
    """
    min_data = period * 3  # need enough data for smoothing

    def adx_indicator(df: pd.DataFrame) -> bool:
        if len(df) < min_data:
            return False

        high = df['High']
        low = df['Low']
        close = df['Close']
        if isinstance(high, pd.DataFrame):
            high = high.squeeze()
        if isinstance(low, pd.DataFrame):
            low = low.squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index,
        )

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

        # ADX
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        current_adx = float(adx.iloc[-1])
        if pd.isna(current_adx):
            return False
        return current_adx > threshold

    adx_indicator.__name__ = f'ADX({period},>{threshold})'
    return adx_indicator
