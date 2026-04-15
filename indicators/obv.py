"""
On-Balance Volume (OBV) indicator and divergence detection.

OBV adds volume on up days and subtracts on down days, creating a
cumulative volume-weighted trend line. Divergences between price and
OBV often precede reversals:
  - Bullish divergence: price makes lower low, OBV makes higher low
    → accumulation happening despite falling prices
  - Bearish divergence: price makes higher high, OBV makes lower high
    → distribution happening despite rising prices
"""

import pandas as pd
import numpy as np
from indicators.base import Indicator


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    close = df['Close']
    volume = df['Volume']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    if isinstance(volume, pd.DataFrame):
        volume = volume.squeeze()

    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def create_obv_trend_indicator(period: int = 20) -> Indicator:
    """
    Returns True when OBV is trending up (OBV > its own SMA).

    Args:
        period: SMA period for OBV trend
    """
    def obv_trend(df: pd.DataFrame) -> bool:
        if len(df) < period + 5:
            return False
        obv = _compute_obv(df)
        obv_sma = obv.rolling(window=period).mean()

        current_obv = float(obv.iloc[-1])
        current_sma = float(obv_sma.iloc[-1])

        if pd.isna(current_obv) or pd.isna(current_sma):
            return False
        return current_obv > current_sma

    obv_trend.__name__ = f'OBV_Trend({period})'
    return obv_trend


def create_obv_divergence_indicator(
    lookback: int = 20,
    divergence_type: str = 'bullish',
) -> Indicator:
    """
    Detects OBV divergences by comparing recent price lows/highs with OBV lows/highs.

    Bullish divergence: price made a lower low over lookback period,
    but OBV made a higher low → buying pressure building.

    Bearish divergence: price made a higher high over lookback period,
    but OBV made a lower high → selling pressure building.

    Uses a two-window approach: compares the low/high in the first half
    of the lookback to the low/high in the second half.

    Args:
        lookback: Total lookback window (split into two halves)
        divergence_type: 'bullish' or 'bearish'
    """
    min_data = lookback + 5

    def obv_divergence(df: pd.DataFrame) -> bool:
        if len(df) < min_data:
            return False

        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        obv = _compute_obv(df)

        half = lookback // 2
        recent = slice(-half, None)
        earlier = slice(-lookback, -half)

        if divergence_type == 'bullish':
            # Price: recent low < earlier low (new low)
            price_recent_low = float(close.iloc[recent].min())
            price_earlier_low = float(close.iloc[earlier].min())
            # OBV: recent low > earlier low (higher low = accumulation)
            obv_recent_low = float(obv.iloc[recent].min())
            obv_earlier_low = float(obv.iloc[earlier].min())

            if any(pd.isna(v) for v in [price_recent_low, price_earlier_low,
                                         obv_recent_low, obv_earlier_low]):
                return False

            return price_recent_low < price_earlier_low and obv_recent_low > obv_earlier_low

        else:  # bearish
            price_recent_high = float(close.iloc[recent].max())
            price_earlier_high = float(close.iloc[earlier].max())
            obv_recent_high = float(obv.iloc[recent].max())
            obv_earlier_high = float(obv.iloc[earlier].max())

            if any(pd.isna(v) for v in [price_recent_high, price_earlier_high,
                                         obv_recent_high, obv_earlier_high]):
                return False

            return price_recent_high > price_earlier_high and obv_recent_high < obv_earlier_high

    obv_divergence.__name__ = f'OBV_Divergence({lookback},{divergence_type})'
    return obv_divergence
