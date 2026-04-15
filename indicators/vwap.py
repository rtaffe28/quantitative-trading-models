"""
VWAP (Volume-Weighted Average Price) deviation indicator.

VWAP represents the "fair value" price weighted by volume. Stocks trading
significantly below their rolling VWAP are potentially undervalued in the
short term — institutions use VWAP as a benchmark for execution quality.

A rolling VWAP approximation over N days serves as a mean-reversion anchor:
  - Price well below VWAP = cheaper than where most volume transacted
  - Price well above VWAP = more expensive than institutional avg cost
"""

import pandas as pd
import numpy as np
from indicators.base import Indicator


def _compute_rolling_vwap(df: pd.DataFrame, period: int) -> pd.Series:
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume

    rolling_tp_vol = tp_volume.rolling(window=period).sum()
    rolling_vol = volume.rolling(window=period).sum()

    vwap = rolling_tp_vol / rolling_vol
    return vwap


def create_vwap_discount_indicator(
    period: int = 20,
    discount_pct: float = 0.03,
) -> Indicator:
    """
    Returns True when price is at least discount_pct below the rolling VWAP.

    Args:
        period: Rolling VWAP lookback in trading days
        discount_pct: Required discount below VWAP (0.03 = 3%)
    """
    def vwap_discount(df: pd.DataFrame) -> bool:
        if len(df) < period + 5:
            return False

        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        vwap = _compute_rolling_vwap(df, period)

        current_price = float(close.iloc[-1])
        current_vwap = float(vwap.iloc[-1])

        if pd.isna(current_vwap) or current_vwap == 0:
            return False

        return current_price < current_vwap * (1 - discount_pct)

    vwap_discount.__name__ = f'VWAP_Discount({period}d,{discount_pct:.0%})'
    return vwap_discount


def compute_vwap_deviation(df: pd.DataFrame, period: int = 20) -> float:
    """Returns the % deviation of current price from rolling VWAP."""
    if len(df) < period + 5:
        return float('nan')

    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    vwap = _compute_rolling_vwap(df, period)

    current_price = float(close.iloc[-1])
    current_vwap = float(vwap.iloc[-1])

    if pd.isna(current_vwap) or current_vwap == 0:
        return float('nan')

    return (current_price - current_vwap) / current_vwap
