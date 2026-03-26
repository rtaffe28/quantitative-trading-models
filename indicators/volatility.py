"""
Volatility indicators.
"""

import numpy as np
import pandas as pd
from indicators.base import Indicator


def create_volatility_spike_indicator(
    lookback: int = 60,
    spike_factor: float = 1.5,
    short_window: int = 20,
) -> Indicator:
    """
    Returns True when recent realized volatility exceeds the longer-term
    average by a configurable factor.

    Args:
        lookback: Long-term window for baseline volatility (trading days)
        spike_factor: Multiplier — current vol must exceed baseline × spike_factor
        short_window: Short-term window for "current" volatility
    """
    min_data = lookback + short_window

    def vol_spike(df: pd.DataFrame) -> bool:
        if len(df) < min_data:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        log_returns = np.log(prices / prices.shift(1))
        short_vol = float(log_returns.rolling(window=short_window).std().iloc[-1])
        long_vol = float(log_returns.rolling(window=lookback).std().iloc[-1])

        if pd.isna(short_vol) or pd.isna(long_vol) or long_vol == 0:
            return False
        return short_vol > long_vol * spike_factor

    vol_spike.__name__ = f'VolSpike({short_window}d>{spike_factor}x{lookback}d)'
    return vol_spike


def create_bollinger_squeeze_indicator(
    period: int = 20,
    std_dev: float = 2.0,
    squeeze_pct: float = 0.05,
) -> Indicator:
    """
    Returns True when Bollinger Band width is unusually narrow, signaling
    low volatility likely to expand.

    Args:
        period: Lookback window for SMA and standard deviation
        std_dev: Number of standard deviations for bands
        squeeze_pct: Width threshold — True when (upper-lower)/middle < squeeze_pct
    """
    def bollinger_squeeze(df: pd.DataFrame) -> bool:
        if len(df) < period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std

        mid = float(sma.iloc[-1])
        width = float(upper.iloc[-1]) - float(lower.iloc[-1])

        if pd.isna(mid) or mid == 0 or pd.isna(width):
            return False
        return (width / mid) < squeeze_pct

    bollinger_squeeze.__name__ = f'BollingerSqueeze({period},<{squeeze_pct})'
    return bollinger_squeeze
