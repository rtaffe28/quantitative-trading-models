"""
Momentum indicators.
"""

import pandas as pd
from indicators.base import Indicator


def create_rsi_indicator(
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    signal: str = 'oversold',
) -> Indicator:
    """
    Returns True when RSI crosses below the oversold threshold (buy signal)
    or above the overbought threshold (sell signal).

    Args:
        period: RSI lookback window
        oversold: Lower threshold
        overbought: Upper threshold
        signal: 'oversold' to trigger when RSI < oversold,
                'overbought' to trigger when RSI > overbought
    """
    def rsi_indicator(df: pd.DataFrame) -> bool:
        if len(df) < period + 1:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

        if pd.isna(current_rsi):
            return False
        if signal == 'oversold':
            return current_rsi < oversold
        return current_rsi > overbought

    rsi_indicator.__name__ = f'RSI({period},{signal}<{oversold if signal == "oversold" else overbought})'
    return rsi_indicator


def create_macd_crossover_indicator(
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Indicator:
    """
    Returns True when the MACD line is above the signal line (bullish).

    Args:
        fast: Fast EMA period
        slow: Slow EMA period
        signal_period: Signal line EMA period
    """
    def macd_crossover(df: pd.DataFrame) -> bool:
        if len(df) < slow + signal_period:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])

        if pd.isna(current_macd) or pd.isna(current_signal):
            return False
        return current_macd > current_signal

    macd_crossover.__name__ = f'MACD({fast},{slow},{signal_period})'
    return macd_crossover
