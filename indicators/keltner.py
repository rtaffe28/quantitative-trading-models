"""
Keltner Channel indicator.

Keltner Channels use ATR (Average True Range) instead of standard deviation
(like Bollinger Bands). When Bollinger Bands contract inside Keltner Channels,
it signals a "squeeze" — compressed volatility likely to expand.
"""

import numpy as np
import pandas as pd
from indicators.base import Indicator


def create_keltner_breakout_indicator(
    ema_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 1.5,
    direction: str = 'above',
) -> Indicator:
    """
    Returns True when price breaks above (or below) the Keltner Channel.

    Keltner Channel:
      - Middle = EMA(ema_period)
      - Upper  = Middle + atr_mult * ATR(atr_period)
      - Lower  = Middle - atr_mult * ATR(atr_period)

    Args:
        ema_period: EMA lookback for the middle line
        atr_period: ATR calculation period
        atr_mult: ATR multiplier for channel width
        direction: 'above' triggers when price > upper band,
                   'below' triggers when price < lower band
    """
    min_data = max(ema_period, atr_period) + 5

    def keltner_breakout(df: pd.DataFrame) -> bool:
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

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()

        # Keltner Channel
        ema = close.ewm(span=ema_period, adjust=False).mean()
        upper = ema + atr_mult * atr
        lower = ema - atr_mult * atr

        current_price = float(close.iloc[-1])
        current_upper = float(upper.iloc[-1])
        current_lower = float(lower.iloc[-1])

        if pd.isna(current_upper) or pd.isna(current_lower):
            return False

        if direction == 'above':
            return current_price > current_upper
        return current_price < current_lower

    keltner_breakout.__name__ = f'Keltner({ema_period},{atr_mult}x,{direction})'
    return keltner_breakout


def create_squeeze_indicator(
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_ema_period: int = 20,
    kc_atr_period: int = 14,
    kc_atr_mult: float = 1.5,
) -> Indicator:
    """
    Returns True when Bollinger Bands are INSIDE Keltner Channels (the "squeeze").

    This signals extremely compressed volatility — a breakout is likely imminent.
    Combine with a momentum or breakout indicator to determine direction.

    Args:
        bb_period: Bollinger Band SMA period
        bb_std: Bollinger Band standard deviation multiplier
        kc_ema_period: Keltner Channel EMA period
        kc_atr_period: Keltner Channel ATR period
        kc_atr_mult: Keltner Channel ATR multiplier
    """
    min_data = max(bb_period, kc_ema_period, kc_atr_period) + 5

    def squeeze(df: pd.DataFrame) -> bool:
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

        # Bollinger Bands
        bb_sma = close.rolling(window=bb_period).mean()
        bb_std_val = close.rolling(window=bb_period).std()
        bb_upper = bb_sma + bb_std * bb_std_val
        bb_lower = bb_sma - bb_std * bb_std_val

        # Keltner Channels
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=kc_atr_period).mean()

        kc_ema = close.ewm(span=kc_ema_period, adjust=False).mean()
        kc_upper = kc_ema + kc_atr_mult * atr
        kc_lower = kc_ema - kc_atr_mult * atr

        # Squeeze = BB inside KC
        bb_up = float(bb_upper.iloc[-1])
        bb_lo = float(bb_lower.iloc[-1])
        kc_up = float(kc_upper.iloc[-1])
        kc_lo = float(kc_lower.iloc[-1])

        if any(pd.isna(v) for v in [bb_up, bb_lo, kc_up, kc_lo]):
            return False

        return bb_up < kc_up and bb_lo > kc_lo

    squeeze.__name__ = f'Squeeze(BB{bb_period}/{bb_std}σ,KC{kc_ema_period}/{kc_atr_mult}x)'
    return squeeze
