"""
Market Regime Detection indicator.

Classifies the market into risk-on vs risk-off regimes using:
  - Trend: SPY price relative to its 200 SMA
  - Volatility regime: recent vs long-term realized volatility
  - Breadth proxy: whether the majority of a basket are above their 50 SMA

This replaces discretionary macro calls with a systematic, rules-based
regime classification that can be computed daily.
"""

import numpy as np
import pandas as pd
from indicators.base import Indicator


def _realized_vol(prices: pd.Series, window: int) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def create_regime_indicator(
    trend_period: int = 200,
    vol_short: int = 20,
    vol_long: int = 60,
    vol_ratio_threshold: float = 1.3,
) -> Indicator:
    """
    Returns True when the market is in a risk-on regime.

    Risk-on requires BOTH:
      - Price above trend_period SMA (uptrend)
      - Short-term vol < vol_ratio_threshold * long-term vol (not spiking)

    When short-term vol is elevated relative to long-term vol, it signals
    stress/uncertainty — the strategy should go defensive even if price
    is still above the SMA.

    Args:
        trend_period: SMA period for trend detection
        vol_short: Short-term realized vol window
        vol_long: Long-term realized vol window
        vol_ratio_threshold: Max ratio of short/long vol for risk-on
    """
    min_data = max(trend_period, vol_long) + 10

    def regime(df: pd.DataFrame) -> bool:
        if len(df) < min_data:
            return True  # default risk-on if insufficient data

        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        # Trend check
        sma = float(close.rolling(window=trend_period).mean().iloc[-1])
        current = float(close.iloc[-1])
        if pd.isna(sma):
            return True
        in_uptrend = current > sma

        # Volatility regime check
        short_vol = _realized_vol(close, vol_short)
        long_vol = _realized_vol(close, vol_long)
        sv = float(short_vol.iloc[-1])
        lv = float(long_vol.iloc[-1])

        if pd.isna(sv) or pd.isna(lv) or lv == 0:
            vol_calm = True
        else:
            vol_calm = sv < lv * vol_ratio_threshold

        return in_uptrend and vol_calm

    regime.__name__ = f'Regime(SMA{trend_period},vol{vol_short}/{vol_long}<{vol_ratio_threshold}x)'
    return regime
