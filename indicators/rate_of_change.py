"""
Rate of Change (ROC) indicator.

Measures the percentage change over a lookback period.
Used for momentum ranking — higher ROC = stronger momentum.
Unlike RSI which oscillates 0-100, ROC is unbounded and directly
comparable across assets for relative ranking.
"""

import pandas as pd
from indicators.base import Indicator


def create_roc_indicator(
    period: int = 63,
    threshold: float = 0.0,
) -> Indicator:
    """
    Returns True when the rate of change over the lookback period exceeds
    the threshold.

    Args:
        period: Lookback in trading days (63 ≈ 3 months, 126 ≈ 6 months)
        threshold: Minimum ROC to trigger (0.0 = positive return)
    """
    def roc_indicator(df: pd.DataFrame) -> bool:
        if len(df) < period + 1:
            return False
        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        current = float(prices.iloc[-1])
        past = float(prices.iloc[-period])

        if pd.isna(current) or pd.isna(past) or past == 0:
            return False
        roc = (current - past) / past
        return roc > threshold

    roc_indicator.__name__ = f'ROC({period}d,>{threshold:.1%})'
    return roc_indicator


def compute_roc(prices: pd.Series, period: int) -> float:
    """Compute the raw ROC value for ranking purposes."""
    if len(prices) < period + 1:
        return float('nan')
    current = float(prices.iloc[-1])
    past = float(prices.iloc[-period])
    if pd.isna(current) or pd.isna(past) or past == 0:
        return float('nan')
    return (current - past) / past
