"""
Indicator Protocol & Composition Utilities

An Indicator is any callable that takes an OHLCV DataFrame and returns True/False.
This module defines the type alias and provides AND/OR composition functions.
"""

from typing import Callable
import pandas as pd


# Core type: takes OHLCV DataFrame (same shape as market_data['data'][ticker]),
# returns True when the signal is active.
Indicator = Callable[[pd.DataFrame], bool]


def composite_and(*indicators: Indicator) -> Indicator:
    """Combine indicators with AND logic — all must return True."""
    def combined(df: pd.DataFrame) -> bool:
        return all(ind(df) for ind in indicators)
    combined.__name__ = ' AND '.join(getattr(i, '__name__', '?') for i in indicators)
    return combined


def composite_or(*indicators: Indicator) -> Indicator:
    """Combine indicators with OR logic — any must return True."""
    def combined(df: pd.DataFrame) -> bool:
        return any(ind(df) for ind in indicators)
    combined.__name__ = ' OR '.join(getattr(i, '__name__', '?') for i in indicators)
    return combined
