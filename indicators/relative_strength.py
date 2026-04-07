"""
Relative Strength indicator — compares a stock's performance to a benchmark.

Unlike RSI (which is absolute momentum), relative strength measures whether
a stock is outperforming or underperforming a benchmark index over a lookback period.
"""

import pandas as pd
import yfinance as yf
from indicators.base import Indicator


def create_relative_strength_indicator(
    benchmark: str = 'SPY',
    lookback: int = 63,
) -> Indicator:
    """
    Returns True when the stock has outperformed the benchmark over the lookback period.

    Relative strength = stock return / benchmark return over the lookback window.
    Signal fires when the stock's return exceeds the benchmark's return.

    Args:
        benchmark: Benchmark ticker (default SPY for S&P 500)
        lookback: Number of trading days to compare (63 ≈ 3 months)
    """
    _cache = {}

    def _get_benchmark_data(start_date, end_date):
        key = (str(start_date.date()), str(end_date.date()))
        if key not in _cache:
            bm = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            if isinstance(bm.columns, pd.MultiIndex):
                bm.columns = bm.columns.get_level_values(0)
            _cache[key] = bm
        return _cache[key]

    def relative_strength(df: pd.DataFrame) -> bool:
        if len(df) < lookback + 1:
            return False

        prices = df['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        stock_return = float(prices.iloc[-1] / prices.iloc[-lookback] - 1)

        # Get benchmark data for the same period
        try:
            bm_data = _get_benchmark_data(df.index[0], df.index[-1])
            bm_prices = bm_data['Close']
            if isinstance(bm_prices, pd.DataFrame):
                bm_prices = bm_prices.squeeze()

            # Align to available dates
            bm_at_end = bm_prices.loc[:df.index[-1]]
            if len(bm_at_end) < lookback:
                return False
            bm_return = float(bm_at_end.iloc[-1] / bm_at_end.iloc[-lookback] - 1)
        except Exception:
            return False

        return stock_return > bm_return

    relative_strength.__name__ = f'RelStrength(vs {benchmark},{lookback}d)'
    return relative_strength
