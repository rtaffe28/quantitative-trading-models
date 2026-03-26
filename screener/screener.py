"""
Stock Screener — run indicators across a universe of tickers.
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from indicators.base import Indicator
from screener.data_loader import MarketDataLoader


class StockScreener:
    """Evaluates indicators against a universe of stocks."""

    def __init__(self, data_loader: Optional[MarketDataLoader] = None):
        self.data_loader = data_loader or MarketDataLoader()

    def screen(
        self,
        indicator: Indicator,
        tickers: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_history_days: int = 200,
    ) -> List[str]:
        """
        Run an indicator against all tickers and return those where it fires True.

        Args:
            indicator: An Indicator callable (or composite)
            tickers: List of tickers to screen (default: full US universe)
            start: Start date for data (default: end - 2 years)
            end: End date / evaluation date (default: today)
            min_history_days: Skip tickers with fewer trading days

        Returns:
            Sorted list of tickers where the indicator returned True
        """
        if tickers is None:
            from screener.universe import get_us_stock_universe
            tickers = get_us_stock_universe()

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365 * 2)

        # Load data (uses any data already in the loader)
        all_data = self.data_loader.load_batch(tickers, start, end)

        matches = []
        errors = 0
        screened = 0

        for ticker, df in all_data.items():
            if len(df) < min_history_days:
                continue
            screened += 1
            try:
                if indicator(df):
                    matches.append(ticker)
            except Exception:
                errors += 1

        name = getattr(indicator, '__name__', 'indicator')
        print(f"[{name}] Screened {screened} tickers, "
              f"{len(matches)} matches, {errors} errors")
        return sorted(matches)

    def screen_detail(
        self,
        indicators: Dict[str, Indicator],
        tickers: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_history_days: int = 200,
    ) -> pd.DataFrame:
        """
        Run multiple named indicators and return a DataFrame showing
        which indicators fired for each ticker.

        Args:
            indicators: Dict mapping indicator name → Indicator callable
            tickers: List of tickers (default: full US universe)
            start: Start date
            end: End date
            min_history_days: Minimum data requirement

        Returns:
            DataFrame with tickers as index, indicator names as columns, bool values.
            Only includes tickers where at least one indicator fired.
        """
        if tickers is None:
            from screener.universe import get_us_stock_universe
            tickers = get_us_stock_universe()

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=365 * 2)

        all_data = self.data_loader.load_batch(tickers, start, end)

        rows = []
        for ticker, df in all_data.items():
            if len(df) < min_history_days:
                continue
            row = {'ticker': ticker}
            any_true = False
            for name, ind in indicators.items():
                try:
                    result = ind(df)
                    row[name] = result
                    if result:
                        any_true = True
                except Exception:
                    row[name] = False
            if any_true:
                rows.append(row)

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.set_index('ticker').sort_index()
        print(f"Screened {len(all_data)} tickers, "
              f"{len(result_df)} had at least one indicator fire")
        return result_df
