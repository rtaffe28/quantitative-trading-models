"""
Batch market data loader with threading support for screening thousands of tickers.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """Downloads and caches OHLCV data for large ticker universes."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self._data: Dict[str, pd.DataFrame] = {}

    @property
    def data(self) -> Dict[str, pd.DataFrame]:
        """Access all loaded data."""
        return self._data

    def load_batch(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        batch_size: int = 100,
        max_workers: int = 4,
        delay_between_batches: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for many tickers in parallel batches.

        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            batch_size: Tickers per yfinance download call
            max_workers: Concurrent download threads
            delay_between_batches: Seconds between batch submissions

        Returns:
            Dict mapping ticker → OHLCV DataFrame
        """
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        total = len(batches)
        completed = 0

        print(f"Loading data for {len(tickers)} tickers in {total} batches "
              f"({batch_size}/batch, {max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, batch in enumerate(batches):
                future = executor.submit(self._download_batch, batch, start, end)
                futures[future] = i
                if i < total - 1:
                    time.sleep(delay_between_batches)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    self._data.update(result)
                except Exception:
                    pass
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} batches, "
                          f"{len(self._data)} tickers loaded")

        print(f"Loaded {len(self._data)} tickers successfully")
        return self._data

    def load_single(self, ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """Load data for a single ticker."""
        if ticker in self._data:
            return self._data[ticker]
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                # Handle MultiIndex columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(level=1, axis=1)
                self._data[ticker] = df
                return df
        except Exception:
            pass
        return None

    def _download_batch(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Download a batch of tickers via yf.download(list)."""
        result = {}
        try:
            df = yf.download(tickers, start=start, end=end, progress=False, threads=True)
            if df.empty:
                return result

            # yfinance returns MultiIndex columns (metric, ticker) for multiple tickers
            if isinstance(df.columns, pd.MultiIndex):
                available_tickers = df.columns.get_level_values(1).unique()
                for ticker in available_tickers:
                    try:
                        ticker_df = df.xs(ticker, level=1, axis=1).copy()
                        if not ticker_df.empty and not ticker_df['Close'].isna().all():
                            result[ticker] = ticker_df
                    except (KeyError, TypeError):
                        continue
            else:
                # Single ticker returned — columns are just metric names
                if len(tickers) == 1 and not df.empty:
                    result[tickers[0]] = df
        except Exception:
            # Fall back to individual downloads for this batch
            for ticker in tickers:
                try:
                    single = yf.download(ticker, start=start, end=end, progress=False)
                    if not single.empty:
                        if isinstance(single.columns, pd.MultiIndex):
                            single = single.droplevel(level=1, axis=1)
                        result[ticker] = single
                except Exception:
                    continue
        return result
