"""
Universe Provider — fetches all US common stock tickers (no ETFs/mutual funds).
"""

import os
import csv
import time
import pandas as pd
from typing import List, Optional
from io import StringIO


_CACHE_FILENAME = 'universe_cache.csv'
_CACHE_MAX_AGE_DAYS = 7


def get_us_stock_universe(cache_dir: Optional[str] = None) -> List[str]:
    """
    Returns a deduplicated, sorted list of US common stock tickers.
    Excludes ETFs, mutual funds, and test symbols.

    Sources tried in order:
      1. Local cache (if < 7 days old)
      2. NASDAQ FTP symbol directory
      3. SEC EDGAR company tickers JSON

    Args:
        cache_dir: Directory for caching. Defaults to ~/.quantitative-trading-models/
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.quantitative-trading-models')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, _CACHE_FILENAME)

    # Try cache first
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    # Try NASDAQ FTP
    tickers = _fetch_from_nasdaq_ftp()
    if tickers:
        _save_cache(cache_path, tickers)
        return tickers

    # Fallback: SEC EDGAR
    tickers = _fetch_from_sec_edgar()
    if tickers:
        _save_cache(cache_path, tickers)
        return tickers

    raise RuntimeError('Failed to fetch stock universe from any source')


def _load_cache(cache_path: str) -> Optional[List[str]]:
    """Load tickers from cache if it exists and is fresh."""
    if not os.path.exists(cache_path):
        return None
    age_days = (time.time() - os.path.getmtime(cache_path)) / 86400
    if age_days > _CACHE_MAX_AGE_DAYS:
        return None
    with open(cache_path, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers if tickers else None


def _save_cache(cache_path: str, tickers: List[str]):
    """Save tickers to cache file."""
    with open(cache_path, 'w') as f:
        f.write('\n'.join(tickers))


def _fetch_from_nasdaq_ftp() -> List[str]:
    """Download ticker lists from NASDAQ's public symbol directory."""
    import urllib.request

    tickers = set()

    urls = [
        'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
        'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt',
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/3'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8')
            lines = text.strip().split('\n')

            # Both files are pipe-delimited with a header row and a footer line
            header = lines[0].split('|')
            for line in lines[1:]:
                if line.startswith('File Creation Time'):
                    continue
                fields = line.split('|')
                if len(fields) < 2:
                    continue

                symbol = fields[0].strip()

                # Filter test symbols and special characters
                if not symbol or ' ' in symbol or '$' in symbol or '.' in symbol:
                    continue

                # nasdaqlisted.txt: column "ETF" (index 6) = 'Y' for ETFs
                if 'ETF' in header:
                    etf_idx = header.index('ETF')
                    if etf_idx < len(fields) and fields[etf_idx].strip() == 'Y':
                        continue

                # otherlisted.txt: column "ETF" (index 4) = 'Y' for ETFs
                if 'ETF' in header:
                    etf_idx = header.index('ETF')
                    if etf_idx < len(fields) and fields[etf_idx].strip() == 'Y':
                        continue

                tickers.add(symbol)
        except Exception:
            continue

    return sorted(tickers) if tickers else []


def _fetch_from_sec_edgar() -> List[str]:
    """Fallback: parse SEC EDGAR company tickers JSON."""
    import urllib.request
    import json

    url = 'https://www.sec.gov/files/company_tickers.json'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'QuantTradingModels research@example.com',
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        tickers = set()
        for entry in data.values():
            ticker = entry.get('ticker', '').strip()
            if ticker and ' ' not in ticker and '$' not in ticker and '.' not in ticker:
                tickers.add(ticker)

        return sorted(tickers)
    except Exception:
        return []
