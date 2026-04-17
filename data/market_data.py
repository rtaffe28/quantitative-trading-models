"""
Pre-built queries for common market data access patterns.

All methods return DataFrames. For custom queries, use data.db.query() directly.
"""

from datetime import date, datetime
from typing import List, Optional, Union

import pandas as pd

from data.db import query

DateLike = Union[str, date, datetime]


def get_ticker(symbol: str, start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
    """OHLCV history for a single ticker."""
    sql = "SELECT * FROM prices WHERE symbol = %s"
    params = [symbol]
    if start:
        sql += " AND date >= %s"
        params.append(start)
    if end:
        sql += " AND date <= %s"
        params.append(end)
    sql += " ORDER BY date"
    return query(sql, tuple(params))


def get_tickers(symbols: List[str], start: DateLike = None, end: DateLike = None) -> pd.DataFrame:
    """OHLCV history for multiple tickers."""
    sql = "SELECT * FROM prices WHERE symbol = ANY(%s)"
    params = [symbols]
    if start:
        sql += " AND date >= %s"
        params.append(start)
    if end:
        sql += " AND date <= %s"
        params.append(end)
    sql += " ORDER BY symbol, date"
    return query(sql, tuple(params))


def symbols_on_date(target_date: DateLike) -> List[str]:
    """All ticker symbols that have data on a given date."""
    df = query(
        "SELECT DISTINCT symbol FROM prices WHERE date = %s ORDER BY symbol",
        (target_date,),
    )
    return df["symbol"].tolist()


def universe_on_date(target_date: DateLike, lookback_days: int = 0) -> pd.DataFrame:
    """
    All stocks on a given date (nearest prior trading day).

    With lookback_days > 0, includes N prior trading days of history
    per symbol — useful for indicators that need lookback.
    """
    if lookback_days > 0:
        sql = """
            WITH date_bounds AS (
                SELECT DISTINCT date FROM prices
                WHERE date <= %s
                ORDER BY date DESC
                LIMIT %s
            )
            SELECT p.* FROM prices p
            WHERE p.date IN (SELECT date FROM date_bounds)
            ORDER BY p.symbol, p.date
        """
        return query(sql, (target_date, lookback_days + 1))
    else:
        sql = """
            WITH target AS (
                SELECT MAX(date) AS d FROM prices WHERE date <= %s
            )
            SELECT p.* FROM prices p, target t
            WHERE p.date = t.d
            ORDER BY p.symbol
        """
        return query(sql, (target_date,))


def universe_history(start: DateLike, end: DateLike, min_rows: int = 0) -> pd.DataFrame:
    """Full OHLCV history for every symbol active in the date range."""
    if min_rows > 0:
        sql = """
            SELECT p.* FROM prices p
            JOIN (
                SELECT symbol FROM prices
                WHERE date BETWEEN %s AND %s
                GROUP BY symbol HAVING COUNT(*) >= %s
            ) active ON p.symbol = active.symbol
            WHERE p.date BETWEEN %s AND %s
            ORDER BY p.symbol, p.date
        """
        return query(sql, (start, end, min_rows, start, end))
    else:
        sql = """
            SELECT * FROM prices
            WHERE date BETWEEN %s AND %s
            ORDER BY symbol, date
        """
        return query(sql, (start, end))


def top_movers(target_date: DateLike, limit: int = 20) -> pd.DataFrame:
    """Biggest single-day percent movers on a given date."""
    sql = """
        SELECT symbol, open, close, volume,
               (close - open) / NULLIF(open, 0) * 100 AS pct_change
        FROM prices WHERE date = %s
        ORDER BY ABS((close - open) / NULLIF(open, 0)) DESC
        LIMIT %s
    """
    return query(sql, (target_date, limit))


def date_range() -> tuple:
    """(min_date, max_date) across all data."""
    df = query("SELECT MIN(date) AS min_d, MAX(date) AS max_d FROM prices")
    return df["min_d"].iloc[0], df["max_d"].iloc[0]


def list_symbols() -> List[str]:
    """All ticker symbols in the database."""
    df = query("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
    return df["symbol"].tolist()


def symbol_count() -> int:
    df = query("SELECT COUNT(DISTINCT symbol) AS n FROM prices")
    return df["n"].iloc[0]


def row_count() -> int:
    df = query("SELECT COUNT(*) AS n FROM prices")
    return df["n"].iloc[0]
