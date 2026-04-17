"""
PostgreSQL market data store for backtesting and screening.

Thin wrapper around psycopg2 — write raw SQL, get DataFrames back.
"""

import os

import pandas as pd
import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quant:quant@localhost:5432/stocks",
)


def _connect():
    return psycopg2.connect(DATABASE_URL)


def query(sql: str, params: tuple = None) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    with _connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def execute(sql: str, params: tuple = None):
    """Run a SQL statement (INSERT, UPDATE, etc.)."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
