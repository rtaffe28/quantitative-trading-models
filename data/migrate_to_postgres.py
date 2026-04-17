"""
Migrate historical price data from DuckDB to PostgreSQL.

Usage:
    docker compose up -d
    python data/migrate_to_postgres.py

Reads from data/stocks.duckdb and bulk-loads into the Postgres
container using COPY for speed. Handles 21M+ rows in chunks.
"""

import io
import os

import duckdb
import psycopg2

DUCKDB_PATH = os.path.join(os.path.dirname(__file__), "stocks.duckdb")

PG_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://quant:quant@localhost:5432/stocks",
)

CHUNK_SIZE = 500_000


def migrate():
    src = duckdb.connect(DUCKDB_PATH, read_only=True)
    dst = psycopg2.connect(PG_DSN)
    cur = dst.cursor()

    total = src.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"Migrating {total:,} rows from DuckDB to PostgreSQL...")

    # Stream chunks from DuckDB and COPY into Postgres
    offset = 0
    migrated = 0

    while offset < total:
        rows = src.execute(
            "SELECT symbol, date, open, high, low, close, adjusted_close, "
            "volume, dividend_amount, split_coefficient "
            "FROM prices ORDER BY symbol, date "
            f"LIMIT {CHUNK_SIZE} OFFSET {offset}"
        ).fetchall()

        if not rows:
            break

        # Build a TSV buffer for COPY
        buf = io.StringIO()
        for row in rows:
            line = "\t".join(
                "\\N" if v is None else str(v) for v in row
            )
            buf.write(line + "\n")
        buf.seek(0)

        cur.copy_from(
            buf,
            "prices",
            columns=(
                "symbol", "date", "open", "high", "low", "close",
                "adjusted_close", "volume", "dividend_amount",
                "split_coefficient",
            ),
        )
        dst.commit()

        migrated += len(rows)
        pct = migrated / total * 100
        print(f"  {migrated:>12,} / {total:,}  ({pct:.1f}%)")

        offset += CHUNK_SIZE

    cur.close()
    dst.close()
    src.close()
    print("Done.")


if __name__ == "__main__":
    migrate()
