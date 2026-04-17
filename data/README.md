# Market Data

Historical stock price data stored in PostgreSQL.

## Schema

The `prices` table is defined in [`init.sql`](init.sql) and created automatically when the Postgres container starts for the first time.

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR(20) | Ticker symbol (PK) |
| date | DATE | Trading date (PK) |
| open | DOUBLE PRECISION | Opening price |
| high | DOUBLE PRECISION | Daily high |
| low | DOUBLE PRECISION | Daily low |
| close | DOUBLE PRECISION | Closing price |
| adjusted_close | DOUBLE PRECISION | Split/dividend adjusted close |
| volume | BIGINT | Shares traded |
| dividend_amount | DOUBLE PRECISION | Dividend paid on date |
| split_coefficient | DOUBLE PRECISION | Split ratio (1.0 = no split) |

## Setup

### 1. Start PostgreSQL

```bash
docker compose up -d
```

This starts a Postgres 17 container with the `prices` table already created. Data persists in a Docker volume (`pgdata`).

### 2. Seed the database

The database starts empty. You have a few options for loading data:

**Option A: Download from yfinance (recommended for fresh setup)**

Two scripts, run in order:

```bash
# 1. Build the ticker list (only needed once, or to refresh)
python data/build_ticker_list.py

# 2. Download price data and load into Postgres
python data/seed.py
```

`build_ticker_list.py` fetches ~8,000+ US common stock tickers from NASDAQ FTP and SEC EDGAR (excludes ETFs, index funds, mutual funds, OTC) and writes them to `data/tickers.json` (committed to the repo).

`seed.py` reads `tickers.json` and downloads full OHLCV history from yfinance. It's resumable — re-run to pick up where you left off. Failed tickers are logged to `data/failed_tickers.json`.

**Updating with new data:** just run `seed.py` again. It checks the last date per ticker in the DB and only fetches new days. You can also update specific tickers:

```bash
python data/seed.py AAPL MSFT NVDA
```

**Option B: Migrate from DuckDB (if you have `stocks.duckdb`)**

If you have the original DuckDB file, install duckdb temporarily and run the migration script:

```bash
pip install duckdb
python data/migrate_to_postgres.py
pip uninstall duckdb
```

**Option C: Load from a pg_dump**

If someone has shared a database dump:

```bash
pg_restore -h localhost -U quant -d stocks dump.sql
```

**Option D: Load from CSV**

Place a CSV with columns matching the schema above and load it:

```bash
psql -h localhost -U quant -d stocks -c "\copy prices FROM 'data.csv' WITH (FORMAT csv, HEADER true)"
```

### 3. Verify

```bash
psql -h localhost -U quant -d stocks -c "SELECT COUNT(*) FROM prices;"
```

Or from Python:

```python
from data import market_data
print(market_data.symbol_count())
print(market_data.date_range())
```

## Connection

Default: `postgresql://quant:quant@localhost:5432/stocks`

Override with the `DATABASE_URL` environment variable:

```bash
export DATABASE_URL=postgresql://quant:quant@192.168.1.50:5432/stocks
```

## Usage

```python
from data import query, market_data

# Raw SQL
df = query("SELECT * FROM prices WHERE symbol = %s AND date >= %s", ("AAPL", "2022-01-01"))

# Pre-built queries
df = market_data.get_ticker("AAPL", start="2022-01-01")
df = market_data.get_tickers(["AAPL", "MSFT", "NVDA"])
df = market_data.top_movers("2023-03-15")
tickers = market_data.symbols_on_date("2023-03-15")
```
