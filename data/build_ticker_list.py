"""
Build a comprehensive list of US common stock tickers.

Sources: NASDAQ FTP (with ETF flag) + SEC EDGAR.
Excludes: ETFs, index funds, mutual funds, preferred shares, warrants, OTC.

Writes to data/tickers.json. Commit this file to the repo so seed.py
doesn't depend on external APIs to know what to download.

Usage:
    python data/build_ticker_list.py
"""

import json
import os
import urllib.request

DATA_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(DATA_DIR, "tickers.json")


def fetch_nasdaq_ftp() -> tuple[set[str], set[str]]:
    """Returns (stocks, etfs) from NASDAQ FTP."""
    stocks = set()
    etfs = set()

    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Python/3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")

            lines = text.strip().split("\n")
            header = lines[0].split("|")
            etf_idx = header.index("ETF") if "ETF" in header else None

            for line in lines[1:]:
                if line.startswith("File Creation Time"):
                    continue
                fields = line.split("|")
                if len(fields) < 2:
                    continue

                symbol = fields[0].strip()
                if not symbol or " " in symbol:
                    continue

                is_etf = (
                    etf_idx is not None
                    and etf_idx < len(fields)
                    and fields[etf_idx].strip() == "Y"
                )

                if is_etf:
                    etfs.add(symbol)
                else:
                    stocks.add(symbol)
        except Exception as e:
            print(f"  Warning: failed to fetch {url}: {e}")

    return stocks, etfs


def fetch_sec_edgar() -> set[str]:
    """Fetch tickers from SEC EDGAR, excluding OTC."""
    # Use local copy if available
    local_path = os.path.join(DATA_DIR, "sec_tickers.json")
    if os.path.exists(local_path):
        with open(local_path) as f:
            data = json.load(f)
    else:
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(
            url, headers={"User-Agent": "QuantTradingModels research@example.com"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(local_path, "w") as f:
            json.dump(data, f)

    tickers = set()
    for entry in data.values():
        ticker = entry.get("ticker", "").strip()
        exchange = entry.get("exchange", "")
        if exchange == "OTC":
            continue
        if ticker:
            tickers.add(ticker)
    return tickers


def is_excluded(symbol: str, known_etfs: set[str]) -> bool:
    """Filter out non-common-stock symbols."""
    if symbol in known_etfs:
        return True
    if "-P" in symbol:
        return True
    for suffix in ["-WT", "-WS", "-UN", "-RT"]:
        if symbol.endswith(suffix):
            return True
    return False


def main():
    print("Fetching ticker lists...")

    nasdaq_stocks, known_etfs = fetch_nasdaq_ftp()
    print(f"  NASDAQ FTP: {len(nasdaq_stocks)} stocks, {len(known_etfs)} ETFs")

    sec_tickers = fetch_sec_edgar()
    print(f"  SEC EDGAR: {len(sec_tickers)} tickers (excl. OTC)")

    all_tickers = nasdaq_stocks | sec_tickers
    filtered = sorted(t for t in all_tickers if not is_excluded(t, known_etfs))

    print(f"  Combined: {len(filtered)} tickers after filtering")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"  Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
