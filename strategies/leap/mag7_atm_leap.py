"""
Mag 7 ATM LEAP Strategy Backtest

Strategy:
- Tickers: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- Strike: At-the-money (strike factor = 1.0)
- Duration: 1-year LEAPs (365 days)
- Roll: When position gets within 90 days of expiration
- Allocation: Equal weight across all 7 names
- Period: 10 years (2016-2026)
"""

import sys
sys.path.insert(0, "../..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

from utils.simulation import BacktestSimulation
from strategies.mag7_leap_strategy import create_mag7_leap_strategy, MAG7_TICKERS


# Configuration
tickers = MAG7_TICKERS
start = datetime(2016, 1, 1)
end = datetime(2026, 1, 1)
initial_cash = 10000


def run_backtest():
    strategy = create_mag7_leap_strategy(
        tickers=tickers,
        strike_factor=1,
        days=365,
        interest_rate=0.05,
        roll_threshold=90,
    )

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy,
    )

    results_df = sim.run()

    # Performance summary
    sim.print_performance_stats()
    sim.plot_portfolio_history()

    # Transaction log
    tx = sim.get_transactions()
    if not tx.empty:
        print(f"Total transactions: {len(tx)}")
        print(f"\nTransaction breakdown:")
        print(tx["action"].value_counts())
        print(f"\nTransactions by ticker:")
        print(tx["ticker"].value_counts())
        print(f"\nFirst 20 transactions:")
        print(tx.head(20).to_string())

    return sim


def run_comparison(sim):
    """Compare LEAP strategy against equal-weight buy-and-hold of Mag 7."""
    benchmark_data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)
        benchmark_data[ticker] = df["Close"]
    breakpoint()
    benchmark_returns = pd.DataFrame()
    for ticker in tickers:
        prices = benchmark_data[ticker]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        benchmark_returns[ticker] = prices / prices.iloc[0]

    benchmark_returns["equal_weight"] = benchmark_returns.mean(axis=1)
    benchmark_returns["equal_weight_value"] = benchmark_returns["equal_weight"] * initial_cash

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 7))

    leap_df = pd.DataFrame(sim.history)
    ax.plot(leap_df["date"], leap_df["total_value"], label="Mag 7 ATM LEAP Strategy", linewidth=2)
    ax.plot(
        benchmark_returns.index,
        benchmark_returns["equal_weight_value"],
        label="Mag 7 Equal-Weight Buy & Hold",
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Mag 7 ATM LEAP Strategy vs Equal-Weight Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    plt.show()

    # Print benchmark stats
    bh_final = benchmark_returns["equal_weight_value"].iloc[-1]
    bh_return = (bh_final / initial_cash - 1) * 100
    leap_final = leap_df["total_value"].iloc[-1]
    leap_return = (leap_final / initial_cash - 1) * 100

    print(f"\n{'='*50}")
    print(f"LEAP Strategy Final Value:     ${leap_final:>12,.2f}  ({leap_return:>+.1f}%)")
    print(f"Buy & Hold Final Value:        ${bh_final:>12,.2f}  ({bh_return:>+.1f}%)")
    print(f"{'='*50}")


if __name__ == "__main__":
    sim = run_backtest()
    run_comparison(sim)
