"""Covered call strategy backtest with buy-and-hold comparison."""

import sys
sys.path.insert(0, "../..")

import pandas as pd
from datetime import datetime

from utils.simulation import BacktestSimulation
from strategies.covered_call_strategy import create_covered_call_strategy


# Configuration
ticker = "MSTR"
strike_factor = 1.06
time = 15 / 365

tickers = [ticker]
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
initial_cash = 100000


def run_covered_call():
    covered_call_strategy = create_covered_call_strategy(
        ticker=ticker,
        strike_factor=strike_factor,
        time_to_expiration=time,
    )

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=covered_call_strategy,
    )

    sim.run()

    print("Performance Stats")
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")
    print("\nTransaction History:")
    print(transactions.to_string())
    sim.plot_portfolio_history()


def run_buy_and_hold():
    def buy_and_hold(date, portfolio, market_data, actions):
        current_price = market_data["prices"][ticker]
        max_shares = int(portfolio.cash / current_price)
        if max_shares > 0:
            actions.buy_stock(portfolio, ticker, max_shares, current_price)

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=buy_and_hold,
    )

    sim.run()

    print("Performance Stats")
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")
    print("\nTransaction History:")
    print(transactions.to_string())
    sim.plot_portfolio_history()


if __name__ == "__main__":
    run_covered_call()
    run_buy_and_hold()
