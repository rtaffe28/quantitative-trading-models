"""Single-stock LEAP strategy backtest with buy-and-hold comparison."""

import sys
sys.path.insert(0, "../..")

import pandas as pd
from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.leap_strategy import create_leap_strategy


# Configuration
ticker = "VOO"
strike_factor = 0.8
interest_rate = 0.05
days = 365

tickers = [ticker]
start = datetime(2016, 1, 1)
end = datetime(2026, 1, 1)
initial_cash = 10000


def run_leap():
    leap_strategy = create_leap_strategy(ticker, strike_factor, days, interest_rate)

    sim_leap = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=leap_strategy,
    )

    results_leap = sim_leap.run()

    print("\n" + "=" * 50)
    print("LEAP STRATEGY PERFORMANCE")
    print("=" * 50)
    sim_leap.print_performance_stats()

    transactions_leap = sim_leap.get_transactions()
    print(f"\nTotal transactions: {len(transactions_leap)}")
    print("\nTransaction History:")
    print(transactions_leap.to_string())

    sim_leap.plot_portfolio_history()


def run_buy_and_hold():
    def buy_and_hold(date, portfolio, market_data, actions):
        """Simple buy and hold strategy for comparison."""
        current_price = market_data["prices"][ticker]
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares == 0:
            max_shares = int(portfolio.cash / current_price)
            if max_shares > 0:
                actions.buy_stock(portfolio, ticker, max_shares, current_price)

    sim_bh = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=buy_and_hold,
    )

    results_bh = sim_bh.run()

    print("\n" + "=" * 50)
    print("BUY AND HOLD STRATEGY PERFORMANCE")
    print("=" * 50)
    sim_bh.print_performance_stats()

    transactions_bh = sim_bh.get_transactions()
    print(f"\nTotal transactions: {len(transactions_bh)}")

    sim_bh.plot_portfolio_history()


if __name__ == "__main__":
    run_leap()
    run_buy_and_hold()
