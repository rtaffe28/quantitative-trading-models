"""
Wheel Strategy Backtest

Tests standard, aggressive, and conservative wheel strategies against buy-and-hold.
"""

import sys
sys.path.insert(0, "../..")

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from utils.simulation import BacktestSimulation
from strategies.wheel_strategy import (
    create_wheel_strategy,
    create_aggressive_wheel_strategy,
    create_conservative_wheel_strategy,
)


# Configuration
ticker = "MSTR"
put_strike_factor = 0.95
call_strike_factor = 1.05
days_to_expiration = 30
interest_rate = 0.05

tickers = [ticker]
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
initial_cash = 100000


def run_standard_wheel():
    """Standard Wheel: 5% OTM puts/calls, 30-day expiration."""
    wheel_strategy = create_wheel_strategy(
        ticker=ticker,
        put_strike_factor=put_strike_factor,
        call_strike_factor=call_strike_factor,
        days_to_expiration=days_to_expiration,
        interest_rate=interest_rate,
    )

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=wheel_strategy,
    )
    sim.run()

    print("\n" + "=" * 50)
    print("WHEEL STRATEGY PERFORMANCE")
    print("=" * 50)
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")
    print("\nTransaction History:")
    print(transactions.to_string())

    sim.plot_portfolio_history()
    return sim, transactions


def run_aggressive_wheel():
    """Aggressive Wheel: 2% OTM, 14-day expiration."""
    strategy = create_aggressive_wheel_strategy(ticker=ticker, interest_rate=interest_rate)

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy,
    )
    sim.run()

    print("\n" + "=" * 50)
    print("AGGRESSIVE WHEEL STRATEGY PERFORMANCE")
    print("=" * 50)
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")

    sim.plot_portfolio_history()
    return sim, transactions


def run_conservative_wheel():
    """Conservative Wheel: 10% OTM, 45-day expiration."""
    strategy = create_conservative_wheel_strategy(ticker=ticker, interest_rate=interest_rate)

    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy,
    )
    sim.run()

    print("\n" + "=" * 50)
    print("CONSERVATIVE WHEEL STRATEGY PERFORMANCE")
    print("=" * 50)
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")

    sim.plot_portfolio_history()
    return sim, transactions


def run_buy_and_hold():
    def buy_and_hold(date, portfolio, market_data, actions):
        current_price = market_data["prices"][ticker]
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares == 0:
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

    print("\n" + "=" * 50)
    print("BUY AND HOLD STRATEGY PERFORMANCE")
    print("=" * 50)
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")

    sim.plot_portfolio_history()
    return sim, transactions


def run_comparison(sim_wheel, sim_aggressive, sim_conservative, sim_bh,
                   tx_wheel, tx_aggressive, tx_conservative, tx_bh):
    """Compare all strategies side by side."""
    sims = [sim_wheel, sim_aggressive, sim_conservative, sim_bh]
    txs = [tx_wheel, tx_aggressive, tx_conservative, tx_bh]
    names = ["Standard Wheel", "Aggressive Wheel", "Conservative Wheel", "Buy & Hold"]

    history = [pd.DataFrame(s.history) for s in sims]
    final_values = [h["total_value"].iloc[-1] for h in history]

    comparison_data = {
        "Strategy": names,
        "Final Value": final_values,
        "Total Return (%)": [((v / initial_cash) - 1) * 100 for v in final_values],
        "Transactions": [len(t) for t in txs],
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for h, name in zip(history, names):
        linestyle = "--" if name == "Buy & Hold" else "-"
        axes[0].plot(h["date"], h["total_value"], label=name, linewidth=2, linestyle=linestyle)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Portfolio Value Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    axes[1].bar(comparison_df["Strategy"], comparison_df["Total Return (%)"], color=colors)
    axes[1].set_ylabel("Total Return (%)")
    axes[1].set_title("Final Returns Comparison")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def analyze_transactions(transactions):
    """Analyze transaction types for the standard wheel strategy."""
    if len(transactions) == 0:
        print("No transactions recorded for standard wheel strategy.")
        return

    action_counts = transactions["action"].value_counts()

    print("\n" + "=" * 50)
    print("TRANSACTION BREAKDOWN (Standard Wheel)")
    print("=" * 50)
    print(action_counts)

    put_premiums = (
        transactions[transactions["action"] == "SELL_PUT"]["total_premium"].sum()
        if "SELL_PUT" in action_counts
        else 0
    )
    call_premiums = (
        transactions[transactions["action"] == "SELL_CALL"]["total_premium"].sum()
        if "SELL_CALL" in action_counts
        else 0
    )

    print(f"\nTotal Put Premiums Collected: ${put_premiums:,.2f}")
    print(f"Total Call Premiums Collected: ${call_premiums:,.2f}")
    print(f"Total Option Premiums: ${put_premiums + call_premiums:,.2f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    action_counts.plot(kind="bar", ax=ax, color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"])
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    ax.set_title("Transaction Type Distribution (Standard Wheel)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim_w, tx_w = run_standard_wheel()
    sim_a, tx_a = run_aggressive_wheel()
    sim_c, tx_c = run_conservative_wheel()
    sim_bh, tx_bh = run_buy_and_hold()
    run_comparison(sim_w, sim_a, sim_c, sim_bh, tx_w, tx_a, tx_c, tx_bh)
    analyze_transactions(tx_w)
