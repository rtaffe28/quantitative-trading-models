"""
Moving Average Strategy Backtest

Tests SMA crossover, EMA crossover, triple MA, and adaptive MA strategies
on a single ticker vs buy-and-hold.
"""

import sys
sys.path.insert(0, "../..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

from utils.simulation import BacktestSimulation
from strategies.moving_average_strategy import (
    create_sma_crossover_strategy,
    create_ema_crossover_strategy,
    create_triple_ma_strategy,
    create_adaptive_ma_strategy,
)


# Configuration
ticker = "MSTR"
tickers = [ticker]
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
initial_cash = 100000


def run_strategy(name, strategy_callback):
    sim = BacktestSimulation(
        tickers=tickers,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy_callback,
    )

    results = sim.run()

    print(f"\n{'='*50}")
    print(f"{name} PERFORMANCE")
    print(f"{'='*50}")
    sim.print_performance_stats()

    transactions = sim.get_transactions()
    print(f"\nTotal transactions: {len(transactions)}")
    if len(transactions) > 0:
        print("\nTransaction History:")
        print(transactions.to_string())

    sim.plot_portfolio_history()
    return sim, results, transactions


def run_buy_and_hold():
    def buy_and_hold(date, portfolio, market_data, actions):
        current_price = market_data["prices"][ticker]
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares == 0:
            max_shares = int(portfolio.cash / current_price)
            if max_shares > 0:
                actions.buy_stock(portfolio, ticker, max_shares, current_price)

    return run_strategy("BUY AND HOLD", buy_and_hold)


def run_comparison(results_dict):
    """Comprehensive comparison of all strategies."""
    names = list(results_dict.keys())
    sims = [results_dict[n][0] for n in names]
    txs = [results_dict[n][2] for n in names]

    histories = [pd.DataFrame(s.history) for s in sims]
    final_values = [h["total_value"].iloc[-1] for h in histories]

    comparison_data = {
        "Strategy": names,
        "Final Value": final_values,
        "Total Return (%)": [((v / initial_cash) - 1) * 100 for v in final_values],
        "Transactions": [len(t) for t in txs],
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{'='*70}")
    print("MOVING AVERAGE STRATEGIES COMPARISON")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#06A77D"]

    # 1. Portfolio values over time
    for h, name, color in zip(histories, names, colors):
        linestyle = "--" if name == "Buy & Hold" else "-"
        axes[0, 0].plot(h["date"], h["total_value"], label=name, linewidth=2, alpha=0.8, color=color, linestyle=linestyle)
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Portfolio Value ($)")
    axes[0, 0].set_title("Portfolio Value Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Final returns comparison
    returns = comparison_df["Total Return (%)"]
    bars = axes[0, 1].bar(range(len(names)), returns, color=colors)
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Total Return (%)")
    axes[0, 1].set_title("Final Returns Comparison")
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    axes[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
        )

    # 3. Transaction count comparison
    axes[1, 0].bar(range(len(names)), comparison_df["Transactions"], color=colors)
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Number of Transactions")
    axes[1, 0].set_title("Transaction Activity")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # 4. Drawdown analysis
    for h, name, color in zip(histories, names, colors):
        rolling_max = h["total_value"].expanding().max()
        drawdown = (h["total_value"] - rolling_max) / rolling_max * 100
        axes[1, 1].plot(h["date"], drawdown, label=name, linewidth=2, alpha=0.8, color=color)

    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Drawdown (%)")
    axes[1, 1].set_title("Drawdown Analysis")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Detailed performance metrics
    print(f"\n{'='*70}")
    print("DETAILED PERFORMANCE METRICS")
    print(f"{'='*70}")

    for h, name in zip(histories, names):
        rolling_max = h["total_value"].expanding().max()
        drawdown = (h["total_value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        daily_returns = h["total_value"].pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

        print(f"\n{name}:")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Volatility: {daily_returns.std() * np.sqrt(252) * 100:.2f}%")


def plot_sma_signals(sim_sma, transactions_sma):
    """Visualize SMA moving averages and buy/sell signals."""
    price_data = yf.download(ticker, start=start, end=end, progress=False)

    price_data["SMA_50"] = price_data["Close"].rolling(window=50).mean()
    price_data["SMA_200"] = price_data["Close"].rolling(window=200).mean()

    buy_dates, sell_dates = [], []
    buy_prices, sell_prices = [], []

    if len(transactions_sma) > 0:
        for _, row in transactions_sma.iterrows():
            if row["action"] == "BUY_STOCK":
                buy_dates.append(row["date"])
                buy_prices.append(row["price"])
            elif row["action"] == "SELL_STOCK":
                sell_dates.append(row["date"])
                sell_prices.append(row["price"])

    history = pd.DataFrame(sim_sma.history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax1.plot(price_data.index, price_data["Close"], label="Price", linewidth=2, color="black", alpha=0.7)
    ax1.plot(price_data.index, price_data["SMA_50"], label="50-day SMA", linewidth=2, color="blue", alpha=0.7)
    ax1.plot(price_data.index, price_data["SMA_200"], label="200-day SMA", linewidth=2, color="red", alpha=0.7)

    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, color="green", marker="^", s=200, label="Buy Signal", zorder=5)
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, color="red", marker="v", s=200, label="Sell Signal", zorder=5)

    ax1.set_ylabel("Price ($)")
    ax1.set_title(f"{ticker} Price with SMA 50/200 Crossover Signals")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["date"], history["total_value"], linewidth=2, color="#2E86AB")
    ax2.fill_between(
        history["date"], history["total_value"], initial_cash,
        where=(history["total_value"] >= initial_cash), color="green", alpha=0.2, label="Profit",
    )
    ax2.fill_between(
        history["date"], history["total_value"], initial_cash,
        where=(history["total_value"] < initial_cash), color="red", alpha=0.2, label="Loss",
    )
    ax2.axhline(y=initial_cash, color="black", linestyle="--", linewidth=1, label="Initial Capital")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.set_title("Portfolio Value Over Time")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*50}")
    print("SIGNAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total Buy Signals: {len(buy_dates)}")
    print(f"Total Sell Signals: {len(sell_dates)}")
    if buy_dates:
        print(f"\nBuy Signal Dates: {[d.strftime('%Y-%m-%d') for d in buy_dates]}")
    if sell_dates:
        print(f"Sell Signal Dates: {[d.strftime('%Y-%m-%d') for d in sell_dates]}")


if __name__ == "__main__":
    results = {}

    # SMA Crossover (50/200)
    sma_strategy = create_sma_crossover_strategy(ticker=ticker, short_window=50, long_window=200)
    results["SMA 50/200"] = run_strategy("SMA CROSSOVER (50/200)", sma_strategy)

    # EMA Crossover (12/26)
    ema_strategy = create_ema_crossover_strategy(ticker=ticker, short_window=12, long_window=26)
    results["EMA 12/26"] = run_strategy("EMA CROSSOVER (12/26)", ema_strategy)

    # Triple MA (10/50/200)
    triple_strategy = create_triple_ma_strategy(ticker=ticker, fast_window=10, medium_window=50, slow_window=200)
    results["Triple MA"] = run_strategy("TRIPLE MA (10/50/200)", triple_strategy)

    # Adaptive MA (20/50)
    adaptive_strategy = create_adaptive_ma_strategy(ticker=ticker, short_window=20, long_window=50, volatility_threshold=0.02)
    results["Adaptive MA"] = run_strategy("ADAPTIVE MA (20/50)", adaptive_strategy)

    # Buy and Hold
    results["Buy & Hold"] = run_buy_and_hold()

    # Comparison
    run_comparison(results)

    # SMA signal visualization
    plot_sma_signals(results["SMA 50/200"][0], results["SMA 50/200"][2])
