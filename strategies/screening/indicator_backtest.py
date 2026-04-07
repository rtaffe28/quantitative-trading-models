"""
Indicator Screening & Backtesting Framework

Demonstrates the full indicator framework:
1. Individual indicators — backtest a single indicator on one stock
2. Composite indicators — AND/OR multiple indicators together
3. Universe screening — find all stocks matching an indicator
4. Multi-ticker backtesting — trade a basket of screened stocks
"""

import sys
sys.path.insert(0, "../..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from indicators import (
    composite_and,
    composite_or,
    create_sma_trend_indicator,
    create_ema_crossover_indicator,
    create_rsi_indicator,
    create_macd_crossover_indicator,
    create_volatility_spike_indicator,
    create_bollinger_squeeze_indicator,
    create_volume_spike_indicator,
    create_mean_reversion_indicator,
)
from strategies.indicator_strategy import create_indicator_strategy, create_screener_strategy
from utils.simulation import BacktestSimulation
from screener import StockScreener


# Configuration
ticker = "AAPL"
start = datetime(2016, 1, 1)
end = datetime(2026, 1, 1)
initial_cash = 10000
demo_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]


def backtest_individual_indicators():
    """Test each indicator as a standalone buy/sell strategy on a single stock."""
    indicators = {
        "SMA Trend (200d)": create_sma_trend_indicator(period=200),
        "EMA Crossover (12/26)": create_ema_crossover_indicator(short_period=12, long_period=26),
        "RSI Oversold (14d, <30)": create_rsi_indicator(period=14, signal="oversold"),
        "MACD Crossover": create_macd_crossover_indicator(),
        "Mean Reversion (200d, 10%)": create_mean_reversion_indicator(period=200, deviation_pct=0.10),
    }

    results = {}

    for name, indicator in indicators.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print("=" * 60)

        strategy = create_indicator_strategy(ticker=ticker, indicator=indicator)

        sim = BacktestSimulation(
            tickers=[ticker],
            start_date=start,
            end_date=end,
            initial_cash=initial_cash,
            strategy_callback=strategy,
        )
        sim.run()
        sim.print_performance_stats()

        history = pd.DataFrame(sim.history)
        results[name] = history

    # Compare all indicators on a single chart
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, history in results.items():
        ax.plot(history["date"], history["total_value"], label=name, linewidth=1.5)

    ax.axhline(y=initial_cash, color="black", linestyle="--", alpha=0.4, label="Initial Capital")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(f"Indicator Strategy Comparison — {ticker}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    plt.tight_layout()
    plt.show()


def backtest_composite_bullish():
    """Composite: uptrend + momentum confirmation."""
    composite_bullish = composite_and(
        create_sma_trend_indicator(period=200),
        create_ema_crossover_indicator(12, 26),
        create_macd_crossover_indicator(),
    )

    print(f"Composite indicator: {composite_bullish.__name__}")

    strategy = create_indicator_strategy(ticker=ticker, indicator=composite_bullish)

    sim = BacktestSimulation(
        tickers=[ticker],
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy,
    )
    sim.run()
    sim.print_performance_stats()
    sim.plot_portfolio_history()


def backtest_composite_dip_buy():
    """Composite: mean reversion + oversold (contrarian buy signal)."""
    composite_dip_buy = composite_and(
        create_mean_reversion_indicator(period=200, deviation_pct=0.10),
        create_rsi_indicator(period=14, signal="oversold"),
        create_volume_spike_indicator(lookback=20, spike_factor=1.5),
    )

    print(f"Composite indicator: {composite_dip_buy.__name__}")

    strategy = create_indicator_strategy(ticker=ticker, indicator=composite_dip_buy)

    sim = BacktestSimulation(
        tickers=[ticker],
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=strategy,
    )
    sim.run()
    sim.print_performance_stats()
    sim.plot_portfolio_history()


def screen_universe():
    """Run indicators across a set of tickers to find matches."""
    screener = StockScreener()

    # Which Mag 7 stocks are currently in an uptrend?
    trend_indicator = create_sma_trend_indicator(period=200)
    uptrend_tickers = screener.screen(
        indicator=trend_indicator,
        tickers=demo_universe,
        min_history_days=200,
    )
    print(f"\nMag 7 stocks above 200-day SMA: {uptrend_tickers}")

    # Detailed screen: show which indicators fire for each ticker
    detail_indicators = {
        "Above 200d SMA": create_sma_trend_indicator(200),
        "EMA Bullish": create_ema_crossover_indicator(12, 26),
        "RSI Oversold": create_rsi_indicator(14, signal="oversold"),
        "MACD Bullish": create_macd_crossover_indicator(),
        "Vol Spike": create_volatility_spike_indicator(60, 1.5),
        "Volume Spike": create_volume_spike_indicator(20, 2.0),
    }

    detail_df = screener.screen_detail(
        indicators=detail_indicators,
        tickers=demo_universe,
        min_history_days=200,
    )
    print("\nDetailed Indicator Screen:")
    print(detail_df.to_string())


def backtest_screener_strategy():
    """Backtest trading on indicator signals across multiple tickers simultaneously."""
    multi_indicator = composite_and(
        create_sma_trend_indicator(200),
        create_macd_crossover_indicator(),
    )

    screener_strategy = create_screener_strategy(
        indicator=multi_indicator,
        max_positions=7,
    )

    sim = BacktestSimulation(
        tickers=demo_universe,
        start_date=start,
        end_date=end,
        initial_cash=initial_cash,
        strategy_callback=screener_strategy,
    )
    sim.run()
    sim.print_performance_stats()
    sim.plot_portfolio_history()

    # Transaction summary
    tx = sim.get_transactions()
    if not tx.empty:
        print(f"Total transactions: {len(tx)}")
        print(f"\nBy action:")
        print(tx["action"].value_counts())
        print(f"\nBy ticker:")
        print(tx["ticker"].value_counts())
    else:
        print("No transactions recorded")


if __name__ == "__main__":
    print("=" * 60)
    print("1. INDIVIDUAL INDICATOR BACKTESTS")
    print("=" * 60)
    backtest_individual_indicators()

    print("\n" + "=" * 60)
    print("2. COMPOSITE INDICATOR BACKTESTS")
    print("=" * 60)
    backtest_composite_bullish()
    backtest_composite_dip_buy()

    print("\n" + "=" * 60)
    print("3. UNIVERSE SCREENING")
    print("=" * 60)
    screen_universe()

    print("\n" + "=" * 60)
    print("4. MULTI-TICKER SCREENER BACKTEST")
    print("=" * 60)
    backtest_screener_strategy()
