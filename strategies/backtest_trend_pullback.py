"""
Backtest: Trend Pullback Momentum Strategy

Runs the trend pullback strategy against a diversified universe of liquid
large-caps and compares performance to buy-and-hold SPY.

Universe: 30 stocks across sectors — tech, healthcare, financials, consumer,
industrials, energy, communications. This avoids sector concentration bias
and tests whether the signal generalizes.
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.trend_pullback_strategy import create_trend_pullback_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy


# Diversified large-cap universe across sectors
UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'GS',
    # Consumer
    'AMZN', 'COST', 'WMT', 'PG', 'KO',
    # Industrials
    'CAT', 'HON', 'UNP', 'GE', 'DE',
    # Energy / Communications
    'XOM', 'CVX', 'GOOGL', 'META', 'NFLX',
]

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_strategy_backtest():
    print("=" * 60)
    print("TREND PULLBACK MOMENTUM STRATEGY")
    print("=" * 60)
    print(f"Universe: {len(UNIVERSE)} stocks across 6 sectors")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Rules:")
    print("  Entry: Price > 150 SMA AND ADX > 20 AND RSI < 45")
    print("  Exit:  RSI > 75 OR Price < 150 SMA")
    print("  Max positions: 8 (equal weight)")
    print()

    strategy = create_trend_pullback_strategy(
        tickers=UNIVERSE,
        sma_period=150,
        adx_period=14,
        adx_threshold=20.0,
        rsi_period=14,
        rsi_entry=45.0,
        rsi_exit=75.0,
        max_positions=8,
    )

    sim = BacktestSimulation(
        tickers=UNIVERSE,
        start_date=START,
        end_date=END,
        initial_cash=INITIAL_CASH,
        strategy_callback=strategy,
    )

    results = sim.run()
    sim.print_performance_stats()

    txns = sim.get_transactions()
    if not txns.empty:
        buys = txns[txns['action'] == 'BUY_STOCK']
        sells = txns[txns['action'] == 'SELL_STOCK']
        print(f"\nTrade count: {len(buys)} buys, {len(sells)} sells")

        if not buys.empty:
            tickers_traded = buys['ticker'].unique()
            print(f"Tickers traded: {len(tickers_traded)} — {', '.join(sorted(tickers_traded))}")

    print()

    return sim, results


def run_benchmark():
    print("=" * 60)
    print("BENCHMARK: BUY & HOLD SPY")
    print("=" * 60)

    benchmark = BacktestSimulation(
        tickers=['SPY'],
        start_date=START,
        end_date=END,
        initial_cash=INITIAL_CASH,
        strategy_callback=create_buy_and_hold_strategy('SPY'),
    )
    bench_results = benchmark.run()
    benchmark.print_performance_stats()
    print()
    return benchmark, bench_results


def compare(strategy_sim, bench_sim):
    strat_final = strategy_sim.history[-1]['total_value']
    bench_final = bench_sim.history[-1]['total_value']

    strat_return = (strat_final / INITIAL_CASH - 1) * 100
    bench_return = (bench_final / INITIAL_CASH - 1) * 100
    alpha = strat_return - bench_return

    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Strategy final value:  ${strat_final:>12,.2f}  ({strat_return:+.2f}%)")
    print(f"SPY final value:       ${bench_final:>12,.2f}  ({bench_return:+.2f}%)")
    print(f"Alpha vs SPY:          {alpha:>12.2f}%")
    print()

    if alpha > 5:
        print(">>> POTENTIALLY PROFITABLE — strategy materially outperforms SPY.")
    elif alpha > 0:
        print(">>> MARGINAL — slight outperformance, may not survive transaction costs.")
    else:
        print(">>> UNDERPERFORMS SPY over this period.")


if __name__ == '__main__':
    strat_sim, strat_results = run_strategy_backtest()
    bench_sim, bench_results = run_benchmark()
    compare(strat_sim, bench_sim)
