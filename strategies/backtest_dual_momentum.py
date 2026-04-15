"""
Backtest: Dual Momentum Sector Rotation Strategy

Uses sector ETFs to test the dual momentum rotation concept.
Sector ETFs give clean exposure without single-stock risk.

Two variants tested:
  1. Sector ETFs — rotate between 11 GICS sectors
  2. Thematic — rotate between growth, value, momentum factor ETFs + sectors

Compared against buy & hold SPY.
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.dual_momentum_rotation_strategy import create_dual_momentum_rotation_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy


# 11 GICS sector ETFs
SECTOR_ETFS = [
    'XLK',   # Technology
    'XLV',   # Healthcare
    'XLF',   # Financials
    'XLY',   # Consumer Discretionary
    'XLP',   # Consumer Staples
    'XLE',   # Energy
    'XLI',   # Industrials
    'XLB',   # Materials
    'XLRE',  # Real Estate
    'XLC',   # Communication Services
    'XLU',   # Utilities
]

# Thematic / factor mix for the second variant
THEMATIC_UNIVERSE = [
    # Sector leaders
    'XLK', 'XLV', 'XLE', 'XLI', 'XLF', 'XLC',
    # Individual high-momentum names (large, liquid)
    'NVDA', 'LLY', 'COST', 'META', 'GE',
    'JPM', 'CAT', 'NFLX', 'AVGO', 'AMZN',
]

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_sector_rotation():
    print("=" * 60)
    print("VARIANT 1: SECTOR ETF ROTATION (top 3 of 11 sectors)")
    print("=" * 60)
    print(f"Universe: {', '.join(SECTOR_ETFS)}")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Rules:")
    print("  Lookback: blended 3-month + 6-month momentum")
    print("  Filter: absolute momentum > 0 (positive returns only)")
    print("  Select: top 3 sectors, equal weight, monthly rebalance")
    print("  Risk-off: 100% cash when no sector has positive momentum")
    print()

    strategy = create_dual_momentum_rotation_strategy(
        tickers=SECTOR_ETFS,
        short_lookback=63,
        long_lookback=126,
        top_n=3,
        rebalance_frequency=21,
        absolute_threshold=0.0,
    )

    sim = BacktestSimulation(
        tickers=SECTOR_ETFS,
        start_date=START,
        end_date=END,
        initial_cash=INITIAL_CASH,
        strategy_callback=strategy,
    )

    results = sim.run()
    sim.print_performance_stats()
    _print_trade_summary(sim)
    return sim, results


def run_thematic_rotation():
    print("=" * 60)
    print("VARIANT 2: THEMATIC ROTATION (top 4 of 16 mixed assets)")
    print("=" * 60)
    print(f"Universe: {', '.join(THEMATIC_UNIVERSE)}")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Rules:")
    print("  Same momentum scoring, top 4, monthly rebalance")
    print()

    strategy = create_dual_momentum_rotation_strategy(
        tickers=THEMATIC_UNIVERSE,
        short_lookback=63,
        long_lookback=126,
        top_n=4,
        rebalance_frequency=21,
        absolute_threshold=0.0,
    )

    sim = BacktestSimulation(
        tickers=THEMATIC_UNIVERSE,
        start_date=START,
        end_date=END,
        initial_cash=INITIAL_CASH,
        strategy_callback=strategy,
    )

    results = sim.run()
    sim.print_performance_stats()
    _print_trade_summary(sim)
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


def _print_trade_summary(sim):
    txns = sim.get_transactions()
    if not txns.empty:
        buys = txns[txns['action'] == 'BUY_STOCK']
        sells = txns[txns['action'] == 'SELL_STOCK']
        print(f"\nTrade count: {len(buys)} buys, {len(sells)} sells")
        if not buys.empty:
            tickers_traded = buys['ticker'].unique()
            print(f"Tickers rotated through: {', '.join(sorted(tickers_traded))}")
    print()


def compare(sims: dict):
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<30} {'Final Value':>14} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 70)

    for name, sim in sims.items():
        final = sim.history[-1]['total_value']
        ret = (final / INITIAL_CASH - 1) * 100
        import numpy as np
        import pandas as pd
        df = pd.DataFrame(sim.history)
        returns = df['total_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        maxdd = ((df['total_value'].cummax() - df['total_value']) / df['total_value'].cummax()).max() * 100
        print(f"{name:<30} ${final:>12,.2f} {ret:>+9.2f}% {sharpe:>7.2f} {maxdd:>7.2f}%")

    print()
    # Flag profitability
    spy_return = (sims['SPY Buy & Hold'].history[-1]['total_value'] / INITIAL_CASH - 1) * 100
    for name, sim in sims.items():
        if name == 'SPY Buy & Hold':
            continue
        ret = (sim.history[-1]['total_value'] / INITIAL_CASH - 1) * 100
        alpha = ret - spy_return
        if alpha > 5:
            print(f">>> {name}: POTENTIALLY PROFITABLE (+{alpha:.1f}% alpha)")
        elif alpha > 0:
            print(f">>> {name}: MARGINAL ({alpha:+.1f}% alpha)")
        else:
            print(f">>> {name}: UNDERPERFORMS ({alpha:+.1f}% vs SPY)")


if __name__ == '__main__':
    s1, _ = run_sector_rotation()
    s2, _ = run_thematic_rotation()
    bench, _ = run_benchmark()

    compare({
        'Sector Rotation (ETFs)': s1,
        'Thematic Rotation (Mixed)': s2,
        'SPY Buy & Hold': bench,
    })
