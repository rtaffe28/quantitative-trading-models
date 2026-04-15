"""
Backtest: Adaptive Momentum with Regime Detection

Tests the always-invested adaptive approach that shifts between
offensive momentum picks and defensive holdings based on SPY regime.

Three variants:
  1. Pure offensive (no regime switch) — baseline for momentum-only
  2. Adaptive (offensive + defensive switch) — the core strategy
  3. Conservative adaptive (smaller offensive allocation)
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.adaptive_momentum_strategy import create_adaptive_momentum_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy
import pandas as pd
import numpy as np


# Offensive: broad large-cap growth + cyclical universe
# Deliberately NOT hand-picking just mega-cap winners to reduce bias
OFFENSIVE = [
    # Tech (broad, not just NVDA/META)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'CRM', 'ORCL', 'ADBE',
    # Healthcare
    'UNH', 'LLY', 'ABBV', 'ISRG', 'MRK',
    # Financials
    'JPM', 'GS', 'V', 'MA', 'AXP',
    # Consumer / Comms
    'AMZN', 'NFLX', 'META', 'GOOGL', 'COST',
    # Industrials / Energy
    'CAT', 'DE', 'GE', 'XOM', 'COP',
]

# Defensive: low-beta staples, utilities, healthcare, REITs
DEFENSIVE = [
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT', 'CL',
    # Utilities
    'NEE', 'DUK', 'SO', 'AEP',
    # Healthcare (defensive)
    'JNJ', 'MRK', 'ABT',
    # REITs
    'PLD', 'AMT',
    # Low-vol industrials
    'WM', 'RSG',
]

# SPY needed for regime detection — add to both simulation ticker lists
ALL_TICKERS = sorted(set(OFFENSIVE + DEFENSIVE + ['SPY']))

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_variant(name, offensive, defensive, top_n, desc=""):
    print("=" * 60)
    print(name)
    print("=" * 60)
    if desc:
        print(desc)
    print(f"Offensive pool: {len(offensive)} stocks → top {top_n}")
    print(f"Defensive pool: {len(defensive)} stocks (equal weight)")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Rebalance: monthly | Regime: SPY 200SMA + vol ratio")
    print()

    strategy = create_adaptive_momentum_strategy(
        offensive_tickers=offensive,
        defensive_tickers=defensive,
        spy_ticker='SPY',
        top_n_offensive=top_n,
        rebalance_frequency=21,
        momentum_short=63,
        momentum_long=126,
    )

    sim = BacktestSimulation(
        tickers=ALL_TICKERS,
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
        print(f"\nTrades: {len(buys)} buys, {len(sells)} sells")
        if not buys.empty:
            print(f"Tickers rotated: {', '.join(sorted(buys['ticker'].unique()))}")

    print()
    return sim


def run_benchmark():
    print("=" * 60)
    print("BENCHMARK: BUY & HOLD SPY")
    print("=" * 60)

    bench = BacktestSimulation(
        tickers=['SPY'],
        start_date=START,
        end_date=END,
        initial_cash=INITIAL_CASH,
        strategy_callback=create_buy_and_hold_strategy('SPY'),
    )
    bench.run()
    bench.print_performance_stats()
    print()
    return bench


def compare(sims: dict):
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<35} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Vol':>8} {'Calmar':>8}")
    print("-" * 78)

    for name, sim in sims.items():
        final = sim.history[-1]['total_value']
        ret = (final / INITIAL_CASH - 1) * 100
        df = pd.DataFrame(sim.history)
        returns = df['total_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        maxdd = ((df['total_value'].cummax() - df['total_value']) / df['total_value'].cummax()).max() * 100
        vol = returns.std() * np.sqrt(252) * 100
        calmar = (ret / 7) / maxdd if maxdd > 0 else 0
        print(f"{name:<35} {ret:>+9.1f}% {sharpe:>7.2f} {maxdd:>7.1f}% {vol:>7.1f}% {calmar:>7.2f}")

    print()
    spy_ret = (sims['SPY Buy & Hold'].history[-1]['total_value'] / INITIAL_CASH - 1) * 100
    for name, sim in sims.items():
        if name == 'SPY Buy & Hold':
            continue
        ret = (sim.history[-1]['total_value'] / INITIAL_CASH - 1) * 100
        alpha = ret - spy_ret
        tag = "PROFITABLE" if alpha > 20 else "MARGINAL" if alpha > 0 else "UNDERPERFORMS"
        print(f">>> {name}: {tag} ({alpha:+.1f}% vs SPY)")


if __name__ == '__main__':
    # Pure offensive — always in top momentum, no defensive switch
    s1 = run_variant(
        "V1: PURE MOMENTUM (always top 5, no regime switch)",
        OFFENSIVE, OFFENSIVE[:5],  # defensive=offensive means no real switch
        top_n=5,
        desc="Always picks top 5 momentum — no regime awareness\n",
    )

    # Adaptive — the core strategy
    s2 = run_variant(
        "V2: ADAPTIVE MOMENTUM (offensive/defensive switch)",
        OFFENSIVE, DEFENSIVE,
        top_n=5,
        desc="Risk-on: top 5 momentum | Risk-off: defensive basket\n",
    )

    # Conservative — fewer concentrated bets
    s3 = run_variant(
        "V3: CONSERVATIVE ADAPTIVE (top 8, wider)",
        OFFENSIVE, DEFENSIVE,
        top_n=8,
        desc="Risk-on: top 8 momentum (less concentrated) | Risk-off: defensive\n",
    )

    bench = run_benchmark()

    compare({
        'Pure Momentum (top 5)': s1,
        'Adaptive (5 off / def switch)': s2,
        'Conservative (8 off / def)': s3,
        'SPY Buy & Hold': bench,
    })
