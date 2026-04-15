"""
Backtest: OBV Divergence Mean Reversion Strategy

Uses a broad, systematic universe to avoid survivorship bias:
  - Variant 1: Large-cap blue chips (stable, liquid)
  - Variant 2: Growth + cyclical mix (more volatile, more divergences)

Both use the same entry/exit rules, just different universes to test
whether the signal generalizes.
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.obv_divergence_strategy import create_obv_divergence_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy
import pandas as pd
import numpy as np


# Broad large-cap universe — stable blue chips across all sectors
# Selected systematically: 2-3 of the largest names per GICS sector
BLUE_CHIP_UNIVERSE = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL',
    # Healthcare
    'UNH', 'JNJ', 'PFE',
    # Financials
    'JPM', 'BAC', 'WFC',
    # Consumer Disc
    'AMZN', 'HD', 'MCD',
    # Consumer Staples
    'PG', 'KO', 'PEP',
    # Industrials
    'HON', 'UNP', 'RTX',
    # Energy
    'XOM', 'CVX', 'COP',
    # Materials
    'LIN', 'APD', 'SHW',
    # Utilities
    'NEE', 'DUK', 'SO',
    # Real Estate
    'PLD', 'AMT', 'EQIX',
    # Communications
    'META', 'DIS', 'CMCSA',
]

# Higher-beta universe — more volatile = more divergence opportunities
GROWTH_CYCLICAL_UNIVERSE = [
    # Growth tech
    'NVDA', 'AMD', 'CRM', 'NFLX', 'SHOP',
    # Biotech / med-tech
    'ISRG', 'REGN', 'VRTX', 'DXCM', 'ALGN',
    # Cyclicals
    'CAT', 'DE', 'FCX', 'FSLR', 'URI',
    # Financials (higher beta)
    'GS', 'MS', 'SCHW', 'AXP', 'BLK',
    # Consumer growth
    'LULU', 'DECK', 'CMG', 'ABNB', 'UBER',
]

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_backtest(name, universe, max_positions=6):
    print("=" * 60)
    print(f"{name}")
    print("=" * 60)
    print(f"Universe: {len(universe)} stocks")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Rules:")
    print("  Filter: Price > 200 SMA, Dollar Vol > $20M, Vol >= avg")
    print("  Entry:  Bullish OBV divergence (price lower low, OBV higher low)")
    print("  Exit:   Price hits 20-day SMA (target) OR 7% stop OR 25-day hold")
    print(f"  Max positions: {max_positions}")
    print()

    strategy = create_obv_divergence_strategy(
        tickers=universe,
        sma_period=200,
        divergence_lookback=30,
        min_dollar_volume=20_000_000,
        volume_avg_period=20,
        volume_min_ratio=1.0,
        mean_reversion_sma=20,
        max_hold_days=25,
        stop_loss_pct=0.07,
        max_positions=max_positions,
    )

    sim = BacktestSimulation(
        tickers=universe,
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
            tickers_traded = sorted(buys['ticker'].unique())
            print(f"Tickers traded: {len(tickers_traded)} — {', '.join(tickers_traded)}")

        # Win rate analysis
        if not sells.empty:
            trades = []
            buy_log = {}
            for _, row in txns.iterrows():
                t = row['ticker']
                if row['action'] == 'BUY_STOCK':
                    buy_log[t] = row['price']
                elif row['action'] == 'SELL_STOCK' and t in buy_log:
                    pnl = (row['price'] - buy_log[t]) / buy_log[t]
                    trades.append(pnl)
                    del buy_log[t]

            if trades:
                wins = sum(1 for t in trades if t > 0)
                losses = len(trades) - wins
                avg_win = np.mean([t for t in trades if t > 0]) if wins else 0
                avg_loss = np.mean([t for t in trades if t <= 0]) if losses else 0
                total_pnl = sum(t for t in trades if t > 0) + sum(t for t in trades if t <= 0)
                expectancy = total_pnl / len(trades)

                print(f"\nWin rate: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
                print(f"Avg win:  {avg_win*100:+.2f}%")
                print(f"Avg loss: {avg_loss*100:+.2f}%")
                if losses > 0 and avg_loss != 0:
                    print(f"Profit factor: {abs(avg_win * wins / (avg_loss * losses)):.2f}")
                print(f"Expectancy per trade: {expectancy*100:+.3f}%")

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


def compare(sims: dict):
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Strategy':<35} {'Final Value':>14} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 75)

    for name, sim in sims.items():
        final = sim.history[-1]['total_value']
        ret = (final / INITIAL_CASH - 1) * 100
        df = pd.DataFrame(sim.history)
        returns = df['total_value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        maxdd = ((df['total_value'].cummax() - df['total_value']) / df['total_value'].cummax()).max() * 100
        print(f"{name:<35} ${final:>12,.2f} {ret:>+9.2f}% {sharpe:>7.2f} {maxdd:>7.2f}%")

    print()
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
    s1, _ = run_backtest("VARIANT 1: BLUE CHIP OBV DIVERGENCE", BLUE_CHIP_UNIVERSE, max_positions=6)
    s2, _ = run_backtest("VARIANT 2: GROWTH/CYCLICAL OBV DIVERGENCE", GROWTH_CYCLICAL_UNIVERSE, max_positions=6)
    bench, _ = run_benchmark()

    compare({
        'Blue Chip OBV Divergence': s1,
        'Growth/Cyclical OBV Div': s2,
        'SPY Buy & Hold': bench,
    })
