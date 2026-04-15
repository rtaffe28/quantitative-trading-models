"""
Backtest: Multi-Signal Ensemble Strategy

Tests the hypothesis that combining 4 uncorrelated entry signals
(trend pullback, squeeze breakout, OBV divergence, VWAP discount)
solves the capital utilization problem seen in individual strategies.

Uses a broad 40-stock universe spanning all major sectors.
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.ensemble_strategy import create_ensemble_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy
import pandas as pd
import numpy as np


# Broad 40-stock universe: 3-4 per sector, mix of mega/large cap
UNIVERSE = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'AMD',
    # Healthcare
    'UNH', 'LLY', 'JNJ', 'ABBV', 'ISRG',
    # Financials
    'JPM', 'GS', 'V', 'MA', 'BLK',
    # Consumer Discretionary
    'AMZN', 'HD', 'COST', 'NFLX',
    # Consumer Staples
    'PG', 'KO', 'PEP', 'WMT',
    # Industrials
    'CAT', 'DE', 'GE', 'HON', 'URI',
    # Energy
    'XOM', 'CVX', 'COP',
    # Communications
    'META', 'GOOGL', 'DIS',
    # Materials
    'LIN', 'FCX',
    # Real Estate / Utilities
    'PLD', 'NEE', 'AMT',
]

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_ensemble():
    print("=" * 60)
    print("MULTI-SIGNAL ENSEMBLE STRATEGY")
    print("=" * 60)
    print(f"Universe: {len(UNIVERSE)} stocks across all sectors")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Entry signals (any one triggers):")
    print("  A. Trend Pullback:  ADX > 20 + RSI < 45 + price > 150 SMA")
    print("  B. Squeeze Breakout: BB inside KC + momentum accelerating")
    print("  C. OBV Divergence:  Price lower low + OBV higher low")
    print("  D. VWAP Discount:   Price > 3% below 20-day VWAP")
    print()
    print("Exit rules (unified):")
    print("  - 8% trailing stop from peak")
    print("  - RSI > 75 (overextended)")
    print("  - Price < 200 SMA (trend break)")
    print()
    print("Max positions: 10 (equal weight)")
    print()

    strategy = create_ensemble_strategy(
        tickers=UNIVERSE,
        min_dollar_volume=15_000_000,
        max_positions=10,
        trailing_stop_pct=0.08,
        rsi_exit=75.0,
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
            tickers_traded = sorted(buys['ticker'].unique())
            print(f"Tickers traded: {len(tickers_traded)}/{len(UNIVERSE)}")

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

                print(f"\nWin rate: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
                print(f"Avg win:  {avg_win*100:+.2f}%")
                print(f"Avg loss: {avg_loss*100:+.2f}%")
                if losses > 0 and avg_loss != 0:
                    print(f"Profit factor: {abs(avg_win * wins / (avg_loss * losses)):.2f}")

                total_pnl = sum(trades)
                expectancy = total_pnl / len(trades)
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


def compare(strat_sim, bench_sim):
    strat_final = strat_sim.history[-1]['total_value']
    bench_final = bench_sim.history[-1]['total_value']

    strat_return = (strat_final / INITIAL_CASH - 1) * 100
    bench_return = (bench_final / INITIAL_CASH - 1) * 100
    alpha = strat_return - bench_return

    strat_df = pd.DataFrame(strat_sim.history)
    bench_df = pd.DataFrame(bench_sim.history)

    strat_returns = strat_df['total_value'].pct_change().dropna()
    bench_returns = bench_df['total_value'].pct_change().dropna()

    strat_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(252)

    strat_dd = ((strat_df['total_value'].cummax() - strat_df['total_value']) / strat_df['total_value'].cummax()).max() * 100
    bench_dd = ((bench_df['total_value'].cummax() - bench_df['total_value']) / bench_df['total_value'].cummax()).max() * 100

    strat_vol = strat_returns.std() * np.sqrt(252) * 100
    bench_vol = bench_returns.std() * np.sqrt(252) * 100

    # Calmar ratio (return / max drawdown)
    years = 7
    strat_calmar = (strat_return / years) / strat_dd if strat_dd > 0 else 0
    bench_calmar = (bench_return / years) / bench_dd if bench_dd > 0 else 0

    print("=" * 60)
    print("DETAILED COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Ensemble':>15} {'SPY B&H':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {strat_return:>+14.2f}% {bench_return:>+14.2f}%")
    print(f"{'Sharpe Ratio':<25} {strat_sharpe:>15.2f} {bench_sharpe:>15.2f}")
    print(f"{'Max Drawdown':<25} {strat_dd:>14.2f}% {bench_dd:>14.2f}%")
    print(f"{'Annualized Vol':<25} {strat_vol:>14.2f}% {bench_vol:>14.2f}%")
    print(f"{'Calmar Ratio':<25} {strat_calmar:>15.2f} {bench_calmar:>15.2f}")
    print(f"{'Alpha vs SPY':<25} {alpha:>+14.2f}%")
    print()

    if alpha > 5:
        print(">>> POTENTIALLY PROFITABLE — ensemble outperforms SPY.")
    elif alpha > 0:
        print(">>> MARGINAL — slight outperformance.")
    else:
        print(">>> UNDERPERFORMS SPY on raw return.")

    if strat_sharpe > bench_sharpe:
        print(f">>> BETTER RISK-ADJUSTED: Sharpe {strat_sharpe:.2f} vs {bench_sharpe:.2f}")
    if strat_dd < bench_dd:
        print(f">>> LOWER DRAWDOWN: {strat_dd:.1f}% vs {bench_dd:.1f}%")
    if strat_calmar > bench_calmar:
        print(f">>> BETTER CALMAR: {strat_calmar:.2f} vs {bench_calmar:.2f} (return per unit drawdown)")


if __name__ == '__main__':
    strat_sim, _ = run_ensemble()
    bench_sim, _ = run_benchmark()
    compare(strat_sim, bench_sim)
