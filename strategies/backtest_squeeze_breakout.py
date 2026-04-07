"""
Backtest: Volatility Squeeze Breakout Strategy

Tests the TTM Squeeze concept against a mix of growth and value stocks
across sectors. Uses a different universe than the trend pullback backtest
to test generalizability.

Filters:
  - Price > 50 SMA (uptrend only)
  - Avg daily dollar volume > $10M (liquidity)
  - Squeeze detected (BB inside KC)
  - Momentum positive and accelerating
"""

from datetime import datetime
from utils.simulation import BacktestSimulation
from strategies.squeeze_breakout_strategy import create_squeeze_breakout_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy


# Mix of growth + cyclical + defensive — different from trend pullback universe
UNIVERSE = [
    # High-growth tech
    'NVDA', 'AMD', 'NFLX', 'CRM', 'SHOP',
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    # Healthcare / biotech
    'LLY', 'UNH', 'ISRG', 'ABBV', 'TMO',
    # Financials
    'JPM', 'GS', 'BLK', 'SCHW', 'AXP',
    # Industrials / cyclicals
    'CAT', 'DE', 'GE', 'URI', 'PWR',
    # Consumer / retail
    'COST', 'TJX', 'DECK', 'LULU', 'CMG',
]

START = datetime(2019, 1, 1)
END = datetime(2025, 12, 31)
INITIAL_CASH = 100_000


def run_strategy_backtest():
    print("=" * 60)
    print("VOLATILITY SQUEEZE BREAKOUT STRATEGY")
    print("=" * 60)
    print(f"Universe: {len(UNIVERSE)} stocks (growth + cyclical + defensive)")
    print(f"Period: {START.date()} to {END.date()}")
    print(f"Initial capital: ${INITIAL_CASH:,.0f}")
    print()
    print("Rules:")
    print("  Filter: Price > 50 SMA, Dollar Vol > $10M")
    print("  Entry:  Squeeze detected + momentum positive & accelerating")
    print("  Exit:   Momentum turns negative OR 15-day hold OR 5% stop")
    print("  Max positions: 6 (equal weight)")
    print()

    strategy = create_squeeze_breakout_strategy(
        tickers=UNIVERSE,
        sma_filter_period=50,
        min_dollar_volume=10_000_000,
        bb_period=20,
        bb_std=2.0,
        kc_atr_mult=1.5,
        momentum_period=20,
        max_positions=6,
        hold_days=15,
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
                    avg_win = sum(t for t in trades if t > 0) / max(wins, 1)
                    losses = sum(1 for t in trades if t <= 0)
                    avg_loss = sum(t for t in trades if t <= 0) / max(losses, 1)
                    print(f"\nWin rate: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
                    print(f"Avg win:  {avg_win*100:+.2f}%")
                    print(f"Avg loss: {avg_loss*100:+.2f}%")
                    if avg_loss != 0:
                        print(f"Profit factor: {abs(avg_win * wins / (avg_loss * losses)):.2f}")

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
        print(">>> UNDERPERFORMS SPY on raw return over this period.")


if __name__ == '__main__':
    strat_sim, strat_results = run_strategy_backtest()
    bench_sim, bench_results = run_benchmark()
    compare(strat_sim, bench_sim)
