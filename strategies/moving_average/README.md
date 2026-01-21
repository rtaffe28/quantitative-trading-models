# Moving Average Trading Strategies

This folder contains implementations and backtests of various moving average-based trading strategies.

## Overview

Moving average strategies are classic technical analysis approaches that smooth out price data to identify trends. These strategies work by generating buy and sell signals based on moving average crossovers.

## Strategies Implemented

### 1. SMA Crossover Strategy (50/200)
**File**: `create_sma_crossover_strategy()`

The classic "Golden Cross" and "Death Cross" strategy using simple moving averages.

- **Buy Signal (Golden Cross)**: When the 50-day SMA crosses above the 200-day SMA
- **Sell Signal (Death Cross)**: When the 50-day SMA crosses below the 200-day SMA
- **Best For**: Long-term trend following, strong trending markets
- **Parameters**:
  - `short_window`: Default 50 days
  - `long_window`: Default 200 days
  - `allocation`: Portfolio allocation (0-1)

### 2. EMA Crossover Strategy (12/26)
**File**: `create_ema_crossover_strategy()`

More responsive strategy using exponential moving averages.

- **Buy Signal**: When the 12-day EMA crosses above the 26-day EMA
- **Sell Signal**: When the 12-day EMA crosses below the 26-day EMA
- **Best For**: Medium-term trading, markets with momentum
- **Advantages**: More responsive to recent price changes than SMA
- **Parameters**:
  - `short_window`: Default 12 days
  - `long_window`: Default 26 days
  - `allocation`: Portfolio allocation (0-1)

### 3. Triple MA Strategy (10/50/200)
**File**: `create_triple_ma_strategy()`

Confirms strong trends using three moving averages.

- **Buy Signal**: When Fast MA > Medium MA > Slow MA (aligned uptrend)
- **Sell Signal**: When Fast MA crosses below Medium MA (trend weakening)
- **Best For**: Catching strong, sustained trends with high confidence
- **Advantages**: Fewer false signals, better trend confirmation
- **Parameters**:
  - `fast_window`: Default 10 days
  - `medium_window`: Default 50 days
  - `slow_window`: Default 200 days
  - `allocation`: Portfolio allocation (0-1)

### 4. Adaptive MA Strategy (20/50)
**File**: `create_adaptive_ma_strategy()`

Intelligent strategy that adjusts behavior based on market volatility.

- **Low Volatility**: Standard crossover signals (1-day confirmation)
- **High Volatility**: Requires 3 days of confirmation to avoid whipsaws
- **Best For**: Volatile markets, reducing false signals
- **Advantages**: Adapts to market conditions automatically
- **Parameters**:
  - `short_window`: Default 20 days
  - `long_window`: Default 50 days
  - `allocation`: Portfolio allocation (0-1)
  - `volatility_threshold`: Default 0.02 (2% daily volatility)

## Usage Example

```python
from strategies.moving_average_strategy import create_sma_crossover_strategy
from utils.simulation import BacktestSimulation

# Create strategy
strategy = create_sma_crossover_strategy(
    ticker="AAPL",
    short_window=50,
    long_window=200,
    allocation=1.0
)

# Run backtest
sim = BacktestSimulation(
    tickers=["AAPL"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_cash=100000,
    strategy_callback=strategy
)

results = sim.run()
```

## Performance Characteristics

### When Moving Average Strategies Work Well:
- **Strong Trends**: MA strategies excel in sustained uptrends or downtrends
- **Lower Volatility**: Smoother price action leads to clearer signals
- **Medium to Long-Term**: Best for holding periods of weeks to months

### When They Struggle:
- **Sideways Markets**: Choppy, range-bound markets generate false signals
- **High Volatility**: Rapid price swings can cause whipsaws
- **Lagging Nature**: MAs are backward-looking and may miss sudden reversals

## Strategy Selection Guide

| Market Condition | Best Strategy | Reason |
|-----------------|---------------|---------|
| Strong Trend | SMA 50/200 or Triple MA | Confirms sustained moves |
| Volatile Market | Adaptive MA | Adjusts confirmation period |
| Medium-term Trading | EMA 12/26 | More responsive to changes |
| Risk-Averse | Triple MA | Fewer but higher-quality signals |
| Active Trading | EMA 12/26 | More frequent signals |

## Key Metrics to Monitor

1. **Total Return**: Overall profit/loss
2. **Number of Transactions**: More trades = higher transaction costs
3. **Max Drawdown**: Largest peak-to-trough decline
4. **Sharpe Ratio**: Risk-adjusted returns
5. **Win Rate**: Percentage of profitable trades
6. **Average Holding Period**: Time in market

## Notebooks

- `MSTR_backtest.ipynb`: Comprehensive backtest comparing all four strategies on MicroStrategy (MSTR) stock

## Tips for Optimization

1. **Backtest Multiple Periods**: Test on different market conditions
2. **Consider Transaction Costs**: Frequent trading reduces net returns
3. **Combine with Other Indicators**: Add volume, RSI, or support/resistance
4. **Risk Management**: Use stop-losses and position sizing
5. **Walk-Forward Testing**: Test on out-of-sample data

## References

- **SMA**: Simple Moving Average - arithmetic mean of prices
- **EMA**: Exponential Moving Average - weighted toward recent prices
- **Golden Cross**: Classic bullish signal (short MA crosses above long MA)
- **Death Cross**: Classic bearish signal (short MA crosses below long MA)
- **Whipsaw**: False signal that quickly reverses, causing losses
