"""
OBV Divergence Mean Reversion Strategy

Contrarian strategy that buys when institutional accumulation is detected
beneath a falling price surface:

  1. Bullish OBV divergence: price makes lower low but OBV makes higher low
     → smart money is accumulating despite headline weakness
  2. Long-term uptrend filter: price > 200 SMA (only buy dips in uptrends,
     avoid catching falling knives)
  3. Volume confirmation: current volume above average (active participation)
  4. Exit when price reverts to 20-day SMA (mean reversion target),
     or via stop-loss / time stop

This is fundamentally different from the momentum strategies in the repo:
  - Trend Pullback uses RSI to time dips — purely price-based
  - Squeeze Breakout trades volatility expansion — follows breakouts
  - Dual Momentum rotates into strength — trend-following

OBV Divergence trades *against* short-term price action based on volume
intelligence. It's contrarian at the micro level but aligned with the
macro trend via the 200 SMA filter.

Stock filter: uses a broad systematic universe (not hand-picked),
filters by liquidity and trend, then screens for the divergence signal.
"""

import pandas as pd
import numpy as np


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()


def _detect_bullish_divergence(close: pd.Series, obv: pd.Series, lookback: int) -> bool:
    """Check if price made lower low but OBV made higher low."""
    if len(close) < lookback:
        return False

    half = lookback // 2
    recent = slice(-half, None)
    earlier = slice(-lookback, -half)

    price_recent_low = float(close.iloc[recent].min())
    price_earlier_low = float(close.iloc[earlier].min())
    obv_recent_low = float(obv.iloc[recent].min())
    obv_earlier_low = float(obv.iloc[earlier].min())

    if any(pd.isna(v) for v in [price_recent_low, price_earlier_low,
                                 obv_recent_low, obv_earlier_low]):
        return False

    return price_recent_low < price_earlier_low and obv_recent_low > obv_earlier_low


def create_obv_divergence_strategy(
    tickers: list,
    sma_period: int = 200,
    divergence_lookback: int = 30,
    min_dollar_volume: float = 20_000_000,
    volume_avg_period: int = 20,
    volume_min_ratio: float = 1.0,
    mean_reversion_sma: int = 20,
    max_hold_days: int = 25,
    stop_loss_pct: float = 0.07,
    max_positions: int = 6,
):
    """
    Multi-ticker OBV divergence strategy.

    Entry conditions:
      - Price > 200 SMA (long-term uptrend)
      - Bullish OBV divergence detected (accumulation)
      - Current volume >= volume_min_ratio * 20-day avg (active market)
      - Avg daily dollar volume > min_dollar_volume (liquidity)

    Exit conditions:
      - Price reaches 20-day SMA (mean reversion target)
      - Stop loss triggered (price drops stop_loss_pct below entry)
      - Time stop: held for max_hold_days

    Args:
        tickers: Universe to scan
        sma_period: Long-term trend filter SMA
        divergence_lookback: Window for divergence detection
        min_dollar_volume: Minimum avg daily dollar volume
        volume_avg_period: Period for average volume calculation
        volume_min_ratio: Current vol must be >= this * avg vol
        mean_reversion_sma: SMA period for the reversion target
        max_hold_days: Maximum holding period
        stop_loss_pct: Stop loss percentage below entry
        max_positions: Max simultaneous positions
    """
    state = {'held': {}}

    def strategy(date, portfolio, market_data, actions):
        # --- EXIT pass ---
        for ticker in list(state['held'].keys()):
            if ticker not in market_data['data']:
                continue
            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0:
                continue

            info = state['held'][ticker]
            info['days_held'] += 1

            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

            should_exit = False
            exit_reason = None

            # Mean reversion target: price reached the short-term SMA
            if len(close) >= mean_reversion_sma:
                target_sma = float(close.rolling(window=mean_reversion_sma).mean().iloc[-1])
                if not pd.isna(target_sma) and current_price >= target_sma:
                    should_exit = True
                    exit_reason = 'target'

            # Stop loss
            if current_price < info['entry_price'] * (1 - stop_loss_pct):
                should_exit = True
                exit_reason = 'stop_loss'

            # Time stop
            if info['days_held'] >= max_hold_days:
                should_exit = True
                exit_reason = 'time_stop'

            if should_exit and ticker in portfolio.positions:
                shares = int(portfolio.positions[ticker].shares)
                if shares > 0:
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                    info['exit_reason'] = exit_reason
                    del state['held'][ticker]

        # --- ENTRY pass ---
        if len(state['held']) >= max_positions:
            return

        min_data = max(sma_period, divergence_lookback + 10, volume_avg_period + 5)

        candidates = []
        for ticker in tickers:
            if ticker in state['held']:
                continue
            if ticker not in market_data['data']:
                continue

            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0 or len(df) < min_data:
                continue

            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

            # Liquidity filter
            avg_vol = float(volume.tail(volume_avg_period).mean())
            if pd.isna(avg_vol) or avg_vol * current_price < min_dollar_volume:
                continue

            # Long-term uptrend filter
            sma_200 = float(close.rolling(window=sma_period).mean().iloc[-1])
            if pd.isna(sma_200) or current_price <= sma_200:
                continue

            # Volume activity filter
            current_vol = float(volume.iloc[-1])
            if pd.isna(current_vol) or current_vol < avg_vol * volume_min_ratio:
                continue

            # OBV divergence detection
            obv = _compute_obv(close, volume)
            if not _detect_bullish_divergence(close, obv, divergence_lookback):
                continue

            # Score: how far below the 20-day SMA (bigger dip = more reversion potential)
            sma_short = float(close.rolling(window=mean_reversion_sma).mean().iloc[-1])
            if pd.isna(sma_short) or sma_short == 0:
                continue
            discount = (sma_short - current_price) / sma_short
            candidates.append((ticker, discount, current_price))

        # Sort by discount descending (most oversold with divergence first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = max_positions - len(state['held'])
        for ticker, discount, price in candidates[:slots]:
            allocation = 1.0 / max_positions
            available_cash = portfolio.cash * allocation
            shares = int(available_cash / price)
            if shares > 0:
                actions.buy_stock(portfolio, ticker, shares, price)
                state['held'][ticker] = {
                    'entry_price': price,
                    'entry_date': date,
                    'days_held': 0,
                    'discount_at_entry': discount,
                }

    return strategy
