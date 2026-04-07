"""
Volatility Squeeze Breakout Strategy

Based on the TTM Squeeze concept (John Carter):
  1. Detect when Bollinger Bands contract inside Keltner Channels ("squeeze")
  2. Track momentum direction using a linear regression oscillator
  3. Enter long when the squeeze fires and momentum is positive + accelerating
  4. Exit when momentum decelerates or turns negative

The thesis: periods of low volatility compress price like a spring. When the
squeeze releases, the resulting move tends to be directional and persistent.
Using momentum direction ensures we're on the right side of the breakout.

Stock filter: price > 50 SMA (don't buy breakouts in downtrends) and
average daily dollar volume > $10M (need liquidity for clean breakouts).
"""

import numpy as np
import pandas as pd


def _compute_squeeze(df, bb_period=20, bb_std=2.0, kc_ema=20, kc_atr=14, kc_mult=1.5):
    """Returns (is_squeezing, was_squeezing_yesterday) tuple."""
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

    # Bollinger Bands
    bb_sma = close.rolling(window=bb_period).mean()
    bb_std_val = close.rolling(window=bb_period).std()
    bb_upper = bb_sma + bb_std * bb_std_val
    bb_lower = bb_sma - bb_std * bb_std_val

    # Keltner Channels
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=kc_atr).mean()
    kc_ema_line = close.ewm(span=kc_ema, adjust=False).mean()
    kc_upper = kc_ema_line + kc_mult * atr
    kc_lower = kc_ema_line - kc_mult * atr

    # Squeeze series: True when BB inside KC
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    if len(squeeze_on) < 2:
        return False, False

    return bool(squeeze_on.iloc[-1]), bool(squeeze_on.iloc[-2])


def _compute_momentum(close, period=20):
    """Linear regression oscillator — measures momentum magnitude and direction."""
    if len(close) < period:
        return None, None

    # Simple momentum: deviation of price from its SMA, normalized
    sma = close.rolling(window=period).mean()
    mom = close - sma

    current = float(mom.iloc[-1])
    prev = float(mom.iloc[-2]) if len(mom) >= 2 else None

    if pd.isna(current):
        return None, None
    if prev is not None and pd.isna(prev):
        prev = None

    return current, prev


def create_squeeze_breakout_strategy(
    tickers: list,
    sma_filter_period: int = 50,
    min_dollar_volume: float = 10_000_000,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_atr_mult: float = 1.5,
    momentum_period: int = 20,
    max_positions: int = 6,
    hold_days: int = 15,
):
    """
    Multi-ticker squeeze breakout strategy.

    Entry conditions (all must be true):
      - Squeeze was ON (BB inside KC) and just released OR is releasing
      - Momentum is positive and accelerating (current > previous)
      - Price > 50 SMA (uptrend filter)
      - Stock has sufficient dollar volume (liquidity filter)

    Exit conditions (any):
      - Held for hold_days (time stop — squeezes are short-term setups)
      - Momentum turns negative
      - Price drops below entry price * 0.95 (5% stop loss)

    Args:
        tickers: Universe to scan
        sma_filter_period: SMA period for trend filter
        min_dollar_volume: Minimum 20-day avg dollar volume
        bb_period: Bollinger Band period
        bb_std: Bollinger Band std multiplier
        kc_atr_mult: Keltner Channel ATR multiplier
        momentum_period: Momentum oscillator lookback
        max_positions: Max simultaneous positions
        hold_days: Max holding period per trade
    """
    state = {'held': {}}  # ticker -> {entry_price, entry_date, days_held}

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
            mom_cur, _ = _compute_momentum(close, momentum_period)

            should_exit = False

            # Time stop
            if info['days_held'] >= hold_days:
                should_exit = True
            # Momentum turned negative
            elif mom_cur is not None and mom_cur < 0:
                should_exit = True
            # Stop loss at 5%
            elif current_price < info['entry_price'] * 0.95:
                should_exit = True

            if should_exit and ticker in portfolio.positions:
                shares = int(portfolio.positions[ticker].shares)
                if shares > 0:
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                    del state['held'][ticker]

        # --- ENTRY pass ---
        if len(state['held']) >= max_positions:
            return

        candidates = []
        for ticker in tickers:
            if ticker in state['held']:
                continue
            if ticker not in market_data['data']:
                continue

            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0 or len(df) < max(sma_filter_period, bb_period, 30):
                continue

            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

            # Trend filter: price above SMA
            sma = float(close.rolling(window=sma_filter_period).mean().iloc[-1])
            if pd.isna(sma) or current_price <= sma:
                continue

            # Liquidity filter: avg dollar volume
            avg_volume = float(volume.tail(20).mean())
            avg_dollar_vol = avg_volume * current_price
            if avg_dollar_vol < min_dollar_volume:
                continue

            # Squeeze detection: was squeezing, now releasing or just released
            is_sq, was_sq = _compute_squeeze(df, bb_period, bb_std, bb_period, 14, kc_atr_mult)

            # We want: squeeze just released (was on, now off) OR still on with strong momentum
            squeeze_firing = (was_sq and not is_sq) or is_sq

            if not squeeze_firing:
                continue

            # Momentum must be positive and accelerating
            mom_cur, mom_prev = _compute_momentum(close, momentum_period)
            if mom_cur is None or mom_cur <= 0:
                continue
            if mom_prev is not None and mom_cur <= mom_prev:
                continue  # not accelerating

            # Score by momentum strength (higher = better breakout)
            candidates.append((ticker, mom_cur, current_price))

        # Sort by momentum descending (strongest breakouts first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = max_positions - len(state['held'])
        for ticker, mom_val, price in candidates[:slots]:
            allocation = 1.0 / max_positions
            available_cash = portfolio.cash * allocation
            shares = int(available_cash / price)
            if shares > 0:
                actions.buy_stock(portfolio, ticker, shares, price)
                state['held'][ticker] = {
                    'entry_price': price,
                    'entry_date': date,
                    'days_held': 0,
                }

    return strategy
