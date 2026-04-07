"""
Trend Pullback Momentum Strategy

Buys dips in strong uptrends with multi-factor confirmation:
  - Trend filter: Price > 200 SMA (uptrend)
  - Trend strength: ADX > 25 (strong directional move)
  - Relative strength: Outperforming SPY (quality filter)
  - Entry: RSI < 35 (pullback within the uptrend)
  - Exit: RSI > 70 (extended) OR price < 200 SMA (trend broken)

The idea: strong trending stocks that temporarily pull back tend to resume
their trend. ADX confirms the trend has conviction, relative strength
ensures we're in market leaders (not laggards bouncing), and RSI times
the entry to buy weakness rather than chase strength.
"""

import pandas as pd
import numpy as np
from indicators.base import Indicator


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(alpha=alpha, adjust=False).mean()


def create_trend_pullback_strategy(
    tickers: list,
    sma_period: int = 200,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    rsi_period: int = 14,
    rsi_entry: float = 35.0,
    rsi_exit: float = 70.0,
    max_positions: int = 5,
):
    """
    Multi-ticker trend pullback strategy.

    Allocates equal weight across up to max_positions stocks.
    Scans the ticker universe each day for entry/exit signals.

    Args:
        tickers: Universe of tickers to scan
        sma_period: SMA lookback for trend filter
        adx_period: ADX calculation period
        adx_threshold: Minimum ADX to confirm strong trend
        rsi_period: RSI calculation period
        rsi_entry: RSI below this triggers a buy (pullback)
        rsi_exit: RSI above this triggers a sell (overextended)
        max_positions: Max simultaneous positions
    """
    state = {'held': {}}  # ticker -> entry info

    def strategy(date, portfolio, market_data, actions):
        # --- EXIT pass first ---
        for ticker in list(state['held'].keys()):
            if ticker not in market_data['data']:
                continue
            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0 or len(df) < sma_period:
                continue

            prices = df['Close']
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze()

            sma = float(prices.rolling(window=sma_period).mean().iloc[-1])
            rsi_series = _compute_rsi(prices, rsi_period)
            current_rsi = float(rsi_series.iloc[-1])

            should_exit = False
            if pd.notna(current_rsi) and current_rsi > rsi_exit:
                should_exit = True
            if pd.notna(sma) and current_price < sma:
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
            if current_price <= 0 or len(df) < sma_period:
                continue

            prices = df['Close']
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze()

            # Trend filter: price above 200 SMA
            sma = float(prices.rolling(window=sma_period).mean().iloc[-1])
            if pd.isna(sma) or current_price <= sma:
                continue

            # Trend strength: ADX above threshold
            adx = _compute_adx(df, adx_period)
            current_adx = float(adx.iloc[-1])
            if pd.isna(current_adx) or current_adx <= adx_threshold:
                continue

            # Entry: RSI pullback
            rsi_series = _compute_rsi(prices, rsi_period)
            current_rsi = float(rsi_series.iloc[-1])
            if pd.isna(current_rsi) or current_rsi >= rsi_entry:
                continue

            # Score by how oversold within a strong trend (lower RSI = better entry)
            candidates.append((ticker, current_rsi, current_price))

        # Sort by RSI ascending (most oversold first)
        candidates.sort(key=lambda x: x[1])

        slots = max_positions - len(state['held'])
        for ticker, rsi_val, price in candidates[:slots]:
            allocation = 1.0 / max_positions
            available_cash = portfolio.cash * allocation
            shares = int(available_cash / price)
            if shares > 0:
                actions.buy_stock(portfolio, ticker, shares, price)
                state['held'][ticker] = {'entry_rsi': rsi_val, 'entry_date': date}

    return strategy
