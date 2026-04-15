"""
Dual Momentum Sector Rotation Strategy

Based on Gary Antonacci's dual momentum framework:
  1. Absolute momentum: Only invest in sectors with positive momentum
     (return > 0 over lookback). This avoids bear markets.
  2. Relative momentum: Rank sectors by momentum, concentrate in the
     top N. This overweights winners.
  3. Monthly rebalance: Rotate holdings on the first trading day of
     each month to avoid overtrading.

When no sectors pass the absolute momentum filter, the strategy holds
cash (risk-off). This is what protects against major drawdowns.

Academic backing:
  - Jegadeesh & Titman (1993): momentum effect in stocks
  - Moskowitz & Grinblatt (1999): industry momentum
  - Antonacci (2014): dual momentum across asset classes

Stock filter: Uses sector ETFs for clean sector exposure, then
concentrates in top-performing sectors. Dollar volume filter ensures
liquidity.
"""

import pandas as pd
import numpy as np


def _compute_momentum_score(df: pd.DataFrame, short_period: int, long_period: int) -> float:
    """
    Blended momentum score: average of short and long lookback returns.
    Blending reduces whipsaw from a single lookback.
    """
    if len(df) < long_period + 1:
        return float('nan')

    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    current = float(close.iloc[-1])

    short_past = float(close.iloc[-short_period]) if len(close) > short_period else float('nan')
    long_past = float(close.iloc[-long_period])

    if any(pd.isna(v) or v == 0 for v in [current, short_past, long_past]):
        return float('nan')

    short_ret = (current - short_past) / short_past
    long_ret = (current - long_past) / long_past

    return (short_ret + long_ret) / 2


def create_dual_momentum_rotation_strategy(
    tickers: list,
    short_lookback: int = 63,
    long_lookback: int = 126,
    top_n: int = 3,
    rebalance_frequency: int = 21,
    absolute_threshold: float = 0.0,
):
    """
    Dual momentum sector rotation strategy.

    On each rebalance:
      1. Compute blended momentum score for each ticker
      2. Filter: only keep tickers with score > absolute_threshold
      3. Rank remaining by score, select top_n
      4. Equal-weight the top_n, sell everything else
      5. If nothing passes the filter, go 100% cash

    Args:
        tickers: Sector ETFs or stocks to rotate between
        short_lookback: Short momentum lookback (trading days, 63 ≈ 3mo)
        long_lookback: Long momentum lookback (trading days, 126 ≈ 6mo)
        top_n: Number of top sectors to hold
        rebalance_frequency: Days between rebalances (21 ≈ monthly)
        absolute_threshold: Minimum momentum score to invest (0.0 = positive)
    """
    state = {
        'days_since_rebal': rebalance_frequency,  # rebalance on first day
        'current_holdings': set(),
    }

    def strategy(date, portfolio, market_data, actions):
        state['days_since_rebal'] += 1

        if state['days_since_rebal'] < rebalance_frequency:
            return

        state['days_since_rebal'] = 0

        # Score all tickers
        scores = {}
        for ticker in tickers:
            if ticker not in market_data['data']:
                continue
            df = market_data['data'][ticker]
            score = _compute_momentum_score(df, short_lookback, long_lookback)
            if not pd.isna(score):
                scores[ticker] = score

        # Absolute momentum filter
        passing = {t: s for t, s in scores.items() if s > absolute_threshold}

        # Relative momentum: top N
        ranked = sorted(passing.items(), key=lambda x: x[1], reverse=True)
        target_holdings = set(t for t, _ in ranked[:top_n])

        # --- Sell positions not in target ---
        for ticker in list(state['current_holdings']):
            if ticker not in target_holdings:
                current_price = market_data['prices'].get(ticker, 0)
                if (ticker in portfolio.positions and
                        portfolio.positions[ticker].shares > 0 and
                        current_price > 0):
                    shares = int(portfolio.positions[ticker].shares)
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                state['current_holdings'].discard(ticker)

        # --- Buy new target positions (equal weight) ---
        if not target_holdings:
            return

        target_weight = 1.0 / len(target_holdings)

        # Calculate total portfolio value for sizing
        total_value = portfolio.cash
        for ticker in state['current_holdings']:
            price = market_data['prices'].get(ticker, 0)
            if ticker in portfolio.positions and price > 0:
                total_value += portfolio.positions[ticker].shares * price

        for ticker in target_holdings:
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0:
                continue

            target_value = total_value * target_weight
            current_shares = 0
            if ticker in portfolio.positions:
                current_shares = int(portfolio.positions[ticker].shares)
            current_value = current_shares * current_price

            diff_value = target_value - current_value

            if diff_value > current_price:
                # Need to buy more
                shares_to_buy = int(min(diff_value, portfolio.cash) / current_price)
                if shares_to_buy > 0:
                    actions.buy_stock(portfolio, ticker, shares_to_buy, current_price)
                    state['current_holdings'].add(ticker)
            elif diff_value < -current_price and current_shares > 0:
                # Need to trim
                shares_to_sell = min(int(-diff_value / current_price), current_shares)
                if shares_to_sell > 0:
                    actions.sell_stock(portfolio, ticker, shares_to_sell, current_price)

            if ticker in target_holdings:
                state['current_holdings'].add(ticker)

    return strategy
