"""
Adaptive Momentum with Regime Detection Strategy

Always-invested strategy that shifts allocation based on market regime:

  RISK-ON (SPY uptrend + low vol):
    - Concentrate in top 5 momentum stocks (aggressive)
    - Use 3-month + 6-month blended momentum for ranking
    - Monthly rebalance into strongest performers

  RISK-OFF (SPY downtrend OR vol spike):
    - Rotate into defensive names (low-beta staples, utilities, healthcare)
    - Equal-weight across defensive basket
    - Preserve capital while staying invested

This solves the capital utilization problem: instead of going to cash
in bad regimes (which killed returns in strategies #1-5), we rotate
into defensive positions that still earn returns.

Regime detection uses SPY's own data:
  - Trend: SPY price vs 200 SMA
  - Volatility: 20-day vol vs 60-day vol (spike detection)

Stock filter:
  - Offensive universe: high-momentum large caps, ranked by blended ROC
  - Defensive universe: low-beta staples/utilities/healthcare
  - Both filtered for liquidity (avg dollar vol > $10M)
"""

import pandas as pd
import numpy as np


def _blended_momentum(close, short: int = 63, long: int = 126) -> float:
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    if not hasattr(close, '__len__') or len(close) < long + 1:
        return float('nan')
    current = float(close.iloc[-1])
    short_past = float(close.iloc[-short]) if len(close) > short else float('nan')
    long_past = float(close.iloc[-long])
    if any(pd.isna(v) or v == 0 for v in [current, short_past, long_past]):
        return float('nan')
    return ((current / short_past - 1) + (current / long_past - 1)) / 2


def _detect_regime(spy_df: pd.DataFrame) -> str:
    """Returns 'risk_on' or 'risk_off'."""
    if len(spy_df) < 210:
        return 'risk_on'

    close = spy_df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()

    current = float(close.iloc[-1])
    sma_200 = float(close.rolling(window=200).mean().iloc[-1])
    if pd.isna(sma_200):
        return 'risk_on'

    in_uptrend = current > sma_200

    # Volatility regime
    log_ret = np.log(close / close.shift(1))
    short_vol = float(log_ret.rolling(window=20).std().iloc[-1]) * np.sqrt(252)
    long_vol = float(log_ret.rolling(window=60).std().iloc[-1]) * np.sqrt(252)

    if pd.isna(short_vol) or pd.isna(long_vol) or long_vol == 0:
        vol_calm = True
    else:
        vol_calm = short_vol < long_vol * 1.3

    if in_uptrend and vol_calm:
        return 'risk_on'
    return 'risk_off'


def create_adaptive_momentum_strategy(
    offensive_tickers: list,
    defensive_tickers: list,
    spy_ticker: str = 'SPY',
    top_n_offensive: int = 5,
    rebalance_frequency: int = 21,
    momentum_short: int = 63,
    momentum_long: int = 126,
):
    """
    Adaptive momentum strategy with regime switching.

    Args:
        offensive_tickers: High-growth stocks for risk-on
        defensive_tickers: Low-beta stocks for risk-off
        spy_ticker: Ticker used for regime detection (must be in one of the lists
                    or added to the simulation tickers)
        top_n_offensive: How many top momentum stocks to hold in risk-on
        rebalance_frequency: Days between rebalances
        momentum_short: Short lookback for momentum (trading days)
        momentum_long: Long lookback for momentum (trading days)
    """
    state = {
        'days_since_rebal': rebalance_frequency,  # trigger on first day
        'current_holdings': set(),
        'current_regime': None,
    }

    def strategy(date, portfolio, market_data, actions):
        state['days_since_rebal'] += 1

        if state['days_since_rebal'] < rebalance_frequency:
            return
        state['days_since_rebal'] = 0

        # Detect regime
        if spy_ticker in market_data['data']:
            regime = _detect_regime(market_data['data'][spy_ticker])
        else:
            regime = 'risk_on'

        state['current_regime'] = regime

        # Determine target holdings
        if regime == 'risk_on':
            # Rank offensive tickers by momentum, pick top N
            scores = {}
            for ticker in offensive_tickers:
                if ticker not in market_data['data']:
                    continue
                df = market_data['data'][ticker]
                close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
                score = _blended_momentum(close, momentum_short, momentum_long)
                if not pd.isna(score) and score > 0:
                    scores[ticker] = score

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            target_tickers = [t for t, _ in ranked[:top_n_offensive]]
        else:
            # Risk-off: equal weight all defensive tickers that have positive data
            target_tickers = [
                t for t in defensive_tickers
                if t in market_data['data'] and
                market_data['prices'].get(t, 0) > 0
            ]

        target_set = set(target_tickers)

        # Sell positions not in target
        for ticker in list(state['current_holdings']):
            if ticker not in target_set:
                current_price = market_data['prices'].get(ticker, 0)
                if (ticker in portfolio.positions and
                        portfolio.positions[ticker].shares > 0 and
                        current_price > 0):
                    shares = int(portfolio.positions[ticker].shares)
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                state['current_holdings'].discard(ticker)

        if not target_tickers:
            return

        # Calculate total portfolio value
        total_value = portfolio.cash
        for ticker in state['current_holdings']:
            price = market_data['prices'].get(ticker, 0)
            if ticker in portfolio.positions and price > 0:
                total_value += portfolio.positions[ticker].shares * price

        target_weight = 1.0 / len(target_tickers)

        # Rebalance into target weights
        for ticker in target_tickers:
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0:
                continue

            target_value = total_value * target_weight
            current_shares = 0
            if ticker in portfolio.positions:
                current_shares = int(portfolio.positions[ticker].shares)
            current_value = current_shares * current_price

            diff = target_value - current_value

            if diff > current_price:
                shares_to_buy = int(min(diff, portfolio.cash) / current_price)
                if shares_to_buy > 0:
                    actions.buy_stock(portfolio, ticker, shares_to_buy, current_price)
            elif diff < -current_price and current_shares > 0:
                shares_to_sell = min(int(-diff / current_price), current_shares)
                if shares_to_sell > 0:
                    actions.sell_stock(portfolio, ticker, shares_to_sell, current_price)

            state['current_holdings'].add(ticker)

    return strategy
