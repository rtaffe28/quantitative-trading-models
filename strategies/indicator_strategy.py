"""
Indicator Strategy — converts boolean indicators into BacktestSimulation callbacks.

Provides two factory functions:
  - create_indicator_strategy: single-ticker, indicator-driven buy/sell
  - create_screener_strategy: multi-ticker, dynamically enters/exits based on signals
"""

from indicators.base import Indicator


def create_indicator_strategy(ticker: str, indicator: Indicator, allocation: float = 1.0):
    """
    Convert an indicator into a single-ticker strategy callback.

    Signal logic:
      - indicator(df) → True  AND no position → BUY with allocation % of cash
      - indicator(df) → False AND in position  → SELL all shares

    Args:
        ticker: Stock ticker to trade
        indicator: An Indicator callable
        allocation: Fraction of cash to deploy (0.0–1.0)

    Returns:
        Strategy callback for BacktestSimulation
    """
    state = {'in_position': False}

    def strategy(date, portfolio, market_data, actions):
        if ticker not in market_data['data']:
            return

        df = market_data['data'][ticker]
        current_price = market_data['prices'].get(ticker, 0)
        if current_price <= 0:
            return

        try:
            signal = indicator(df)
        except Exception:
            return

        if signal and not state['in_position']:
            available_cash = portfolio.cash * allocation
            shares = int(available_cash / current_price)
            if shares > 0:
                actions.buy_stock(portfolio, ticker, shares, current_price)
                state['in_position'] = True

        elif not signal and state['in_position']:
            if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
                shares = int(portfolio.positions[ticker].shares)
                actions.sell_stock(portfolio, ticker, shares, current_price)
                state['in_position'] = False

    return strategy


def create_screener_strategy(
    indicator: Indicator,
    max_positions: int = 10,
):
    """
    Multi-ticker strategy that dynamically enters/exits positions across all
    tickers in the simulation based on indicator signals.

    On each trading day:
      1. Evaluate indicator for every ticker in market_data
      2. Buy tickers where signal is True (up to max_positions, equal weight)
      3. Sell tickers where signal turned False

    Args:
        indicator: An Indicator callable (or composite)
        max_positions: Maximum simultaneous positions

    Returns:
        Strategy callback for BacktestSimulation
    """
    state = {'held': set()}

    def strategy(date, portfolio, market_data, actions):
        # First pass: sell positions where signal turned off
        for ticker in list(state['held']):
            if ticker not in market_data['data']:
                continue
            df = market_data['data'][ticker]
            try:
                signal = indicator(df)
            except Exception:
                signal = False

            if not signal:
                current_price = market_data['prices'].get(ticker, 0)
                if (ticker in portfolio.positions and
                        portfolio.positions[ticker].shares > 0 and
                        current_price > 0):
                    shares = int(portfolio.positions[ticker].shares)
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                state['held'].discard(ticker)

        # Second pass: buy new positions where signal fires
        for ticker in market_data['data']:
            if ticker in state['held']:
                continue
            if len(state['held']) >= max_positions:
                break

            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0:
                continue

            try:
                signal = indicator(df)
            except Exception:
                continue

            if signal:
                allocation = 1.0 / max_positions
                available_cash = portfolio.cash * allocation
                shares = int(available_cash / current_price)
                if shares > 0:
                    actions.buy_stock(portfolio, ticker, shares, current_price)
                    state['held'].add(ticker)

    return strategy
