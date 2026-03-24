"""
Mag 7 ATM LEAP Strategy

Buys at-the-money LEAP calls on all Magnificent 7 stocks (AAPL, MSFT, GOOGL,
AMZN, META, NVDA, TSLA) with capital split evenly across all names. Holds for
~1 year and rolls positions when they get within a configurable number of days
to expiration.
"""

import pandas as pd
from datetime import timedelta
from utils.black_scholes import black_scholes_call


MAG7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]


def create_mag7_leap_strategy(
    tickers=None,
    strike_factor=1.0,
    days=365,
    interest_rate=0.05,
    roll_threshold=90,
):
    """
    Factory function to create a Mag 7 ATM LEAP strategy.

    Capital is split evenly across all tickers. Each slot buys ATM LEAP calls
    and rolls them when they approach expiration.

    Args:
        tickers: List of tickers (defaults to Mag 7)
        strike_factor: Multiplier for strike price (1.0 = ATM)
        days: Days to expiration for LEAPs
        interest_rate: Risk-free interest rate for Black-Scholes
        roll_threshold: Days before expiration to roll the position

    Returns:
        A strategy callback function for BacktestSimulation
    """
    if tickers is None:
        tickers = MAG7_TICKERS

    num_slots = len(tickers)
    time_years = days / 365

    def strategy(date, portfolio, market_data, actions):
        # Determine per-ticker allocation based on total portfolio value
        # Use cash for initial allocation, then maintain equal weight on rolls
        allocation_per_ticker = portfolio.cash / num_slots

        for ticker in tickers:
            if ticker not in market_data['prices']:
                continue

            current_price = market_data['prices'][ticker]
            if current_price <= 0:
                continue

            # Get volatility
            vol = _get_volatility(ticker, market_data)
            if vol is None:
                continue

            # Check for existing LEAP on this ticker
            existing = [
                opt for opt in portfolio.options
                if opt.ticker == ticker
                and opt.option_type == 'call'
                and opt.position == 'long'
            ]

            # Roll positions approaching expiration
            for opt in existing:
                days_to_exp = (opt.expiration_date - date).days
                if 0 < days_to_exp < roll_threshold:
                    # Close the old LEAP
                    current_premium = black_scholes_call(
                        S=current_price,
                        K=opt.strike,
                        sigma=vol,
                        r=interest_rate,
                        t=days_to_exp / 365,
                    )
                    actions.close_call(
                        portfolio=portfolio,
                        ticker=ticker,
                        strike=opt.strike,
                        expiration=opt.expiration_date,
                        contracts=opt.contracts,
                        premium=current_premium,
                    )

                    # Open a new LEAP with the freed cash (equal share)
                    _buy_leap(
                        date, portfolio, ticker, current_price, vol,
                        strike_factor, days, time_years, interest_rate,
                        portfolio.cash / num_slots, actions,
                    )
                    break  # one roll per ticker per day

            # If no position exists for this ticker, open one
            if not existing and allocation_per_ticker > 100:
                _buy_leap(
                    date, portfolio, ticker, current_price, vol,
                    strike_factor, days, time_years, interest_rate,
                    allocation_per_ticker, actions,
                )

    return strategy


def _get_volatility(ticker, market_data):
    """Extract the latest volatility value for a ticker."""
    if ticker not in market_data.get('volatility', {}):
        return None
    vol_data = market_data['volatility'][ticker]
    if len(vol_data) == 0:
        return None
    vol = vol_data.iloc[-1, 0]
    if pd.isna(vol):
        return None
    return vol


def _buy_leap(date, portfolio, ticker, price, vol, strike_factor, days,
              time_years, interest_rate, budget, actions):
    """Buy LEAP calls for a single ticker within a given budget."""
    strike = price * strike_factor
    expiration = date + timedelta(days=days)

    premium = black_scholes_call(
        S=price, K=strike, sigma=vol, r=interest_rate, t=time_years,
    )
    if premium <= 0:
        return

    premium_per_contract = premium * 100
    max_contracts = int(budget / premium_per_contract)

    if max_contracts > 0:
        actions.buy_call(
            portfolio=portfolio,
            ticker=ticker,
            strike=strike,
            expiration=expiration,
            contracts=max_contracts,
            premium=premium,
        )
