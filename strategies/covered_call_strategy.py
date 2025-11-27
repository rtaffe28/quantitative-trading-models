"""
Covered Call Strategy

Sells call options against owned stock to generate income from premiums.
"""

import pandas as pd
from datetime import timedelta
from utils.black_scholes import black_scholes_call


def create_covered_call_strategy(ticker, strike_factor=1.06, interest_rate=0.05, time_to_expiration=15/365):
    """
    Factory function to create a covered call strategy with specific parameters.
    
    Args:
        ticker: Stock ticker symbol
        strike_factor: Multiplier for strike price (typically >1.0 for OTM calls)
        interest_rate: Risk-free interest rate for Black-Scholes
        time_to_expiration: Time to expiration in years (default 15 days)
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    days = int(time_to_expiration * 365)
    
    def covered_call_strategy(date, portfolio, market_data, actions):
        """
        Covered call strategy:
        1. Buy stock with available cash
        2. Sell OTM call options against stock holdings
        3. Handle option exercises and roll positions
        """
        current_price = market_data['prices'][ticker]
        
        # Buy stock with available cash
        max_shares = int(portfolio.cash / current_price)
        if max_shares > 0:
            actions.buy_stock(portfolio, ticker, max_shares, current_price)
        
        # Sell covered calls if we have shares and no active options
        if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
            has_active_call = any(opt.ticker == ticker and opt.option_type == 'call' 
                                  for opt in portfolio.options)
            
            if not has_active_call:
                shares = portfolio.positions[ticker].shares
                contracts = shares // 100  # Each contract is 100 shares
                
                if contracts > 0:
                    strike = current_price * strike_factor
                    expiration = date + timedelta(days=days)
                    
                    # Get volatility for Black-Scholes
                    if ticker in market_data['volatility']:
                        vol_data = market_data['volatility'][ticker]
                        if len(vol_data) > 0:
                            vol = vol_data.iloc[-1, 0]
                            
                            if not pd.isna(vol):
                                # Calculate premium using Black-Scholes
                                premium = black_scholes_call(
                                    S=current_price,
                                    K=strike,
                                    sigma=vol,
                                    r=interest_rate,
                                    t=time_to_expiration
                                )
                                
                                actions.sell_call(
                                    portfolio=portfolio,
                                    ticker=ticker,
                                    strike=strike,
                                    expiration=expiration,
                                    contracts=contracts,
                                    premium=premium
                                )
    
    return covered_call_strategy
