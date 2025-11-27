"""
LEAP (Long-term Equity Anticipation Securities) Strategy

This strategy buys long-dated call options instead of stock directly,
providing leverage and limited downside risk.
"""

import pandas as pd
from datetime import timedelta
from utils.black_scholes import black_scholes_call


def create_leap_strategy(ticker, strike_factor=1.1, days=365, interest_rate=0.05, roll_threshold=90):
    """
    Factory function to create a LEAP strategy with specific parameters.
    
    Args:
        ticker: Stock ticker symbol
        strike_factor: Multiplier for strike price (1.0 = ATM, >1.0 = OTM, <1.0 = ITM)
        days: Days to expiration for LEAPs
        interest_rate: Risk-free interest rate for Black-Scholes
        roll_threshold: Days before expiration to roll the position
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    time = days / 365  # Convert to years for Black-Scholes
    
    def leap_strategy(date, portfolio, market_data, actions):
        current_price = market_data['prices'][ticker]
        
        # Check if we have any active LEAP calls
        has_active_leap = any(opt.ticker == ticker and 
                              opt.option_type == 'call' and 
                              opt.position == 'long'
                              for opt in portfolio.options)
        
        # Buy LEAPs if we don't have any and have cash
        if not has_active_leap and portfolio.cash > 1000:
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
                            t=time
                        )
                        
                        # Calculate how many contracts we can afford
                        premium_per_contract = premium * 100
                        max_contracts = int(portfolio.cash / premium_per_contract)
                        
                        if max_contracts > 0:
                            actions.buy_call(
                                portfolio=portfolio,
                                ticker=ticker,
                                strike=strike,
                                expiration=expiration,
                                contracts=max_contracts,
                                premium=premium
                            )
        
        # Roll LEAPs when they get close to expiration
        for opt in portfolio.options:
            if (opt.ticker == ticker and 
                opt.option_type == 'call' and 
                opt.position == 'long'):
                days_to_expiration = (opt.expiration_date - date).days
                
                if days_to_expiration < roll_threshold and days_to_expiration > 0:
                    # Close the current LEAP
                    if ticker in market_data['volatility']:
                        vol_data = market_data['volatility'][ticker]
                        if len(vol_data) > 0:
                            vol = vol_data.iloc[-1, 0]
                            
                            if not pd.isna(vol):
                                # Calculate current value of the option
                                current_premium = black_scholes_call(
                                    S=current_price,
                                    K=opt.strike,
                                    sigma=vol,
                                    r=interest_rate,
                                    t=days_to_expiration / 365
                                )
                                
                                # Close the old LEAP
                                actions.close_call(
                                    portfolio=portfolio,
                                    ticker=ticker,
                                    strike=opt.strike,
                                    expiration=opt.expiration_date,
                                    contracts=opt.contracts,
                                    premium=current_premium
                                )
                                
                                # Buy new LEAP with the proceeds
                                new_strike = current_price * strike_factor
                                new_expiration = date + timedelta(days=days)
                                
                                new_premium = black_scholes_call(
                                    S=current_price,
                                    K=new_strike,
                                    sigma=vol,
                                    r=interest_rate,
                                    t=time
                                )
                                
                                new_premium_per_contract = new_premium * 100
                                max_new_contracts = int(portfolio.cash / new_premium_per_contract)
                                
                                if max_new_contracts > 0:
                                    actions.buy_call(
                                        portfolio=portfolio,
                                        ticker=ticker,
                                        strike=new_strike,
                                        expiration=new_expiration,
                                        contracts=max_new_contracts,
                                        premium=new_premium
                                    )
                                break  # Only roll one position at a time
    
    return leap_strategy
