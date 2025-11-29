"""
Wheel Strategy

The wheel strategy is an options trading strategy that generates income by:
1. Selling cash-secured puts when you don't own the stock
2. If assigned, selling covered calls against the stock position
3. Repeating the cycle when stock is called away

This is a neutral to bullish strategy that works best in sideways to moderately
bullish markets.
"""

import pandas as pd
from datetime import timedelta
from utils.black_scholes import black_scholes_call, black_scholes_put


def create_wheel_strategy(
    ticker,
    put_strike_factor=0.95,
    call_strike_factor=1.05,
    days_to_expiration=30,
    interest_rate=0.05,
    min_premium_threshold=0.01
):
    """
    Factory function to create a wheel strategy with specific parameters.
    
    Args:
        ticker: Stock ticker symbol
        put_strike_factor: Multiplier for put strike price (<1.0 for OTM puts)
        call_strike_factor: Multiplier for call strike price (>1.0 for OTM calls)
        days_to_expiration: Days to expiration for options (default 30)
        interest_rate: Risk-free interest rate for Black-Scholes
        min_premium_threshold: Minimum premium as % of stock price to trade
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    time = days_to_expiration / 365  # Convert to years for Black-Scholes
    
    def wheel_strategy(date, portfolio, market_data, actions):
        """
        Wheel strategy implementation:
        1. If no stock position: Sell cash-secured puts
        2. If holding stock: Sell covered calls
        3. Manage assignments and roll positions
        """
        current_price = market_data['prices'][ticker]
        
        # Check current positions
        has_stock = ticker in portfolio.positions and portfolio.positions[ticker].shares > 0
        has_active_put = any(
            opt.ticker == ticker and 
            opt.option_type == 'put' and 
            opt.position == 'short'
            for opt in portfolio.options
        )
        has_active_call = any(
            opt.ticker == ticker and 
            opt.option_type == 'call' and 
            opt.position == 'short'
            for opt in portfolio.options
        )
        
        # Get volatility for pricing
        vol = None
        if ticker in market_data['volatility']:
            vol_data = market_data['volatility'][ticker]
            if len(vol_data) > 0:
                vol = vol_data.iloc[-1, 0]
                if pd.isna(vol):
                    vol = None
        
        if vol is None:
            return  # Can't price options without volatility
        
        # PHASE 1: Sell cash-secured puts when we don't own stock
        if not has_stock and not has_active_put:
            # Calculate how many contracts we can secure with our cash
            put_strike = current_price * put_strike_factor
            
            # Cash needed per contract to secure the put
            cash_per_contract = put_strike * 100
            max_contracts = int(portfolio.cash / cash_per_contract)
            
            if max_contracts > 0:
                expiration = date + timedelta(days=days_to_expiration)
                
                # Calculate put premium using Black-Scholes
                put_premium = black_scholes_put(
                    S=current_price,
                    K=put_strike,
                    sigma=vol,
                    r=interest_rate,
                    t=time
                )
                
                # Only sell if premium meets minimum threshold
                if put_premium / current_price >= min_premium_threshold:
                    # Sell puts to generate income and possibly acquire stock
                    actions.sell_put(
                        portfolio=portfolio,
                        ticker=ticker,
                        strike=put_strike,
                        expiration=expiration,
                        contracts=max_contracts,
                        premium=put_premium
                    )
        
        # PHASE 2: Sell covered calls when we own stock
        elif has_stock and not has_active_call:
            shares = portfolio.positions[ticker].shares
            contracts = shares // 100  # Each contract covers 100 shares
            
            if contracts > 0:
                call_strike = current_price * call_strike_factor
                expiration = date + timedelta(days=days_to_expiration)
                
                # Calculate call premium using Black-Scholes
                call_premium = black_scholes_call(
                    S=current_price,
                    K=call_strike,
                    sigma=vol,
                    r=interest_rate,
                    t=time
                )
                
                # Only sell if premium meets minimum threshold
                if call_premium / current_price >= min_premium_threshold:
                    # Sell covered calls to generate income
                    actions.sell_call(
                        portfolio=portfolio,
                        ticker=ticker,
                        strike=call_strike,
                        expiration=expiration,
                        contracts=contracts,
                        premium=call_premium
                    )
    
    return wheel_strategy


def create_aggressive_wheel_strategy(ticker, interest_rate=0.05):
    """
    More aggressive wheel strategy with closer strikes and shorter duration.
    Higher income potential but higher assignment probability.
    """
    return create_wheel_strategy(
        ticker=ticker,
        put_strike_factor=0.98,  # Closer to current price
        call_strike_factor=1.02,  # Closer to current price
        days_to_expiration=14,  # Shorter duration for more frequent trades
        interest_rate=interest_rate,
        min_premium_threshold=0.005
    )


def create_conservative_wheel_strategy(ticker, interest_rate=0.05):
    """
    More conservative wheel strategy with further OTM strikes and longer duration.
    Lower income but lower assignment probability.
    """
    return create_wheel_strategy(
        ticker=ticker,
        put_strike_factor=0.90,  # Further OTM
        call_strike_factor=1.10,  # Further OTM
        days_to_expiration=45,  # Longer duration
        interest_rate=interest_rate,
        min_premium_threshold=0.015
    )
