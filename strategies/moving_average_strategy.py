"""
Moving Average Strategy

Technical analysis strategy that uses moving average crossovers to generate
buy and sell signals.

Common approaches:
- Simple Moving Average (SMA): Average price over N days
- Golden Cross: Short MA crosses above long MA (bullish signal)
- Death Cross: Short MA crosses below long MA (bearish signal)
"""

import pandas as pd
import numpy as np


def create_sma_crossover_strategy(
    ticker,
    short_window=50,
    long_window=200,
    allocation=1.0
):
    """
    Factory function to create a simple moving average (SMA) crossover strategy.
    
    Strategy Rules:
    - BUY signal: When short-term MA crosses above long-term MA (Golden Cross)
    - SELL signal: When short-term MA crosses below long-term MA (Death Cross)
    
    Args:
        ticker: Stock ticker symbol
        short_window: Number of days for short-term moving average (default: 50)
        long_window: Number of days for long-term moving average (default: 200)
        allocation: Percentage of portfolio to allocate to trades (0-1, default: 1.0)
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    
    # Track previous crossover state
    state = {
        'prev_short_ma': None,
        'prev_long_ma': None,
        'position_entered': False
    }
    
    def sma_crossover(date, portfolio, market_data, actions):
        """
        Execute SMA crossover strategy.
        Buy when short MA crosses above long MA.
        Sell when short MA crosses below long MA.
        """
        current_price = float(market_data['prices'][ticker])
        
        # Get historical data up to current date
        if ticker not in market_data['data']:
            return
        
        hist_data = market_data['data'][ticker]
        
        # Need enough data to calculate both moving averages
        if len(hist_data) < long_window:
            return
        
        # Calculate moving averages
        prices = hist_data['Close']
        short_ma = float(prices.rolling(window=short_window).mean().iloc[-1])
        long_ma = float(prices.rolling(window=long_window).mean().iloc[-1])
        
        # Skip if MA calculation failed
        if pd.isna(short_ma) or pd.isna(long_ma):
            return
        
        # Check for crossover signals
        if state['prev_short_ma'] is not None and state['prev_long_ma'] is not None:
            
            # Golden Cross: Short MA crosses above Long MA (BUY signal)
            if (state['prev_short_ma'] <= state['prev_long_ma'] and 
                short_ma > long_ma and 
                not state['position_entered']):
                
                # Buy stock with allocated cash
                available_cash = portfolio.cash * allocation
                max_shares = int(available_cash / current_price)
                
                if max_shares > 0:
                    actions.buy_stock(portfolio, ticker, max_shares, current_price)
                    state['position_entered'] = True
            
            # Death Cross: Short MA crosses below Long MA (SELL signal)
            elif (state['prev_short_ma'] >= state['prev_long_ma'] and 
                  short_ma < long_ma and 
                  state['position_entered']):
                
                # Sell all shares
                if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
                    shares = portfolio.positions[ticker].shares
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                    state['position_entered'] = False
        
        # Update state for next iteration
        state['prev_short_ma'] = short_ma
        state['prev_long_ma'] = long_ma
    
    return sma_crossover


def create_ema_crossover_strategy(
    ticker,
    short_window=12,
    long_window=26,
    allocation=1.0
):
    """
    Factory function to create an exponential moving average (EMA) crossover strategy.
    
    EMA gives more weight to recent prices, making it more responsive to new information
    than SMA. Commonly used with shorter periods (12/26 or 9/21).
    
    Args:
        ticker: Stock ticker symbol
        short_window: Number of days for short-term EMA (default: 12)
        long_window: Number of days for long-term EMA (default: 26)
        allocation: Percentage of portfolio to allocate to trades (0-1, default: 1.0)
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    
    state = {
        'prev_short_ema': None,
        'prev_long_ema': None,
        'position_entered': False
    }
    
    def ema_crossover(date, portfolio, market_data, actions):
        """
        Execute EMA crossover strategy.
        """
        current_price = float(market_data['prices'][ticker])
        
        if ticker not in market_data['data']:
            return
        
        hist_data = market_data['data'][ticker]
        
        if len(hist_data) < long_window:
            return
        
        # Calculate exponential moving averages
        prices = hist_data['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()  # Convert DataFrame to Series
        short_ema = float(prices.ewm(span=short_window, adjust=False).mean().iloc[-1])
        long_ema = float(prices.ewm(span=long_window, adjust=False).mean().iloc[-1])
        
        if pd.isna(short_ema) or pd.isna(long_ema):
            return
        
        # Check for crossover signals
        if state['prev_short_ema'] is not None and state['prev_long_ema'] is not None:
            
            # Bullish crossover: Buy signal
            if (state['prev_short_ema'] <= state['prev_long_ema'] and 
                short_ema > long_ema and 
                not state['position_entered']):
                
                available_cash = portfolio.cash * allocation
                max_shares = int(available_cash / current_price)
                
                if max_shares > 0:
                    actions.buy_stock(portfolio, ticker, max_shares, current_price)
                    state['position_entered'] = True
            
            # Bearish crossover: Sell signal
            elif (state['prev_short_ema'] >= state['prev_long_ema'] and 
                  short_ema < long_ema and 
                  state['position_entered']):
                
                if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
                    shares = portfolio.positions[ticker].shares
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                    state['position_entered'] = False
        
        # Update state
        state['prev_short_ema'] = short_ema
        state['prev_long_ema'] = long_ema
    
    return ema_crossover


def create_triple_ma_strategy(
    ticker,
    fast_window=10,
    medium_window=50,
    slow_window=200,
    allocation=1.0
):
    """
    Factory function to create a triple moving average strategy.
    
    This strategy uses three moving averages to confirm trends:
    - BUY: Fast > Medium > Slow (strong uptrend)
    - SELL: Fast < Medium (trend weakening)
    
    Args:
        ticker: Stock ticker symbol
        fast_window: Fast MA period (default: 10)
        medium_window: Medium MA period (default: 50)
        slow_window: Slow MA period (default: 200)
        allocation: Percentage of portfolio to allocate (0-1, default: 1.0)
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    
    state = {'position_entered': False}
    
    def triple_ma(date, portfolio, market_data, actions):
        """
        Execute triple MA strategy with trend confirmation.
        """
        current_price = float(market_data['prices'][ticker])
        
        if ticker not in market_data['data']:
            return
        
        hist_data = market_data['data'][ticker]
        
        if len(hist_data) < slow_window:
            return
        
        # Calculate all three moving averages
        prices = hist_data['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()  # Convert DataFrame to Series
        fast_ma = float(prices.rolling(window=fast_window).mean().iloc[-1])
        medium_ma = float(prices.rolling(window=medium_window).mean().iloc[-1])
        slow_ma = float(prices.rolling(window=slow_window).mean().iloc[-1])
        
        if pd.isna(fast_ma) or pd.isna(medium_ma) or pd.isna(slow_ma):
            return
        
        # Strong uptrend: Fast > Medium > Slow
        if (fast_ma > medium_ma > slow_ma and not state['position_entered']):
            available_cash = portfolio.cash * allocation
            max_shares = int(available_cash / current_price)
            
            if max_shares > 0:
                actions.buy_stock(portfolio, ticker, max_shares, current_price)
                state['position_entered'] = True
        
        # Trend weakening: Fast crosses below Medium
        elif (fast_ma < medium_ma and state['position_entered']):
            if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
                shares = portfolio.positions[ticker].shares
                actions.sell_stock(portfolio, ticker, shares, current_price)
                state['position_entered'] = False
    
    return triple_ma


def create_adaptive_ma_strategy(
    ticker,
    short_window=20,
    long_window=50,
    allocation=1.0,
    volatility_threshold=0.02
):
    """
    Factory function to create an adaptive moving average strategy.
    
    This strategy adjusts its behavior based on market volatility:
    - In low volatility: Uses standard MA crossover
    - In high volatility: Requires additional confirmation to avoid whipsaws
    
    Args:
        ticker: Stock ticker symbol
        short_window: Short MA period (default: 20)
        long_window: Long MA period (default: 50)
        allocation: Percentage of portfolio to allocate (0-1, default: 1.0)
        volatility_threshold: Threshold for high volatility regime (default: 0.02)
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    
    state = {
        'prev_short_ma': None,
        'prev_long_ma': None,
        'position_entered': False,
        'confirmation_days': 0
    }
    
    def adaptive_ma(date, portfolio, market_data, actions):
        """
        Execute adaptive MA strategy with volatility adjustment.
        """
        current_price = float(market_data['prices'][ticker])
        
        if ticker not in market_data['data']:
            return
        
        hist_data = market_data['data'][ticker]
        
        if len(hist_data) < long_window:
            return
        
        # Calculate moving averages
        prices = hist_data['Close']
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()  # Convert DataFrame to Series
        short_ma = float(prices.rolling(window=short_window).mean().iloc[-1])
        long_ma = float(prices.rolling(window=long_window).mean().iloc[-1])
        
        if pd.isna(short_ma) or pd.isna(long_ma):
            return
        
        # Calculate recent volatility
        returns = prices.pct_change().dropna()
        if len(returns) >= 20:
            current_volatility = float(returns.tail(20).std())
        else:
            current_volatility = 0
        
        # Determine if we're in high volatility regime
        high_volatility = current_volatility > volatility_threshold
        required_confirmation = 3 if high_volatility else 1
        
        if state['prev_short_ma'] is not None and state['prev_long_ma'] is not None:
            
            # Check for bullish signal
            if short_ma > long_ma and not state['position_entered']:
                # In high volatility, require multiple days of confirmation
                if state['prev_short_ma'] > state['prev_long_ma']:
                    state['confirmation_days'] += 1
                else:
                    state['confirmation_days'] = 1
                
                if state['confirmation_days'] >= required_confirmation:
                    available_cash = portfolio.cash * allocation
                    max_shares = int(available_cash / current_price)
                    
                    if max_shares > 0:
                        actions.buy_stock(portfolio, ticker, max_shares, current_price)
                        state['position_entered'] = True
                        state['confirmation_days'] = 0
            
            # Check for bearish signal
            elif short_ma < long_ma and state['position_entered']:
                if state['prev_short_ma'] < state['prev_long_ma']:
                    state['confirmation_days'] += 1
                else:
                    state['confirmation_days'] = 1
                
                if state['confirmation_days'] >= required_confirmation:
                    if ticker in portfolio.positions and portfolio.positions[ticker].shares > 0:
                        shares = portfolio.positions[ticker].shares
                        actions.sell_stock(portfolio, ticker, shares, current_price)
                        state['position_entered'] = False
                        state['confirmation_days'] = 0
            else:
                state['confirmation_days'] = 0
        
        # Update state
        state['prev_short_ma'] = short_ma
        state['prev_long_ma'] = long_ma
    
    return adaptive_ma
