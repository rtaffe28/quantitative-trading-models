"""
Buy and Hold Strategy

Simple passive strategy that buys and holds stock for the entire period.
"""


def create_buy_and_hold_strategy(ticker):
    """
    Factory function to create a buy and hold strategy.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        A strategy callback function for BacktestSimulation
    """
    
    def buy_and_hold(date, portfolio, market_data, actions):
        """Buy stock once with all available cash and hold"""
        current_price = market_data['prices'][ticker]
        
        # Buy stock with all available cash on first opportunity
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares == 0:
            max_shares = int(portfolio.cash / current_price)
            if max_shares > 0:
                actions.buy_stock(portfolio, ticker, max_shares, current_price)
    
    return buy_and_hold
