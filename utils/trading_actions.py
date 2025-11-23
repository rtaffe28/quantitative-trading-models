from datetime import datetime
from typing import List, Dict
from utils.portfolio import Portfolio, Position, OptionContract


class TradingAction:
    _transaction_log = None
    _current_date = None
    
    @classmethod
    def set_transaction_log(cls, transaction_log: List[Dict], current_date: datetime):
        cls._transaction_log = transaction_log
        cls._current_date = current_date
    
    @staticmethod
    def buy_stock(portfolio: Portfolio, ticker: str, shares: int, price: float) -> bool:
        cost = shares * price
        if cost > portfolio.cash:
            return False
        
        portfolio.cash -= cost
        if ticker in portfolio.positions:
            pos = portfolio.positions[ticker]
            total_shares = pos.shares + shares
            portfolio.positions[ticker].avg_cost = (pos.avg_cost * pos.shares + cost) / total_shares
            portfolio.positions[ticker].shares = total_shares
        else:
            portfolio.positions[ticker] = Position(ticker, shares, price)
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'BUY_STOCK',
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'total': cost
            })
        
        return True
    
    @staticmethod
    def sell_stock(portfolio: Portfolio, ticker: str, shares: int, price: float) -> bool:
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares < shares:
            return False
        
        proceeds = shares * price
        portfolio.cash += proceeds
        portfolio.positions[ticker].shares -= shares
        
        if portfolio.positions[ticker].shares == 0:
            del portfolio.positions[ticker]
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'SELL_STOCK',
                'ticker': ticker,
                'shares': shares,
                'price': price,
                'total': proceeds
            })
        
        return True
    
    @staticmethod
    def buy_call(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                  contracts: int, premium: float) -> bool:
        premium_cost = premium * contracts * 100
        if portfolio.cash < premium_cost:
            return False
        
        portfolio.cash -= premium_cost
        
        portfolio.options.append(OptionContract(
            ticker=ticker,
            strike=strike,
            expiration_date=expiration,
            option_type='call',
            contracts=contracts,
            premium_received=-premium_cost,
            position='long'
        ))
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'BUY_CALL',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_cost,
                'expiration': expiration
            })
        
        return True
    
    @staticmethod
    def sell_call(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                  contracts: int, premium: float) -> bool:
        required_shares = contracts * 100
        if ticker not in portfolio.positions or portfolio.positions[ticker].shares < required_shares:
            return False
        
        premium_collected = premium * contracts * 100
        portfolio.cash += premium_collected
        
        portfolio.options.append(OptionContract(
            ticker=ticker,
            strike=strike,
            expiration_date=expiration,
            option_type='call',
            contracts=contracts,
            premium_received=premium_collected,
            position='short'
        ))
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'SELL_CALL',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_collected,
                'expiration': expiration
            })
        
        return True
    
    @staticmethod
    def sell_put(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                 contracts: int, premium: float) -> bool:
        premium_collected = premium * contracts * 100
        portfolio.cash += premium_collected
        
        portfolio.options.append(OptionContract(
            ticker=ticker,
            strike=strike,
            expiration_date=expiration,
            option_type='put',
            contracts=contracts,
            premium_received=premium_collected,
            position='short'
        ))
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'SELL_PUT',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_collected,
                'expiration': expiration
            })
        
        return True
    
    @staticmethod
    def buy_put(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                 contracts: int, premium: float) -> bool:
        premium_cost = premium * contracts * 100
        if portfolio.cash < premium_cost:
            return False
        
        portfolio.cash -= premium_cost
        
        portfolio.options.append(OptionContract(
            ticker=ticker,
            strike=strike,
            expiration_date=expiration,
            option_type='put',
            contracts=contracts,
            premium_received=-premium_cost,
            position='long'
        ))
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'BUY_PUT',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_cost,
                'expiration': expiration
            })
        
        return True
    
    @staticmethod
    def close_call(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                   contracts: int, premium: float) -> bool:
        option_index = None
        for i, opt in enumerate(portfolio.options):
            if (opt.ticker == ticker and 
                opt.option_type == 'call' and 
                opt.position == 'long' and
                opt.strike == strike and
                opt.expiration_date == expiration and
                opt.contracts >= contracts):
                option_index = i
                break
        
        if option_index is None:
            return False
        
        premium_collected = premium * contracts * 100
        portfolio.cash += premium_collected
        
        opt = portfolio.options[option_index]
        if opt.contracts == contracts:
            portfolio.options.pop(option_index)
        else:
            portfolio.options[option_index].contracts -= contracts
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'CLOSE_CALL',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_collected,
                'expiration': expiration
            })
        
        return True
    
    @staticmethod
    def close_put(portfolio: Portfolio, ticker: str, strike: float, expiration: datetime,
                  contracts: int, premium: float) -> bool:
        option_index = None
        for i, opt in enumerate(portfolio.options):
            if (opt.ticker == ticker and 
                opt.option_type == 'put' and 
                opt.position == 'long' and
                opt.strike == strike and
                opt.expiration_date == expiration and
                opt.contracts >= contracts):
                option_index = i
                break
        
        if option_index is None:
            return False
        
        premium_collected = premium * contracts * 100
        portfolio.cash += premium_collected
        
        opt = portfolio.options[option_index]
        if opt.contracts == contracts:
            portfolio.options.pop(option_index)
        else:
            portfolio.options[option_index].contracts -= contracts
        
        if TradingAction._transaction_log is not None:
            TradingAction._transaction_log.append({
                'date': TradingAction._current_date,
                'action': 'CLOSE_PUT',
                'ticker': ticker,
                'contracts': contracts,
                'strike': strike,
                'premium_per_share': premium,
                'total_premium': premium_collected,
                'expiration': expiration
            })
        
        return True
