from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from utils.black_scholes import black_scholes_call, black_scholes_put


@dataclass
class Position:
    """Represents a stock position"""
    ticker: str
    shares: float
    avg_cost: float


@dataclass
class OptionContract:
    """Represents an option contract"""
    ticker: str
    strike: float
    expiration_date: datetime
    option_type: str  # 'call' or 'put'
    contracts: int  # Number of contracts (100 shares each)
    premium_received: float  # Premium received when sold (negative if bought)
    position: str  # 'long' or 'short'


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    options: List[OptionContract] = field(default_factory=list)
    
    def get_stock_value(self, prices: Dict[str, float]) -> float:
        return sum(pos.shares * prices.get(pos.ticker, 0) for pos in self.positions.values())
    
    def get_options_value(self, current_date: datetime, prices: Dict[str, float], 
                          volatility: Optional[float] = None, risk_free_rate: float = 0.05) -> float:
        total = 0
        for opt in self.options:
            if current_date >= opt.expiration_date:
                continue
            
            current_price = prices.get(opt.ticker, 0)
            if current_price == 0:
                continue
            
            # Calculate time to expiration in years
            days_to_expiration = (opt.expiration_date - current_date).days
            if days_to_expiration <= 0:
                continue
            time_to_expiration = days_to_expiration / 365.0
            
            # Calculate option value
            if volatility is not None and time_to_expiration > 0:
                # Use Black-Scholes for full option value (intrinsic + time value)
                if opt.option_type == 'call':
                    option_value = black_scholes_call(
                        S=current_price,
                        K=opt.strike,
                        sigma=volatility,
                        r=risk_free_rate,
                        t=time_to_expiration
                    )
                else:  # put
                    option_value = black_scholes_put(
                        S=current_price,
                        K=opt.strike,
                        sigma=volatility,
                        r=risk_free_rate,
                        t=time_to_expiration
                    )
            else:
                option_value = self._get_intrinsic_value_simple(opt, prices)
            
            if opt.position == 'long':
                total += option_value * opt.contracts * 100
            else:
                total -= option_value * opt.contracts * 100
                
        return total
    
    def _get_intrinsic_value_simple(self, opt: OptionContract, prices: Dict[str, float]) -> float:
        """Calculate just the intrinsic value (no time value)"""
        current_price = prices.get(opt.ticker, 0)
        
        if opt.option_type == 'call':
            return max(0, current_price - opt.strike)
        else:
            return max(0, opt.strike - current_price)
    
    def _get_intrinsic_value(self, opt: OptionContract, prices: Dict[str, float]) -> float:
        current_price = prices.get(opt.ticker, 0)
        
        if opt.option_type == 'call':
            intrinsic = max(0, current_price - opt.strike)
        else:
            intrinsic = max(0, opt.strike - current_price)
        
        if opt.position == 'short':
            return -intrinsic
        return intrinsic
    
    def get_total_value(self, current_date: datetime, prices: Dict[str, float],
                       volatility: Optional[float] = None, risk_free_rate: float = 0.05) -> float:
        return (self.cash + 
                self.get_stock_value(prices) + 
                self.get_options_value(current_date, prices, volatility, risk_free_rate))
