from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable


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
                          pricing_func: Optional[Callable] = None) -> float:
        total = 0
        for opt in self.options:
            if current_date >= opt.expiration_date:
                continue
            
            if pricing_func:
                intrinsic = self._get_intrinsic_value(opt, prices)
                total += intrinsic * opt.contracts * 100
            else:
                intrinsic = self._get_intrinsic_value(opt, prices)
                total += intrinsic * opt.contracts * 100
                
        return total
    
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
                       pricing_func: Optional[Callable] = None) -> float:
        return (self.cash + 
                self.get_stock_value(prices) + 
                self.get_options_value(current_date, prices, pricing_func))
