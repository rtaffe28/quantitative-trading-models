"""
Trading Strategies Module

This module contains reusable trading strategy implementations that can be
imported into any notebook or script.

Available strategies:
- LEAP Strategy: Buy long-dated call options for leverage
- Covered Call Strategy: Sell calls against stock holdings for income
- Wheel Strategy: Sell cash-secured puts and covered calls for income
- Moving Average Strategies: Technical analysis with MA crossovers
- Buy and Hold Strategy: Passive long-term holding

Usage:
    from strategies.leap_strategy import create_leap_strategy
    from strategies.covered_call_strategy import create_covered_call_strategy
    from strategies.wheel_strategy import create_wheel_strategy
    from strategies.moving_average_strategy import create_sma_crossover_strategy
    from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy
"""

from strategies.leap_strategy import create_leap_strategy
from strategies.covered_call_strategy import create_covered_call_strategy
from strategies.wheel_strategy import (
    create_wheel_strategy,
    create_aggressive_wheel_strategy,
    create_conservative_wheel_strategy
)
from strategies.moving_average_strategy import (
    create_sma_crossover_strategy,
    create_ema_crossover_strategy,
    create_triple_ma_strategy,
    create_adaptive_ma_strategy
)
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy

__all__ = [
    'create_leap_strategy',
    'create_covered_call_strategy',
    'create_wheel_strategy',
    'create_aggressive_wheel_strategy',
    'create_conservative_wheel_strategy',
    'create_sma_crossover_strategy',
    'create_ema_crossover_strategy',
    'create_triple_ma_strategy',
    'create_adaptive_ma_strategy',
    'create_buy_and_hold_strategy',
]
