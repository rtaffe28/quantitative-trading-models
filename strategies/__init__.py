"""
Trading Strategies Module

This module contains reusable trading strategy implementations that can be
imported into any notebook or script.

Available strategies:
- LEAP Strategy: Buy long-dated call options for leverage
- Covered Call Strategy: Sell calls against stock holdings for income
- Buy and Hold Strategy: Passive long-term holding

Usage:
    from strategies.leap_strategy import create_leap_strategy
    from strategies.covered_call_strategy import create_covered_call_strategy
    from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy
"""

from strategies.leap_strategy import create_leap_strategy
from strategies.covered_call_strategy import create_covered_call_strategy
from strategies.buy_and_hold_strategy import create_buy_and_hold_strategy

__all__ = [
    'create_leap_strategy',
    'create_covered_call_strategy',
    'create_buy_and_hold_strategy',
]
