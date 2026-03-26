"""
Volume indicators.
"""

import pandas as pd
from indicators.base import Indicator


def create_volume_spike_indicator(
    lookback: int = 20,
    spike_factor: float = 2.0,
) -> Indicator:
    """
    Returns True when today's volume exceeds the lookback-period average
    by a configurable factor.

    Args:
        lookback: Window for average volume calculation
        spike_factor: Multiplier — current volume must exceed avg × spike_factor
    """
    def volume_spike(df: pd.DataFrame) -> bool:
        if len(df) < lookback + 1:
            return False
        volume = df['Volume']
        if isinstance(volume, pd.DataFrame):
            volume = volume.squeeze()

        avg_vol = float(volume.rolling(window=lookback).mean().iloc[-1])
        current_vol = float(volume.iloc[-1])

        if pd.isna(avg_vol) or avg_vol == 0 or pd.isna(current_vol):
            return False
        return current_vol > avg_vol * spike_factor

    volume_spike.__name__ = f'VolumeSpike({lookback}d,>{spike_factor}x)'
    return volume_spike
