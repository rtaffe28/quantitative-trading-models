import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

def historical_volatility(ticker: str, start: datetime, end: datetime, window: int = 30) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    
    data = tk.history(start=start-timedelta(days=window*2), end=end)
    returns = np.log(data['Close'] / data['Close'].shift(1))
    
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    if vol.index.tz is not None and start.tzinfo is None:
        start = pd.Timestamp(start).tz_localize(vol.index.tz)
    
    filtered_vol = vol[vol.index >= start]

    return pd.DataFrame({"volatility": filtered_vol})