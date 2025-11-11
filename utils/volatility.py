import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# returns the historical volatility over a period of time for a given stock
def volatility(ticker, start, end, window=30):
    tk = yf.Ticker(ticker)
    data = tk.history(start=start, end=end)
    
    returns = np.log(data['Close'] / data['Close'].shift(1)l)
    
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    return vol