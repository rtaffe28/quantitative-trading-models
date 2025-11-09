import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# returns the historical volatility over a period of time for a given stock
def volatility(ticker, start, end, window=30):
    tk = yf.Ticker(ticker)
    data = tk.history(start=start, end=end)
    
    print(data)
    returns = np.log(data['Close'] / data['Close'].shift(1))
    
    vol = returns.rolling(window=window).std() * np.sqrt(252)
    
    return vol


vol = volatility("GOOG", datetime.now()-timedelta(days=1000), datetime.now())
print(vol)

# Plot the volatility
plt.figure(figsize=(12, 6))
plt.plot(vol.index, vol * 100, label='Historical Volatility (30-day)', linewidth=2)
plt.title('GOOG Historical Volatility', fontsize=14)
plt.ylabel('Volatility (%)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()