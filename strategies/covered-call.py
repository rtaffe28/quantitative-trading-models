import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

ticker = "MSTR"
tk = yf.Ticker(ticker)

def black_scholes_call(S, K, T, r, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call

def calculate_historical_volatility(prices, window=30):
    returns = np.log(prices / prices.shift(1))
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

data = yf.download(ticker, start="2024-01-01", end="2025-01-01")
data = data[['Close']].copy()
data.reset_index(inplace=True)

data['Volatility'] = calculate_historical_volatility(data['Close'], window=30)

r = 0.045
strike_otm = 0.20
holding_days = 30
shares_per_call = 100

initial_price = data.iloc[0]['Close']
if isinstance(initial_price, pd.Series):
    initial_price = initial_price.iloc[0]
cash = -initial_price * shares_per_call
shares = shares_per_call
portfolio = []
dates = []
premiums_collected = []
calls_exercised = 0

for i in range(30, len(data) - holding_days, holding_days):
    start_price = data.iloc[i]['Close']
    if isinstance(start_price, pd.Series):
        start_price = start_price.iloc[0]
    
    strike = start_price * (1 + strike_otm)
    
    sigma = data.iloc[i]['Volatility']
    if isinstance(sigma, pd.Series):
        sigma = sigma.iloc[0]
    if pd.isna(sigma) or sigma <= 0:
        sigma = 0.50
    
    call_price = black_scholes_call(start_price, strike, holding_days/365, r, sigma)
    premium = call_price * shares_per_call
    cash += premium 
    premiums_collected.append(premium)
    
    next_price = data.iloc[i + holding_days]['Close']
    if isinstance(next_price, pd.Series):
        next_price = next_price.iloc[0]
    
    if next_price > strike:
        cash += strike * shares_per_call
        shares = 0
        calls_exercised += 1
        cash -= next_price * shares_per_call
        shares = shares_per_call
    
    total_value = cash + shares * next_price
    portfolio.append(total_value)
    
    date_val = data.iloc[i + holding_days]['Date']
    if isinstance(date_val, pd.Series):
        date_val = date_val.iloc[0]
    dates.append(date_val)

buy_hold_shares = shares_per_call
buy_hold_values = []
for d in dates:
    date_match = data[data['Date'] == d]
    if len(date_match) > 0:
        idx = date_match.index[0]
        price_val = data.iloc[idx]['Close']
        if isinstance(price_val, pd.Series):
            price_val = price_val.iloc[0]
        buy_hold_values.append(buy_hold_shares * price_val)

final_value = portfolio[-1]
bh_value = buy_hold_values[-1]
initial_investment = initial_price * shares_per_call

covered_call_return = (final_value - initial_investment) / initial_investment
buy_hold_return = (bh_value - initial_investment) / initial_investment
total_premiums = sum(premiums_collected)

# --- 8. Results ---
print(f"=== {ticker} Covered Call Strategy (2024) ===")
print(f"Strike: 20% OTM, Expiry: ~30 days")
print(f"\nInitial Investment: ${initial_investment:,.2f}")
print(f"Final Portfolio Value: ${final_value:,.2f}")
print(f"Buy & Hold Value: ${bh_value:,.2f}")
print(f"\nCovered Call Return: {covered_call_return*100:.2f}%")
print(f"Buy & Hold Return: {buy_hold_return*100:.2f}%")
print(f"Outperformance: {(covered_call_return - buy_hold_return)*100:.2f}%")
print(f"\nTotal Premiums Collected: ${total_premiums:,.2f}")
print(f"Number of Calls Exercised: {calls_exercised}/{len(portfolio)}")

# --- 9. Plot ---
plt.figure(figsize=(12,6))
plt.plot(dates, portfolio, label='Covered Call Portfolio', linewidth=2)
plt.plot(dates, buy_hold_values, label='Buy & Hold', linewidth=2, alpha=0.7)
plt.title(f"Covered Call vs Buy & Hold - {ticker} (2024)", fontsize=14)
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(data['Date'], data['Volatility']*100, label='Historical Volatility (30-day)', color='orange')
plt.title(f"{ticker} Historical Volatility (2024)", fontsize=14)
plt.ylabel("Volatility (%)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

