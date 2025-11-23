import numpy as np
from scipy.stats import norm

def black_scholes_call(S: float, K: float, sigma: float, r: float, t: float) -> float:
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C


def black_scholes_put(S: float, K: float, sigma: float, r: float, t: float) -> float:
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    P = K * np.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return P