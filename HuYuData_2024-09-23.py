import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime


# Define the Black-Scholes formula
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S : float : current stock price
    K : float : strike price
    T : float : time to expiration (in years)
    r : float : risk-free interest rate
    sigma : float : volatility of the stock price
    option_type : str : 'call' for call option, 'put' for put option
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # For Call Option
    if option_type == 'call':
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

    # For Put Option
    elif option_type == 'put':
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)

    return price


# Fetch Walmart's current stock price
ticker = 'WMT'
data = yf.Ticker(ticker)
current_price = data.history(period='1d')['Close'].iloc[-1]

# Parameters for the Black-Scholes model
S = current_price  # Current stock price of Walmart (WMT)
K = 150  # Strike price (you can change this value)
T = (datetime(2024, 12, 31) - datetime.now()).days / 365  # Time to expiration (1 year from today)
r = 0.05  # Risk-free interest rate (e.g., 5%)
sigma = 0.25  # Volatility of the stock price (e.g., 25%)

# Calculate Call and Put Option prices
call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

# Print the results
print(f"Walmart (WMT) Current Stock Price: {S:.2f}")
print(f"Strike Price: {K}")
print(f"Time to Expiration (years): {T:.2f}")
print(f"Risk-Free Interest Rate: {r}")
print(f"Volatility: {sigma:.2f}")
print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")

# Walmart (WMT) Current Stock Price: 158.67
# Strike Price: 150
# Time to Expiration (years): 1.10
# Risk-Free Interest Rate: 0.05
# Volatility: 0.25
# Call Option Price: 16.78
# Put Option Price: 9.32