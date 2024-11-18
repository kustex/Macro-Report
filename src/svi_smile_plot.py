import datetime as dt
import numpy as np
import yfinance as yf
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from svi import SVICalculator  # Import the SVICalculator class

# Initialize variables and load data
ticker = 'SPY'
stock_data = yf.Ticker(ticker)

# Convert option expiration dates to datetime and find the closest to 1 month
expiries = [dt.datetime.strptime(date, '%Y-%m-%d') for date in stock_data.options]
t_1m = dt.datetime.now() + dt.timedelta(weeks=4)
expiry_t_1m = {abs(t_1m.timestamp() - date.timestamp()): date for date in expiries}
res = expiry_t_1m[min(expiry_t_1m.keys())].strftime("%Y-%m-%d")

# Get option chains for the selected expiry date
option_chain = stock_data.option_chain(res)
chain_calls = option_chain.calls
chain_puts = option_chain.puts

# Set parameters for Black-Scholes and SVI
T = 1 / 12  # Time to maturity (in years)
r = 0.03  # Risk-free rate
delta = 0.2 # Moneyness range for strike prices
S = stock_data.history(period='1d')['Close'].iloc[0]  # Underlying spot price
sigma_guess = 0.1  # Initial volatility guess

# Create an instance of SVICalculator
svi_calc = SVICalculator(S, r, delta, sigma_guess)

# Calculate Black-Scholes implied volatilities for calls and puts
strikes_calls, bs_vols_calls, strikes_puts, bs_vols_puts = svi_calc.black_scholes_implied_vol(T, chain_calls, chain_puts)

# Combine strike prices and implied volatilities
strikes = np.concatenate((strikes_puts, strikes_calls))
implied_vols = np.concatenate((bs_vols_puts, bs_vols_calls))

# # Remove NaN values
valid_indices = ~np.isnan(implied_vols)
strikes = strikes[valid_indices]
implied_vols = implied_vols[valid_indices]

# Calculate log moneyness and market variance (w_market)
k_values = np.log(strikes / S)
w_market = implied_vols ** 2

# Fit SVI parameters using least squares
svi_params = svi_calc.fit_svi_ls(k_values, w_market)

# Calculate SVI volatilities based on fitted parameters
strikes_svi, svi_vols_combined = svi_calc.svi(chain_calls, chain_puts, svi_params)

# Sort the moneyness and implied volatility data for plotting
moneyness = strikes / S
sorted_indices = np.argsort(moneyness)
sorted_moneyness = moneyness[sorted_indices]
sorted_implied_vols = implied_vols[sorted_indices]

# Sort the SVI volatilities for plotting
moneyness_svi = strikes_svi / S
sorted_indices_svi = np.argsort(moneyness_svi)
sorted_moneyness_svi = moneyness_svi[sorted_indices_svi]
sorted_svi_vols = np.array(svi_vols_combined)[sorted_indices_svi]

# Spline interpolation for smoother plot
smoothing_param = 25
spline = CubicSpline(sorted_moneyness, sorted_implied_vols, extrapolate=True)
new_x = np.linspace(sorted_moneyness.min(), sorted_moneyness.max(), smoothing_param)
new_y = spline(new_x)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(sorted_moneyness, sorted_implied_vols, label='Implied Volatilities')
plt.plot(sorted_moneyness_svi, sorted_svi_vols, label='SVI', color='orange')

plt.title(f'Implied Volatility Surface for {ticker} ({res})')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.legend()

plt.grid()
plt.show()
