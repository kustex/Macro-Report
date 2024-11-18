import datetime as dt
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import CubicSpline
from svi import SVICalculator

class SVIOptions:
    def __init__(self, ticker, time_to_maturity_weeks=4, risk_free_rate=0.03, moneyness=0.1, sigma_guess=0.1):
        self.ticker = ticker
        self.T_weeks = time_to_maturity_weeks
        self.r = risk_free_rate
        self.delta = moneyness
        self.sigma_guess = sigma_guess
        self.stock_data = yf.Ticker(ticker)

    def get_closest_expiry(self):
        expiries = [dt.datetime.strptime(date, '%Y-%m-%d') for date in self.stock_data.options]
        target_date = dt.datetime.now() + dt.timedelta(weeks=self.T_weeks)
        closest_expiry = min(expiries, key=lambda date: abs((date - target_date).total_seconds()))
        return closest_expiry.strftime("%Y-%m-%d")

    def calculate_and_plot_volatility_smile(self):
        expiry_date = self.get_closest_expiry()
        option_chain = self.stock_data.option_chain(expiry_date)
        chain_calls, chain_puts = option_chain.calls, option_chain.puts
        S = self.stock_data.history(period='1d')['Close'].iloc[0]
        T = 1 / 12  # Use the same T as in svi_smile_plot.py

        # Initialize SVICalculator with consistent parameters
        svi_calc = SVICalculator(S, self.r, self.delta, self.sigma_guess)

        # Calculate Black-Scholes implied volatilities
        strikes_calls, bs_vols_calls, strikes_puts, bs_vols_puts = svi_calc.black_scholes_implied_vol(T, chain_calls, chain_puts)

        # Combine and filter data
        strikes = np.concatenate((strikes_puts, strikes_calls))
        implied_vols = np.concatenate((bs_vols_puts, bs_vols_calls))
        valid_indices = ~np.isnan(implied_vols)
        strikes = strikes[valid_indices]
        implied_vols = implied_vols[valid_indices]

        # Log key variables for comparison
        print("strikes:", strikes)
        print("implied_vols:", implied_vols)

        # Calculate log moneyness and market variance (w_market)
        k_values = np.log(strikes / S)
        w_market = implied_vols ** 2

        # Fit SVI parameters
        svi_params = svi_calc.fit_svi_ls(k_values, w_market)
        if svi_params is None:
            st.error("Failed to fit SVI parameters.")
            return

        # Calculate SVI volatilities based on fitted parameters
        strikes_svi, svi_vols_combined = svi_calc.svi(chain_calls, chain_puts, svi_params)

        # Sort data for plotting
        moneyness = strikes / S
        sorted_indices = np.argsort(moneyness)
        sorted_moneyness = moneyness[sorted_indices]
        sorted_implied_vols = implied_vols[sorted_indices]

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.scatter(sorted_moneyness, sorted_implied_vols, label='Implied Volatilities')
        plt.plot(strikes_svi / S, svi_vols_combined, label='SVI', color='orange')
        plt.title(f'Implied Volatility Surface for {self.ticker} ({expiry_date})')
        plt.xlabel('Moneyness')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        st.pyplot(plt)
