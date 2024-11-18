import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
import datetime as dt

class SVICalculator:
    def __init__(self, S, r=0.03, delta=0.1, sigma_guess=0.1):
        self.S = S  # Spot price
        self.r = r  # Risk-free rate
        self.delta = delta  # Moneyness range
        self.sigma_guess = sigma_guess  # Initial guess for volatility

    def black_scholes(self, K, T, sigma, option_type):
        """Calculate the Black-Scholes price for a given option."""
        d1 = (np.log(self.S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'C':
            return self.S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        return K * np.exp(-self.r * T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def implied_volatility(self, K, T, option_price, option_type):
        """Calculate implied volatility for a given option using Black-Scholes."""
        def objective(sigma):
            return (self.black_scholes(K, T, sigma, option_type) - option_price) ** 2
        result = minimize(objective, x0=self.sigma_guess, bounds=[(0.0002, 5)])
        return result.x[0] if result.success else np.nan

    @staticmethod
    def svi_parametrization(k, a, b, rho, m, sigma):
        """SVI (Stochastic Volatility Inspired) parametrization for variance."""
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    @staticmethod
    def svi_residuals(params, k_values, w_market):
        """Residuals between market and SVI model variances for optimization."""
        w_svi = SVICalculator.svi_parametrization(k_values, *params)
        return w_svi - w_market

    def fit_svi_ls(self, k_values, w_market):
        """Fit SVI parameters to market data using least squares optimization."""
        initial_params = [0.04, 0.2, -0.3, 0.0, 0.1]
        result = least_squares(
            self.svi_residuals, initial_params, args=(k_values, w_market),
            bounds=([0, 0, -1, -1, 0], [1, 2, 1, 1, 2]), method='trf'
        )
        if result.success:
            print("Optimization successful!")
            return result.x
        print("Optimization failed:", result.message)
        return None

    def filter_options(self, chain, option_type, T):
        """Filter options within moneyness range and calculate implied volatilities."""
        if option_type == 'C':
            filtered_chain = chain[(chain['strike'] >= self.S * (1 - self.delta)) &
                                   (chain['strike'] <= self.S * (1 + self.delta))]
        else:
            filtered_chain = chain[(chain['strike'] >= self.S * (1 - self.delta)) &
                                   (chain['strike'] <= self.S * (1 + self.delta))]
        
        strikes, implied_vols = [], []
        for _, row in filtered_chain.iterrows():
            K, price = row['strike'], row['lastPrice']
            vol = self.implied_volatility(K, T, price, option_type)
            if not np.isnan(vol):
                strikes.append(K)
                implied_vols.append(vol ** 2)  # Store variance for SVI fitting

        return np.array(strikes), np.array(implied_vols)

    def black_scholes_implied_vol(self, T, chain_calls, chain_puts):
        """Calculate implied volatilities for both calls and puts."""
        strikes_calls, implied_vols_calls = self.filter_options(chain_calls, 'C', T)
        strikes_puts, implied_vols_puts = self.filter_options(chain_puts, 'P', T)
        return strikes_calls, implied_vols_calls, strikes_puts, implied_vols_puts

    def svi(self, chain_calls, chain_puts, svi_params):
        """Calculate SVI volatilities based on fitted parameters."""
        strikes, svi_vols = [], []
        combined_chain = pd.concat([chain_calls, chain_puts])
        
        for _, row in combined_chain.iterrows():
            K = row['strike']
            log_moneyness = np.log(K / self.S)
            svi_variance = self.svi_parametrization(log_moneyness, *svi_params)
            svi_vol = np.sqrt(svi_variance)
            strikes.append(K)
            svi_vols.append(svi_vol)

        return np.array(strikes), np.array(svi_vols)

    def calculate_vol_surface(self, symbol, T_months=1):
        """Calculate the SVI volatility surface for a given ticker."""
        stock_data = yf.Ticker(symbol)
        expiries = [dt.datetime.strptime(date, '%Y-%m-%d') for date in stock_data.options]
        target_expiry = min(expiries, key=lambda date: abs((date - (dt.datetime.now() + dt.timedelta(days=30 * T_months))).days))
        
        strikes_surface, vol_surface, maturities_surface = [], [], []
        T = (target_expiry - dt.datetime.now()).days / 365.0
        self.S = stock_data.history(period='1d')['Close'].iloc[0]

        option_chain = stock_data.option_chain(target_expiry.strftime('%Y-%m-%d'))
        strikes_calls, w_market_calls = self.filter_options(option_chain.calls, 'C', T)
        strikes_puts, w_market_puts = self.filter_options(option_chain.puts, 'P', T)

        # Fit SVI parameters
        k_values = np.log(np.concatenate((strikes_puts, strikes_calls)) / self.S)
        w_market = np.concatenate((w_market_puts, w_market_calls))
        svi_params = self.fit_svi_ls(k_values, w_market)

        if svi_params is not None:
            strikes_svi, svi_vols_combined = self.svi(option_chain.calls, option_chain.puts, svi_params)
            strikes_surface.extend(strikes_svi)
            maturities_surface.extend([T] * len(strikes_svi))
            vol_surface.extend(svi_vols_combined)

        return np.array(strikes_surface), np.array(maturities_surface), np.array(vol_surface)
