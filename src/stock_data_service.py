import logging
import pandas as pd
import pandas_datareader as pdr
import os
import yfinance as yf

from datetime import datetime, timedelta


# ---------------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)


class StockDataService:
    def __init__(self, db_client):
        self.db_client = db_client

    def get_prices_for_tickers(self, tickers, start_date, end_date):
        """
        Check and download the latest closing prices for the given tickers, 
        then insert the data directly into the database. Does not fetch existing data from the database.
        """
        tickers_to_download = []
        for ticker in tickers:
            # Check if the database already has up-to-date data
            if not self.db_client.data_exists(ticker):
                tickers_to_download.append(ticker)
                logging.info(f"Ticker {ticker} needs to be downloaded.")
            else:
                logging.info(f"Ticker {ticker} is already up-to-date.")

        if tickers_to_download:
            try:
                fetched_data = yf.download(
                    tickers=tickers_to_download, 
                    start=start_date, 
                    end=end_date, 
                    progress=False, 
                    group_by='ticker'
                )

                # If single ticker, adjust format
                if isinstance(fetched_data, pd.DataFrame) and 'Date' in fetched_data.columns:
                    fetched_data = {tickers_to_download[0]: fetched_data}

                for ticker in tickers_to_download:
                    if ticker in fetched_data:
                        df = fetched_data[ticker].reset_index()
                        df.rename(columns={
                            'Date': 'date',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        }, inplace=True)
                        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                        
                        # Insert into database
                        self.db_client.insert_stock_data(df, ticker)
                        logging.info(f"Downloaded and inserted data for {ticker}.")
                    else:
                        logging.warning(f"No data found for {ticker}.")
            except Exception as e:
                logging.error(f"Error downloading data: {str(e)}")
        else:
            logging.info("All tickers are already up-to-date.")

        # No nested dictionary returned, as fetching is not needed here
        return tickers_to_download


    def fetch_prices_from_db(self, tickers, start_date, end_date):
        """
        Fetch closing prices, volume, and date for multiple tickers directly from the database.
        """
        nested_dict = {}
        for ticker in tickers:
            df = self.db_client.fetch_prices(ticker, start_date, end_date)
            if not df.empty:
                nested_dict[ticker] = {
                    'date': df['date'].tolist(),
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist(),
                }
                logging.info(f"Fetched data for {ticker} from the database.")
            else:
                logging.warning(f"No data found in the database for {ticker}.")
        return nested_dict

    def get_tickers(self, dir, filename):
        """Retrieve tickers from a specified CSV file."""
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath) and filepath.endswith('.csv'):
            with open(filepath, 'r') as file:
                tickers = [line.strip().split(',')[0] for line in file][1:]
                if filename == 'fx.csv':
                    tickers = [f"{ticker}=X" for ticker in tickers]
                return tickers
        return [] 

    @staticmethod
    def time_delta(years):
        return (datetime.today() - timedelta(weeks=years * 52)).strftime('%Y-%m-%d')

    def fetch_fred_data(self, ticker):
        """Fetch data from FRED for a specific ticker using pandas_datareader."""
        return pdr.DataReader(ticker, 'fred')

    def get_rates_spreads_data(self):
        """Fetch and combine FRED data into a pandas DataFrame."""
        names = [
            '2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield',
            '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note'
        ]
        tickers = [
            'T10Y2Y', 'T5YIE', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLH0A0HYM2EY',
            'DTB3', 'DGS2', 'DGS5', 'DGS10', 'DGS30'
        ]
        
        data = {}
        for ticker, name in zip(tickers, names):
            df = self.fetch_fred_data(ticker).interpolate()
            data[name] = {
                'date': df.index.tolist(),
                'close': df[ticker].tolist()
            }
        return data
