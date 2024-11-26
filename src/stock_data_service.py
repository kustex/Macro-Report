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
        Download and store data for the given tickers in one batch.
        """
        # Step 1: Identify tickers that need downloading
        tickers_to_download = [
            ticker for ticker in tickers if not self.db_client.data_exists(ticker)
        ]
        logging.info(f"Tickers to download: {tickers_to_download}")

        if not tickers_to_download:
            logging.info("All tickers are already up-to-date.")
            return []

        # Step 2: Download data in one batch
        data_dict = self.download_data_for_tickers(tickers_to_download, start_date, end_date)

        # Step 3: Store data in the database
        self.store_data_in_database(data_dict)

        return tickers_to_download


    def download_data_for_tickers(self, tickers, start_date, end_date):
        """
        Download data for all tickers in one batch using yfinance.
        Returns a dictionary of DataFrames where keys are ticker symbols.
        """
        try:
            logging.info(f"Downloading data for tickers: {tickers}")

            # Download data for all tickers in one batch
            fetched_data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker'
            )

            # If single ticker, adjust format
            if isinstance(fetched_data, pd.DataFrame) and 'Date' in fetched_data.columns:
                fetched_data = {tickers[0]: fetched_data}

            data_dict = {}
            for ticker in tickers:
                if ticker in fetched_data:
                    # Process the ticker's DataFrame
                    df = fetched_data[ticker].reset_index()
                    df.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)

                    # Ensure date is in string format
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    # Clean and validate data
                    df = df[["date", "open", "high", "low", "close", "volume"]]
                    df = df.astype({
                        'open': 'float',
                        'high': 'float',
                        'low': 'float',
                        'close': 'float',
                        'volume': 'float'
                    }, errors='ignore')  # Ignore if conversion isn't needed

                    data_dict[ticker] = df
                    logging.info(f"Successfully processed data for {ticker}.")
                else:
                    logging.warning(f"No data found for {ticker}. Skipping.")

            return data_dict
        except Exception as e:
            logging.error(f"Error downloading data in batch: {e}")
            return {}

    def store_data_in_database(self, data_dict):
        """
        Store the downloaded data into MongoDB.
        `data_dict` is a dictionary where keys are ticker symbols and values are DataFrames.
        """
        for ticker, df in data_dict.items():
            try:
                logging.info(f"Storing data for {ticker} in the database.")
                self.db_client.insert_stock_data(df, ticker)
                logging.info(f"Successfully stored data for {ticker}.")
            except Exception as e:
                logging.error(f"Error storing data for {ticker}: {e}")

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