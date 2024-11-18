import asyncio
import aiohttp
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

    async def get_prices_for_tickers(self, tickers, start_date, end_date):
        """Fetch closing prices, volume, and date for multiple tickers asynchronously and return a nested dictionary."""
        
        nested_dict = {}  # Initialize a nested dictionary to store results
        tasks = []  # List to store asyncio tasks

        # Loop over tickers to create tasks
        for ticker in tickers:
            tasks.append(self.process_ticker(ticker, start_date, end_date))
        
        # Await all tasks to complete asynchronously
        results = await asyncio.gather(*tasks)
        
        # Iterate over results and build the nested dictionary
        for result in results:
            if result is not None:
                symbol = result['symbol'].iloc[0]  # Get the symbol (assumes all rows in result have the same symbol)
                
                # Initialize an empty dictionary for this symbol if not already present
                if symbol not in nested_dict:
                    nested_dict[symbol] = {'date': [], 'close': [], 'volume': []}
                
                # Append date, close, and volume data to their respective lists in the dictionary
                nested_dict[symbol]['date'].extend(result['date'].tolist())
                nested_dict[symbol]['close'].extend(result['close'].tolist())
                nested_dict[symbol]['volume'].extend(result['volume'].tolist())
        
        # Check if nested_dict is populated
        if nested_dict:
            logging.info("Successfully created nested dictionary for closing prices, volume, and date.")
        else:
            logging.warning("No closing prices data available after processing.")
        
        return nested_dict, tickers

    async def process_ticker(self, ticker, start_date, end_date):
        """Process individual ticker data."""
        try:
            if self.db_client.data_exists(ticker, start_date, end_date):
                df = self.db_client.fetch_prices(ticker)
                df['symbol'] = ticker  
                logging.info(f"Fetched closing prices for {ticker} from the database.")
                return df
            else:
                fetched_data = await self.get_historical_data([ticker])
                if not fetched_data.empty:
                    self.store_data(fetched_data, ticker)
                    logging.info(f"Inserted closing prices for {ticker} into the database.")
                    return fetched_data
                else:
                    logging.warning(f"No historical data found for {ticker}.")
                    return None

        except Exception as e:
            logging.error(f"Error processing {ticker}: {str(e)}")
            return None

    async def get_historical_data(self, tickers):
        """Fetch historical data for multiple tickers asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_historical_data(session, ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            return pd.concat([result for result in results if result is not None], ignore_index=True)

    async def fetch_historical_data(self, session, ticker):
        """Fetch historical data for a single ticker using yfinance."""
        try:
            df = yf.download(ticker, progress=False)
            if not df.empty:
                df.reset_index(inplace=True)
                df['symbol'] = ticker
                df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                logging.info(f"Fetched historical data for {ticker}.")
                return df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]
            else:
                logging.warning(f"No data returned for {ticker}.")
        except Exception as e:
            logging.error(f"Error fetching historical data for {ticker}: {str(e)}")
        return None

    def store_data(self, df, ticker):
        """Store the fetched data in the database."""
        self.db_client.insert_stock_data(df)
        logging.info(f"Stored {len(df)} ---- {ticker} records in the database.")

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

    async def fetch_fred_data(self, ticker):
        ''' Asynchronously fetch data from FRED for a specific ticker using pandas_datareader. '''
        return await asyncio.to_thread(pdr.DataReader, ticker, 'fred')

    async def df_rates_spreads(self):
        '''
        Connects to the Federal Reserve Economic Data and retrieves data into a pandas DataFrame asynchronously.
        '''
        names = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', 
                '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']
        tickers = ['T10Y2Y', 'T5YIE', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLH0A0HYM2EY', 
                'DTB3', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
        
        tasks = [self.fetch_fred_data(tick) for tick in tickers]
        results = await asyncio.gather(*tasks)
        combined_df = pd.concat(results, axis=1)
        combined_df.columns = names 
        return combined_df.interpolate()