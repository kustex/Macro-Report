import asyncio
import aiohttp
import logging
import pandas as pd
import pandas_datareader as pdr
import os
import yfinance as yf

from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)


class StockDataService:
    def __init__(self, db_client):
        self.db_client = db_client

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

    async def get_historical_data(self, tickers):
        """Fetch historical data for multiple tickers asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_historical_data(session, ticker) for ticker in tickers]
            results = await asyncio.gather(*tasks)
            return pd.concat([result for result in results if result is not None], ignore_index=True)

    def store_data(self, df):
        """Store the fetched data in the database."""
        self.db_client.insert_stock_data(df)
        logging.info(f"Stored {len(df)} records in the database.")

    async def get_closing_prices_for_tickers(self, tickers, start_date, end_date):
        """Fetch closing prices for multiple tickers asynchronously."""
        closing_prices_df = pd.DataFrame()  

        tasks = [] 

        for ticker in tickers:
            tasks.append(self.process_ticker(ticker, start_date, end_date))
        
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result is not None:
                closing_prices_df = pd.concat([closing_prices_df, result], ignore_index=True)

        if not closing_prices_df.empty:
            closing_prices_df = closing_prices_df.pivot(index='date', columns='symbol', values='close')
            logging.info("Successfully created closing prices DataFrame.")
        else:
            logging.warning("No closing prices data available after processing.")

        return closing_prices_df, tickers
    
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

    async def process_ticker(self, ticker, start_date, end_date):
        """Process individual ticker data."""
        try:
            if self.db_client.data_exists(ticker, start_date, end_date):
                df = self.db_client.fetch_closing_prices(ticker)
                df['symbol'] = ticker  
                logging.info(f"Fetched closing prices for {ticker} from the database.")
                return df
            else:
                fetched_data = await self.get_historical_data([ticker])
                if not fetched_data.empty:
                    self.store_data(fetched_data)
                    logging.info(f"Inserted closing prices for {ticker} into the database.")
                    return fetched_data
                else:
                    logging.warning(f"No historical data found for {ticker}.")
                    return None

        except Exception as e:
            logging.error(f"Error processing {ticker}: {str(e)}")
            return None

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