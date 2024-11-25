import asyncio
import itertools
import os
import pandas as pd

from datetime import datetime
from stock_data_service import StockDataService
from database_client import DatabaseClient

db_client = DatabaseClient('stock_data.db')
ap = StockDataService(db_client)

dir_t = 'res/tickers/'
dir_t_corr = 'res/tickers_corr/'

unique_tickers = list(set(
    itertools.chain.from_iterable(
        ap.get_tickers(dir_path, file) for dir_path in [dir_t, dir_t_corr] for file in os.listdir(dir_path)
    )
))

start_date = (datetime.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

def fetch_and_update_data():
    print(f"Fetching data at {datetime.now()}")
    ap.get_prices_for_tickers(unique_tickers, start_date, end_date)

if __name__ == "__main__":
    fetch_and_update_data()

