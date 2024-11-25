import itertools
import matplotlib.pyplot as plt
import logging
import os
import pandas as pd
import sqlite3

from src.calculations import StockCalculations
from src.database_client import DatabaseClient
from src.stock_data_service import StockDataService
from datetime import datetime, timedelta


# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

def adapt_datetime(dt):
    return dt.strftime('%Y-%m-%d')

def convert_datetime(s):
    return datetime.strptime(s.decode('utf-8'), '%Y-%m-%d')

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("DATE", convert_datetime)

db_client = DatabaseClient('stock_data.db')  
ap = StockDataService(db_client)
dir_t = 'res/tickers/'
dir_t_corr = 'res/tickers_corr/'

unique_tickers = list(set(
    itertools.chain.from_iterable(
        ap.get_tickers(dir_path, file) for dir_path in [dir_t, dir_t_corr] for file in os.listdir(dir_path)
    )
))

def main():
    start_date = (datetime.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = ap.fetch_prices_from_db(unique_tickers, start_date, end_date) 
    print(data)
    return data

if __name__ == "__main__":
    main()