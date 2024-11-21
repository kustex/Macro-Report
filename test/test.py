import asyncio
import matplotlib.pyplot as plt
import logging
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

def main():
    ap = StockDataService('stock_data.db')
    calc = StockCalculations()
    df = ap.get_rates_spreads_data()


    df = pd.DataFrame.from_dict(df)
    columns = df.columns.tolist()

    for c in columns:
        date = df[c].loc['date']
        close = df[c].loc['close']
        print(c, close)

    # results = calc.df_performance_rates_spreads(df)
    # print(results)


if __name__ == "__main__":
    main()