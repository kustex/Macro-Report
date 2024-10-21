import asyncio
import matplotlib.pyplot as plt
import logging
import sqlite3

from calculations import StockCalculations
from database_client import DatabaseClient
from stock_data_service import StockDataService
from datetime import datetime, timedelta


logging.basicConfig(level=logging.INFO)

def adapt_datetime(dt):
    return dt.strftime('%Y-%m-%d')

def convert_datetime(s):
    return datetime.strptime(s.decode('utf-8'), '%Y-%m-%d')

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("DATE", convert_datetime)

async def main():
    db_client = DatabaseClient('stock_data.db')  
    ap = StockDataService(db_client)
    calc = StockCalculations()
    dir = 'tickers_corr/'
    filename = 'correlations_etfs.csv'

    # Set date range
    start_date = ap.time_delta(2)
    end_date = datetime.today().strftime('%Y-%m-%d')

    tickers = ap.get_tickers(dir, 'correlations_etfs.csv')
    df, ticker_list = await ap.get_closing_prices_for_tickers(tickers, start_date, end_date)

    df_correlation, dataframe = calc.get_correlation_table_window_x(df, 'UUP')
    fig = calc.create_correlation_graph(dataframe, ticker_list, 'UUP')
    plt.show()
    # return fig

if __name__ == "__main__":
    asyncio.run(main())