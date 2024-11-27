import itertools
import os
import pandas as pd
from datetime import datetime
from stock_data_service import StockDataService
from database_client import DatabaseClient

# Initialize MongoDB client
db_client = DatabaseClient(mongo_uri="mongodb://ip-172-31-87-70.ec2.internal:27017", db_name="macro_report")
# db_client = DatabaseClient(db_name="macro_report")
ap = StockDataService(db_client)

# Directory structure for tickers
dir_t = "res/tickers/"
dir_t_corr = "res/tickers_corr/"

# Get unique tickers
unique_tickers = list(
    set(
        itertools.chain.from_iterable(
            ap.get_tickers(dir_path, file)
            for dir_path in [dir_t, dir_t_corr]
            for file in os.listdir(dir_path)
        )
    )
)

# Define date range for fetching data
start_date = (datetime.today() - pd.DateOffset(years=100)).strftime("%Y-%m-%d")
end_date = datetime.today().strftime("%Y-%m-%d")


def fetch_and_update_data():
    """
    Fetch and update data for all unique tickers.
    """
    print(f"Fetching data at {datetime.now()}")
    ap.get_prices_for_tickers(unique_tickers, start_date, end_date)


if __name__ == "__main__":
    fetch_and_update_data()
