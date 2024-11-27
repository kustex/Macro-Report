import pandas as pd
import logging

from pymongo import MongoClient, UpdateOne
from datetime import datetime

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------------------

class DatabaseClient:
    def __init__(self, mongo_uri="mongodb://ip-172-31-31-149.ec2.internal:27017", db_name="macro_report"):
    # def __init__(self, db_name="macro_report"):
        """
        Initialize the MongoDB client and specify the database and collection.
        """
        # self.client = MongoClient("mongodb://localhost:27017")
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["stock_data"]
        self.collection.create_index([("symbol", 1), ("date", 1)], unique=True)

    def data_exists(self, symbol):
        """
        Check if the latest data for the given symbol exists in the database.
        """
        latest_entry = self.collection.find_one({"symbol": symbol}, sort=[("date", -1)])
        if not latest_entry:
            return False

        # Check if the latest date matches the expected business day
        now = datetime.now()
        expected_date = pd.date_range(end=now, periods=1, freq="B")[-1].date()
        return latest_entry["date"] == expected_date.strftime("%Y-%m-%d")

    def fetch_prices(self, symbol, start_date=None, end_date=None):
        """
        Fetch prices for a given symbol and date range from the database.
        Returns a dictionary of lists for the symbol.
        """
        query = {"symbol": symbol}
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}

        cursor = self.collection.find(
            query,
            {"_id": 0, "date": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        )

        # Convert cursor to DataFrame and then to dictionary
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return {
                "date": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }

        # Convert DataFrame to dictionary of lists
        return df.to_dict(orient="list")

    def insert_stock_data(self, df, ticker, batch_size=1000):
        """
        Insert stock data into MongoDB in batches, avoiding duplicates.
        """
        if df.empty:
            logging.warning(f"No data to insert for {ticker}.")
            return

        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict(orient="records")

            # Prepare bulk update operations
            operations = [
                UpdateOne(
                    {"symbol": ticker, "date": record["date"]},
                    {"$set": record},
                    upsert=True
                )
                for record in records
            ]

            # Execute bulk operations in batches
            for i in range(0, len(operations), batch_size):
                self.collection.bulk_write(operations[i:i + batch_size])
        except Exception as e:
            logging.error(f"Error storing data for {ticker}: {e}")


    def close(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()