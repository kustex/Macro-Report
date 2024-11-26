from pymongo import MongoClient, UpdateOne
from datetime import datetime
import pandas as pd


class DatabaseClient:
    def __init__(self, mongo_uri="mongodb://ip-172-31-87-70.ec2.internal:27017", db_name="macro_report"):
        """
        Initialize the MongoDB client and specify the database and collection.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db["stock_data"]

    def data_exists(self, symbol):
        """
        Check if the latest data for the given symbol exists in the database.
        """
        latest_entry = self.collection.find_one({"symbol": symbol}, sort=[("date", -1)])
        if not latest_entry:
            return False

        # Check if the latest date matches the expected business day
        now = datetime.utcnow()
        expected_date = pd.date_range(end=now, periods=1, freq="B")[-1].date()
        return latest_entry["date"] == expected_date.strftime("%Y-%m-%d")

    def fetch_prices(self, symbol, start_date, end_date):
        """
        Fetch prices for a given symbol and date range from the database.
        """
        cursor = self.collection.find(
            {
                "symbol": symbol,
                "date": {"$gte": start_date, "$lte": end_date},
            },
            {"_id": 0, "date": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        )
        return pd.DataFrame(list(cursor))

    def insert_stock_data(self, df, ticker):
        """
        Insert stock data into MongoDB, avoiding duplicates.
        """
        operations = []
        for _, row in df.iterrows():
            operations.append(
                UpdateOne(
                    {"symbol": ticker, "date": row["date"]},
                    {
                        "$set": {
                            "symbol": ticker,
                            "date": row["date"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"],
                        }
                    },
                    upsert=True,
                )
            )

        if operations:
            self.collection.bulk_write(operations)

    def close(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()
