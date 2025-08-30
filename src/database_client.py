import pandas as pd
import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class DatabaseClient:
    def __init__(self, db_name="macro_report.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False, timeout=30.0)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA busy_timeout=30000;")
        self._create_table()

    def _create_table(self):
        """
        Create the stock_data table if it doesn't exist.
        """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        """)
        self.conn.commit()

    def data_exists(self, symbol):
        """
        Check if the latest data for the given symbol exists in the database.
        """
        self.cursor.execute("""
            SELECT date FROM stock_data
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1
        """, (symbol,))
        result = self.cursor.fetchone()

        if not result:
            return False

        latest_date = result[0]
        expected_date = pd.date_range(end=datetime.now(), periods=1, freq="B")[-1].strftime("%Y-%m-%d")
        return latest_date == expected_date

    def fetch_prices(self, symbol, start_date=None, end_date=None):
        """
        Fetch prices for a given symbol and date range from the database.
        Returns a dictionary of lists for the symbol.
        """
        query = "SELECT date, open, high, low, close, volume FROM stock_data WHERE symbol = ?"
        params = [symbol]

        if start_date and end_date:
            query += " AND date BETWEEN ? AND ?"
            params.extend([start_date, end_date])

        df = pd.read_sql_query(query, self.conn, params=params)

        if df.empty:
            return {
                "date": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }

        return df.to_dict(orient="list")

    def insert_stock_data(self, df, ticker, batch_size=1000):
        """
        Insert stock data into SQLite in batches, avoiding duplicates.
        """
        if df.empty:
            return

        try:
            records = df.to_dict(orient="records")
            batch = [
                (
                    ticker,
                    r["date"],
                    r["open"],
                    r["high"],
                    r["low"],
                    r["close"],
                    r["volume"]
                ) for r in records
            ]

            for i in range(0, len(batch), batch_size):
                self.cursor.executemany("""
                    INSERT OR REPLACE INTO stock_data
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, batch[i:i + batch_size])
                self.conn.commit()

        except Exception as e:
            logging.error(f"Error inserting data for {ticker}: {e}")

    def close(self):
        self.conn.close()
