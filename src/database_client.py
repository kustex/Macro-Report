import os
import pandas as pd
import sqlite3

from datetime import datetime, timedelta

class DatabaseClient:
    def __init__(self, db_name):
        self.db_dir_name = 'data'
        self.db_name = os.path.join(self.db_dir_name, db_name)
        self.ensure_data_folder_exists()
        self.create_table()  
    
    def ensure_data_folder_exists(self):
        if not os.path.exists(self.db_dir_name):
            os.makedirs(self.db_dir_name)

    def create_table(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    UNIQUE (symbol, date)
                )
            """)
            conn.commit()

    def data_exists(self, symbol):
        """
        Check if the database is up-to-date with the latest closing price
        for the given symbol, considering business days and time (CET).
        """
        now = datetime.now()
        cet_offset = timedelta(hours=1)  # Adjust to CET
        current_cet_time = now + cet_offset

        # Determine the latest expected date
        if current_cet_time.hour >= 22 and current_cet_time.isoweekday() <= 5:  # After 22:00 CET Mon-Fri
            expected_date = current_cet_time.date()
        else:
            # For early hours (before 22:00 CET or weekend), get the last business day
            last_business_day = pd.date_range(end=current_cet_time, periods=1, freq="B")
            expected_date = last_business_day[-1].date()

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(date) FROM stock_data WHERE symbol = ?
            """, (symbol,))
            latest_date_in_db = cursor.fetchone()[0]

        if latest_date_in_db is None:
            return False

        latest_date_in_db = datetime.strptime(latest_date_in_db, "%Y-%m-%d").date()
        return latest_date_in_db == expected_date


    def fetch_prices(self, symbol, start_date, end_date):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                SELECT date, close, volume FROM stock_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """, (symbol, start_date, end_date))
            return pd.DataFrame(cursor.fetchall(), columns=['date', 'close', 'volume'])

    def insert_stock_data(self, df, ticker):
        df = df.copy()
        df['symbol'] = ticker
        """Insert stock data into the database, avoiding duplicates."""
        sql = """
            INSERT OR IGNORE INTO stock_data (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        df = df[columns]
        df['date'] = df['date'].astype(str)
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, df.values.tolist())
            conn.commit()

    def close(self):
        pass  # No need to close connection since it's handled in context managers
