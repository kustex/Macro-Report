import os
import pandas as pd
import sqlite3


class DatabaseClient:
    def __init__(self, db_name):
        self.db_dir_name = 'data'
        self.db_name = os.path.join(self.db_dir_name, db_name)
        self.ensure_data_folder_exists()
        self.create_table()  
    
    def ensure_data_folder_exists(self):
        # Create the 'data' directory if it doesn't exist
        if not os.path.exists(self.db_dir_name):
            os.makedirs(self.db_dir_name)

    def create_table(self):
        # Create the table if it does not exist
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

    def data_exists(self, symbol, start_date, end_date):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                SELECT COUNT(*) FROM stock_data 
                WHERE symbol = ? AND date BETWEEN ? AND ?
            """, (symbol, start_date, end_date))
            count = cursor.fetchone()[0]
        return count > 0

    def fetch_closing_prices(self, symbol):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
                SELECT date, close FROM stock_data 
                WHERE symbol = ? 
                ORDER BY date
            """, (symbol,))
            return pd.DataFrame(cursor.fetchall(), columns=['date', 'close'])

    def insert_stock_data(self, df):
        """Insert stock data into the database, avoiding duplicates."""
        # Prepare the SQL command
        sql = """
            INSERT OR IGNORE INTO stock_data (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        # Convert 'date' column to string format if it's a Timestamp
        df['date'] = df['date'].astype(str)  # Convert Timestamps to string
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, df.values)
            conn.commit()

    def close(self):
        pass  # No need to close connection since it's handled in context managers