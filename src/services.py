# src/services.py
from src.database_client import DatabaseClient
from src.stock_data_service import StockDataService

# Use ONE DB file path everywhere (unify these!)
DB_PATH = "data/macro_report.db"  # pick this and stick to it

db_client = DatabaseClient(db_name=DB_PATH)
ap = StockDataService(db_client)
