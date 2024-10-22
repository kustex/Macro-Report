import asyncio
import itertools
import time
import os

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from app import app, ap  
from calculations import StockCalculations 
from stock_data_service import StockDataService
from database_client import DatabaseClient


# Initialize global instances for database and services
db_client = DatabaseClient('stock_data.db')  
ap = StockDataService(db_client)
calc = StockCalculations()

# Set date range
start_date = ap.time_delta(2)
end_date = datetime.today().strftime('%Y-%m-%d')
dir_t = 'res/tickers/'
dir_t_corr = 'res/tickers_corr/' 

tickers = []
for file in os.listdir(dir_t):
    tickers.append(ap.get_tickers(dir_t, file))
tickers = list(itertools.chain.from_iterable(tickers))

tickers_corr = []
for file in os.listdir(dir_t_corr):
    tickers_corr.append(ap.get_tickers(dir_t_corr, file))
tickers_corr = list(itertools.chain.from_iterable(tickers_corr))

all_tickers = list(itertools.chain.from_iterable([tickers + tickers_corr]))
unique_tickers = list(set(all_tickers))


# Function to fetch the latest financial data
async def fetch_latest_data():
    print(f"Fetching latest data at {datetime.now()}")
    await ap.get_closing_prices_for_tickers(unique_tickers, start_date, end_date)  

def run_async_job(job):
    # Create a new event loop for the thread and run the job
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(job())

def start_scheduler():
    scheduler = BackgroundScheduler()
    # Schedule the job to run every day at 10:00 PM CET
    scheduler.add_job(run_async_job, 'cron', hour=14, minute=14, args=[fetch_latest_data], timezone='CET')
    scheduler.start()
    print("Scheduler started")

if __name__ == '__main__':
    start_scheduler()
    app.run_server(debug=True)
