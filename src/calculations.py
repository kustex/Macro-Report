import asyncio
import datetime as dt
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import seaborn as sns

from datetime import datetime, timedelta, date
from plotly.subplots import make_subplots
from plotly.tools import mpl_to_plotly
from stock_data_service import StockDataService
from database_client import DatabaseClient

# -----------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

db_client = DatabaseClient('stock_data.db')  
ap = StockDataService(db_client)


class StockCalculations:
    def df_performance_rates_spreads(self, df):
        '''
        Imports historical data using df_rates_spreads() and calculates percentage changes of each financial instrument over different time horizons.
        '''
        window_names = ['Ticker', 'Price', '1D', '1W', '1M', '3M', 'vs 52w max', 'vs 52w min', 'vs 3Y ave', 'vs 5Y ave']
        ticker_list = df.columns.tolist()

        len_week, len_1m, len_3m = self.get_lengths_periods()[:3]
        results = []

        for ticker in ticker_list:
            data = df[ticker]
            latest = data.iloc[-1]

            periods = [2, len_week, len_1m, len_3m]
            changes = [latest - data.iloc[-time] if latest > data.iloc[-time] 
                    else -(data.iloc[-time] - latest) for time in periods]

            yearly_high = data.iloc[-252:].max()
            yearly_low = data.iloc[-252:].min()
            y3_avg = data.iloc[-756:].mean()
            y5_avg = data.iloc[-1260:].mean()

            vs_52_max = -(yearly_high - latest)
            vs_52_min = (latest - yearly_low)
            vs_y3_avg = latest - y3_avg if latest > y3_avg else -(y3_avg - latest)
            vs_y5_avg = latest - y5_avg if latest > y5_avg else -(y5_avg - latest)

            results.append([ticker, data.iloc[-2].round(2)] + 
                        ["{:.1f}".format(value * 100) for value in changes + 
                            [vs_52_max, vs_52_min, vs_y3_avg, vs_y5_avg]])

        df = pd.DataFrame(results, columns=window_names)
        return df
    
    def get_end_date(self):
        belgium_tz = pytz.timezone('Europe/Brussels')
        current_time = datetime.now(belgium_tz)
        cutoff_time = current_time.replace(hour=22, minute=0, second=0, microsecond=0)

        if current_time > cutoff_time:
            return current_time.strftime('%Y-%m-%d')  # Use today's date
        else:
            return (current_time - timedelta(days=1)).strftime('%Y-%m-%d')

    def get_volume_for_ticker(self, data, ticker):
        """
        Processes volume data to compute rolling averages in reverse.
        
        Returns:
            pd.DataFrame: A DataFrame containing dates as index and rolling average volumes as columns.
        """
        volume = pd.Series(data[ticker]['volume'])
        dates = pd.Series(data[ticker]['date'])
        
        # Ensure there are no missing values in the volume data
        volume.fillna(0, inplace=True)
        
        # Define the rolling windows
        windows = {
            'Vol 5D Avg': 5,
            'Vol 3W Avg': 15,  
            'Vol 1M Avg': 20,  
            'Vol 3M Avg': 60,  
            'Vol 1Y Avg': 252  
        }

        rolling_avgs = {}
        for avg_name, window in windows.items():
            rolling_avg = volume.rolling(window=window).mean()
            rolling_avgs[avg_name] = rolling_avg

        df = pd.DataFrame(rolling_avgs)
        df['Date'] = dates.values  # Assign dates as a new column, not the index yet
        df.dropna(how='all', inplace=True)
        df.set_index('Date', inplace=True)
        return df

    def plot_rolling_avg_volumes(self, df, ticker):
        """
        Plots the rolling average volumes against dates using Plotly, selecting the last 3 years.

        Parameters:
            df (pd.DataFrame): DataFrame containing rolling average volumes with dates as index.
            ticker (str): The ticker symbol for the title of the plot.
        """
        df.index = pd.to_datetime(df.index)

        # Get today's date and filter for the last 3 years
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=3)
        
        # Filter the DataFrame for the last 3 years
        df_filtered = df.loc[start_date:end_date]

        # Create a Plotly figure
        fig = go.Figure()

        # Add traces for each rolling average
        for column in df_filtered.columns:
            fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[column], mode='lines', name=column))

        # Update the layout of the figure
        fig.update_layout(
            title=f'Rolling Average Volume for {ticker} (Last 3 Years)',
            xaxis_title='Date',
            yaxis_title='Volume',
            hovermode='x unified',
            template='plotly_dark'
        )

        # Show the figure
        fig.show()


    def get_performance(self, data):
        """
        Creates a DataFrame showing price changes over different time horizons in percentage terms.
        This version processes the nested dictionary input and only calculates price performance.
        """
        if not data:
            raise ValueError("Input dictionary is empty. Cannot fetch data.")

        window_names = ['Ticker', 'Price', '1D', '1W', '3W', '1M', 'MTD', '3M', 'QTD', 
                        'YTD', 'vs 52w max', 'vs 52w min']
        
        end_date = self.get_end_date()  
        performance_data = []

        periods = {
            '1D': dt.timedelta(days=1),
            '1W': dt.timedelta(weeks=1),
            '3W': dt.timedelta(weeks=3),
            '1M': dt.timedelta(days=30),
            'MTD': dt.datetime.today().replace(day=1),
            '3M': dt.timedelta(days=90),
            'QTD': dt.datetime.today().replace(month=((dt.datetime.today().month - 1) // 3) * 3 + 1, day=1),
            'YTD': dt.datetime(dt.datetime.today().year, 1, 1)
        }

        for ticker, ticker_data in data.items():
            dates = pd.to_datetime(ticker_data['date'])
            close_prices = pd.Series(ticker_data['close'], index=dates)
            close_prices = close_prices[close_prices.index <= end_date]

            if close_prices.empty:
                continue
            latest_price = close_prices.iloc[-1]  
            results = [latest_price.round(2)]  
            current_date = close_prices.index[-1]  
            for period_name, delta in periods.items():
                if isinstance(delta, dt.date):
                    period_date = delta
                else:
                    period_date = current_date - delta
                try:
                    period_price = close_prices.loc[close_prices.index <= period_date].iloc[-1]
                    change = (latest_price - period_price) / period_price  
                    results.append("{:.2%}".format(change))
                except (IndexError, KeyError):  
                    results.append(None)

            one_year_ago = current_date - dt.timedelta(weeks=52)
            one_year_data = close_prices.loc[one_year_ago:current_date]
            yearly_high = one_year_data.max()
            yearly_low = one_year_data.min()
            vs_52_max = (latest_price - yearly_high) / yearly_high if yearly_high else None
            vs_52_min = (latest_price - yearly_low) / yearly_low if yearly_low else None
            results.extend(["{:.2%}".format(vs_52_max) if vs_52_max is not None else None,
                            "{:.2%}".format(vs_52_min) if vs_52_min is not None else None])
            performance_data.append([ticker] + results)
        performance_df = pd.DataFrame(performance_data, columns=window_names)
        return performance_df.sort_values(by='Ticker').set_index('Ticker')
    
    def get_performance_vs_rolling_mean(self, data):
        """
        Creates a DataFrame showing price changes vs rolling averages over different time horizons in percentage terms.
        This version processes the nested dictionary input and calculates performance relative to rolling means.
        """
        if not data:
            raise ValueError("Input dictionary is empty. Cannot fetch data.")

        window_names = ['Ticker', 'Price', '1W_avg', '3W_avg', '1M_avg', '3M_avg', '1Y_avg', '3Y_avg']
        # window_names = ['1W_avg', '3W_avg', '1M_avg', '3M_avg', '1Y_avg', '3Y_avg']
        end_date = self.get_end_date()
        performance_data = []

        windows = {
            '1W': 5,
            '3W': 15,
            '1M': 21,
            '3M': 63,
            '1Y': 252,
            '3Y': 756
        }

        for ticker, ticker_data in data.items():
            dates = pd.to_datetime(ticker_data['date'])
            close_prices = pd.Series(ticker_data['volume'], index=dates)
            close_prices = close_prices[close_prices.index <= end_date]

            if close_prices.empty:
                continue
            
            latest_price = close_prices.iloc[-1]
            # results = []
            results = [ticker, latest_price.round(2)]
            
            for label, window in windows.items():
                rolling_mean = close_prices.rolling(window=window).mean()
                recent_mean = rolling_mean.iloc[-1] if not rolling_mean.empty else None
                if recent_mean:
                    performance_vs_avg = (latest_price - recent_mean) / recent_mean * 100
                    results.append("{:.2f}%".format(performance_vs_avg))
                else:
                    results.append(None)

            performance_data.append(results)

        performance_df = pd.DataFrame(performance_data, columns=window_names).sort_values(by='Ticker').set_index('Ticker')
        return performance_df.drop(columns='Price')
    
    def get_one_year_range(self, index):
        one_year_ago = pd.Timestamp(dt.date.today() - dt.timedelta(weeks=52))
        return index >= one_year_ago

    def filter_data_by_date(self, data, from_date):
        """
        Filters the DataFrame to only include data from the specified start date.
        """
        data_filtered = data[data.index >= from_date]
        return data_filtered

    def get_lengths_periods(self):
            """
            Define the lengths of various periods for performance calculations.
            """
            len_week = 5
            len_3w = 15
            len_1m = 22
            len_mtd = datetime.today().day
            len_3m = 66
            len_qtd = len_mtd + 22 * (datetime.today().month % 3)
            len_ytd = (datetime.today() - datetime(datetime.today().year, 1, 1)).days // 7 * 5
            return len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd
    
    def get_correlation_table_window_x(self, data, value):
        '''
        This function creates a pandas dataframe. In it, correlations (pearson method) between different contracts are being calculated and shown, over different time horizons.
        '''
        end_date = self.get_end_date()
        window_list = [15, 30, 90, 120, 180]
        tickerlist = list(data.keys())
        TICKER = value
        MULTP_TICKERS = [item for item in tickerlist if value not in item]

        dataframe = pd.DataFrame()
        for window in window_list:
                for i in MULTP_TICKERS:
                    dates_ticker = pd.to_datetime(data[TICKER]['date'])
                    close_prices_ticker = pd.Series(data[TICKER]['close'], index=dates_ticker)
                    close_prices_ticker = close_prices_ticker[close_prices_ticker.index <= end_date]

                    dates_i = pd.to_datetime(data[i]['date'])
                    close_prices_i = pd.Series(data[i]['close'], index=dates_i)
                    close_prices_i = close_prices_i[close_prices_i.index <= end_date]

                    combined_data = pd.concat([close_prices_ticker, close_prices_i], axis=1).dropna()
                    combined_data.columns = [TICKER, i]

                    dataframe[f'{TICKER}_{i}_{window}'] = combined_data[TICKER].rolling(window).corr(combined_data[i])

        day_15 = dataframe.filter(regex='15')
        day_30 = dataframe.filter(regex='30')
        day_90 = dataframe.filter(regex='90')
        day_120 = dataframe.filter(regex='120')
        day_180 = dataframe.filter(regex='180')

        start_date = pd.Timestamp(dt.date.today() - timedelta(weeks=52))
        day_30 = day_30[day_30.index >= start_date]
        rolling_30d_max = [np.nanmax(day_30[i]) for i in list(day_30.columns)]
        rolling_30d_min = [np.nanmin(day_30[i]) for i in list(day_30.columns)]

        data = pd.DataFrame()
        data['15D'] = np.array(day_15.iloc[-1, :])
        data['30D'] = np.array(day_30.iloc[-1, :])
        data['90D'] = np.array(day_90.iloc[-1, :])
        data['120D'] = np.array(day_120.iloc[-1, :])
        data['180D'] = np.array(day_180.iloc[-1, :])
        data['1y 30d high'] = rolling_30d_max
        data['1y 30d low'] = rolling_30d_min

        data[f"{TICKER} vs"] = MULTP_TICKERS
        cols = list(data.columns)
        cols = [cols[-1]] + cols[:-1]
        data = data[cols]
        return data.round(2), dataframe

    def create_correlation_graph(self, dataframe, ticker_list, selected_ticker):
        '''
        This function automates creation of scatterplots of correlations between tickers over different time horizons.
        '''
        dt_3m = date.today() - pd.DateOffset(months=2)
        dataframe.index = pd.to_datetime(dataframe.index)
        dataframe = dataframe[dataframe.index >= dt_3m]
        
        time_horizons = ['15', '30', '90', '120', '180']
        time_horizon_names = [f'{i}D' for i in time_horizons]

        color_palette = plt.cm.tab20.colors 
        colors = {ticker: f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)' 
                  for ticker, (r, g, b) in zip(ticker_list, color_palette)}

        # Initialize the figure with subplots
        num_rows = int(np.ceil(len(time_horizons) / 2))
        fig = make_subplots(
            rows=num_rows, 
            cols=2,
            subplot_titles=time_horizon_names
        )

        for idx, t in enumerate(time_horizons):
            row = idx // 2 + 1
            col = idx % 2 + 1

            for ticker in ticker_list:
                if ticker != selected_ticker:
                    fig.add_trace(go.Scatter(
                        x=dataframe[f'{selected_ticker}_{ticker}_{t}'].index,
                        y=dataframe[f'{selected_ticker}_{ticker}_{t}'],
                        showlegend=False,
                        name=ticker,
                        marker=dict(color=colors[ticker]) 
                    ), col=col, row=row)
            fig.update_yaxes(range=[1,-1], title_text='rho', row=row, col=col)
            fig.update_xaxes(title_text="date", row=row, col=col)

        fig.update_layout(
            autosize=True,
            height=1000,
            title=f'{ticker} rolling correlations',
            title_x=0.5
        )

        return fig
    
    def chart_rates_spreads(self, data, TICKER):
        '''
        This function creates a interactive line chart from the dataframe inputed'''
        x = data.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.array(data), name=TICKER))
        fig.update_xaxes(title='date')
        fig.update_yaxes(title='performance')
        return fig

    def generate_returns_graph(self, selected_ticker, start_date, end_date):
        data, _ = asyncio.run(ap.get_prices_for_tickers([selected_ticker], start_date, end_date))
        date = pd.to_datetime(data[selected_ticker]['date'])
        returns = data[selected_ticker]['close']
        df = pd.DataFrame({'date': date, 'returns': returns}).set_index('date')

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['returns'],
            mode='lines',
            name='Returns'
        ))

        # Update layout with an x-axis range slider
        fig.update_layout(
            xaxis=dict(
                title="Date",
            ),
            yaxis=dict(
                title="Returns",
                autorange=True  # Enable autoscaling for the y-axis
            ),
            height=550
        )
        return fig

    def create_volume_and_rolling_avg_graph(self, selected_ticker, start_date, end_date):
        # begin_data_date = (dt.date.today() - pd.DateOffset(years=10)).strftime('%Y-%m-%d') 
        data, _ = asyncio.run(ap.get_prices_for_tickers([selected_ticker], start_date, end_date))
        date = pd.to_datetime(data[selected_ticker]['date'])
        volume = data[selected_ticker]['volume']
        df = pd.DataFrame({'date': date, 'volume': volume}).set_index('date')

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]

        windows = {
            '1W': 5,
            '3W': 15,
            '1M': 21,
            '3M': 63,
            '1Y': 252,
            '3Y': 756
        }

        fig = go.Figure()
        for label, window in windows.items():
            rolling_avg = df['volume'].rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=rolling_avg.index,
                    y=rolling_avg.values,
                    mode='lines',
                    name=f"{label} Avg",
                    # line=dict(dash='dash')
                )
            )
        fig.update_layout(
            # title=f"{selected_ticker} Volume and Rolling Averages",
            xaxis=dict(
                title="Date",
            ),
            yaxis=dict(
                title="Volume",
                autorange=True  # Enable autoscaling for the y-axis
            ),
            height=550
        )
        return fig
