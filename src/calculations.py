import datetime as dt
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz

from datetime import datetime, timedelta, date
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)


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

    def get_performance(self, data, start_date=None):
        """
        Creates a DataFrame showing price changes over different time horizons in percentage terms.
        """
        if data.empty:
            raise ValueError("DataFrame is empty. Cannot fetch latest date.")

        window_names = ['Ticker', 'Price', '1D', '1W', '3W', '1M', 'MTD', '3M', 'QTD', 
                        'YTD', 'vs 52w max', 'vs 52w min']

        end_date = self.get_end_date()
        data.index = pd.to_datetime(data.index)
        data = data[data.index <= end_date]
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

        for ticker in data.columns:
            data_perf = data[ticker]
            latest_price = data_perf.iloc[-1]
            results = [latest_price.round(2)]
            
            current_date = data_perf.index[-1]

            for period_name, delta in periods.items():
                if isinstance(delta, dt.date):
                    period_date = delta
                else:
                    period_date = current_date - delta
                
                # If the period date is in the index, fetch the price directly
                if period_date in data_perf.index:
                    period_price = data_perf.loc[period_date]
                else:
                    # If the period date is not available, look for the closest earlier date
                    try:
                        period_price = data_perf.loc[data_perf.index <= period_date].iloc[-1]
                    except IndexError:
                        period_price = None
                
                if period_price is not None:
                    change = (latest_price - period_price) / period_price
                    results.append("{:.2%}".format(change))
                else:
                    results.append(None)

            # Calculate yearly high and low performance metrics
            one_year_ago = data_perf.index[-1] - dt.timedelta(weeks=52)
            one_year_data = data_perf.loc[one_year_ago:data_perf.index[-1]]

            yearly_high = one_year_data.max()
            yearly_low = one_year_data.min()

            vs_52_max = (latest_price - yearly_high) / yearly_high
            vs_52_min = (latest_price - yearly_low) / yearly_low

            results.extend(["{:.2%}".format(vs_52_max), "{:.2%}".format(vs_52_min)])
            performance_data.append([ticker] + results)

        performance_df = pd.DataFrame(performance_data, columns=window_names)

        return performance_df.sort_values(by='Ticker').set_index('Ticker')

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
    
    def get_correlation_table_window_x(self, df, value):
        '''
        This function creates a pandas dataframe. In it, correlations (pearson method) between different contracts are being calculated and shown, over different time horizons.
        '''
        end_date = self.get_end_date()
        df.index = pd.to_datetime(df.index)
        df = df[df.index <= end_date] 

        window_list = [15, 30, 90, 120, 180]
        tickerlist = list(df.columns)
        TICKER = value
        MULTP_TICKERS = [item for item in tickerlist if value not in item]

        dataframe = pd.DataFrame()
        for window in window_list:
            for i in MULTP_TICKERS:
                dataframe[f'{TICKER}_{i}_{window}'] = df[TICKER].rolling(window).corr(df[i])

        day_15 = dataframe.filter(regex='15')
        day_30 = dataframe.filter(regex='30')
        day_90 = dataframe.filter(regex='90')
        day_120 = dataframe.filter(regex='120')
        day_180 = dataframe.filter(regex='180')

        rolling_30d_max = [np.nanmax(dataframe[i]) for i in list(day_30.columns)]
        rolling_30d_min = [np.nanmin(dataframe[i]) for i in list(day_30.columns)]

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
        time_horizon_names = ['%sD' % (i) for i in time_horizons]

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
                        x=dataframe['%s_%s_%s' % (selected_ticker, ticker, t)].index,
                        y=dataframe['%s_%s_%s' % (selected_ticker, ticker, t)],
                        showlegend=False,
                        name=ticker
                    ), col=col, row=row)

        fig.update_layout(
            autosize=True,
            height=1000,
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
