import numpy as np
import pandas as pd
import os
import requests
import csv
import yfinance as yf
import itertools
import plotly.graph_objects as go

from datetime import datetime, timedelta, date
from iexfinance.stocks import Stock, get_historical_data


class app_yf:

    def sort(self, list):
        list.sort()
        return list

    def getDaterange(self, numdays):
        end_date = datetime.datetime.today()
        delta = datetime.timedelta(days=numdays)
        start_date = end_date - delta
        date_list = pd.Series(pd.bdate_range(start_date, end_date))
        return date_list

    def get_df_all_data(self):
        tickers_list = []
        files = os.listdir('tickers')
        for file in files:
            securities = pd.read_csv('tickers/{}'.format(file))
            tickers_list.append(list(securities.columns))

        tickers_list = list(itertools.chain(*tickers_list))
        tickers = " ".join(tickers_list)

        end = datetime.today() - timedelta(days=1)
        start = end - timedelta(weeks=52)
        df = yf.download(tickers, start=start, end=end, group_by="ticker")
        date_index = df.index
        latest_date = date_index[-2]

        close_prices = pd.DataFrame()
        for ticker in tickers_list:
            close_prices[ticker] = np.array(df[ticker]['Close'])
        close_prices.index = date_index
        return close_prices, tickers_list, latest_date

    def get_df_all_data_iex(self):
        token = 'sk_1a591a5ab2b346b38240aaf0ecb96db9'
        files = os.listdir('tickers')
        df = pd.DataFrame()
        tickerlist = []
        for file in files:
            securities = pd.read_csv('tickers/{}'.format(file))
            tickers = list(securities.columns)
            tickerlist.append(tickers)
            for tick in tickers:
                data = get_historical_data(tick, start, end, close_only=True, output_format='pandas', token=token)
                df[f'{tick}_close'] = np.array(data['close'])
        return df, tickers

    def daterange(self, date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

    def get_yesterday(self):
        today = date.today()
        yesterday = today - timedelta(days=1)
        if today.weekday() == 6:
            yesterday = date.today() - timedelta(days=2)
        elif today.weekday() == 0:
            yesterday = date.today() - timedelta(days=3)
        return yesterday

    def get_lengths_periods(self):
        end_dt = date.today()
        start_dt_ytd = date(2021, 12, 31)
        start_dt_qtd = date(2022, 3, 31)
        start_dt_3m = end_dt - timedelta(weeks=12)
        start_dt_mtd = end_dt.replace(day=1)
        start_dt_week = end_dt - timedelta(weeks=1)
        start_dt_3week = end_dt - timedelta(weeks=3)
        # print(start_dt_3week)
        start_dt_4week = end_dt - timedelta(weeks=4)
        len_week, len_3w, len_4w, len_mtd, len_3m, len_qtd, len_ytd = 0, 0, 0, 0, 0, 0, 0
        weekenddays = [5, 6]
        for dt in self.daterange(start_dt_ytd, end_dt):
            if dt.weekday() not in weekenddays:
                len_ytd += 1
        for dt in self.daterange(start_dt_qtd, end_dt):
            if dt.weekday() not in weekenddays:
                len_qtd += 1
        for dt in self.daterange(start_dt_mtd, end_dt):
            if dt.weekday() not in weekenddays:
                len_mtd += 1
        for dt in self.daterange(start_dt_3week, end_dt):
            if dt.weekday() not in weekenddays:
                len_3w += 1
        for dt in self.daterange(start_dt_4week, end_dt):
            if dt.weekday() not in weekenddays:
                len_4w += 1
        for dt in self.daterange(start_dt_week, end_dt):
            if dt.weekday() not in weekenddays:
                len_week += 1
        for dt in self.daterange(start_dt_3m, end_dt):
            if dt.weekday() not in weekenddays:
                len_3m += 1
        return len_week, len_3w, len_4w, len_mtd, len_3m, len_qtd, len_ytd

    def get_correlation_table_window_x(self, df, TICKER, MULTP_TICKERS):
        window_list = [15, 30, 90, 120, 180]
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

    def get_correlation_chart(self, df, TICKER, WINDOW):
        fig = go.Figure()
        x = df.index
        for i in list(df.columns):
            fig.add_trace(go.Scatter(x=x, y=df[i], name=i[len(TICKER)+1:-3]))
        fig.update_xaxes(title='date')
        fig.update_yaxes(title='correlation')
        fig.update_layout(
            yaxis_range=[-1, 1],
            title={
                'text': f"{TICKER} {WINDOW}D correlations",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        return fig

    def get_performance(self, dataframe, TICKER, MULTP_TICKERS):
        window_names = ['Ticker', 'Price', '1D', '1W', '3W', '1M', 'MTD', '3M', 'QTD', 'YTD', 'vs 52w max', 'vs 52w min']
        df = pd.DataFrame()
        ticker_list = list(TICKER.split(" ")) + MULTP_TICKERS
        len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd = self.get_lengths_periods()
        for ticker in ticker_list:
            data = dataframe.loc[:, ticker]
            latest = data[-2]
            range = [3, len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd]
            results = []
            for time in range:
                if data[-time] < latest:
                    results.append((latest - data[-time]) / latest)
                else:
                    results.append(-(data[-time] - latest) / data[-time])
                # print(data[-time], latest)
            yearly_high = data.max()
            yearly_low = data.min()
            vs_52_max = -(yearly_high - latest) / yearly_high
            vs_52_min = (latest - yearly_low) / yearly_low
            results.append(vs_52_max)
            results.append(vs_52_min)
            results = ["{:.2%}".format(y) for y in results]
            results.insert(0, data[-2].round(2))
            df[ticker] = results
        df = df.T.reset_index()
        df.columns = window_names
        return df

    def relative_performance(self, df, TICKER, MULTP_TICKERS):
        window_names_rel_performance = [f'{TICKER} vs', '1D', '1W', '3W', '1M', 'MTD', 'QTD', 'YTD']
        len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd = self.get_lengths_periods()
        range = [3, len_week, len_3w, len_1m, len_mtd, len_qtd, len_ytd]
        df_new = pd.DataFrame()
        for i in MULTP_TICKERS:
            results = []
            latest = df[i].iloc[-2]
            latest_relative = df[TICKER].iloc[-2]
            for time in range:
                results.append(((df[TICKER].iloc[-time])/ latest_relative) - (df[i].iloc[-time]) / latest)
            results = [i.tolist() for i in results]
            results = ["{:.2%}".format(y) for y in results]
            df_new[i] = results
        df_new = df_new.T.reset_index()
        df_new.columns = window_names_rel_performance
        return df_new

