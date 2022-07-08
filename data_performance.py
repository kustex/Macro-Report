import time
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import os
import glob
import asyncio
import ib_insync as ibi
import itertools
import datetime
import re
import random
import pyEX as px
import plotly.graph_objects as go

from datetime import timedelta, date
from itertools import chain


class app_performance:
    def getTickerlists(self, filename):
        stocks = {'name': [], 'exchange': [], 'currency': []}
        cash = {'name': [], 'exchange': [], 'symbol': [], 'currency': []}
        index = {'name': [], 'exchange': [], 'currency': []}
        cont_futures = {'name': [], 'exchange': [], 'currency': []}
        conid = {'name': [], 'exchange': [], 'contract_id': []}
        securities = pd.read_csv(filename)
        for security in securities.iterrows():
            if security[1][1] == 'STK':
                stocks['name'].append(security[1][0])
                stocks['exchange'].append(security[1][2])
                stocks['currency'].append(security[1][5])
            elif security[1][1] == 'CASH':
                cash['name'].append(security[1][0])
                cash['exchange'].append(security[1][2])
                cash['symbol'].append(security[1][5])
                cash['currency'].append(security[1][7])
            elif security[1][1] == 'IND':
                index['name'].append(security[1][0])
                index['exchange'].append(security[1][2])
                index['currency'].append(security[1][5])
            elif security[1][1] == 'FUT':
                cont_futures['name'].append(security[1][0])
                cont_futures['exchange'].append(security[1][2])
                cont_futures['currency'].append(security[1][5])
            else:
                conid['name'].append(security[1][0])
                conid['exchange'].append(security[1][2])
                conid['contract_id'].append(security[1][8])

        stocks_df = pd.DataFrame.from_dict(stocks)
        index_df = pd.DataFrame.from_dict(index)
        cont_futures_df = pd.DataFrame.from_dict(cont_futures)
        contract_ids_df = pd.DataFrame.from_dict(conid)
        cash_df = pd.DataFrame.from_dict(cash)
        tickers = pd.concat([stocks_df, index_df, cont_futures_df, contract_ids_df, cash_df])['name']
        return stocks_df, cash_df, index_df, cont_futures_df, contract_ids_df, tickers

    def getContractlists(self, filename):
        stocks, cash, index, futures, contract_ids, tickers = self.getTickerlists(filename)
        c_stocks = [ibi.Stock(symbol=name, exchange=exchange, currency=currency) for name, exchange, currency in
                    zip(stocks['name'], stocks['exchange'], stocks['currency'])]
        c_cash = [ibi.Forex(pair=name, exchange=exchange, symbol=symbol, currency=currency) for
                  name, exchange, symbol, currency in
                  zip(cash['name'], cash['exchange'], cash['symbol'], cash['currency'])]
        c_index = [ibi.Index(symbol=name, exchange=exchange, currency=currency) for name, exchange, currency in
                   zip(index['name'], index['exchange'], index['currency'])]
        c_fut = [ibi.ContFuture(symbol=name, exchange=exchange, currency=currency) for name, exchange, currency in
                 zip(futures['name'], futures['exchange'], futures['currency'])]
        c_conid = [ibi.Contract(localSymbol=name, conId=contractid, exchange=exchange) for name, contractid, exchange in
                   zip(contract_ids['name'], contract_ids['contract_id'], contract_ids['exchange'])]
        contracts_adjusted_last = list(chain(c_stocks, c_index, c_fut, c_conid))
        contracts_midpoint = list(chain(c_cash))
        return contracts_adjusted_last, contracts_midpoint, tickers

    async def req_historical_data_async_adjusted_last(self, contract):
        hist_data = await self.ib.reqHistoricalDataAsync(contract, '', barSizeSetting='1 day', durationStr='1 Y',
                                                         whatToShow='ADJUSTED_LAST', useRTH=False)
        return hist_data

    async def req_historical_data_async_midpoint(self, contract):
        hist_data = await self.ib.reqHistoricalDataAsync(contract, '', barSizeSetting='1 day', durationStr='1 Y',
                                                         whatToShow='MIDPOINT', useRTH=False)
        return hist_data

    async def getHistoricaldata(self, filename):
        self.ib = ibi.IB()
        random_id = random.randint(0, 9999)
        with await self.ib.connectAsync(host='127.0.0.1', port=7496, clientId=random_id):
            contracts_adjusted_last, contracts_midpoint, tickers = self.getContractlists(filename)
            all_bars_adj_last = await asyncio.gather(*[
                self.req_historical_data_async_adjusted_last(contract)
                for contract in contracts_adjusted_last
            ])
            all_bars_midpoint = await asyncio.gather(*[
                self.req_historical_data_async_midpoint(contract)
                for contract in contracts_midpoint
            ])
            all_bars = all_bars_adj_last + all_bars_midpoint
            # self.ib.disconnect()
            return all_bars, tickers

    def get_df_all_data(self, filename):
        all_bars_total = []
        all_bars, tickers = asyncio.run(self.getHistoricaldata(filename))
        tickers = list(tickers)
        all_bars_total.extend(all_bars)
        df = pd.DataFrame()
        df['date'] = self.getDaterange(365)
        df = df.set_index('date')
        for ticker, bars in zip(tickers, all_bars_total):
            dataframe = ibi.util.df(bars).iloc[:, [0, 4]]
            dataframe = dataframe.set_index('date').rename(
                columns={'close': f'{ticker}'})
            df = df.join(dataframe)
        return df.interpolate(), tickers

    def get_df_all_data_iex(self, filename):
        token = os.environ.get('IEX_TOKEN')
        stocks, cash, index, futures, contract_ids, tickers = self.getTickerlists(filename)
        df = pd.DataFrame()
        for ticker in tickers:
            df[ticker] = px.chartDF(ticker, timeframe= '1y', token=token).close
        # print(df[::-1])
        return df[::-1], tickers

    def get_performance(self, data):
        data = data
        tickerlist = list(data.columns)
        tickerlist = " ".join(tickerlist)
        tickerlist = tickerlist.split()
        window_names = ['Ticker', 'Price', '1D', '1W', '3W', '1M', 'MTD', '3M', 'QTD', 'YTD', 'vs 52w max', 'vs 52w min']
        df = pd.DataFrame()
        len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd = self.get_lengths_periods()
        for ticker in tickerlist:
            data_perf = data[ticker]
            latest = data_perf[-1]
            range = [2, len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd]
            results = []
            for time in range:
                if data_perf[-time] < latest:
                    results.append((latest - data_perf[-time]) / latest)
                else:
                    results.append(-(data_perf[-time] - latest) / data_perf[-time])
            yearly_high = data_perf.max()
            yearly_low = data_perf.min()
            vs_52_max = -(yearly_high - latest) / yearly_high
            vs_52_min = (latest - yearly_low) / yearly_low
            results.append(vs_52_max)
            results.append(vs_52_min)
            results = ["{:.2%}".format(y) for y in results]
            results.insert(0, data_perf[-1].round(2))
            df[ticker] = results
        df = df.T.reset_index()
        df.columns = window_names
        return df.sort_values(by = 'Ticker')

    def get_correlation_table_window_x(self, df, value):
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

    def df_rates_spreads(self):
        names = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']
        tickers = ['T5YIE', 'BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'BAMLH0A0HYM2EY', 'DTB3', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
        df = pdr.get_data_fred('T10Y2Y')
        for tick, name in zip(tickers, names):
            df[tick] = pdr.get_data_fred(tick)
        df.columns = names
        return df.dropna()

    def df_performance_rates_spreads(self):
        dataframe = self.df_rates_spreads()
        # print(dataframe)
        window_names = ['Ticker', 'Price', '1D', '1W', '1M', '3M', 'vs 52w max', 'vs 52w min', 'vs 3Y ave', 'vs 5Y ave']
        df = pd.DataFrame()
        ticker_list = ['2Y-10Y Spread', '5Y Breakeven', 'HY-OAS', 'IG Spread', 'High Yield', '3M t-bill', '2Y t-note', '5Y t-note', '10Y t-note', '30Y t-note']
        len_week, len_3w, len_1m, len_mtd, len_3m, len_qtd, len_ytd = self.get_lengths_periods()
        for ticker in ticker_list:
            data = dataframe.loc[:, ticker]
            latest = data[-1]
            range = [2, len_week, len_1m, len_3m]
            results = []
            for time in range:
                if data[-time] < latest:
                    results.append(latest - data[-time])
                else:
                    results.append(-(data[-time] - latest))
            yearly_high = data[-252:].max()
            yearly_low = data[-252:].min()
            y3_ave = data[-756:].sum()/756
            y5_ave = data[-1260:].sum()/1260
            vs_52_max = -(yearly_high - latest)
            vs_52_min = (latest - yearly_low)

            if y3_ave < latest:
                vs_y3_ave = (latest - y3_ave)
            else:
                vs_y3_ave = (-y3_ave - latest)

            if y5_ave < latest:
                vs_y5_ave = (latest - y5_ave)
            else:
                vs_y5_ave = (-y5_ave - latest)
            results_vs = [vs_52_max, vs_52_min, vs_y3_ave, vs_y5_ave]
            for i in results_vs:
                results.append(i)
            results = ["{0:.1f}".format(y*100) for y in results]
            results.insert(0, data[-1].round(2))
            df[ticker] = results
        df = df.T.reset_index()
        df.columns = window_names
        return df

    def chart_rates_spreads(self, data, TICKER):
        x = data.index
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.array(data), name=TICKER))
        fig.update_xaxes(title='date')
        fig.update_yaxes(title='performance')
        fig.update_layout(
            title={
                'text': f"{TICKER} performance",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        return fig

    def getDaterange(self, numdays):
        end_date = datetime.datetime.today()
        delta = datetime.timedelta(days=numdays)
        start_date = end_date - delta
        date_list = pd.Series(pd.bdate_range(start_date, end_date))
        return date_list

    def get_yesterday(self):
        today = date.today()
        yesterday = today - timedelta(days=1)
        if today.weekday() == 6:
            yesterday = date.today() - timedelta(days=2)
        elif today.weekday() == 0:
            yesterday = date.today() - timedelta(days=3)
        return yesterday

    def get_lengths_periods(self):
        end_dt = self.get_yesterday()
        start_dt_ytd = date(2021, 12, 31)
        start_dt_qtd = date(2022, 6, 30)
        start_dt_3m = end_dt - timedelta(weeks=12)
        start_dt_mtd = end_dt.replace(day=1) - timedelta(days=1)
        start_dt_week = end_dt - timedelta(weeks=1)
        start_dt_3week = end_dt - timedelta(weeks=3)
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

    def daterange(self, date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)


# ap = app_performance()
# df = ap.get_df_all_data_iex('tickers/sectors.csv')
# print(df)