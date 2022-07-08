import numpy as np
import os
import pandas as pd
import pyEX as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def getData(ticker):
    token = os.environ.get('IEX_TOKEN')
    # df = px.chartDF(ticker, timeframe='2y', token=token)
    df = px.chartDF(ticker, timeframe='max', token=token)

    close = df.close[::-1]
    high = df.high[::-1]
    low = df.low[::-1]
    volume = df.fVolume[::-1]
    return close, high, low, volume

def getATR(close, high, low, delta):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(high - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1).dropna()
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(delta).mean()
    std_atr = atr.rolling(delta).std()
    return atr, std_atr

def getSMA(data, delta):
    ma = data.rolling(delta).mean()
    return ma

def getActual_daily_rangeR(high, low, close, delta):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(high - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1).dropna()
    true_range = np.max(ranges, axis=1).dropna()
    adr = getSMA(true_range, delta)
    return adr

def getExpected_range(atr, std_atr, sma_adr):
    df = pd.DataFrame(columns=['atr', 'std_atr', 'sma_adr'])
    df['atr'] = atr
    df['std_atr'] = std_atr
    df['sma_adr'] = sma_adr
    df = df.dropna()
    # print(df.sma_adr, df.std_atr)
    pct_sma_adr_std_atr = (df.std_atr/df.sma_adr)*100
    # print(pct_sma_adr_std_atr)
    expected_range = df.atr * pct_sma_adr_std_atr
    return expected_range



if __name__ == "__main__":
    ticker = 'SPY'
    delta_ma = 10
    close, high, low, volume = getData(ticker)

    atr60, std_atr60 = getATR(close, high, low, 60)
    atr20, std_atr20 = getATR(close, high, low, 20)


    sma_adr = getActual_daily_rangeR(high, low, close, delta_ma)
    # sma_adr = getActual_daily_rangeR(high, low, close, delta_ma)

    expected_range_d60 = getExpected_range(atr60, std_atr60, 60)
    expected_range_d20 = getExpected_range(atr20, std_atr20, 20)

    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(x=close.index[-len(expected_range):], y=close[-len(expected_range):]))
    # fig.add_trace(go.Scatter(x=close.index[-len(expected_range):], y=expected_range), secondary_y=True)
    #



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index[-len(expected_range_d60):], y=close[-len(expected_range_d60):]))
    fig.add_trace(go.Scatter(x=close.index[-len(expected_range_d60):], y=expected_range_d60, yaxis='y2'))
    fig.add_trace(go.Scatter(x=close.index[-len(expected_range_d60):], y=expected_range_d20[-len(expected_range_d60):], yaxis='y3'))




    fig.show()

    # print(expected_range)
    # print(getSMA(close, delta_ma))