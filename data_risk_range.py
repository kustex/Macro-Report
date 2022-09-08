import datetime
import ib_insync as ibi
import os
import pandas
import random


def getDate_timedelta(delta):
    now = datetime.datetime.today()
    date = now - datetime.timedelta(weeks=delta)
    return date.strftime("%Y%m%d %H:%M:%S")
print(getDate_timedelta(3))

ib = ibi.IB()
conid = 320227571
symbol = 'QQQ'
exchange = 'SMART'
random_id = random.randint(0, 9999)
with ib.connect(host='127.0.0.1', port=7496, clientId=random_id):
    contract = ibi.Contract(symbol=symbol, conId=conid, exchange=exchange)
    df = ib.reqHistoricalData(contract, '',durationStr='3 W', barSizeSetting='1 min', useRTH=False, keepUpToDate=False, whatToShow='ADJUSTED_LAST')
    df = ibi.util.df(df)
print(df)